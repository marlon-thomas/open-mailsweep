from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .audit import AuditLog
from .config import Policy, Settings
from .decision import decide
from .gmail_client import GmailClient, GmailRateLimiter, GmailRateLimitError
from .provider import create_mail_provider, provider_is_ready, provider_needs_auth_hint
from .jev_client import JevClient
from .local_classifier import (
    ACTION_CLASSES,
    LocalClassifier,
    Prediction,
    example_from_message,
)
from .models import Classification, Decision, EmailMessage
from .safety import SafetyRules
from .state_store import StateStore, extract_message_features
from .training_data import TrainingDataset, TrainingExample, bootstrap_from_history
from .unsubscribe import UnsubscribeClient


ACTION_LABELS = {
    "keep": "Keep in Inbox",
    "read_later": "Read Later",
    "archive": "Archive",
    "trash": "Clean / Trash",
    "unsubscribe_trash": "Unsubscribe + Clean",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(value: str, limit: int = 90) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


@dataclass
class ClassifyCandidate:
    item_id: int
    message: EmailMessage
    safety_signals: list[str]
    plan: Any


class WorkerStatus:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {
            "scanner": "starting",
            "classifier": "starting",
            "executor": "starting",
            "executor_workers": 0,
            "executor_active": 0,
            "executor_current_subjects": [],
            "last_scan_started": None,
            "last_scan_completed": None,
            "last_scan_error": None,
            "last_progress_at": None,
            "scan_cycle": 0,
            "scan_page": 0,
            "scan_page_count": 0,
            "scan_seen": 0,
            "scan_estimate": None,
            "scan_new": 0,
            "scan_known": 0,
            "scan_protected": 0,
            "scan_rule_matched": 0,
            "scan_discovered": 0,
            "scanner_current_subject": None,
            "classifier_current_subject": None,
            "classifier_batch_size": 0,
            "classifier_total": 0,
            "last_action": None,
            "model_loaded": False,
            "gmail_api_state": "ready",
            "gmail_api_wait_seconds": 0.0,
            "gmail_api_operation": None,
            "gmail_api_detail": None,
            "gmail_api_last_limit_at": None,
            "gmail_api_units_per_minute": None,
            "local_bootstrap": None,
            "local_classifier": None,
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self._data.update(values)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


class OpenMailSweepService:
    """Continuous rule-first OpenMailSweep service.

    Gmail intake applies hard protections and learned rules immediately. Only
    genuinely unknown mail enters the persistent ``discovered`` queue for the
    local classifier.
    Classification and action execution remain independent workers, so slow local
    inference never blocks Gmail pagination, Pending decisions, or learned-rule
    actions.
    """

    def __init__(self, settings: Settings, policy: Policy, store: StateStore, audit: AuditLog):
        self.settings = settings
        self.policy = policy
        self.store = store
        self.audit = audit
        self.safety = SafetyRules(policy)
        self.unsubscriber = UnsubscribeClient(policy.unsubscribe)
        self.status = WorkerStatus()
        self.status.update(gmail_api_units_per_minute=settings.gmail_quota_units_per_minute)
        self.gmail_limiter = GmailRateLimiter(
            settings.gmail_quota_units_per_minute,
            settings.gmail_quota_burst_units,
            event_callback=self._gmail_rate_event,
        )
        # Wake signals are counting semaphores rather than auto-reset events:
        # every set/release is preserved until a worker consumes it, so a wake
        # request can never be erased by another thread's clear() (a classic
        # lost-wakeup race with Event.wait()+Event.clear() among several
        # workers sharing one event).
        self.stop_event = threading.Event()
        self.scan_wake = threading.Semaphore()
        self.classifier_wake = threading.Semaphore()
        self.action_wake = threading.Semaphore()
        self._scanner_thread: threading.Thread | None = None
        self._classifier_thread: threading.Thread | None = None
        self._action_threads: list[threading.Thread] = []
        self._scanner_gmail: Any = None
        self._classifier_gmail: Any = None
        self._classifier = None
        self._prior_correspondent_cache: dict[str, bool] = {}
        self._prior_correspondent_cache_lock = threading.Lock()
        self._started = False
        self._paused_flag = self.store.get_app_state("paused", "0") == "1"
        # Serializes GmailClient construction. Building a client can refresh
        # the OAuth token and rewrite token.json; the action pool constructs
        # its clients concurrently, so those writes must not interleave.
        self._client_init_lock = threading.Lock()
        self._action_state_lock = threading.RLock()
        self._action_active: dict[int, str] = {}
        self._unsubscribe_lock_guard = threading.Lock()
        self._unsubscribe_locks: dict[str, dict[str, Any]] = {}
        # Local personalised classifier. Stateful parts are created lazily in
        # start()/bootstrap so importing this module or constructing the
        # service never touches the filesystem or loads ML dependencies.
        self.uses_local_classifier = settings.classifier == "local"
        self.training = None
        self.local = None

    def report(self, text: str) -> None:
        print(text, flush=True)

    def _gmail_rate_event(self, event: dict[str, Any]) -> None:
        state = str(event.get("state") or "ready")
        wait_seconds = float(event.get("wait_seconds") or 0.0)
        operation = event.get("operation")
        detail = event.get("detail")
        values: dict[str, Any] = {
            "gmail_api_state": state,
            "gmail_api_wait_seconds": wait_seconds,
            "gmail_api_operation": operation,
            "gmail_api_detail": detail,
            "last_progress_at": _now(),
        }
        if state in {"backoff", "error"}:
            values["gmail_api_last_limit_at"] = _now()
        self.status.update(**values)
        if state == "backoff":
            attempt = event.get("attempt")
            maximum = event.get("max_attempts")
            suffix = f" attempt {attempt}/{maximum}" if attempt is not None else ""
            self.report(
                f"[gmail] rate limit during {operation}; backing off {wait_seconds:.1f}s{suffix}"
            )
        elif state == "ready" and detail and "recovered" in str(detail).lower():
            self.report(f"[gmail] {detail}")

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        recovered = self.store.recover_processing()
        if recovered:
            self.report(f"[worker] recovered {recovered} interrupted queue item(s)")
        released = self.store.apply_all_rules_to_open_items()
        if released:
            self.report(
                f"[worker] routed {released} existing queued message(s) by learned rule before classification"
            )
        self._scanner_thread = threading.Thread(target=self._scanner_loop, name="openmailsweep-intake", daemon=True)
        self._classifier_thread = threading.Thread(target=self._classifier_loop, name="openmailsweep-classifier", daemon=True)
        self._action_threads = [
            threading.Thread(
                target=self._action_loop,
                args=(worker_id,),
                name=f"openmailsweep-actions-{worker_id}",
                daemon=True,
            )
            for worker_id in range(1, self.settings.action_workers + 1)
        ]
        self.status.update(executor_workers=self.settings.action_workers, executor_active=0)
        self._scanner_thread.start()
        self._classifier_thread.start()
        for thread in self._action_threads:
            thread.start()
        self.classifier_wake.release()
        self.action_wake.release()

    def stop(self) -> None:
        self._started = False
        self.stop_event.set()
        # Unblock any worker parked on a wake semaphore so join() is quick.
        self.scan_wake.release()
        self.classifier_wake.release()
        for _ in range(self.settings.action_workers):
            self.action_wake.release()
        threads = [self._scanner_thread, self._classifier_thread, *self._action_threads]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5)

    def request_scan(self) -> None:
        self.scan_wake.release()

    def wake_actions(self) -> None:
        self.action_wake.release()

    def wake_classifier(self) -> None:
        self.classifier_wake.release()

    def pause(self) -> None:
        self._paused_flag = True
        self.store.set_app_state("paused", "1")
        self.status.update(scanner="paused", last_progress_at=_now())

    def resume(self) -> None:
        self._paused_flag = False
        self.store.set_app_state("paused", "0")
        self.request_scan()

    def is_paused(self) -> bool:
        return self._paused_flag

    def _new_provider_client(self):
        """Build a Gmail/Yahoo client for the calling thread.

        Construction is serialized: Gmail token refresh rewrites token.json,
        and the action pool builds clients concurrently at startup.
        """
        with self._client_init_lock:
            return create_mail_provider(
                self.settings,
                interactive=False,
                limiter=self.gmail_limiter,
                rate_event=self._gmail_rate_event,
            )

    def _provider_for_scanner(self):
        if self._scanner_gmail is None:
            self._scanner_gmail = self._new_provider_client()
        return self._scanner_gmail

    def _provider_for_classifier(self):
        if self._classifier_gmail is None:
            self._classifier_gmail = self._new_provider_client()
        return self._classifier_gmail

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier
        self.status.update(classifier="loading-model", last_progress_at=_now())
        self._classifier = JevClient(
            self.settings.jev_api_key,
            self.settings.jev_model,
            self.settings.jev_base_url,
            max_body_chars=self.policy.classification.max_body_chars,
        )
        self.status.update(model_loaded=True, classifier="idle", last_progress_at=_now())
        return self._classifier

    def _prior_correspondent(self, gmail: Any, address: str) -> bool:
        address = (address or "").strip().lower()
        if not address or not self.policy.protected.protect_prior_correspondents:
            return False
        with self._prior_correspondent_cache_lock:
            if address not in self._prior_correspondent_cache:
                self._prior_correspondent_cache[address] = gmail.has_sent_to_address(address)
            return self._prior_correspondent_cache[address]

    def _scanner_loop(self) -> None:
        if not provider_is_ready(self.settings):
            self.status.update(scanner="needs-auth", last_scan_error=provider_needs_auth_hint(self.settings))
        while not self.stop_event.is_set():
            if self.is_paused():
                self.status.update(scanner="paused", last_progress_at=_now())
                # Sleep instead of consuming a wake token: requests made while
                # paused must still trigger a scan after resume().
                time.sleep(2.0)
                continue
            try:
                self.scan_once()
            except GmailRateLimitError as exc:
                # The request layer already backed off/retried. Keep the service
                # healthy and resume on the next pass rather than spinning.
                self.status.update(
                    scanner="rate-limited",
                    last_scan_error=str(exc),
                    last_scan_completed=_now(),
                    last_progress_at=_now(),
                )
                self.report(f"[intake] Gmail quota pause: {exc}")
            except Exception as exc:  # keep the long-running service alive
                self.status.update(
                    scanner="error",
                    last_scan_error=str(exc),
                    last_scan_completed=_now(),
                    last_progress_at=_now(),
                )
                self.report(f"[intake] ERROR: {exc}")
            # Block until the next interval or an explicit scan request. Wake
            # tokens released meanwhile are consumed here, so nothing is lost.
            self.scan_wake.acquire(timeout=self.settings.scan_interval_seconds)

    def _classifier_loop(self) -> None:
        self.status.update(classifier="idle")
        while not self.stop_event.is_set():
            if not provider_is_ready(self.settings):
                self.status.update(classifier="needs-auth")
                time.sleep(2.0)
                continue

            if self.uses_local_classifier and self.local is None:
                self._bootstrap_local_classifier()
                self.status.update(local_classifier=self.local.public_state() if self.local else None)

            if self.uses_local_classifier and self.local is not None:
                self._maybe_periodic_retrain()

            claimed = self.store.claim_discovered_batch(self.settings.classify_batch_size)
            if not claimed:
                self.status.update(classifier="idle", classifier_current_subject=None, classifier_batch_size=0)
                self.classifier_wake.acquire(timeout=1.0)
                continue

            try:
                self._classify_claimed(claimed)
            except Exception as exc:
                # Do not lose claimed work if an unexpected worker-level error occurs.
                for item in claimed:
                    self.store.return_to_discovered(int(item["id"]), str(exc)[:2000])
                self.status.update(classifier="error", classifier_current_subject=None, last_progress_at=_now())
                self.report(f"[classifier] ERROR: {exc}")
                time.sleep(1.0)

    # ---------------------------------------------------------------- local classifier

    def _bootstrap_local_classifier(self) -> None:
        """Train-or-load the personalised model before classifying unknowns.

        Runs on the classifier thread only. Gmail intake, Pending interaction,
        and action workers continue meanwhile; unknown messages simply stay in
        the persistent discovered queue until the model is READY (spec 8.1).
        """
        if not self.uses_local_classifier or self.local is not None:
            return
        self.status.update(classifier="bootstrapping", last_progress_at=_now())
        try:
            training = TrainingDataset(self.settings.state_db)
            local = LocalClassifier(self.settings, self.policy, report=self.report)
            seeded = bootstrap_from_history(
                self.store, training, include_rules=self.settings.local_bootstrap_from_rules
            )
            revision = self.store.data_revision()
            self.report(
                f"[classifier-local] bootstrap starting rules_revision={revision}"
                f" seeded={seeded} examples={training.count()}"
            )
            examples = self._dataset_examples(training)
            state = local.bootstrap(examples, revision)
            try:
                saved = json.loads(self.store.get_app_state("local_counters", "{}"))
                if isinstance(saved, dict):
                    for key in local.counters:
                        if key in saved:
                            local.counters[key] = int(saved[key])
            except Exception:
                pass
            self.training = training
            self.local = local
            self.status.update(
                classifier="idle",
                local_bootstrap=state,
                local_classifier=local.public_state(),
                last_progress_at=_now(),
            )
            self._reeval_pending_sweep("startup")
        except Exception as exc:
            self.report(f"[classifier-local] bootstrap FAILED; unknowns stay queued: {exc}")
            self.status.update(classifier="idle", last_progress_at=_now())

    @staticmethod
    def _dataset_examples(training: TrainingDataset) -> list[Any]:
        # Content features only: the stored rows already carry subject,
        # snippet, sender domain, and structural signals. Per-sender history is
        # evidence for Pending explanations, never a training feature (the
        # rules table already handles known sources mechanically).
        return training.all_examples()

    def _resync_dataset(self) -> None:
        if self.training is None:
            return
        bootstrap_from_history(
            self.store, self.training, include_rules=self.settings.local_bootstrap_from_rules
        )
        live_sources = {
            rule.get("source_message_id")
            for rule in self.store.list_rules()
            if rule.get("source_message_id")
        }
        self.training.prune_rule_bootstrap(live_sources)

    def _maybe_periodic_retrain(self) -> None:
        if self.local is None or not self.local.bootstrapped:
            return
        due, reason = self.local.retrain_due(self.training.count() if self.training else 0)
        if not due:
            return
        try:
            self._resync_dataset()
            examples = self._dataset_examples(self.training)
            thresholds_before = self.local.current_thresholds()
            ok = self.local.full_retrain(examples, self.store.data_revision(), reason)
            if ok:
                self.local.reset_incremental_counter_after_full()
                if self.local.current_thresholds() != thresholds_before:
                    self._reeval_pending_sweep(f"threshold update after retrain ({reason})")
        except Exception as exc:
            self.report(f"[classifier-local] periodic retrain error: {exc}")
        self.status.update(local_classifier=self.local.public_state(), last_progress_at=_now())

    @staticmethod
    def _is_permanent_message_missing(exc: Exception) -> bool:
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status == 404:
            return True
        text = str(exc).lower()
        return "requested entity was not found" in text or "notfound" in text

    @staticmethod
    def _mixed_history(history: dict[str, int]) -> bool:
        labels = {
            key.split("hist_sender_", 1)[-1].split("hist_list_", 1)[-1]
            for key, count in (history or {}).items()
            if count > 0
        }
        return len(labels) > 1

    def _reeval_pending_sweep(self, trigger: str) -> None:
        """Re-run the classifier over every unanswered Pending item.

        Runs on the classifier thread at startup and whenever a retrain changes
        the effective gates: a threshold calibrated *after* a message went
        Pending should release the messages that the new evidence now clears.
        Pure CPU/DB work (no Gmail calls) - the action executor re-verifies
        hard protection before mutating anything, because these items may have
        waited long enough for replies or star flags to appear.
        """
        if (
            self.local is None
            or self.training is None
            or not self.local.bootstrapped
            or self.local.example_count == 0
        ):
            return
        rows = self.store.list_all_pending()
        if not rows:
            return
        from .models import UnsubscribePlan

        started = time.perf_counter()
        released = 0
        for index, row in enumerate(rows):
            if self.stop_event.is_set():
                break
            if index and index % 200 == 0:
                self.status.update(last_progress_at=_now())
            try:
                features = json.loads(row.get("features_json") or "{}")
            except Exception:
                features = {}
            features["body_excerpt"] = row.get("body_excerpt") or features.get("body_excerpt") or ""
            try:
                signals = json.loads(row.get("safety_signals_json") or "[]")
            except Exception:
                signals = []
            history = self.training.history_for(
                sender_address=row.get("sender_address") or "",
                list_id=row.get("list_id") or "",
                exclude_message_id=row.get("message_id") or "",
            )
            example = TrainingExample(
                message_id=row["message_id"],
                label="",
                sender=row.get("sender") or "",
                sender_address=row.get("sender_address") or "",
                subject=row.get("subject") or "",
                snippet=row.get("snippet") or "",
                list_id=row.get("list_id") or "",
                features=features,
                history=history,
            )
            prediction = self.local.predict(example)
            plan = UnsubscribePlan(
                method=str(row.get("unsubscribe_method") or "none"),
                one_click=bool(features.get("one_click")),
                one_click_auto_eligible=bool(features.get("one_click")),
                auto_eligible=bool(row.get("unsubscribe_auto_eligible")),
                reason="stored",
            )
            gate = self.local.gate(
                prediction,
                plan=plan,
                safety_signals=[s for s in signals if isinstance(s, str)],
                mixed_history=self._mixed_history(history),
            )
            if gate.mode == "actionable" and gate.action:
                self.store.set_actionable(
                    int(row["id"]),
                    gate.action,
                    source="local-classifier-reeval",
                    reason=f"re-scored after {trigger}: {gate.reason}",
                )
                released += 1
        if released:
            self.local.counters["reeval_released"] = int(self.local.counters.get("reeval_released", 0)) + released
            self.action_wake.release()
        self._persist_local_counters()
        self.report(
            f"[classifier-local] pending re-score ({trigger}): scanned={len(rows)}"
            f" released={released} duration={time.perf_counter() - started:.1f}s"
        )
        self.status.update(local_classifier=self.local.public_state(), last_progress_at=_now())

    def _classifier_example(self, msg: EmailMessage, plan: Any) -> Any:
        history: dict[str, int] = {}
        if self.training is not None:
            # History is attached for Pending *explanations* and the mixed-source
            # safety brake only. It never reaches build_feature_text.
            history = self.training.history_for(
                sender_address=msg.sender_address, list_id=msg.headers.get("list-id")
            )
        return example_from_message(msg, plan, history=history)

    def handle_user_answer(self, item_id: int, action: str, scope_type: str) -> None:
        """Online learning hook for the web thread (spec 8.2/28).

        Persists the confirmed decision as a training example, applies an
        immediate incremental model update, and wakes the classifier so similar
        queued mail is re-evaluated against the fresher model.
        """
        if not self.uses_local_classifier or self.training is None or self.local is None:
            return
        try:
            row = self.store.get_item(item_id)
            if not row:
                return
            try:
                features = json.loads(row.get("features_json") or "{}")
            except Exception:
                features = {}
            # The live full body is gone by answer time; use the excerpt the
            # classifier stage persisted so the model trains on actual content.
            features["body_excerpt"] = row.get("body_excerpt") or ""
            corrected = self.training.upsert_example(
                message_id=row["message_id"],
                label=action,
                sender=row.get("sender") or "",
                sender_address=row.get("sender_address") or "",
                subject=row.get("subject") or "",
                snippet=row.get("snippet") or "",
                list_id=row.get("list_id") or "",
                features=features,
                source="pending-answer",
            )
            example = TrainingExample(
                message_id=row["message_id"],
                label=action,
                sender=row.get("sender") or "",
                sender_address=row.get("sender_address") or "",
                subject=row.get("subject") or "",
                snippet=row.get("snippet") or "",
                list_id=row.get("list_id") or "",
                features=features,
                history=self.training.history_for(
                    sender_address=row.get("sender_address") or "",
                    list_id=row.get("list_id") or "",
                    exclude_message_id=row["message_id"],
                ),
                weight=2.0 if corrected else 1.0,
                corrected=corrected,
            )
            self.local.incremental_update([example], data_revision=self.store.data_revision())
            self.report(
                f"[classifier-local] training update examples={self.local.example_count}"
                f" action={action} corrected={int(corrected)}"
            )
            self.classifier_wake.release()
        except Exception as exc:
            self.report(f"[classifier-local] online update failed (decision still applied): {exc}")

    def handle_rule_change(self) -> None:
        if self.local is None or not self.settings.local_retrain_on_rule_change:
            return
        self.local.retrain_requested = True
        self.classifier_wake.release()

    def _classify_claimed_locally(self, remaining: list[ClassifyCandidate]) -> list[ClassifyCandidate]:
        """Route unknown candidates through the personalised local model.

        Exact learned rules are re-checked first; a newly learned rule always
        beats the classifier.
        """
        model_ready = (
            self.local is not None and self.local.bootstrapped and self.local.example_count > 0
        )
        for candidate in remaining:
            current = self.store.get_item(candidate.item_id)
            if not current or current.get("status") != "classifying":
                continue
            rule = self.store.matching_rule_for_message(candidate.message)
            if rule is not None:
                self.store.set_actionable(
                    candidate.item_id,
                    rule["action"],
                    source="learned-rule",
                    reason=f"learned {rule['scope_type']} rule",
                    rule_id=int(rule["id"]),
                )
                self.action_wake.release()
                continue

            if not model_ready:
                self._local_pending(candidate, None, None)
                continue

            example = self._classifier_example(candidate.message, candidate.plan)
            # Sender/list history is NOT a model feature (identity memorising is
            # the rules table's job). It is only consulted as a brake: a source
            # with genuinely mixed previous decisions is never auto-cleaned.
            mixed_history = self._mixed_history(example.history)
            prediction = self.local.predict(example)
            gate = self.local.gate(
                prediction,
                plan=candidate.plan,
                safety_signals=candidate.safety_signals,
                mixed_history=mixed_history,
            )
            self.report(
                f"[classifier-local] {candidate.message.id[:10]} predicted={prediction.action}"
                f" conf={prediction.confidence:.2f} latency={prediction.latency_ms:.0f}ms => {gate.mode}"
            )
            if gate.mode == "actionable" and gate.action:
                classification = self._local_classification(prediction)
                self.audit.record(
                    candidate.message,
                    Decision(gate.action, f"local classifier: {gate.reason}", False, classification),
                    candidate.plan,
                    None,
                )
                self.store.set_actionable(
                    candidate.item_id,
                    gate.action,
                    source="local-classifier",
                    reason=gate.reason,
                )
                self.local.counters["auto_actioned"] += 1
                self.action_wake.release()
                self.report(
                    f"[classifier-local] auto -> {gate.action} | conf={prediction.confidence:.2f}"
                    f" | {_short(candidate.message.subject)}"
                )
                continue

            self._local_pending(candidate, prediction, gate)

        if self.local is not None:
            self._persist_local_counters()
            self.status.update(local_classifier=self.local.public_state(), last_progress_at=_now())

    def _persist_local_counters(self) -> None:
        if self.local is not None:
            try:
                self.store.set_app_state("local_counters", json.dumps(self.local.counters))
            except Exception:
                pass

    def _local_pending(self, candidate: ClassifyCandidate, prediction: Prediction | None, gate: Any) -> None:
        classification = self._local_classification(prediction) if prediction else None
        reason = gate.reason if gate else "cold start: waiting for confirmed decisions"
        self.audit.record(
            candidate.message,
            Decision("review", f"local classifier: {reason}", False, classification),
            candidate.plan,
            None,
        )
        proposed = gate.action if gate is not None and gate.action in ACTION_CLASSES else None
        if proposed and prediction is not None:
            question = (
                f"The local classifier suggests {ACTION_LABELS.get(proposed, proposed)}"
                f" ({prediction.confidence:.0%} confidence), below its auto-action threshold."
                " What should OpenMailSweep do with this source?"
            )
        else:
            question = "OpenMailSweep is still learning your preferences. What should it do with this source?"
        self.store.mark_pending(
            candidate.item_id,
            classification,
            reason=reason,
            question=question,
            proposed_action=proposed,
            safety_signals=candidate.safety_signals,
        )
        if self.local is not None:
            self.local.counters["pending"] += 1
        self.report(f"[classifier-local] pending | {reason} | {_short(candidate.message.subject)}")

    @staticmethod
    def _local_classification(prediction: Prediction) -> Classification:
        probs = prediction.probabilities or {}
        return Classification(
            category=prediction.action or "uncertain",
            confidence=round(float(prediction.confidence), 4),
            spam_probability=round(float(probs.get("trash", 0.0)), 4),
            bulk_probability=round(float(probs.get("unsubscribe_trash", 0.0) + probs.get("archive", 0.0)), 4),
            useful_probability=round(float(probs.get("keep", 0.0)), 4),
            importance_score=round(float(probs.get("keep", 0.0)) * 5.0, 3),
            raw={
                "_backend": "local-classifier",
                "probabilities": probs,
                "evidence": prediction.evidence,
                "latency_ms": prediction.latency_ms,
            },
        )

    def _set_action_worker(self, worker_id: int, subject: str | None) -> None:
        with self._action_state_lock:
            if subject:
                self._action_active[worker_id] = subject
            else:
                self._action_active.pop(worker_id, None)
            active = len(self._action_active)
            subjects = [self._action_active[k] for k in sorted(self._action_active)]
            self.status.update(
                executor="processing" if active else "idle",
                executor_workers=self.settings.action_workers,
                executor_active=active,
                executor_current_subject=subjects[0] if subjects else None,
                executor_current_subjects=subjects,
                last_progress_at=_now(),
            )

    def _action_loop(self, worker_id: int) -> None:
        gmail: GmailClient | None = None
        self._set_action_worker(worker_id, None)
        while not self.stop_event.is_set():
            if not provider_is_ready(self.settings):
                self.status.update(executor="needs-auth")
                time.sleep(2.0)
                continue
            if gmail is None:
                # google-api-python-client resources are not shared across action
                # threads. Every executor has its own Gmail client while all of
                # them share the same weighted quota limiter.
                gmail = self._new_provider_client()
            item = self.store.claim_next_action()
            if item is None:
                self._set_action_worker(worker_id, None)
                self.action_wake.acquire(timeout=self.settings.action_poll_seconds)
                continue
            subject = item.get("subject") or "(no subject)"
            self._set_action_worker(worker_id, subject)
            try:
                self._execute_action(item, gmail, worker_id=worker_id)
            except GmailRateLimitError as exc:
                self.store.fail_action(int(item["id"]), str(exc), retry=True)
                self.status.update(last_action=f"RETRY worker {worker_id}: {_short(subject)}")
                self.report(f"[action:{worker_id}] Gmail quota pause; requeued item={item['id']}")
                time.sleep(2.0)
            except Exception as exc:
                self.store.fail_action(int(item["id"]), str(exc))
                self.status.update(last_action=f"FAILED worker {worker_id}: {_short(subject)}")
                self.report(f"[action:{worker_id}] ERROR item={item['id']}: {exc}")
            finally:
                self._set_action_worker(worker_id, None)

    def _unsubscribe_key(self, item: dict[str, Any], message: EmailMessage) -> str:
        list_id = (item.get("list_id") or message.headers.get("list-id") or "").strip().lower()
        if list_id.startswith("<") and list_id.endswith(">"):
            list_id = list_id[1:-1].strip()
        if list_id:
            return f"list:{list_id}"
        sender = (item.get("sender_address") or message.sender_address or "").strip().lower()
        if sender:
            return f"sender:{sender}"
        return f"message:{message.id}"

    @contextmanager
    def _unsubscribe_lock(self, dedupe_key: str):
        """Serialize unsubscribe work for one mailing-list/sender key.

        Lock entries are reference-counted and evicted once idle so the registry
        does not grow without bound over a long-running service. Eviction only
        happens when no other thread holds or waits for the entry, and always
        after the holder's work is complete, so two workers can never end up
        using different locks for the same key.
        """
        with self._unsubscribe_lock_guard:
            entry = self._unsubscribe_locks.get(dedupe_key)
            if entry is None:
                entry = {"lock": threading.Lock(), "refs": 0}
                self._unsubscribe_locks[dedupe_key] = entry
            entry["refs"] += 1
            lock = entry["lock"]
        with lock:
            try:
                yield
            finally:
                with self._unsubscribe_lock_guard:
                    entry["refs"] -= 1
                    if entry["refs"] <= 0:
                        self._unsubscribe_locks.pop(dedupe_key, None)

    def scan_once(self) -> None:
        """Stream every message matching the Gmail query, without a total cap."""
        if not provider_is_ready(self.settings):
            self.status.update(scanner="needs-auth", last_scan_error=provider_needs_auth_hint(self.settings))
            return

        gmail = self._provider_for_scanner()
        started = time.perf_counter()
        previous = self.status.snapshot()
        cycle = int(previous.get("scan_cycle") or 0) + 1
        counters = {
            "seen": 0,
            "new": 0,
            "known": 0,
            "protected": 0,
            "rule_matched": 0,
            "discovered": 0,
        }
        self.status.update(
            scanner="fetching",
            last_scan_started=_now(),
            last_scan_error=None,
            last_progress_at=_now(),
            scan_cycle=cycle,
            scan_page=0,
            scan_page_count=0,
            scan_seen=0,
            scan_estimate=None,
            scan_new=0,
            scan_known=0,
            scan_protected=0,
            scan_rule_matched=0,
            scan_discovered=0,
            scanner_current_subject="Starting mail intake…",
        )
        self.report(
            f"[intake] cycle {cycle} started; query={self.settings.scan_query!r}; "
            f"page_size={self.settings.fetch_batch_size}; no total message cap"
        )

        page_count = 0
        for page_count, (ids, estimate) in enumerate(
            gmail.iter_message_id_pages(self.settings.scan_query, self.settings.fetch_batch_size), start=1
        ):
            if self.stop_event.is_set() or self.is_paused():
                break
            self.status.update(
                scanner="fetching",
                scan_page=page_count,
                scan_page_count=len(ids),
                scan_estimate=estimate,
                scanner_current_subject=f"Fetching mail batch {page_count} ({len(ids)} IDs)",
                last_progress_at=_now(),
            )
            self.report(
                f"[intake] page {page_count}: {len(ids)} message ID(s)"
                + (f"; Gmail estimate={estimate}" if estimate is not None else "")
            )

            for message_id in ids:
                if self.stop_event.is_set() or self.is_paused():
                    break
                counters["seen"] += 1
                if self.store.has_message(message_id):
                    counters["known"] += 1
                    self._publish_scan_counters(counters, current=f"Known message {message_id[:10]}…")
                    continue

                # Intake deliberately uses metadata-only Gmail reads. Full bodies
                # are fetched later by the classifier worker, so intake can keep
                # moving even when classification is slow.
                msg = gmail.get_message_metadata(message_id)
                counters["new"] += 1
                self._publish_scan_counters(counters, current=_short(msg.subject or "(no subject)"))
                plan = self.unsubscriber.inspect(msg.headers)

                # Cheap hard protections are applied at intake. The expensive
                # thread-history check (threads.get costs more Gmail quota than
                # messages.get) is deferred to the classifier stage. This keeps
                # continuous mailbox intake well below Gmail's per-minute quota.
                protected_reason = self.safety.protect_reason(msg)
                if protected_reason:
                    item_id = self.store.add_discovered(msg, plan, protected_reason=protected_reason)
                    self.store.mark_protected(item_id, protected_reason)
                    self.audit.record(msg, Decision("protected", protected_reason, True, None), plan, None)
                    counters["protected"] += 1
                    self._publish_scan_counters(counters, current=_short(msg.subject))
                    self.report(f"[intake] protected | {_short(msg.subject)} | {protected_reason}")
                    continue

                rule = self.store.matching_rule_for_message(msg)
                if rule is not None:
                    # Learned rules are resolved *before* the classification backlog. The
                    # action worker performs the deferred replied-thread safety
                    # check before executing any learned rule other than KEEP, so
                    # hard protections still outrank learned cleanup rules.
                    item_id = self.store.add_discovered(msg, plan)
                    self.store.set_actionable(
                        item_id,
                        rule["action"],
                        source="learned-rule",
                        reason=f"learned {rule['scope_type']} rule",
                        rule_id=int(rule["id"]),
                    )
                    counters["rule_matched"] += 1
                    self.action_wake.release()
                    self._publish_scan_counters(counters, current=_short(msg.subject))
                    self.report(
                        f"[intake] learned rule -> {rule['action']} | {_short(msg.subject)} | "
                        f"{rule['scope_type']}={rule['scope_value']}"
                    )
                    continue

                self.store.add_discovered(msg, plan)
                counters["discovered"] += 1
                self.classifier_wake.release()
                self._publish_scan_counters(counters, current=_short(msg.subject))

            self.report(
                f"[intake] page {page_count} complete | seen={counters['seen']} new={counters['new']} "
                f"known={counters['known']} queued-for-classifier={counters['discovered']}"
            )

        elapsed = time.perf_counter() - started
        state = "paused" if self.is_paused() else "idle"
        self.status.update(
            scanner=state,
            last_scan_completed=_now(),
            last_scan_error=None,
            last_progress_at=_now(),
            scan_page=page_count,
            scanner_current_subject=None,
        )
        self.report(
            f"[intake] cycle {cycle} complete: seen={counters['seen']} new={counters['new']} "
            f"known={counters['known']} in {elapsed:.1f}s"
        )

    def _publish_scan_counters(self, counters: dict[str, int], *, current: str | None) -> None:
        self.status.update(
            scan_seen=counters["seen"],
            scan_new=counters["new"],
            scan_known=counters["known"],
            scan_protected=counters["protected"],
            scan_rule_matched=counters["rule_matched"],
            scan_discovered=counters["discovered"],
            scanner_current_subject=current,
            last_progress_at=_now(),
        )

    def _career_subject_signal(self, message: EmailMessage) -> str | None:
        subject = (message.subject or "").lower()
        for term in self.policy.classification.career_subject_terms:
            term = term.strip().lower()
            if term and term in subject:
                return term
        return None

    def _classify_claimed(self, claimed: list[dict[str, Any]]) -> None:
        gmail = self._provider_for_classifier()
        candidates: list[ClassifyCandidate] = []
        self.status.update(
            classifier="fetching-bodies",
            classifier_batch_size=len(claimed),
            classifier_current_subject=f"Preparing {len(claimed)} message(s)",
            last_progress_at=_now(),
        )

        for index, row in enumerate(claimed):
            item_id = int(row["id"])
            current = self.store.get_item(item_id)
            if not current or current.get("status") != "classifying":
                continue
            try:
                msg = gmail.get_message(row["message_id"])
            except GmailRateLimitError as exc:
                # The request helper has already backed off/retried. If Gmail is
                # still refusing requests, release the entire unstarted remainder
                # of this batch back to the persistent queue and stop hammering.
                self.store.return_to_discovered(item_id, f"Gmail quota pause: {exc}"[:2000])
                for remaining_row in claimed[index + 1 :]:
                    self.store.return_to_discovered(
                        int(remaining_row["id"]), "Deferred after Gmail quota pause"
                    )
                self.status.update(
                    classifier="rate-limited",
                    classifier_current_subject=None,
                    classifier_batch_size=0,
                    last_progress_at=_now(),
                )
                self.report(f"[classifier] Gmail quota pause; returned batch to queue after item={item_id}")
                time.sleep(2.0)
                return
            except LookupError as exc:
                # Yahoo-style "message not found" is permanent.
                self.store.mark_message_gone(item_id, f"provider message not found: {exc}"[:2000])
                self.report(f"[classifier] item={item_id} no longer exists at provider; marked failed")
                continue
            except Exception as exc:
                if self._is_permanent_message_missing(exc):
                    # Deleted at the provider: retrying forever only burns quota.
                    self.store.mark_message_gone(item_id, f"provider message not found: {exc}"[:2000])
                    self.report(f"[classifier] item={item_id} no longer exists at provider; marked failed")
                    continue
                # Put transient Gmail failures back into the intake/classifier
                # backlog rather than turning them into a user decision.
                self.store.return_to_discovered(item_id, f"Gmail body fetch failed: {exc}"[:2000])
                self.report(f"[classifier] body fetch failed item={item_id}: {exc}")
                continue

            self.status.update(classifier_current_subject=_short(msg.subject), last_progress_at=_now())

            # Persist a local body excerpt so a later user answer can train the
            # classifier on the email's actual content, not just its sender.
            if msg.body:
                self.store.set_body_excerpt(item_id, msg.body)

            # Re-check hard protection using the full message before inference,
            # including replied-thread history. The thread lookup is deliberately
            # deferred from fast intake because Gmail charges it more heavily.
            protected_reason = self.safety.protect_reason(msg)
            if protected_reason is None:
                try:
                    thread_has_sent = gmail.thread_has_sent_message(msg.thread_id)
                except GmailRateLimitError as exc:
                    self.store.return_to_discovered(item_id, f"Gmail quota pause: {exc}"[:2000])
                    for remaining_row in claimed[index + 1 :]:
                        self.store.return_to_discovered(
                            int(remaining_row["id"]), "Deferred after Gmail quota pause"
                        )
                    self.status.update(
                        classifier="rate-limited", classifier_current_subject=None, classifier_batch_size=0, last_progress_at=_now()
                    )
                    self.report(f"[classifier] Gmail quota pause during thread safety check; batch requeued")
                    time.sleep(2.0)
                    return
                except Exception as exc:
                    thread_has_sent = False
                    self.report(f"[classifier] thread-history check failed for {msg.id}: {exc}")
                protected_reason = self.safety.protect_reason(msg, thread_has_sent=thread_has_sent)

            if protected_reason:
                plan = self.unsubscriber.inspect(msg.headers)
                self.store.mark_protected(item_id, protected_reason)
                self.audit.record(msg, Decision("protected", protected_reason, True, None), plan, None)
                continue

            rule = self.store.matching_rule_for_message(msg)
            if rule is not None:
                self.store.set_actionable(
                    item_id,
                    rule["action"],
                    source="learned-rule",
                    reason=f"learned {rule['scope_type']} rule",
                    rule_id=int(rule["id"]),
                )
                self.action_wake.release()
                continue

            prior = self._prior_correspondent(gmail, msg.sender_address)
            signals = self.safety.soft_signals(msg, prior_correspondent=prior)
            plan = self.unsubscriber.inspect(msg.headers)

            # Obvious job-alert subjects are cheap and safe to recognise
            # deterministically. They still go to Pending, so this only avoids a
            # a model call; the user remains in control of the provider action.
            career_signal = self._career_subject_signal(msg)
            if career_signal:
                classification = Classification(
                    category="career",
                    confidence=1.0,
                    spam_probability=0.05,
                    bulk_probability=0.95,
                    useful_probability=0.90,
                    importance_score=2.0,
                    raw={"_backend": "deterministic", "career_subject_term": career_signal},
                )
                decision = decide(classification, self.policy, None, signals)
                self.audit.record(msg, decision, plan, None)
                self.store.mark_pending(
                    item_id,
                    classification,
                    reason=decision.reason,
                    question=self._question(msg, classification, plan),
                    proposed_action="read_later",
                    safety_signals=signals,
                )
                self.report(f"[pending] career(subject:{career_signal}) -> {_short(msg.subject)}")
                continue

            candidates.append(ClassifyCandidate(item_id, msg, signals, plan))

        if not candidates:
            self.status.update(classifier="idle", classifier_current_subject=None, classifier_batch_size=0, last_progress_at=_now())
            return

        # Rules may have been learned while full bodies were being fetched.
        remaining: list[ClassifyCandidate] = []
        for candidate in candidates:
            current = self.store.get_item(candidate.item_id)
            if not current or current.get("status") != "classifying":
                continue
            rule = self.store.matching_rule_for_message(candidate.message)
            if rule is not None:
                self.store.set_actionable(
                    candidate.item_id,
                    rule["action"],
                    source="learned-rule",
                    reason=f"learned {rule['scope_type']} rule",
                    rule_id=int(rule["id"]),
                )
                self.action_wake.release()
            else:
                remaining.append(candidate)
        if not remaining:
            self.status.update(classifier="idle", classifier_current_subject=None, classifier_batch_size=0, last_progress_at=_now())
            return

        # --- routing for the remaining unknowns --------------------------------
        # The fast local personalised classifier owns the normal live path.
        # The hosted Jev backend remains available for explicit CLASSIFIER=jev setups.
        if self.uses_local_classifier:
            self._classify_claimed_locally(remaining)
            self.status.update(classifier="idle", classifier_current_subject=None, classifier_batch_size=0, last_progress_at=_now())
            self.classifier_wake.release()
            return

        self._classify_claimed_with_batch_model(remaining)

    def _classify_claimed_with_batch_model(self, remaining: list[ClassifyCandidate]) -> None:
        classifier = self._get_classifier()
        messages = [c.message for c in remaining]
        self.status.update(
            classifier="classifying",
            classifier_batch_size=len(messages),
            classifier_current_subject=f"Classifier batch: {len(messages)} message(s)",
            last_progress_at=_now(),
        )
        started = time.perf_counter()
        try:
            batch_method = getattr(classifier, "classify_batch", None)
            if callable(batch_method):
                results = batch_method(messages, batch_size=self.settings.classify_batch_size)
            else:
                results = [classifier.classify(m) for m in messages]
        except Exception as exc:
            for candidate in remaining:
                current = self.store.get_item(candidate.item_id)
                if current and current.get("status") == "classifying":
                    self.store.mark_pending(
                        candidate.item_id,
                        None,
                        reason="classifier error",
                        question="OpenMailSweep could not classify this message. What should it do?",
                        proposed_action=None,
                        safety_signals=candidate.safety_signals,
                        error=str(exc),
                    )
            self.status.update(classifier="error", classifier_current_subject=None, classifier_batch_size=0, last_progress_at=_now())
            self.report(f"[classifier] classification ERROR: {exc}")
            return

        elapsed = time.perf_counter() - started
        previous_total = int(self.status.snapshot().get("classifier_total") or 0)
        self.status.update(classifier_total=previous_total + len(results), last_progress_at=_now())
        self.report(
            f"[classifier] classified {len(results)} message(s) in {elapsed:.1f}s "
            f"({elapsed / max(1,len(results)):.1f}s/message)"
        )

        for candidate, classification in zip(remaining, results):
            current = self.store.get_item(candidate.item_id)
            # A learned rule or user decision may have released this message
            # while inference was running. Never overwrite that newer decision.
            if not current or current.get("status") != "classifying":
                continue

            rule = self.store.matching_rule_for_message(candidate.message)
            if rule is not None:
                self.store.set_actionable(
                    candidate.item_id,
                    rule["action"],
                    source="learned-rule",
                    reason=f"learned {rule['scope_type']} rule",
                    rule_id=int(rule["id"]),
                )
                self.action_wake.release()
                continue

            decision = decide(classification, self.policy, None, candidate.safety_signals)
            self.audit.record(candidate.message, decision, candidate.plan, None)
            if decision.action == "keep":
                self.store.mark_model_keep(
                    candidate.item_id, classification, decision.reason, candidate.safety_signals
                )
                continue

            proposed = self._proposed_action(classification, candidate.plan)
            question = self._question(candidate.message, classification, candidate.plan)
            self.store.mark_pending(
                candidate.item_id,
                classification,
                reason=decision.reason,
                question=question,
                proposed_action=proposed,
                safety_signals=candidate.safety_signals,
            )
            self.report(f"[pending] {classification.category} -> {_short(candidate.message.subject)}")

        self.status.update(classifier="idle", classifier_current_subject=None, classifier_batch_size=0, last_progress_at=_now())
        # If intake filled the persistent discovered queue while classification was busy,
        # immediately continue with the next batch instead of waiting for a timer.
        self.classifier_wake.release()

    @staticmethod
    def _proposed_action(classification: Classification, plan: Any) -> str | None:
        if classification.category == "career":
            return "read_later"
        if classification.category in {"promotion", "newsletter"}:
            return "unsubscribe_trash"
        if classification.category in {"junk", "spam"}:
            return "trash"
        if classification.category in {"transactional", "notification", "community", "keep"}:
            return "keep"
        return None

    @staticmethod
    def _question(message: EmailMessage, classification: Classification, plan: Any) -> str:
        if classification.category == "career":
            return "This looks like a job or career opportunity. Keep it in Inbox, move it to Read Later, archive it, or clean future matching mail?"
        if classification.category in {"promotion", "newsletter"}:
            if plan.auto_eligible:
                return "This looks like recurring bulk mail and supports automatic unsubscribe. What should OpenMailSweep do with this source?"
            return "This looks like recurring bulk mail. What should OpenMailSweep do with this source?"
        if classification.category in {"junk", "spam"}:
            return "This looks unwanted, but OpenMailSweep is not confident enough to act without you. What should it do?"
        return "OpenMailSweep is unsure how to handle this message. What should it do with this source?"

    def _execute_action(self, item: dict[str, Any], gmail: Any, *, worker_id: int = 0) -> None:
        action = item.get("desired_action")
        if action == "keep":
            self.store.finish_action(int(item["id"]), "keep", "Kept by user/learned rule")
            self.status.update(last_action=f"kept: {_short(item.get('subject',''))}")
            return

        message_id = item["message_id"]
        action_message = None

        # Learned rules bypass the classifier entirely, but never hard safety.
        # Re-fetch lightweight metadata and check whether the user has replied in
        # the thread before applying archive/read-later/trash/unsubscribe rules.
        if item.get("decision_source") in {"learned-rule", "local-classifier-reeval"}:
            action_message = gmail.get_message_metadata(message_id)
            protected_reason = self.safety.protect_reason(action_message)
            if protected_reason is None:
                thread_has_sent = gmail.thread_has_sent_message(action_message.thread_id)
                protected_reason = self.safety.protect_reason(
                    action_message, thread_has_sent=thread_has_sent
                )
            if protected_reason:
                plan = self.unsubscriber.inspect(action_message.headers)
                self.store.mark_protected(int(item["id"]), protected_reason)
                self.audit.record(
                    action_message, Decision("protected", protected_reason, True, None), plan, None
                )
                self.status.update(
                    last_action=f"protected: {_short(item.get('subject',''))}"
                )
                self.report(
                    f"[action] learned rule blocked by hard protection | "
                    f"{_short(item.get('subject',''))} | {protected_reason}"
                )
                return

        detail = ""
        if action == "archive":
            gmail.archive(message_id)
            detail = "Removed INBOX label"
        elif action == "read_later":
            gmail.move_to_read_later(message_id, self.policy.labels.read_later)
            detail = f"Moved to Gmail label {self.policy.labels.read_later!r} and removed from Inbox"
        elif action == "trash":
            gmail.trash(message_id)
            detail = "Moved to Gmail Trash"
        elif action == "unsubscribe_trash":
            msg = action_message or gmail.get_message_metadata(message_id)
            plan = self.unsubscriber.inspect(msg.headers)
            dedupe_key = self._unsubscribe_key(item, msg)
            # Serialize unsubscribe for one list/sender only. Different lists can
            # still be processed by different workers concurrently. The registry
            # persists success/uncertain outcomes across restarts so a learned
            # rule does not send one unsubscribe request per matching email.
            with self._unsubscribe_lock(dedupe_key):
                previous = self.store.get_unsubscribe_registry(dedupe_key)
                if previous and previous.get("status") in {"success", "uncertain"}:
                    detail = (
                        f"Skipped duplicate unsubscribe for {dedupe_key}; "
                        f"previous outcome={previous.get('status')} via {previous.get('method') or 'unknown'}"
                    )
                elif plan.auto_eligible:
                    result = self.unsubscriber.execute(
                        plan, send_mailto=gmail.send_unsubscribe_email
                    )
                    detail = f"Unsubscribe {result.status} via {result.method}"
                    if result.detail:
                        detail += f": {result.detail}"
                    if len(result.attempts) > 1:
                        attempt_text = " -> ".join(
                            f"{a.get('method')}:{a.get('status')}" for a in result.attempts
                        )
                        detail += f" [attempts {attempt_text}]"
                    self.store.record_unsubscribe_registry(
                        dedupe_key,
                        result.status,
                        result.method,
                        detail,
                        source_message_id=message_id,
                    )
                else:
                    detail = f"No safe automatic unsubscribe method ({plan.reason}); cleaned anyway"
            gmail.trash(message_id)
            detail += "; moved to Gmail Trash"
        else:
            raise RuntimeError(f"Unsupported queued action: {action}")

        self.store.finish_action(int(item["id"]), str(action), detail)
        self.status.update(last_action=f"{action}: {_short(item.get('subject',''))}")
        self.report(f"[action:{worker_id}] {action} | {_short(item.get('subject',''))} | {detail}")
