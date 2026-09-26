from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

from .audit import AuditLog
from .config import Policy
from .decision import decide
from .models import Classification, EmailMessage, UnsubscribeResult
from .safety import SafetyRules
from .unsubscribe import UnsubscribeClient

if TYPE_CHECKING:
    from .gmail_client import GmailClient


Reporter = Callable[[str], None]


class Classifier(Protocol):
    def classify(self, message: EmailMessage) -> Classification: ...


class BatchClassifier(Protocol):
    def classify_batch(
        self,
        messages: Sequence[EmailMessage],
        batch_size: int | None = None,
    ) -> list[Classification]: ...


class Runner:
    def __init__(
        self,
        gmail: "GmailClient",
        classifier: Classifier,
        policy: Policy,
        audit: AuditLog,
        unsubscriber: UnsubscribeClient | None = None,
        report: Reporter | None = None,
        batch_size: int = 16,
    ):
        self.gmail = gmail
        self.classifier = classifier
        self.policy = policy
        self.audit = audit
        self.safety = SafetyRules(policy)
        self.unsubscriber = unsubscriber or UnsubscribeClient(policy.unsubscribe)
        self.report = report or (lambda text: print(text, flush=True))
        self.batch_size = max(1, batch_size)
        self._prior_correspondent_cache: dict[str, bool] = {}

    def _apply_named_label(self, message_id: str, label_name: str, cache: dict[str, str]) -> None:
        if label_name not in cache:
            cache[label_name] = self.gmail.ensure_label(label_name)
        self.gmail.apply_label(message_id, cache[label_name])

    @staticmethod
    def _short(value: str, limit: int = 74) -> str:
        value = " ".join((value or "").split())
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "..."

    def _classify_messages(self, messages: list[EmailMessage]) -> list[Classification]:
        batch_method = getattr(self.classifier, "classify_batch", None)
        if callable(batch_method):
            return batch_method(messages, batch_size=self.batch_size)
        return [self.classifier.classify(message) for message in messages]

    def _prior_correspondent(self, address: str) -> bool:
        address = (address or "").strip().lower()
        if not address or not self.policy.protected.protect_prior_correspondents:
            return False
        if address not in self._prior_correspondent_cache:
            lookup = getattr(self.gmail, "has_sent_to_address", None)
            self._prior_correspondent_cache[address] = bool(lookup(address)) if callable(lookup) else False
        return self._prior_correspondent_cache[address]

    def _handle_one(
        self,
        mode: str,
        msg: EmailMessage,
        classification: Classification | None,
        protected_reason: str | None,
        safety_signals: list[str],
        plan,
        counters: Counter,
        label_cache: dict[str, str],
    ) -> None:
        decision = decide(classification, self.policy, protected_reason, safety_signals)
        unsubscribe_result: UnsubscribeResult | None = None

        counters["processed"] += 1
        counters[decision.action] += 1
        counters[f"unsubscribe_method_{plan.method}"] += 1
        if plan.auto_eligible:
            counters["unsubscribe_auto_eligible"] += 1
        if protected_reason:
            counters["deterministic_protected"] += 1
        if safety_signals:
            counters["safety_signal_messages"] += 1
            for signal in safety_signals:
                counters[f"safety_signal_{signal}"] += 1

        if mode == "audit":
            self.audit.record(msg, decision, plan, unsubscribe_result)
            return

        if mode == "review":
            if decision.action == "protected":
                label_name = self.policy.labels.protected
            elif decision.action == "junk":
                label_name = self.policy.labels.junk
            elif decision.action == "promotion":
                label_name = self.policy.labels.promotion
            elif decision.action == "newsletter":
                label_name = self.policy.labels.newsletter
            elif decision.action == "review":
                label_name = self.policy.labels.review
            else:
                label_name = None

            if label_name:
                self._apply_named_label(msg.id, label_name, label_cache)
            self.audit.record(msg, decision, plan, unsubscribe_result)
            return

        if mode == "sweep":
            if decision.action == "junk":
                self.gmail.trash(msg.id)
                counters["trashed"] += 1
            self.audit.record(msg, decision, plan, unsubscribe_result)
            return

        if decision.action not in {"promotion", "newsletter"}:
            self.audit.record(msg, decision, plan, unsubscribe_result)
            return

        if plan.auto_eligible:
            unsubscribe_result = self.unsubscriber.execute(
                plan, send_mailto=getattr(self.gmail, "send_unsubscribe_email", None)
            )
            counters[f"unsubscribe_{unsubscribe_result.status}"] += 1
            counters[f"unsubscribe_via_{unsubscribe_result.method}"] += 1
            if unsubscribe_result.status == "success":
                self.gmail.trash(msg.id)
                counters["bulk_trashed"] += 1
            elif unsubscribe_result.status in {"unavailable", "skipped"} and self.policy.unsubscribe.trash_if_unavailable:
                self.gmail.trash(msg.id)
                counters["bulk_trashed"] += 1
            elif self.policy.unsubscribe.trash_on_failure:
                self.gmail.trash(msg.id)
                counters["bulk_trashed"] += 1
            else:
                self._apply_named_label(msg.id, self.policy.labels.unsubscribe_review, label_cache)
                counters["unsubscribe_review"] += 1
        else:
            unsubscribe_result = UnsubscribeResult(
                "unavailable", plan.method, plan.reason, plan.target_host
            )
            counters["unsubscribe_unavailable"] += 1
            if self.policy.unsubscribe.trash_if_unavailable:
                self.gmail.trash(msg.id)
                counters["bulk_trashed"] += 1
            else:
                self._apply_named_label(msg.id, self.policy.labels.unsubscribe_review, label_cache)
                counters["unsubscribe_review"] += 1

        self.audit.record(msg, decision, plan, unsubscribe_result)

    def run(
        self,
        mode: str,
        query: str,
        limit: int,
        allow_sweep: bool = False,
        allow_unsubscribe: bool = False,
    ) -> Counter:
        if mode not in {"audit", "review", "sweep", "bulk-clean"}:
            raise RuntimeError(f"Unsupported mode: {mode}")
        if mode in {"sweep", "bulk-clean"} and not allow_sweep:
            raise RuntimeError("Cleanup is disabled. Set ALLOW_SWEEP=true only after reviewing audit/review results.")
        if mode == "bulk-clean" and not allow_unsubscribe:
            raise RuntimeError(
                "Automatic unsubscribe is disabled. Set ALLOW_UNSUBSCRIBE=true after reviewing detected methods."
            )

        counters = Counter()
        label_cache: dict[str, str] = {}
        ids = list(self.gmail.list_message_ids(query, limit))
        total = len(ids)
        self.report(f"[gmail] Query matched {total} message(s); mode={mode}; batch_size={self.batch_size}")
        if total == 0:
            return counters

        overall_started = time.perf_counter()
        for chunk_start in range(0, total, self.batch_size):
            chunk_ids = ids[chunk_start : chunk_start + self.batch_size]
            chunk_messages: list[EmailMessage] = []
            protected_reasons: list[str | None] = []
            safety_signal_sets: list[list[str]] = []
            plans = []

            for offset, message_id in enumerate(chunk_ids):
                index = chunk_start + offset + 1
                fetch_started = time.perf_counter()
                msg = self.gmail.get_message(message_id)

                protected_reason = self.safety.protect_reason(msg)
                if protected_reason is None:
                    sent_in_thread = self.gmail.thread_has_sent_message(msg.thread_id)
                    protected_reason = self.safety.protect_reason(msg, thread_has_sent=sent_in_thread)

                safety_signals: list[str] = []
                if protected_reason is None:
                    prior = self._prior_correspondent(msg.sender_address)
                    safety_signals = self.safety.soft_signals(msg, prior_correspondent=prior)

                plan = self.unsubscriber.inspect(msg.headers)
                chunk_messages.append(msg)
                protected_reasons.append(protected_reason)
                safety_signal_sets.append(safety_signals)
                plans.append(plan)
                elapsed = time.perf_counter() - fetch_started
                protection_note = f" | protect={protected_reason}" if protected_reason else ""
                signal_note = f" | signals={','.join(safety_signals)}" if safety_signals else ""
                self.report(
                    f"[fetch {index:>3}/{total}] {elapsed:4.1f}s | "
                    f"{self._short(msg.sender, 38)} | {self._short(msg.subject)}{protection_note}{signal_note}"
                )

            classifications: list[Classification | None] = [None] * len(chunk_messages)
            classify_indexes = [i for i, reason in enumerate(protected_reasons) if reason is None]
            classify_messages = [chunk_messages[i] for i in classify_indexes]

            first = chunk_start + 1
            last = chunk_start + len(chunk_messages)
            if classify_messages:
                classify_started = time.perf_counter()
                skipped = len(chunk_messages) - len(classify_messages)
                self.report(
                    f"[classify {first}-{last}/{total}] scoring {len(classify_messages)} message(s); "
                    f"skipping {skipped} deterministically protected"
                )
                classified = self._classify_messages(classify_messages)
                for local_index, classification in zip(classify_indexes, classified):
                    classifications[local_index] = classification
                classify_elapsed = time.perf_counter() - classify_started
                self.report(
                    f"[classify {first}-{last}/{total}] done in {classify_elapsed:.2f}s "
                    f"({classify_elapsed / max(1, len(classify_messages)):.3f}s/classified message)"
                )
            else:
                self.report(
                    f"[classify {first}-{last}/{total}] skipped; all {len(chunk_messages)} message(s) protected by rules"
                )

            for offset, (msg, classification, protected_reason, safety_signals, plan) in enumerate(
                zip(chunk_messages, classifications, protected_reasons, safety_signal_sets, plans)
            ):
                index = chunk_start + offset + 1
                decision = decide(classification, self.policy, protected_reason, safety_signals)
                if classification is None:
                    model_text = "rules-only"
                else:
                    model_text = (
                        f"{classification.category:<13} conf={classification.confidence:.3f} "
                        f"spam={classification.spam_probability:.3f} "
                        f"bulk={classification.bulk_probability:.3f} "
                        f"useful={classification.useful_probability:.3f}"
                    )
                self.report(
                    f"[result {index:>3}/{total}] {model_text} => {decision.action:<10} "
                    f"reason={decision.reason} | {self._short(msg.subject)}"
                )
                self._handle_one(
                    mode,
                    msg,
                    classification,
                    protected_reason,
                    safety_signals,
                    plan,
                    counters,
                    label_cache,
                )

        elapsed = time.perf_counter() - overall_started
        rate = total / elapsed if elapsed > 0 else 0.0
        self.report(f"[done] processed {total} message(s) in {elapsed:.1f}s ({rate:.2f} msg/s)")
        return counters
