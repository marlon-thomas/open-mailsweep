"""Concurrency regression tests for the OpenMailSweep threading model.

These pin down the invariants that keep the scanner, classifier, action-pool,
and FastAPI request threads from corrupting queue state:

- action queue claims are exclusive across worker threads;
- queue status transitions never clobber a newer decision made by another
  thread (learned rule / user answer) between check and write;
- per-key unsubscribe locks serialize identical work and are evicted when idle;
- the service starts/stops its worker threads exactly once and joins them all.
"""

import threading
import time
from pathlib import Path

from openmailsweep.audit import AuditLog
from openmailsweep.config import Policy, Settings
from openmailsweep.models import Classification, EmailMessage, UnsubscribePlan
from openmailsweep.state_store import StateStore
from openmailsweep.worker import OpenMailSweepService


def msg(mid: str, sender: str = "offers@example.com") -> EmailMessage:
    return EmailMessage(
        id=mid,
        thread_id=f"t-{mid}",
        subject="Weekly offers",
        sender=f"Sender <{sender}>",
        sender_address=sender,
        body="",
        snippet="preview",
        labels=[],
        headers={"list-id": "<weekly.example.com>"},
    )


def plan() -> UnsubscribePlan:
    return UnsubscribePlan(
        method="one_click",
        target_host="example.com",
        one_click=True,
        auto_eligible=True,
        reason="test",
    )


def classification() -> Classification:
    return Classification("newsletter", 0.4, 0.8, 0.95, 0.2, 0.5, {})


def make_service(tmp_path: Path) -> OpenMailSweepService:
    settings = Settings(
        gmail_token=tmp_path / "missing-token.json",
        state_db=tmp_path / "state.db",
        audit_db=tmp_path / "audit.db",
    )
    store = StateStore(tmp_path / "state.db")
    audit = AuditLog(tmp_path / "audit.db")
    return OpenMailSweepService(settings, Policy(), store, audit)


def test_claim_next_action_is_exclusive_across_threads(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    total = 40
    for i in range(total):
        item_id = store.add_discovered(msg(f"m{i}"), plan())
        store.set_actionable(item_id, "trash", source="user-message", reason="test")

    claimed: list[str] = []
    claimed_lock = threading.Lock()
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        while True:
            item = store.claim_next_action()
            if item is None:
                return
            with claimed_lock:
                claimed.append(item["message_id"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed) == total
    assert len(set(claimed)) == total, "two workers claimed the same action item"


def test_mark_pending_cannot_clobber_learned_rule(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    item_id = store.add_discovered(msg("m1"), plan())
    rule_id = store.upsert_rule("sender", "offers@example.com", "trash")
    assert store.apply_rule_to_open_items(rule_id) == 1

    # Classifier finishes inference after the rule was learned; the pending
    # write must not overwrite the actionable decision.
    store.mark_pending(
        item_id,
        classification(),
        reason="uncertain",
        question="What should OpenMailSweep do?",
        proposed_action="unsubscribe_trash",
    )

    item = store.get_item(item_id)
    assert item["status"] == "actionable"
    assert item["desired_action"] == "trash"


def test_mark_model_keep_cannot_clobber_learned_rule(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    item_id = store.add_discovered(msg("m1"), plan())
    rule_id = store.upsert_rule("sender", "offers@example.com", "archive")
    assert store.apply_rule_to_open_items(rule_id) == 1

    store.mark_model_keep(item_id, classification(), "model says keep")

    item = store.get_item(item_id)
    assert item["status"] == "actionable"
    assert item["desired_action"] == "archive"


def test_mark_pending_still_works_for_claimed_items(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    item_id = store.add_discovered(msg("m1"), plan())
    store.mark_pending(
        item_id,
        classification(),
        reason="uncertain",
        question="q",
        proposed_action=None,
    )
    assert store.get_item(item_id)["status"] == "pending"


def test_mark_protected_covers_processing_and_skips_done(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")

    # Action-worker path: a claimed (processing) learned-rule item blocked by
    # hard protection must be markable as protected.
    blocked_id = store.add_discovered(msg("m1"), plan())
    store.set_actionable(blocked_id, "trash", source="learned-rule", reason="test")
    processing = store.claim_next_action()
    assert processing is not None
    store.mark_protected(blocked_id, "replied thread")
    item = store.get_item(blocked_id)
    assert item["status"] == "done"
    assert item["final_action"] == "protected"

    # A completed item is never rewritten by a late protect call.
    done_id = store.add_discovered(msg("m2"), plan())
    store.mark_model_keep(done_id, classification(), "model keep")
    store.mark_protected(done_id, "late protection")
    item = store.get_item(done_id)
    assert item["final_action"] == "keep"


def test_unsubscribe_lock_serializes_same_key_and_evicts(tmp_path: Path):
    service = make_service(tmp_path)
    events: list[str] = []
    events_lock = threading.Lock()
    start_gate = threading.Barrier(2)

    def worker(name: str) -> None:
        start_gate.wait()
        with service._unsubscribe_lock("list:weekly.example.com"):
            with events_lock:
                events.append(f"{name}-enter")
            time.sleep(0.05)
            with events_lock:
                events.append(f"{name}-exit")

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Strict enter/exit alternation proves mutual exclusion for the key.
    assert events == ["a-enter", "a-exit", "b-enter", "b-exit"] or events == [
        "b-enter",
        "b-exit",
        "a-enter",
        "a-exit",
    ]
    # Idle entries are evicted, so the registry cannot grow without bound.
    assert service._unsubscribe_locks == {}


def test_service_start_is_idempotent_and_stop_joins_all(tmp_path: Path):
    service = make_service(tmp_path)
    service.start()
    first = [service._scanner_thread, service._classifier_thread, *service._action_threads]
    assert all(thread is not None and thread.is_alive() for thread in first)

    service.start()  # must not spawn a second set of workers
    second = [service._scanner_thread, service._classifier_thread, *service._action_threads]
    assert first == second

    service.stop()
    assert all(not thread.is_alive() for thread in first)


def test_wake_tokens_are_not_lost(tmp_path: Path):
    """Releases issued before start() must survive and wake the workers."""
    service = make_service(tmp_path)
    for _ in range(3):
        service.wake_actions()
    service.stop()
    assert service.action_wake._value >= service.settings.action_workers
