from pathlib import Path
from types import SimpleNamespace

from openmailsweep.config import Policy
from openmailsweep.models import EmailMessage, UnsubscribePlan
from openmailsweep.safety import SafetyRules
from openmailsweep.state_store import StateStore
from openmailsweep.worker import OpenMailSweepService


def message(mid: str, subject: str = "Offer") -> EmailMessage:
    return EmailMessage(
        id=mid,
        thread_id=f"t-{mid}",
        subject=subject,
        sender="Sender <offers@example.com>",
        sender_address="offers@example.com",
        body="",
        snippet="preview",
        labels=[],
        headers={},
    )


def plan() -> UnsubscribePlan:
    return UnsubscribePlan(
        method="none", auto_eligible=False, one_click=False, target_host=None, reason="test"
    )


class FakeGmail:
    def __init__(self, msg: EmailMessage, thread_has_sent: bool = False):
        self.msg = msg
        self.thread_has_sent = thread_has_sent
        self.trashed = []
        self.archived = []
        self.read_later = []

    def get_message_metadata(self, message_id: str):
        return self.msg

    def thread_has_sent_message(self, thread_id: str) -> bool:
        return self.thread_has_sent

    def trash(self, message_id: str):
        self.trashed.append(message_id)

    def archive(self, message_id: str):
        self.archived.append(message_id)

    def move_to_read_later(self, message_id: str, label: str):
        self.read_later.append((message_id, label))


class FakeAudit:
    def __init__(self):
        self.records = []

    def record(self, *args):
        self.records.append(args)


class FakeStatus:
    def update(self, **kwargs):
        pass


class FakeUnsubscriber:
    def inspect(self, headers):
        return plan()


def test_learned_cleanup_rule_is_blocked_by_hard_protection(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    msg = message("m1", "Security alert")
    item_id = store.add_discovered(msg, plan())
    store.set_actionable(
        item_id, "trash", source="learned-rule", reason="learned sender rule", rule_id=1
    )
    item = store.claim_next_action()
    gmail = FakeGmail(msg)
    fake = SimpleNamespace(
        store=store,
        safety=SafetyRules(Policy()),
        audit=FakeAudit(),
        unsubscriber=FakeUnsubscriber(),
        status=FakeStatus(),
        policy=Policy(),
        report=lambda text: None,
    )

    OpenMailSweepService._execute_action(fake, item, gmail)

    row = store.get_item(item_id)
    assert row["final_action"] == "protected"
    assert row["decision_reason"] == "protected-subject:security alert"
    assert gmail.trashed == []


def test_learned_cleanup_rule_skips_classifier_and_executes_after_safety(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    msg = message("m2", "Weekly offers")
    item_id = store.add_discovered(msg, plan())
    store.set_actionable(
        item_id, "trash", source="learned-rule", reason="learned sender rule", rule_id=1
    )
    item = store.claim_next_action()
    gmail = FakeGmail(msg)
    fake = SimpleNamespace(
        store=store,
        safety=SafetyRules(Policy()),
        audit=FakeAudit(),
        unsubscriber=FakeUnsubscriber(),
        status=FakeStatus(),
        policy=Policy(),
        report=lambda text: None,
    )

    OpenMailSweepService._execute_action(fake, item, gmail)

    row = store.get_item(item_id)
    assert row["final_action"] == "trash"
    assert gmail.trashed == ["m2"]
