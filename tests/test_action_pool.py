from pathlib import Path

from openmailsweep.audit import AuditLog
from openmailsweep.config import Policy, Settings
from openmailsweep.models import EmailMessage, UnsubscribePlan, UnsubscribeResult
from openmailsweep.state_store import StateStore
from openmailsweep.worker import OpenMailSweepService


def msg(mid: str) -> EmailMessage:
    return EmailMessage(
        id=mid,
        thread_id=f"t-{mid}",
        subject="Weekly offers",
        sender="Sender <offers@example.com>",
        sender_address="offers@example.com",
        body="",
        snippet="preview",
        labels=[],
        headers={"list-id": "<weekly.example.com>"},
    )


class FakeGmail:
    def __init__(self, messages):
        self.messages = {m.id: m for m in messages}
        self.trashed = []

    def get_message_metadata(self, message_id: str):
        return self.messages[message_id]

    def trash(self, message_id: str):
        self.trashed.append(message_id)

    def send_unsubscribe_email(self, *_):
        return "sent"


class FakeUnsubscriber:
    def __init__(self):
        self.calls = 0

    def inspect(self, headers):
        return UnsubscribePlan(
            method="one_click",
            https_url="https://example.com/u/token",
            target_host="example.com",
            one_click=True,
            dkim_authenticated=True,
            one_click_auto_eligible=True,
            auto_eligible=True,
            reason="test",
        )

    def execute(self, plan, *, send_mailto=None):
        self.calls += 1
        return UnsubscribeResult("success", "one_click", "accepted", "example.com", 204)


def test_duplicate_unsubscribe_is_skipped_for_same_list(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    service = OpenMailSweepService(Settings(), Policy(), store, AuditLog(tmp_path / "audit.db"))
    fake_unsub = FakeUnsubscriber()
    service.unsubscriber = fake_unsub
    messages = [msg("m1"), msg("m2")]
    gmail = FakeGmail(messages)

    items = []
    for m in messages:
        item_id = store.add_discovered(m, fake_unsub.inspect(m.headers))
        store.set_actionable(item_id, "unsubscribe_trash", source="user-message", reason="test")
        items.append(store.claim_next_action())

    service._execute_action(items[0], gmail, worker_id=1)
    service._execute_action(items[1], gmail, worker_id=2)

    assert fake_unsub.calls == 1
    assert gmail.trashed == ["m1", "m2"]
    assert "Skipped duplicate unsubscribe" in store.get_item(items[1]["id"])["action_detail"]


def test_settings_default_to_three_action_workers(monkeypatch):
    monkeypatch.delenv("OPENMAILSWEEP_ACTION_WORKERS", raising=False)
    assert Settings.from_env().action_workers == 3
