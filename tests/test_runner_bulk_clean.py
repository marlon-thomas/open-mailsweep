from pathlib import Path

from openmailsweep.audit import AuditLog
from openmailsweep.config import Policy
from openmailsweep.models import (
    Classification,
    EmailMessage,
    UnsubscribePlan,
    UnsubscribeResult,
)
from openmailsweep.runner import Runner


class FakeGmail:
    def __init__(self, message):
        self.message = message
        self.trashed = []
        self.labels = []

    def list_message_ids(self, query, limit):
        yield self.message.id

    def get_message(self, message_id):
        return self.message

    def thread_has_sent_message(self, thread_id):
        return False

    def ensure_label(self, name):
        return name

    def apply_label(self, message_id, label_id):
        self.labels.append((message_id, label_id))

    def trash(self, message_id):
        self.trashed.append(message_id)


class PromotionClassifier:
    def classify(self, message):
        return Classification(
            category="promotion",
            confidence=0.99,
            spam_probability=0.10,
            bulk_probability=0.99,
            useful_probability=0.05,
            importance_score=0.2,
            raw={},
        )


class SuccessfulUnsubscriber:
    def __init__(self):
        self.executed = 0

    def inspect(self, headers):
        return UnsubscribePlan(
            method="one_click",
            https_url="https://example.com/u/token",
            target_host="example.com",
            one_click=True,
            dkim_authenticated=True,
            auto_eligible=True,
            reason="available",
        )

    def execute(self, plan, *, send_mailto=None):
        self.executed += 1
        return UnsubscribeResult("success", "one_click", "accepted", "example.com", 204)


def message():
    return EmailMessage(
        id="m1",
        thread_id="t1",
        subject="Weekly offers",
        sender="Offers <offers@example.com>",
        sender_address="offers@example.com",
        body="Sale now on",
        snippet="Sale now on",
        labels=[],
        headers={},
    )


def test_bulk_clean_unsubscribes_before_trash(tmp_path: Path):
    gmail = FakeGmail(message())
    unsub = SuccessfulUnsubscriber()
    runner = Runner(gmail, PromotionClassifier(), Policy(), AuditLog(tmp_path / "audit.db"), unsub)

    counts = runner.run(
        "bulk-clean",
        "in:inbox",
        10,
        allow_sweep=True,
        allow_unsubscribe=True,
    )

    assert unsub.executed == 1
    assert gmail.trashed == ["m1"]
    assert counts["unsubscribe_success"] == 1
    assert counts["bulk_trashed"] == 1


class BatchPromotionClassifier(PromotionClassifier):
    def __init__(self):
        self.batch_calls = 0

    def classify_batch(self, messages, batch_size=None):
        self.batch_calls += 1
        return [self.classify(m) for m in messages]


def test_runner_uses_batch_classifier_when_available(tmp_path: Path):
    gmail = FakeGmail(message())
    classifier = BatchPromotionClassifier()
    output = []
    runner = Runner(
        gmail,
        classifier,
        Policy(),
        AuditLog(tmp_path / "audit.db"),
        SuccessfulUnsubscriber(),
        report=output.append,
        batch_size=16,
    )

    counts = runner.run("audit", "in:inbox", 10)

    assert classifier.batch_calls == 1
    assert counts["processed"] == 1
    assert any(line.startswith("[fetch") for line in output)
    assert any(line.startswith("[classify") for line in output)
    assert any(line.startswith("[result") for line in output)


class CountingClassifier(PromotionClassifier):
    def __init__(self):
        self.calls = 0

    def classify(self, message):
        self.calls += 1
        return super().classify(message)

    def classify_batch(self, messages, batch_size=None):
        self.calls += len(messages)
        return [super(CountingClassifier, self).classify(m) for m in messages]


def test_deterministically_protected_message_skips_model(tmp_path: Path):
    protected = message()
    protected.subject = "Request for Planning Advice - 26/503288/PAPL"
    protected.sender = "Georgina Quinn <GeorginaQuinn@maidstone.gov.uk>"
    protected.sender_address = "georginaquinn@maidstone.gov.uk"
    gmail = FakeGmail(protected)
    classifier = CountingClassifier()
    output = []
    runner = Runner(
        gmail,
        classifier,
        Policy(),
        AuditLog(tmp_path / "audit.db"),
        SuccessfulUnsubscriber(),
        report=output.append,
        batch_size=16,
    )

    counts = runner.run("audit", "in:inbox", 10)

    assert classifier.calls == 0
    assert counts["protected"] == 1
    assert counts["deterministic_protected"] == 1
    assert any("official-domain:gov.uk" in line for line in output)


def test_gmail_personal_is_classified_but_cannot_be_auto_cleaned(tmp_path: Path):
    personal = message()
    personal.labels = ["CATEGORY_PERSONAL"]
    gmail = FakeGmail(personal)
    classifier = CountingClassifier()
    output = []
    runner = Runner(
        gmail,
        classifier,
        Policy(),
        AuditLog(tmp_path / "audit.db"),
        SuccessfulUnsubscriber(),
        report=output.append,
        batch_size=16,
    )

    counts = runner.run("audit", "in:inbox", 10)

    assert classifier.calls == 1
    assert counts["review"] == 1
    assert counts["safety_signal_messages"] == 1
    assert counts["safety_signal_gmail-personal"] == 1
    assert any("signals=gmail-personal" in line for line in output)
    assert any("reason=safety-signal:gmail-personal" in line for line in output)
