from openmailsweep.config import Policy
from openmailsweep.models import EmailMessage
from openmailsweep.safety import SafetyRules


def msg(**kwargs):
    base = dict(
        id="1",
        thread_id="t",
        subject="hello",
        sender="a@example.com",
        sender_address="a@example.com",
        body="",
        snippet="",
        labels=[],
        headers={},
    )
    base.update(kwargs)
    return EmailMessage(**base)


def test_starred_protected():
    assert SafetyRules(Policy()).protect_reason(msg(labels=["STARRED"])) == "starred"


def test_security_subject_protected():
    assert SafetyRules(Policy()).protect_reason(msg(subject="Security alert")) is not None


def test_sent_thread_protected():
    assert SafetyRules(Policy()).protect_reason(msg(), thread_has_sent=True) == "thread-has-sent-reply"


def test_official_gov_uk_sender_is_protected():
    m = msg(
        sender="Georgina Quinn <GeorginaQuinn@maidstone.gov.uk>",
        sender_address="georginaquinn@maidstone.gov.uk",
        subject="Request for Planning Advice - 26/503288/PAPL",
        labels=["CATEGORY_PERSONAL"],
    )
    reason = SafetyRules(Policy()).protect_reason(m)
    assert reason == "official-domain:gov.uk"


def test_planning_advice_subject_is_protected_even_without_official_domain():
    reason = SafetyRules(Policy()).protect_reason(
        msg(subject="Request for Planning Advice - 26/503288/PAPL")
    )
    assert reason == "transactional-subject:planning advice"


def test_charging_receipt_style_subject_is_protected():
    reason = SafetyRules(Policy()).protect_reason(msg(subject="Thanks for charging at IONITY"))
    assert reason in {"transactional-subject:thanks for charging", "transactional-subject:charging at"}


def test_delivery_subject_is_protected():
    reason = SafetyRules(Policy()).protect_reason(msg(subject="Royal Mail delivered your parcel"))
    assert reason == "transactional-subject:delivered"


def test_gmail_personal_is_soft_signal_not_hard_protection():
    rules = SafetyRules(Policy())
    m = msg(labels=["CATEGORY_PERSONAL"])
    assert rules.protect_reason(m) is None
    assert rules.soft_signals(m) == ["gmail-personal"]


def test_prior_correspondent_is_soft_signal_not_hard_protection():
    rules = SafetyRules(Policy())
    m = msg()
    assert rules.protect_reason(m, prior_correspondent=True) is None
    assert rules.soft_signals(m, prior_correspondent=True) == ["prior-correspondent"]


def test_personal_and_prior_correspondent_signals_can_coexist():
    rules = SafetyRules(Policy())
    assert rules.soft_signals(msg(labels=["CATEGORY_PERSONAL"]), prior_correspondent=True) == [
        "gmail-personal",
        "prior-correspondent",
    ]
