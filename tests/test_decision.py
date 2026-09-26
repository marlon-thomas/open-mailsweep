from openmailsweep.config import Policy
from openmailsweep.decision import decide
from openmailsweep.models import Classification


def c(category="junk", confidence=0.99, spam=0.99, bulk=0.99, useful=0.05, importance=0.5):
    return Classification(category, confidence, spam, bulk, useful, importance, {})


def test_protected_always_wins():
    d = decide(c(), Policy(), "starred")
    assert d.action == "protected"


def test_high_confidence_junk():
    d = decide(c(), Policy(), None)
    assert d.action == "junk"


def test_suspicious_goes_to_review():
    d = decide(c(category="suspicious"), Policy(), None)
    assert d.action == "review"


def test_uncertain_junk_not_auto_cleaned():
    d = decide(c(confidence=0.7, spam=0.7), Policy(), None)
    assert d.action == "review"


def test_spam_and_useful_contradiction_goes_to_review():
    d = decide(c(category="junk", confidence=0.99, spam=0.99, useful=0.95), Policy(), None)
    assert d.action == "review"
    assert d.reason == "contradictory-spam-and-useful"


def test_spam_and_important_contradiction_goes_to_review():
    d = decide(c(category="spam", confidence=0.99, spam=0.99, useful=0.1, importance=3.0), Policy(), None)
    assert d.action == "review"
    assert d.reason == "contradictory-spam-and-important"


def test_transactional_category_is_kept():
    d = decide(c(category="transactional", confidence=0.2, spam=0.8, useful=0.9), Policy(), None)
    assert d.action == "keep"


def test_community_category_is_kept():
    d = decide(c(category="community", confidence=0.2, spam=0.95, useful=0.9), Policy(), None)
    assert d.action == "keep"


def test_protected_can_have_no_model_classification():
    d = decide(None, Policy(), "official-domain:gov.uk")
    assert d.action == "protected"


def test_gmail_personal_signal_blocks_automatic_junk_cleanup():
    d = decide(c(category="junk", confidence=0.999, spam=0.999, useful=0.01), Policy(), None, ["gmail-personal"])
    assert d.action == "review"
    assert d.reason == "safety-signal:gmail-personal"


def test_prior_correspondent_signal_blocks_automatic_promotion_cleanup():
    d = decide(c(category="promotion", confidence=0.999, spam=0.1, bulk=0.999, useful=0.01), Policy(), None, ["prior-correspondent"])
    assert d.action == "review"
    assert d.reason == "safety-signal:prior-correspondent"


def test_legitimate_model_category_with_soft_signal_is_kept():
    d = decide(c(category="notification", confidence=0.7, spam=0.2, useful=0.8), Policy(), None, ["gmail-personal"])
    assert d.action == "keep"
    assert "safety-signal:gmail-personal" in d.reason


def test_career_category_goes_to_review_for_user_choice():
    d = decide(c(category="career", confidence=0.95, spam=0.1, bulk=0.9, useful=0.9), Policy(), None)
    assert d.action == "review"
    assert d.reason == "career-opportunity"
