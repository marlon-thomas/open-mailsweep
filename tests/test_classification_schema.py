from openmailsweep.classification_schema import build_state_and_questions, parse_classification
from openmailsweep.models import EmailMessage


def test_state_truncates_body():
    msg = EmailMessage(
        id="1", thread_id="t", subject="Hello", sender="a@example.com",
        sender_address="a@example.com", body="x" * 5000, snippet="", labels=[], headers={}
    )
    state, questions = build_state_and_questions(msg, 1200)
    assert len(state["body"]) == 1200
    assert "category" in questions
    assert "spam" in questions


def test_parse_system_one_shape():
    raw = {
        "answers": {
            "category": {"choice": "junk", "confidence": 0.99},
            "spam": {"noul": 0.995},
            "bulk": {"noul": 0.98},
            "useful": {"noul": 0.03},
            "importance": {"score": 0.4},
        }
    }
    c = parse_classification(raw)
    assert c.category == "junk"
    assert c.spam_probability == 0.995


def test_category_schema_includes_safe_legitimate_types():
    msg = EmailMessage(
        id="1", thread_id="t", subject="Hello", sender="a@example.com",
        sender_address="a@example.com", body="body", snippet="", labels=[], headers={}
    )
    _, questions = build_state_and_questions(msg, 1200)
    criteria = questions["category"]["criteria"]
    assert "transactional" in criteria
    assert "notification" in criteria
    assert "community" in criteria
    assert "career" in criteria
