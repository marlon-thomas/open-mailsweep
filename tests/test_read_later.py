from types import SimpleNamespace

from openmailsweep.config import Policy
from openmailsweep.models import Classification, EmailMessage
from openmailsweep.worker import OpenMailSweepService


def test_career_proposes_read_later():
    c = Classification("career", 0.9, 0.1, 0.9, 0.9, 2.0, {})
    plan = SimpleNamespace(auto_eligible=False)
    assert OpenMailSweepService._proposed_action(c, plan) == "read_later"
    assert "Read Later" in OpenMailSweepService._question(SimpleNamespace(), c, plan)


def test_job_alert_subject_is_recognised_without_model():
    fake_service = SimpleNamespace(policy=Policy())
    message = EmailMessage(
        id="1", thread_id="t1", subject="Your job alert for java software engineer",
        sender="LinkedIn <jobs-noreply@linkedin.com>", sender_address="jobs-noreply@linkedin.com",
        body="", snippet="", labels=[], headers={}
    )
    assert OpenMailSweepService._career_subject_signal(fake_service, message) == "job alert"
