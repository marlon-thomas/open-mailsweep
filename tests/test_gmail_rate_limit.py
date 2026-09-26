from types import SimpleNamespace

from openmailsweep.gmail_client import GmailClient, GmailRateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds: float):
        self.sleeps.append(seconds)
        self.now += seconds


def test_weighted_rate_limiter_paces_after_burst():
    fake = FakeClock()
    events = []
    limiter = GmailRateLimiter(
        units_per_minute=60,  # 1 quota unit / second
        burst_units=40,
        event_callback=events.append,
        clock=fake.clock,
        sleeper=fake.sleep,
    )

    limiter.acquire(40, "threads.get")
    assert fake.sleeps == []

    limiter.acquire(20, "messages.get")
    assert fake.sleeps == [20.0]
    assert any(e.get("state") == "pacing" for e in events)
    assert events[-1]["state"] == "ready"


def test_rate_limit_error_detection_for_gmail_403():
    exc = SimpleNamespace(
        resp=SimpleNamespace(status=403),
        content=b'{"error":{"errors":[{"reason":"rateLimitExceeded"}],"message":"Quota exceeded"}}',
    )
    assert GmailClient._is_rate_limit_error(exc) is True


def test_non_quota_403_is_not_treated_as_rate_limit():
    exc = SimpleNamespace(
        resp=SimpleNamespace(status=403),
        content=b'{"error":{"errors":[{"reason":"forbidden"}],"message":"Forbidden"}}',
    )
    assert GmailClient._is_rate_limit_error(exc) is False


def test_move_to_read_later_adds_label_and_removes_inbox():
    captured = {}

    class Messages:
        def modify(self, **kwargs):
            captured["modify"] = kwargs
            return "request"

    class Users:
        def messages(self):
            return Messages()

    class Service:
        def users(self):
            return Users()

    fake = SimpleNamespace(
        service=Service(),
        ensure_label=lambda name: "Label_123",
        _execute=lambda request, **kwargs: captured.update({"request": request, "execute": kwargs}),
    )

    GmailClient.move_to_read_later(fake, "m1", "Read Later")

    assert captured["modify"]["userId"] == "me"
    assert captured["modify"]["id"] == "m1"
    assert captured["modify"]["body"] == {
        "addLabelIds": ["Label_123"],
        "removeLabelIds": ["INBOX"],
    }
    assert captured["execute"]["operation"] == "messages.modify(read-later)"
