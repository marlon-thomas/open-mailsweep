import httpx

from openmailsweep.config import UnsubscribePolicy
from openmailsweep.unsubscribe import (
    UnsubscribeClient,
    _default_public_target_checker,
    discover_unsubscribe,
)


def authenticated_headers():
    return {
        "list-unsubscribe": "<mailto:list@example.com?subject=unsubscribe>, <https://example.com/u/opaque-token>",
        "list-unsubscribe-post": "List-Unsubscribe=One-Click",
        "authentication-results": "mx.google.com; dkim=pass header.i=@example.com",
        "dkim-signature": "v=1; a=rsa-sha256; h=from:to:subject:list-unsubscribe:list-unsubscribe-post; bh=x; b=y",
    }


def test_discovers_authenticated_one_click():
    plan = discover_unsubscribe(authenticated_headers())
    assert plan.method == "one_click"
    assert plan.auto_eligible is True
    assert plan.target_host == "example.com"


def test_one_click_without_dkim_is_not_auto_eligible():
    headers = authenticated_headers()
    headers.pop("authentication-results")
    plan = discover_unsubscribe(headers)
    assert plan.method == "one_click"
    assert plan.auto_eligible is False


def test_plain_web_link_is_manual_only():
    plan = discover_unsubscribe({"list-unsubscribe": "<https://example.com/preferences>"})
    assert plan.method == "web"
    assert plan.auto_eligible is False


def test_private_ip_target_is_blocked():
    allowed, _ = _default_public_target_checker("https://127.0.0.1/unsubscribe")
    assert allowed is False


def test_one_click_post_has_no_cookie_or_authorization():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.content.decode()
        seen["cookie"] = request.headers.get("cookie")
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(204)

    plan = discover_unsubscribe(authenticated_headers())
    client = UnsubscribeClient(
        UnsubscribePolicy(),
        transport=httpx.MockTransport(handler),
        public_target_checker=lambda _: (True, ""),
    )
    result = client.execute(plan)
    assert result.status == "success"
    assert seen == {
        "method": "POST",
        "body": "List-Unsubscribe=One-Click",
        "cookie": None,
        "authorization": None,
    }


def authenticated_mailto_only_headers():
    return {
        "list-unsubscribe": "<mailto:unsubscribe@example.com?subject=remove%20me&body=unsubscribe>",
        "authentication-results": "mx.google.com; dkim=pass header.i=@example.com",
        "dkim-signature": "v=1; a=rsa-sha256; h=from:to:subject:list-unsubscribe; bh=x; b=y",
    }


def test_authenticated_mailto_is_auto_eligible():
    plan = discover_unsubscribe(authenticated_mailto_only_headers())
    assert plan.method == "mailto"
    assert plan.auto_eligible is True
    assert plan.mailto_auto_eligible is True


def test_one_click_failure_falls_back_to_mailto():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    plan = discover_unsubscribe(authenticated_headers())
    client = UnsubscribeClient(
        UnsubscribePolicy(),
        transport=httpx.MockTransport(handler),
        public_target_checker=lambda _: (True, ""),
    )
    result = client.execute(
        plan,
        send_mailto=lambda to, subject, body: sent.append((to, subject, body)) or "gmail-msg-1",
    )
    assert result.status == "success"
    assert result.method == "mailto"
    assert [a["method"] for a in result.attempts] == ["one_click", "mailto"]
    assert sent == [("list@example.com", "unsubscribe", "unsubscribe")]


def test_one_click_success_does_not_send_mailto():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    plan = discover_unsubscribe(authenticated_headers())
    client = UnsubscribeClient(
        UnsubscribePolicy(),
        transport=httpx.MockTransport(handler),
        public_target_checker=lambda _: (True, ""),
    )
    result = client.execute(
        plan,
        send_mailto=lambda to, subject, body: sent.append((to, subject, body)) or "gmail-msg-1",
    )
    assert result.status == "success"
    assert result.method == "one_click"
    assert sent == []
    assert len(result.attempts) == 1


def test_network_uncertain_one_click_does_not_fallback():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    plan = discover_unsubscribe(authenticated_headers())
    client = UnsubscribeClient(
        UnsubscribePolicy(),
        transport=httpx.MockTransport(handler),
        public_target_checker=lambda _: (True, ""),
    )
    result = client.execute(
        plan,
        send_mailto=lambda to, subject, body: sent.append((to, subject, body)) or "gmail-msg-1",
    )
    assert result.status == "uncertain"
    assert result.method == "one_click"
    assert sent == []


def test_mailto_rejects_multiple_recipients():
    from openmailsweep.unsubscribe import parse_mailto_unsubscribe

    try:
        parse_mailto_unsubscribe("mailto:a@example.com,b@example.com?subject=unsubscribe")
    except ValueError as exc:
        assert "exactly one recipient" in str(exc)
    else:
        raise AssertionError("multi-recipient mailto should be rejected")


def test_mailto_sender_exceptions_propagate_for_queue_retry():
    plan = discover_unsubscribe(authenticated_mailto_only_headers())
    client = UnsubscribeClient(UnsubscribePolicy())

    class TemporarySendError(RuntimeError):
        pass

    try:
        client.execute(plan, send_mailto=lambda *_: (_ for _ in ()).throw(TemporarySendError("retry me")))
    except TemporarySendError as exc:
        assert "retry me" in str(exc)
    else:
        raise AssertionError("Gmail send errors should propagate to the action worker")
