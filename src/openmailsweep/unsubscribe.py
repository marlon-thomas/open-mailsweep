from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from email.utils import parseaddr
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .config import UnsubscribePolicy
from .models import UnsubscribePlan, UnsubscribeResult


_URI_RE = re.compile(r"<\s*([^>]+?)\s*>")


def _header_uris(value: str) -> list[str]:
    if not value:
        return []
    bracketed = [m.group(1).strip() for m in _URI_RE.finditer(value)]
    if bracketed:
        return bracketed
    return [part.strip() for part in value.split(",") if part.strip()]


def _dkim_covers(headers: dict[str, str], required_headers: set[str]) -> bool:
    auth_results = headers.get("authentication-results", "").lower()
    if "dkim=pass" not in auth_results:
        return False

    signatures = headers.get("dkim-signature", "")
    for signature in signatures.split("\n"):
        match = re.search(r"(?:^|;)\s*h\s*=\s*([^;]+)", signature, flags=re.IGNORECASE)
        if not match:
            continue
        signed_headers = {item.strip().lower() for item in match.group(1).split(":")}
        if required_headers.issubset(signed_headers):
            return True
    return False


def discover_unsubscribe(headers: dict[str, str], require_dkim: bool = True) -> UnsubscribePlan:
    """Discover all standard unsubscribe mechanisms and choose the best safe one.

    The plan keeps both HTTPS one-click and mailto candidates when present. The
    executor prefers authenticated RFC 8058 one-click, then falls back to an
    authenticated List-Unsubscribe mailto address only after a definite failure.
    Ordinary web preference links are intentionally never followed automatically.
    """
    list_unsubscribe = headers.get("list-unsubscribe", "")
    post = headers.get("list-unsubscribe-post", "").strip()
    uris = _header_uris(list_unsubscribe)

    https_url = next((u for u in uris if u.lower().startswith("https://")), None)
    mailto_url = next((u for u in uris if u.lower().startswith("mailto:")), None)
    one_click = post.lower() == "list-unsubscribe=one-click" and https_url is not None

    one_click_dkim = (
        _dkim_covers(headers, {"list-unsubscribe", "list-unsubscribe-post"}) if one_click else False
    )
    mailto_dkim = _dkim_covers(headers, {"list-unsubscribe"}) if mailto_url else False

    one_click_eligible = bool(
        one_click and (not require_dkim or one_click_dkim)
    )
    mailto_eligible = bool(
        mailto_url and (not require_dkim or mailto_dkim)
    )

    host = None
    if https_url:
        try:
            host = urlsplit(https_url).hostname
        except ValueError:
            host = None

    if one_click_eligible:
        method = "one_click"
        reason = "authenticated RFC 8058 one-click unsubscribe available"
        if mailto_eligible:
            reason += "; authenticated mailto fallback also available"
    elif mailto_eligible:
        method = "mailto"
        reason = "authenticated List-Unsubscribe mailto fallback available"
    elif one_click:
        method = "one_click"
        reason = "one-click headers present but required DKIM evidence is missing"
    elif https_url:
        method = "web"
        reason = "ordinary web unsubscribe link requires manual interaction"
    elif mailto_url:
        method = "mailto"
        reason = "mailto unsubscribe detected but required DKIM evidence is missing"
    else:
        method = "none"
        reason = "no standard unsubscribe header found"

    return UnsubscribePlan(
        method=method,
        https_url=https_url,
        mailto_url=mailto_url,
        target_host=host,
        one_click=one_click,
        dkim_authenticated=one_click_dkim,
        one_click_auto_eligible=one_click_eligible,
        mailto_authenticated=mailto_dkim,
        mailto_auto_eligible=mailto_eligible,
        auto_eligible=one_click_eligible or mailto_eligible,
        reason=reason,
    )


def _default_public_target_checker(url: str) -> tuple[bool, str]:
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        return False, f"invalid URL: {exc}"

    if parsed.scheme.lower() != "https":
        return False, "only HTTPS unsubscribe targets are allowed"
    if not parsed.hostname:
        return False, "missing unsubscribe hostname"
    if parsed.username or parsed.password:
        return False, "userinfo in unsubscribe URL is not allowed"

    try:
        port = parsed.port
    except ValueError:
        return False, "invalid unsubscribe port"
    if port not in {None, 443}:
        return False, "only the default HTTPS port is allowed"

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return False, "local unsubscribe targets are blocked"

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return (literal.is_global, "" if literal.is_global else "non-public IP target is blocked")

    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        return False, f"DNS lookup failed: {exc}"

    addresses = {item[4][0] for item in infos}
    if not addresses:
        return False, "unsubscribe host did not resolve"
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False, "unsubscribe host resolved to an invalid IP"
        if not ip.is_global:
            return False, "unsubscribe host resolved to a non-public IP"
    return True, ""


def parse_mailto_unsubscribe(uri: str) -> tuple[str, str, str]:
    """Return a safe single recipient, subject and body from a mailto URI."""
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise ValueError(f"invalid mailto URI: {exc}") from exc
    if parsed.scheme.lower() != "mailto":
        raise ValueError("unsubscribe URI is not mailto")

    raw_to = unquote(parsed.path or "").strip()
    # List-Unsubscribe mailto should designate one address. Reject multi-target
    # forms rather than accidentally sending an automated message to many people.
    if not raw_to or any(ch in raw_to for ch in "\r\n,;"):
        raise ValueError("mailto unsubscribe must contain exactly one recipient")
    _, address = parseaddr(raw_to)
    address = address.strip().lower()
    if not address or "@" not in address or address != raw_to.lower():
        raise ValueError("invalid mailto unsubscribe recipient")

    values = parse_qs(parsed.query, keep_blank_values=True)
    subject = (values.get("subject", ["unsubscribe"])[0] or "unsubscribe").strip()
    body = (values.get("body", ["unsubscribe"])[0] or "unsubscribe").strip()
    if any(ch in subject for ch in "\r\n"):
        raise ValueError("invalid newline in mailto subject")
    # Keep automated unsubscribe messages intentionally tiny.
    subject = subject[:300]
    body = body[:1000]
    return address, subject, body


MailtoSender = Callable[[str, str, str], str | None]


class UnsubscribeClient:
    def __init__(
        self,
        policy: UnsubscribePolicy,
        *,
        transport: httpx.BaseTransport | None = None,
        public_target_checker: Callable[[str], tuple[bool, str]] | None = None,
    ):
        self.policy = policy
        self.transport = transport
        self.public_target_checker = public_target_checker or _default_public_target_checker

    def inspect(self, headers: dict[str, str]) -> UnsubscribePlan:
        return discover_unsubscribe(headers, require_dkim=self.policy.require_dkim)

    def _one_click(self, plan: UnsubscribePlan) -> UnsubscribeResult:
        if not self.policy.auto_one_click:
            return UnsubscribeResult("skipped", "one_click", "automatic one-click unsubscribe is disabled", plan.target_host)
        if not plan.one_click_auto_eligible or not plan.https_url:
            return UnsubscribeResult("unavailable", "one_click", "safe one-click unsubscribe is unavailable", plan.target_host)

        allowed, reason = self.public_target_checker(plan.https_url)
        if not allowed:
            return UnsubscribeResult("blocked", "one_click", reason, plan.target_host)

        try:
            with httpx.Client(
                timeout=self.policy.request_timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={"User-Agent": "OpenMailSweep/0.9.6"},
            ) as client:
                response = client.post(
                    plan.https_url,
                    content="List-Unsubscribe=One-Click",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        except httpx.RequestError as exc:
            # With a transport error it can be impossible to know whether the
            # remote server accepted the request before the connection failed.
            # Do not send a second unsubscribe through another mechanism.
            return UnsubscribeResult(
                "uncertain",
                "one_click",
                f"network result uncertain; no fallback attempted: {exc}",
                plan.target_host,
            )

        if 200 <= response.status_code < 300:
            return UnsubscribeResult("success", "one_click", "one-click unsubscribe accepted", plan.target_host, response.status_code)
        if 300 <= response.status_code < 400:
            return UnsubscribeResult("failed", "one_click", "redirect refused for safety", plan.target_host, response.status_code)
        return UnsubscribeResult(
            "failed",
            "one_click",
            f"unsubscribe endpoint returned HTTP {response.status_code}",
            plan.target_host,
            response.status_code,
        )

    def _mailto(self, plan: UnsubscribePlan, send_mailto: MailtoSender | None) -> UnsubscribeResult:
        if not self.policy.auto_mailto:
            return UnsubscribeResult("skipped", "mailto", "automatic mailto unsubscribe is disabled")
        if not plan.mailto_auto_eligible or not plan.mailto_url:
            return UnsubscribeResult("unavailable", "mailto", "safe mailto unsubscribe is unavailable")
        if send_mailto is None:
            return UnsubscribeResult("unavailable", "mailto", "Gmail mail-sending callback is unavailable")

        try:
            recipient, subject, body = parse_mailto_unsubscribe(plan.mailto_url)
        except ValueError as exc:
            return UnsubscribeResult("failed", "mailto", f"unsubscribe email is unsafe or invalid: {exc}")

        # Let Gmail/API exceptions propagate to the action worker. In
        # particular, GmailRateLimitError must requeue the action rather than
        # being converted into a permanent unsubscribe failure.
        message_id = send_mailto(recipient, subject, body)
        detail = f"unsubscribe email sent to {recipient}"
        if message_id:
            detail += f" (Gmail message {message_id})"
        return UnsubscribeResult("success", "mailto", detail)

    def execute(self, plan: UnsubscribePlan, *, send_mailto: MailtoSender | None = None) -> UnsubscribeResult:
        """Use the best safe unsubscribe method without asking the user to choose.

        Preference order:
          1. authenticated RFC 8058 HTTPS one-click
          2. authenticated List-Unsubscribe mailto via the user's Gmail account

        A mailto fallback is attempted only after a *definite* one-click failure
        (HTTP response or local target block). Network-uncertain outcomes stop to
        avoid duplicate unsubscribe requests. Ordinary web links are never
        browsed automatically.
        """
        attempts: list[dict[str, object]] = []

        if plan.one_click_auto_eligible:
            result = self._one_click(plan)
            attempts.append({
                "method": "one_click",
                "status": result.status,
                "detail": result.detail,
                "http_status": result.http_status,
            })
            if result.status == "success":
                result.attempts = attempts
                return result
            if result.status == "uncertain":
                result.attempts = attempts
                return result
            # blocked/failed/skipped/unavailable are definite enough to try
            # authenticated mailto if the message supplied one.

        if plan.mailto_auto_eligible:
            result = self._mailto(plan, send_mailto)
            attempts.append({
                "method": "mailto",
                "status": result.status,
                "detail": result.detail,
                "http_status": result.http_status,
            })
            result.attempts = attempts
            if result.status == "success":
                return result
            # Return the final attempted method while retaining the full trace.
            return result

        if attempts:
            last = attempts[-1]
            return UnsubscribeResult(
                str(last["status"]),
                str(last["method"]),
                str(last["detail"]),
                plan.target_host,
                last.get("http_status") if isinstance(last.get("http_status"), int) else None,
                attempts=attempts,
            )

        return UnsubscribeResult(
            "unavailable",
            plan.method,
            plan.reason,
            plan.target_host,
            attempts=[],
        )
