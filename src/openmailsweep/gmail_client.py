from __future__ import annotations

import base64
import json
import os
import random
import re
import threading
import time
from email.message import EmailMessage as MimeEmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Callable, Iterator

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .models import EmailMessage

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

# Gmail API quota-unit costs. These are intentionally kept beside the calls so
# OpenMailSweep can pace *weighted* Gmail work, rather than merely limiting request
# count. Current Google docs assign 20 units to messages.get and 40 to
# threads.get, for example.
COST_GET_PROFILE = 1
COST_MESSAGES_LIST = 5
COST_MESSAGES_GET = 20
COST_THREADS_GET = 40
COST_LABELS_LIST = 1
COST_LABELS_CREATE = 5
COST_MESSAGES_MODIFY = 5
COST_MESSAGES_TRASH = 20
COST_MESSAGES_SEND = 100

RateEventCallback = Callable[[dict[str, Any]], None]


class GmailRateLimitError(RuntimeError):
    """Raised only after OpenMailSweep has exhausted Gmail rate-limit retries."""


class GmailRateLimiter:
    """Thread-safe weighted token bucket shared by every Gmail worker.

    The app has intake/classifier threads plus a configurable action-worker pool.
    A single limiter is therefore required: separate per-thread throttles can
    still collectively exceed Gmail's per-user/project quota.
    """

    def __init__(
        self,
        units_per_minute: int = 3600,
        burst_units: int = 240,
        *,
        event_callback: RateEventCallback | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.units_per_minute = max(60, int(units_per_minute))
        self.rate_per_second = self.units_per_minute / 60.0
        # threads.get costs 40 units, so never configure a bucket too small to
        # allow one legitimate Gmail operation through.
        self.capacity = max(40.0, float(burst_units))
        self._tokens = self.capacity
        self._updated = clock()
        self._lock = threading.Lock()
        self._event = event_callback
        self._clock = clock
        self._sleep = sleeper

    def _emit(self, **event: Any) -> None:
        if self._event is not None:
            try:
                self._event(event)
            except Exception:
                # Telemetry must never break Gmail access.
                pass

    def acquire(self, cost: int, operation: str) -> None:
        cost = max(1, int(cost))
        waited = False
        if cost > self.capacity:
            # Allow unusually expensive future methods while preserving the
            # configured refill rate.
            with self._lock:
                self.capacity = float(cost)
                self._tokens = min(self._tokens, self.capacity)

        while True:
            with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated)
                self._updated = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
                if self._tokens >= cost:
                    self._tokens -= cost
                    if waited:
                        self._emit(state="ready", operation=operation, wait_seconds=0.0, detail="Gmail API pacing complete")
                    return
                missing = cost - self._tokens
                wait_seconds = max(0.01, missing / self.rate_per_second)

            waited = True
            self._emit(
                state="pacing",
                operation=operation,
                wait_seconds=wait_seconds,
                detail=f"Local Gmail quota pacing ({self.units_per_minute} units/min)",
            )
            self._sleep(wait_seconds)


class GmailClient:
    def __init__(
        self,
        credentials_file: Path,
        token_file: Path,
        interactive: bool = True,
        *,
        limiter: GmailRateLimiter | None = None,
        rate_event: RateEventCallback | None = None,
        max_rate_retries: int | None = None,
        max_rate_backoff: float | None = None,
    ):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.interactive = interactive
        self.rate_event = rate_event
        self.max_rate_retries = max(
            0,
            int(max_rate_retries if max_rate_retries is not None else os.getenv("GMAIL_RATE_MAX_RETRIES", "7")),
        )
        self.max_rate_backoff = max(
            1.0,
            float(max_rate_backoff if max_rate_backoff is not None else os.getenv("GMAIL_RATE_MAX_BACKOFF", "64")),
        )
        if limiter is None:
            limiter = GmailRateLimiter(
                int(os.getenv("GMAIL_QUOTA_UNITS_PER_MINUTE", "3600")),
                int(os.getenv("GMAIL_QUOTA_BURST_UNITS", "240")),
                event_callback=rate_event,
            )
        self.limiter = limiter
        self.service = build("gmail", "v1", credentials=self._credentials(), cache_discovery=False)
        self._label_cache: dict[str, str] = {}
        self._profile_email: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.token_file.exists()

    def _emit_rate(self, **event: Any) -> None:
        if self.rate_event is not None:
            try:
                self.rate_event(event)
            except Exception:
                pass

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status not in {403, 429}:
            return False
        content = getattr(exc, "content", b"")
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content or "")
        lowered = text.lower()
        if any(
            marker in lowered
            for marker in (
                "ratelimitexceeded",
                "userratelimitexceeded",
                "quota exceeded",
                "usage limits",
                "resource_exhausted",
            )
        ):
            return True
        try:
            payload = json.loads(text)
        except Exception:
            return status == 429
        errors = payload.get("error", {}).get("errors", []) if isinstance(payload, dict) else []
        return any(
            str(item.get("reason", "")).lower() in {"ratelimitexceeded", "userratelimitexceeded"}
            for item in errors
            if isinstance(item, dict)
        ) or status == 429

    def _execute(self, request: Any, *, cost: int, operation: str) -> Any:
        """Execute one Gmail API request with shared pacing and backoff."""
        for attempt in range(self.max_rate_retries + 1):
            self.limiter.acquire(cost, operation)
            try:
                result = request.execute(num_retries=0)
                if attempt:
                    self._emit_rate(
                        state="ready",
                        operation=operation,
                        wait_seconds=0.0,
                        detail=f"Gmail API recovered after {attempt} rate-limit retry/retries",
                    )
                return result
            except HttpError as exc:
                if not self._is_rate_limit_error(exc):
                    raise
                if attempt >= self.max_rate_retries:
                    self._emit_rate(
                        state="error",
                        operation=operation,
                        wait_seconds=0.0,
                        detail="Gmail rate limit still exceeded after all retries",
                    )
                    raise GmailRateLimitError(
                        f"Gmail rate limit exceeded during {operation} after {self.max_rate_retries} retries"
                    ) from exc

                # Google's recommended shape is exponential backoff with jitter.
                wait_seconds = min((2**attempt) + random.random(), self.max_rate_backoff)
                self._emit_rate(
                    state="backoff",
                    operation=operation,
                    wait_seconds=wait_seconds,
                    attempt=attempt + 1,
                    max_attempts=self.max_rate_retries,
                    detail="Gmail returned rateLimitExceeded; backing off and retrying",
                )
                time.sleep(wait_seconds)

        raise GmailRateLimitError(f"Gmail rate limit retry loop unexpectedly exited during {operation}")

    def _credentials(self) -> Credentials:
        creds = None
        if self.token_file.exists():
            # Load the scopes exactly as granted in the existing token. Refresh
            # tokens cannot silently expand OAuth permissions.
            creds = Credentials.from_authorized_user_file(str(self.token_file))
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(creds.to_json(), encoding="utf-8")

        missing_scopes = bool(creds and creds.valid and not creds.has_scopes(SCOPES))
        if missing_scopes and not self.interactive:
            raise RuntimeError(
                "Gmail OAuth token is missing the gmail.send permission required for "
                "automatic mailto unsubscribe fallback. Run 'openmailsweep auth' interactively "
                "or reinstall v0.9.5+ to grant the additional scope."
            )
        if missing_scopes:
            creds = None

        if not creds or not creds.valid:
            if not self.interactive:
                raise RuntimeError("Gmail OAuth token is missing or invalid; run 'openmailsweep auth' interactively")
            if not self.credentials_file.exists():
                raise FileNotFoundError(f"Missing OAuth client file: {self.credentials_file}")
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), SCOPES)
            oauth_port = int(os.getenv("OAUTH_PORT", "8765"))
            creds = flow.run_local_server(host="localhost", bind_addr="0.0.0.0", port=oauth_port, open_browser=False)
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def iter_message_id_pages(self, query: str, page_size: int = 100) -> Iterator[tuple[list[str], int | None]]:
        """Yield every Gmail message ID matching *query*, page by page.

        There is deliberately no total-message cap here. ``page_size`` controls
        only how much is requested from Gmail in one API page (Gmail allows up
        to 500). Weighted pacing prevents continuous streaming from exceeding
        Gmail's per-user request budget.
        """
        page_size = max(1, min(500, int(page_size)))
        page_token = None
        while True:
            request = self.service.users().messages().list(
                userId="me", q=query, maxResults=page_size, pageToken=page_token
            )
            result = self._execute(request, cost=COST_MESSAGES_LIST, operation="messages.list")
            ids = [item["id"] for item in result.get("messages", []) if item.get("id")]
            estimate = result.get("resultSizeEstimate")
            try:
                estimate = int(estimate) if estimate is not None else None
            except (TypeError, ValueError):
                estimate = None
            if ids:
                yield ids, estimate
            page_token = result.get("nextPageToken")
            if not page_token:
                return

    def list_message_ids(self, query: str, limit: int) -> Iterator[str]:
        """Legacy bounded iterator used by explicit CLI commands."""
        yielded = 0
        limit = max(1, int(limit))
        for ids, _ in self.iter_message_id_pages(query, page_size=min(500, limit)):
            for message_id in ids:
                yield message_id
                yielded += 1
                if yielded >= limit:
                    return

    @staticmethod
    def _headers(payload: dict) -> dict[str, str]:
        headers: dict[str, str] = {}
        for header in payload.get("headers", []):
            name = header["name"].lower()
            value = header["value"]
            # Preserve repeated headers (notably DKIM-Signature and
            # Authentication-Results) so unsubscribe authentication checks
            # do not silently discard evidence.
            if name in headers:
                headers[name] = headers[name] + "\n" + value
            else:
                headers[name] = value
        return headers

    @classmethod
    def _message_from_raw(cls, raw: dict, *, include_body: bool) -> EmailMessage:
        payload = raw.get("payload", {})
        headers = cls._headers(payload)
        sender = headers.get("from", "")
        _, sender_address = parseaddr(sender)
        return EmailMessage(
            id=raw["id"],
            thread_id=raw.get("threadId", ""),
            subject=headers.get("subject", ""),
            sender=sender,
            sender_address=sender_address.lower(),
            body=cls._extract_text(payload) if include_body else "",
            snippet=raw.get("snippet", ""),
            labels=raw.get("labelIds", []),
            headers=headers,
        )

    def get_message_metadata(self, message_id: str) -> EmailMessage:
        """Fetch lightweight fields needed for intake/rules/safety checks."""
        request = self.service.users().messages().get(
            userId="me",
            id=message_id,
            format="metadata",
            metadataHeaders=[
                "From",
                "Subject",
                "List-ID",
                "List-Unsubscribe",
                "List-Unsubscribe-Post",
                "DKIM-Signature",
                "Authentication-Results",
                "Precedence",
            ],
        )
        raw = self._execute(request, cost=COST_MESSAGES_GET, operation="messages.get(metadata)")
        return self._message_from_raw(raw, include_body=False)

    def get_message(self, message_id: str) -> EmailMessage:
        request = self.service.users().messages().get(userId="me", id=message_id, format="full")
        raw = self._execute(request, cost=COST_MESSAGES_GET, operation="messages.get(full)")
        return self._message_from_raw(raw, include_body=True)

    def thread_has_sent_message(self, thread_id: str) -> bool:
        request = self.service.users().threads().get(userId="me", id=thread_id, format="metadata")
        thread = self._execute(request, cost=COST_THREADS_GET, operation="threads.get(metadata)")
        return any("SENT" in m.get("labelIds", []) for m in thread.get("messages", []))

    def has_sent_to_address(self, address: str) -> bool:
        """Return True when the user has previously sent mail to this exact address."""
        address = (address or "").strip().lower()
        if not address or "@" not in address:
            return False
        safe = address.replace("\\", "").replace('"', "")
        request = self.service.users().messages().list(userId="me", q=f'in:sent to:"{safe}"', maxResults=1)
        result = self._execute(request, cost=COST_MESSAGES_LIST, operation="messages.list(sent-history)")
        return bool(result.get("messages"))

    def ensure_label(self, name: str) -> str:
        cached = self._label_cache.get(name)
        if cached:
            return cached
        request = self.service.users().labels().list(userId="me")
        labels = self._execute(request, cost=COST_LABELS_LIST, operation="labels.list").get("labels", [])
        for label in labels:
            self._label_cache[label["name"]] = label["id"]
            if label["name"] == name:
                return label["id"]
        request = self.service.users().labels().create(
            userId="me", body={"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
        )
        created = self._execute(request, cost=COST_LABELS_CREATE, operation="labels.create")
        self._label_cache[name] = created["id"]
        return created["id"]

    def apply_label(self, message_id: str, label_id: str) -> None:
        request = self.service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": [label_id]}
        )
        self._execute(request, cost=COST_MESSAGES_MODIFY, operation="messages.modify(label)")

    def archive(self, message_id: str) -> None:
        request = self.service.users().messages().modify(
            userId="me", id=message_id, body={"removeLabelIds": ["INBOX"]}
        )
        self._execute(request, cost=COST_MESSAGES_MODIFY, operation="messages.modify(archive)")

    def move_to_read_later(self, message_id: str, label_name: str = "Read Later") -> None:
        """Move a message out of Inbox and into the configured Read Later label.

        Gmail has labels rather than folders; adding the label and removing INBOX
        in one modify call gives the expected folder-like behaviour.
        """
        label_id = self.ensure_label(label_name)
        request = self.service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": [label_id], "removeLabelIds": ["INBOX"]},
        )
        self._execute(request, cost=COST_MESSAGES_MODIFY, operation="messages.modify(read-later)")

    def _my_email_address(self) -> str:
        if self._profile_email:
            return self._profile_email
        request = self.service.users().getProfile(userId="me")
        profile = self._execute(request, cost=COST_GET_PROFILE, operation="getProfile")
        self._profile_email = str(profile.get("emailAddress") or "").strip()
        if not self._profile_email:
            raise RuntimeError("Gmail profile did not return an email address")
        return self._profile_email

    def send_unsubscribe_email(self, recipient: str, subject: str = "unsubscribe", body: str = "unsubscribe") -> str | None:
        """Send the standard List-Unsubscribe mailto message through Gmail.

        The recipient and content originate only from the authenticated
        List-Unsubscribe header parsed by OpenMailSweep; no arbitrary attachments
        or rich content are sent.
        """
        msg = MimeEmailMessage()
        msg["From"] = self._my_email_address()
        msg["To"] = recipient
        msg["Subject"] = subject or "unsubscribe"
        msg.set_content(body or "unsubscribe")
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        request = self.service.users().messages().send(userId="me", body={"raw": raw})
        result = self._execute(request, cost=COST_MESSAGES_SEND, operation="messages.send(unsubscribe)")
        return result.get("id")

    def trash(self, message_id: str) -> None:
        request = self.service.users().messages().trash(userId="me", id=message_id)
        self._execute(request, cost=COST_MESSAGES_TRASH, operation="messages.trash")

    @classmethod
    def _extract_text(cls, payload: dict) -> str:
        parts: list[str] = []

        def walk(part: dict) -> None:
            mime = part.get("mimeType", "")
            data = part.get("body", {}).get("data")
            if data and mime == "text/plain":
                try:
                    parts.append(base64.urlsafe_b64decode(data + "===").decode("utf-8", errors="replace"))
                except Exception:
                    pass
            for child in part.get("parts", []) or []:
                walk(child)

        walk(payload)
        text = "\n".join(parts)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
