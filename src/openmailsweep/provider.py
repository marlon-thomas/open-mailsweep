from __future__ import annotations

"""Email provider abstraction.

OpenMailSweep drives Gmail (REST API) and Yahoo Mail (IMAP4rev1 + SMTP) through
one interface so intake, safety, learned rules, the local classifier, and the
concurrent action pool behave identically for both. Gmail has no official
public API for consumer accounts beyond the Google client, and Yahoo likewise
exposes standard IMAP/SMTP only, so Yahoo uses imaplib/smtplib with an app
password. Each worker thread owns its own provider client (a Gmail service or
one IMAP connection), exactly as before; clients are never shared across
threads.
"""

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from .models import EmailMessage

if TYPE_CHECKING:  # pragma: no cover
    from .config import Settings

RateEventCallback = Callable[[dict[str, Any]], None]


def encode_message_id(mailbox: str, uid: str) -> str:
    """IMAP UIDs are per-folder; Gmail ids are global. Encode both uniformly."""
    return f"{mailbox}:{uid}"


def split_message_id(encoded_id: str) -> tuple[str, str]:
    if ":" in encoded_id:
        mailbox, uid = encoded_id.split(":", 1)
        return mailbox, uid
    return "INBOX", encoded_id


@runtime_checkable
class MailProvider(Protocol):
    is_authenticated: bool

    def iter_message_id_pages(self, query: str, page_size: int): ...
    def list_message_ids(self, query: str, limit: int): ...
    def get_message_metadata(self, message_id: str) -> EmailMessage: ...
    def get_message(self, message_id: str) -> EmailMessage: ...
    def thread_has_sent_message(self, thread_id: str) -> bool: ...
    def has_sent_to_address(self, address: str) -> bool: ...
    def ensure_label(self, name: str) -> str: ...
    def apply_label(self, message_id: str, label_id: str) -> None: ...
    def archive(self, message_id: str) -> None: ...
    def move_to_read_later(self, message_id: str, label_name: str) -> None: ...
    def trash(self, message_id: str) -> None: ...
    def send_unsubscribe_email(self, recipient: str, subject: str = ..., body: str = ...) -> str | None: ...


def provider_is_ready(settings: "Settings") -> bool:
    """Cheap auth gate used by the worker loops before contacting the provider."""
    if settings.mail_provider == "yahoo":
        return bool(settings.yahoo_email and settings.yahoo_app_password)
    return settings.gmail_token.exists()


def provider_needs_auth_hint(settings: "Settings") -> str:
    if settings.mail_provider == "yahoo":
        return "YAHOO_EMAIL / YAHOO_APP_PASSWORD are missing; set them in .env (Yahoo app password required)"
    return "Gmail OAuth token is missing; run 'openmailsweep auth' first"


def create_mail_provider(settings: "Settings", *, interactive: bool = False, limiter=None, rate_event=None):
    """Build a provider client for the current thread."""
    if settings.mail_provider == "yahoo":
        from .yahoo_client import YahooClient

        return YahooClient(settings, limiter=limiter, rate_event=rate_event)
    from .gmail_client import GmailClient

    return GmailClient(
        settings.gmail_credentials,
        settings.gmail_token,
        interactive=interactive,
        limiter=limiter,
        rate_event=rate_event,
        max_rate_retries=settings.gmail_rate_max_retries,
        max_rate_backoff=settings.gmail_rate_max_backoff,
    )
