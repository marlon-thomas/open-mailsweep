from __future__ import annotations

"""Yahoo Mail provider over IMAP4rev1 + SMTP.

Yahoo has no Gmail-style REST API for consumer accounts, so OpenMailSweep uses
standard IMAP (imap.mail.yahoo.com, SSL) with an app password and SMTP for the
one action that sends mail (authenticated mailto unsubscribe). Message ids are
encoded as ``<mailbox>:<uid>`` because IMAP UIDs are per-folder, unlike Gmail's
global message ids.

One client instance == one IMAP connection. Following the existing worker
design, every thread builds its own client; nothing here is shared across
threads. imaplib is not thread-safe, so this design is a requirement, not an
optimisation. Commands are paced through the same weighted limiter the Gmail
path uses (one IMAP command = one unit) so aggregate traffic stays polite.
"""

import email as email_lib
import imaplib
import re
import smtplib
import ssl
import time
from email.message import EmailMessage as MimeEmailMessage
from email.utils import parseaddr
from typing import Any, Iterator

from .gmail_client import GmailRateLimiter
from .models import EmailMessage
from .provider import RateEventCallback, split_message_id, encode_message_id

METADATA_HEADER_NAMES = [
    "From",
    "To",
    "Subject",
    "Date",
    "Message-ID",
    "In-Reply-To",
    "References",
    "List-ID",
    "List-Unsubscribe",
    "List-Unsubscribe-Post",
    "DKIM-Signature",
    "Authentication-Results",
    "Precedence",
    "Auto-Submitted",
]

YAHOO_INBOX_FLAGS_TO_LABELS = {
    "\\Flagged": "STARRED",
    "$Important": "IMPORTANT",
}


class YahooAuthError(RuntimeError):
    pass


class YahooClient:
    """IMAP-backed provider exposing the OpenMailSweep ``MailProvider`` contract."""

    def __init__(
        self,
        settings: Any,
        *,
        limiter: GmailRateLimiter | None = None,
        rate_event: RateEventCallback | None = None,
    ):
        self.settings = settings
        self.rate_event = rate_event
        self.limiter = limiter or GmailRateLimiter(600, 60, event_callback=rate_event)
        self._imap: imaplib.IMAP4_SSL | None = None
        self._folders: set[str] | None = None
        self._delimiter = "/"
        if not (settings.yahoo_email and settings.yahoo_app_password):
            raise YahooAuthError("MAIL_PROVIDER=yahoo requires YAHOO_EMAIL and YAHOO_APP_PASSWORD")

    # ------------------------------------------------------------- plumbing

    def _emit_rate(self, **event: Any) -> None:
        if self.rate_event is not None:
            try:
                self.rate_event(event)
            except Exception:
                pass

    @property
    def is_authenticated(self) -> bool:
        return True

    def _connect(self) -> imaplib.IMAP4_SSL:
        self._imap = imaplib.IMAP4_SSL(
            self.settings.yahoo_imap_host, self.settings.yahoo_imap_port, timeout=30
        )
        try:
            self._imap.login(self.settings.yahoo_email, self.settings.yahoo_app_password)
        except imaplib.IMAP4.error as exc:
            raise YahooAuthError(
                "Yahoo IMAP login failed. Yahoo rejects normal account passwords for "
                "third-party clients; generate an app password at "
                "https://mail.yahoo.com -> Settings -> Accounts -> Manage app passwords."
            ) from exc
        return self._imap

    def _cmd(self, name: str, fn):
        """Run one paced IMAP command with one reconnect-and-retry on socket loss."""
        self.limiter.acquire(1, f"imap.{name}")
        for attempt in (0, 1):
            conn = self._imap or self._connect()
            try:
                return fn(conn)
            except (imaplib.IMAP4.abort, EOFError, OSError, ssl.SSLError) as exc:
                if attempt:
                    raise RuntimeError(f"Yahoo IMAP command {name!r} failed twice: {exc}") from exc
                self._close_quietly()
                self._imap = None
                self._folders = None
        raise RuntimeError(f"unreachable: Yahoo IMAP command {name!r}")  # pragma: no cover

    def _close_quietly(self) -> None:
        if self._imap is not None:
            try:
                self._imap.logout()
            except Exception:
                pass

    @staticmethod
    def _ok(typ: str) -> bool:
        return typ == "OK"

    def _select(self, conn: imaplib.IMAP4_SSL, mailbox: str, readonly: bool = False):
        typ, data = conn.select(self._quote(mailbox), readonly=readonly)
        if not self._ok(typ):
            raise RuntimeError(f"Yahoo mailbox {mailbox!r} unavailable ({typ})")
        return data

    @staticmethod
    def _quote(mailbox: str) -> str:
        return '"' + mailbox.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def list_folders(self) -> set[str]:
        if self._folders is None:
            def run(conn: imaplib.IMAP4_SSL) -> set[str]:
                typ, lines = conn.list()
                folders: set[str] = set()
                if self._ok(typ):
                    for line in lines or []:
                        try:
                            decoded = line.decode()
                            folders.add(decoded.split(' "/" ')[-1].strip('"'))
                        except Exception:
                            continue
                return folders

            self._folders = self._cmd("list", run) or {"INBOX", "Sent", "Trash", "Archive"}
        return self._folders

    def ensure_folder(self, name: str) -> str:
        folders = self.list_folders()
        if name in folders:
            return name

        def run(conn: imaplib.IMAP4_SSL) -> str:
            typ, _ = conn.create(self._quote(name))
            if not self._ok(typ) and "exist" not in str(typ).lower():
                raise RuntimeError(f"cannot create Yahoo folder {name!r}: {typ}")
            folders.add(name)
            return name

        return self._cmd("create", run)

    # ------------------------------------------------------------ ingestion

    def translate_query(self, query: str) -> tuple[str, str]:
        """Map the (small) Gmail query subset we ship by default onto IMAP.

        Supports ``in:<mailbox>`` and ``newer_than:<N>d``; anything else is
        treated as a full-folder scan of the resolved mailbox. This keeps the
        default ``in:inbox`` workflow working while making the mapping explicit
        rather than silently guessing.
        """
        mailbox = "INBOX"
        criteria_parts: list[str] = ["ALL"]
        lowered = (query or "").lower()
        if "in:inbox" not in lowered and "in:" in lowered:
            token = lowered.split("in:", 1)[1].split()[0].strip("()")
            if token and token != "inbox":
                mailbox = token.title() if token != "sent" else "Sent"
        newer = None
        for term in lowered.replace("(", " ").replace(")", " ").split():
            if term.startswith("newer_than:") and term.endswith("d"):
                try:
                    newer = int(term.split(":", 1)[1][:-1])
                except ValueError:
                    newer = None
        if newer:
            since = time.strftime("%d-%b-%Y", time.gmtime(time.time() - newer * 86400))
            criteria_parts = [f'SINCE "{since}"']
        return mailbox, " ".join(criteria_parts)

    def iter_message_id_pages(self, query: str, page_size: int = 100) -> Iterator[tuple[list[str], int | None]]:
        page_size = max(1, min(500, int(page_size)))
        mailbox, criteria = self.translate_query(query)

        def run(conn: imaplib.IMAP4_SSL) -> list[str]:
            self._select(conn, mailbox, readonly=True)
            typ, data = conn.search(None, criteria)
            if not self._ok(typ):
                raise RuntimeError(f"Yahoo SEARCH failed: {typ}")
            return [str(num) for num in (data[0].split() if data and data[0] else [])]

        uids = self._cmd("search", run)
        estimate = len(uids)
        for start in range(0, len(uids), page_size):
            yield [encode_message_id(mailbox, uid) for uid in uids[start : start + page_size]], estimate

    def list_message_ids(self, query: str, limit: int) -> Iterator[str]:
        yielded = 0
        limit = max(1, int(limit))
        for ids, _ in self.iter_message_id_pages(query, page_size=min(500, limit)):
            for message_id in ids:
                yield message_id
                yielded += 1
                if yielded >= limit:
                    return

    def _fetch(self, encoded_id: str, spec: str):
        mailbox, uid = split_message_id(encoded_id)

        def run(conn: imaplib.IMAP4_SSL):
            self._select(conn, mailbox, readonly=True)
            typ, data = conn.uid("FETCH", uid.encode("ascii"), spec)
            if not self._ok(typ):
                raise LookupError(f"Yahoo message {encoded_id!r} not found ({typ})")
            return data

        return self._cmd("fetch", run)

    @staticmethod
    def _payload_from_fetch(data: list[Any]) -> tuple[bytes, list[str]]:
        raw = b""
        flags: list[str] = []
        for item in data or []:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            meta = item[0].decode("utf-8", errors="replace")
            body = item[1]
            if isinstance(body, (bytes, bytearray)):
                raw = bytes(body)
            match = re.search(r"FLAGS\s*\(([^)]*)\)", meta)
            if match:
                flags = match.group(1).split()
        return raw, flags

    def _message_from_fetch(self, encoded_id: str, spec: str, *, include_body: bool) -> EmailMessage:
        data = self._fetch(encoded_id, spec)
        raw, flags = self._payload_from_fetch(data)
        parsed = email_lib.message_from_bytes(raw)
        headers: dict[str, str] = {}
        for name, value in parsed.items():
            key = name.lower()
            if key in headers:
                headers[key] = headers[key] + "\n" + value
            else:
                headers[key] = value
        sender = headers.get("from", "")
        _, sender_address = parseaddr(sender)
        labels: list[str] = []
        mailbox, _ = split_message_id(encoded_id)
        if mailbox.upper() == "INBOX":
            labels.append("INBOX")
        for flag in flags:
            mapped = YAHOO_INBOX_FLAGS_TO_LABELS.get(flag)
            if mapped:
                labels.append(mapped)
        body = ""
        if include_body:
            body = self._extract_text(parsed)
        message_id = headers.get("message-id", "").strip() or encoded_id
        return EmailMessage(
            id=encoded_id,
            thread_id=message_id,
            subject=headers.get("subject", ""),
            sender=sender,
            sender_address=(sender_address or "").lower(),
            body=body,
            snippet=(body or "")[:200] if include_body else "",
            labels=labels,
            headers=headers,
        )

    def get_message_metadata(self, message_id: str) -> EmailMessage:
        fields = " ".join(METADATA_HEADER_NAMES)
        spec = f"(FLAGS BODY.PEEK[HEADER.FIELDS ({fields})])"
        return self._message_from_fetch(message_id, spec, include_body=False)

    def get_message(self, message_id: str) -> EmailMessage:
        return self._message_from_fetch(message_id, "(FLAGS RFC822)", include_body=True)

    @staticmethod
    def _extract_text(parsed) -> str:
        parts: list[str] = []
        for part in parsed.walk():
            if part.get_content_maintype() == "multipart":
                continue
            ctype = part.get_content_type()
            if ctype not in {"text/plain", "text/html"}:
                continue
            try:
                payload = part.get_payload(decode=True) or b""
                text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            except Exception:
                continue
            if ctype == "text/html":
                import re as _re

                text = _re.sub(r"<[^>]+>", " ", text)
            parts.append(text.strip())
            if ctype == "text/plain":
                break
        return "\n\n".join(parts)

    # ------------------------------------------------------------- history

    def thread_has_sent_message(self, thread_id: str) -> bool:
        mid = (thread_id or "").strip()
        if not mid:
            return False

        def run(conn: imaplib.IMAP4_SSL) -> bool:
            try:
                self._select(conn, "Sent", readonly=True)
                typ, data = conn.search(None, "HEADER", "In-Reply-To", f'"{mid}"')
                if self._ok(typ) and data and data[0]:
                    return True
                typ, data = conn.search(None, "HEADER", "References", f'"{mid}"')
                return bool(self._ok(typ) and data and data[0])
            except Exception:
                return False

        return self._cmd("search", run)

    def has_sent_to_address(self, address: str) -> bool:
        address = (address or "").strip().lower()
        if not address or "@" not in address:
            return False

        def run(conn: imaplib.IMAP4_SSL) -> bool:
            try:
                self._select(conn, "Sent", readonly=True)
                typ, data = conn.search(None, "TO", f'"{address}"')
                return bool(self._ok(typ) and data and data[0])
            except Exception:
                return False

        return self._cmd("search", run)

    # ------------------------------------------------------------- mutations

    def _move(self, encoded_id: str, destination: str) -> bool:
        """COPY then verify then delete, because Yahoo does not advertise MOVE.

        The COPYUID verification avoids the classic IMAP data-loss pattern of
        deleting the original when the copy silently failed.
        """
        mailbox, uid = split_message_id(encoded_id)
        target = self.ensure_folder(destination)
        msgid = ""
        try:
            msgid = self.get_message_metadata(encoded_id).headers.get("message-id", "").strip()
        except Exception:
            pass

        def run(conn: imaplib.IMAP4_SSL) -> bool:
            self._select(conn, mailbox)
            typ, _ = conn.uid("COPY", uid.encode("ascii"), self._quote(target))
            if not self._ok(typ):
                return False
            self._select(conn, target, readonly=True)
            copied = False
            if msgid:
                t2, data = conn.search(None, "HEADER", "MESSAGE-ID", f'"{msgid}"')
                copied = bool(self._ok(t2) and data and data[0])
            if not copied:
                # Without Message-ID confirmation, refuse to delete the source.
                return False
            self._select(conn, mailbox)
            conn.uid("STORE", uid.encode("ascii"), "+FLAGS", r"(\Deleted)")
            conn.expunge()
            return True

        return self._cmd("move", run)

    def _delete_inbox_copy_only(self, encoded_id: str) -> None:
        # "Archive" semantics for Gmail is *remove INBOX label*; Yahoo has no
        # labels, so archive moves the mail out of Inbox (above). apply_label
        # copies. This helper is used if a destination already holds a copy.
        mailbox, uid = split_message_id(encoded_id)

        def run(conn: imaplib.IMAP4_SSL) -> None:
            self._select(conn, mailbox)
            conn.uid("STORE", uid.encode("ascii"), "+FLAGS", r"(\Deleted)")
            conn.expunge()

        self._cmd("delete", run)

    def archive(self, message_id: str) -> None:
        if not self._move(message_id, "Archive"):
            raise RuntimeError(f"Yahoo archive move failed for {message_id}")

    def move_to_read_later(self, message_id: str, label_name: str = "Read Later") -> None:
        if not self._move(message_id, label_name):
            raise RuntimeError(f"Yahoo move to {label_name!r} failed for {message_id}")

    def trash(self, message_id: str) -> None:
        if not self._move(message_id, "Trash"):
            raise RuntimeError(f"Yahoo trash move failed for {message_id}")

    def ensure_label(self, name: str) -> str:
        # Gmail labels map to Yahoo folders; label id == folder name.
        return self.ensure_folder(name)

    def apply_label(self, message_id: str, label_id: str) -> None:
        mailbox, uid = split_message_id(message_id)
        self.ensure_label(label_id)

        def run(conn: imaplib.IMAP4_SSL) -> None:
            self._select(conn, mailbox, readonly=True)
            typ, _ = conn.uid("COPY", uid.encode("ascii"), self._quote(label_id))
            if not self._ok(typ):
                raise RuntimeError(f"Yahoo copy to {label_id!r} failed")

        self._cmd("copy", run)

    # --------------------------------------------------------------- sending

    def send_unsubscribe_email(self, recipient: str, subject: str = "unsubscribe", body: str = "unsubscribe") -> str | None:
        import uuid

        msg = MimeEmailMessage()
        msg["From"] = self.settings.yahoo_email
        msg["To"] = recipient
        msg["Subject"] = subject or "unsubscribe"
        message_id = f"<{uuid.uuid4()}@openmailsweep>"
        msg["Message-ID"] = message_id
        msg.set_content(body or "unsubscribe")
        self.limiter.acquire(20, "smtp.send")
        with smtplib.SMTP(self.settings.yahoo_smtp_host, self.settings.yahoo_smtp_port, timeout=30) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(self.settings.yahoo_email, self.settings.yahoo_app_password)
            smtp.send_message(msg)
        return message_id
