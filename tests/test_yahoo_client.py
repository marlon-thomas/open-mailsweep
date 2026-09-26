"""Yahoo IMAP provider tests using a scripted fake imaplib/smtplib.

Covers the safety-critical behaviours: copy-verified moves (never delete the
original unless the destination copy is confirmed), flag/label mapping to the
shared safety vocabulary (STARRED), query translation, auth gating, and the
provider factory wiring.
"""

from pathlib import Path

import pytest

from openmailsweep.config import Settings
from openmailsweep.models import EmailMessage
from openmailsweep import yahoo_client as yc
from openmailsweep.provider import (
    create_mail_provider,
    encode_message_id,
    provider_is_ready,
    split_message_id,
)

RAW_MESSAGE = (
    b"From: Daily News <news@daily-example.com>\r\n"
    b"To: me@yahoo.com\r\n"
    b"Subject: Your daily digest: 10 headlines\r\n"
    b"Date: Mon, 01 Jan 2024 08:00:00 +0000\r\n"
    b"Message-ID: <abc-123@daily-example.com>\r\n"
    b"List-ID: <digest.daily-example.com>\r\n"
    b"List-Unsubscribe: <https://daily-example.com/unsub?t=xyz>, <mailto:unsub@daily-example.com>\r\n"
    b"List-Unsubscribe-Post: List-Unsubscribe=One-Click\r\n"
    b"Precedence: bulk\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Top stories inside. Manage preferences or unsubscribe anytime.\r\n"
)


class FakeIMAP:
    def __init__(self, *args, **kwargs):
        self.selected = None
        self.readonly = False
        self.calls: list[tuple] = []
        self.copy_ok = True
        self.copy_confirmation = True

    # imaplib surface -----------------------------------------------------
    def login(self, user, password):
        self.calls.append(("login", user, password))
        return ("OK", [b"logged in"])

    def select(self, mailbox, readonly=False):
        self.selected = mailbox
        self.readonly = readonly
        self.calls.append(("select", mailbox, readonly))
        return ("OK", [b"4"])

    def list(self, *args):
        self.calls.append(("list",))
        return (
            "OK",
            [
                b'(\\HasNoChildren) "/" "INBOX"',
                b'(\\HasNoChildren) "/" "Sent"',
                b'(\\HasNoChildren) "/" "Trash"',
            ],
        )

    def create(self, mailbox):
        self.calls.append(("create", mailbox))
        return ("OK", [b"created"])

    def search(self, charset, *criteria):
        self.calls.append(("search", self.selected, criteria))
        joined = b" ".join(c if isinstance(c, bytes) else str(c).encode() for c in criteria).upper()
        if self.selected == '"Sent"' or self.selected == "Sent":
            if b"TO" in joined.split():
                return ("OK", [b"77"] if b"FRIEND@EXAMPLE.COM" in joined else [b""])
            if self.copy_confirmation and b"IN-REPLY-TO" in joined:
                return ("OK", [b"78"])
            return ("OK", [b""])
        if b"MESSAGE-ID" in joined:
            return ("OK", [b"9"] if self.copy_confirmation else [b""])
        return ("OK", [b"1 2 3"])

    def uid(self, command, uid, *args):
        uid_text = uid.decode() if isinstance(uid, bytes) else str(uid)
        self.calls.append(("uid", command, uid_text, args))
        if command == "FETCH":
            spec = args[0]
            flags = r"(UID 4 FLAGS (\Seen \Flagged) BODY[HEADER] {80}"
            return ("OK", [(flags.encode(), RAW_MESSAGE), b")"])
        if command == "COPY":
            return ("OK", [b"[COPYUID 1 4 9]"]) if self.copy_ok else ("NO", [b"failed"])
        if command == "STORE":
            return ("OK", [(b"(UID 4 FLAGS (\\Deleted))", b"")])
        return ("OK", [b""])

    def expunge(self):
        self.calls.append(("expunge",))
        return ("OK", [b"1"])

    def logout(self):
        self.calls.append(("logout",))
        return ("BYE", [b"done"])


class FakeSMTP:
    last = None

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.sent = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.__class__.last = self

    def login(self, user, password):
        self.user = user

    def send_message(self, msg):
        self.sent = msg
        FakeSMTP.last = self


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        mail_provider="yahoo",
        yahoo_email="tester@yahoo.com",
        yahoo_app_password="abcd efgh ijkl mnop",
    )


@pytest.fixture()
def fake_client(monkeypatch, settings):
    fake = FakeIMAP()
    monkeypatch.setattr(yc.imaplib, "IMAP4_SSL", lambda *a, **k: fake)
    monkeypatch.setattr(yc.smtplib, "SMTP", FakeSMTP)
    client = yc.YahooClient(settings)
    return client, fake


def test_metadata_fetch_maps_flags_and_headers(fake_client):
    client, fake = fake_client
    msg = client.get_message_metadata(encode_message_id("INBOX", "4"))
    assert isinstance(msg, EmailMessage)
    assert msg.subject == "Your daily digest: 10 headlines"
    assert msg.sender_address == "news@daily-example.com"
    assert msg.headers["list-unsubscribe-post"] == "List-Unsubscribe=One-Click"
    # \Flagged must map to the shared STARRED label used by hard safety rules.
    assert "STARRED" in msg.labels
    assert msg.thread_id == "<abc-123@daily-example.com>"
    # readonly select: ingestion must never mutate flags implicitly
    assert ("select", '"INBOX"', True) in fake.calls or ("select", "INBOX", True) in fake.calls


def test_message_id_codec_roundtrip():
    mailbox, uid = split_message_id(encode_message_id("Archive", "42"))
    assert (mailbox, uid) == ("Archive", "42")
    # Gmail ids (no mailbox prefix) default to INBOX handling.
    assert split_message_id("18f2abc") == ("INBOX", "18f2abc")


def test_archive_moves_and_deletes_only_after_copy_is_confirmed(fake_client):
    client, fake = fake_client
    client.archive(encode_message_id("INBOX", "4"))
    names = [c[0] if c[0] != "uid" else c[1] for c in fake.calls]
    assert "COPY" in names and "STORE" in names and "expunge" in names


def test_archive_refuses_to_delete_when_copy_unconfirmed(fake_client):
    client, fake = fake_client
    fake.copy_ok = True
    fake.copy_confirmation = False  # destination search comes back empty
    with pytest.raises(RuntimeError, match="archive move failed"):
        client.archive(encode_message_id("INBOX", "4"))
    # The source must never be deleted: no STORE +FLAGS \Deleted happened.
    assert all(
        not (c[0] == "uid" and c[1] == "STORE" and any("\\Deleted" in str(a) for a in c[3]))
        for c in fake.calls
    ), "Yahoo client deleted the source despite an unverified copy"


def test_read_later_creates_folder_and_moves(fake_client):
    client, fake = fake_client
    client.move_to_read_later(encode_message_id("INBOX", "4"), "Read Later")
    assert ("create", '"Read Later"') in fake.calls


def test_sent_history_lookup(fake_client):
    client, fake = fake_client
    assert client.has_sent_to_address("friend@example.com") is True
    assert client.has_sent_to_address("nobody@example.com") is False
    assert client.thread_has_sent_message("<abc-123@daily-example.com>") is True


def test_unsubscribe_email_sent_via_smtp(fake_client, settings):
    client, fake = fake_client
    message_id = client.send_unsubscribe_email("unsub@daily-example.com")
    smtp = FakeSMTP.last
    assert smtp.user == "tester@yahoo.com"
    assert smtp.sent["To"] == "unsub@daily-example.com"
    assert smtp.sent.get_content().strip() == "unsubscribe"
    assert message_id


def test_query_translation():
    settings = Settings(mail_provider="yahoo", yahoo_email="a@b.c", yahoo_app_password="x")
    client = yc.YahooClient(settings)
    assert client.translate_query("in:inbox") == ("INBOX", "ALL")
    mailbox, criteria = client.translate_query("in:inbox newer_than:30d")
    assert mailbox == "INBOX" and criteria.startswith("SINCE")


def test_missing_credentials_fail_fast():
    with pytest.raises(yc.YahooAuthError):
        yc.YahooClient(Settings(mail_provider="yahoo"))


def test_provider_factory_and_auth_gating(settings, monkeypatch):
    assert provider_is_ready(settings) is True
    assert provider_is_ready(Settings(mail_provider="yahoo")) is False

    monkeypatch.setattr(yc.imaplib, "IMAP4_SSL", lambda *a, **k: FakeIMAP())
    client = create_mail_provider(settings)
    assert isinstance(client, yc.YahooClient)

    built = {}
    import openmailsweep.gmail_client as gc
    monkeypatch.setattr(gc, "GmailClient", lambda *a, **k: built.setdefault("gmail", True))
    create_mail_provider(Settings(mail_provider="gmail"))
    assert built.get("gmail") is True

def test_gmail_auth_gating_uses_token_file(tmp_path: Path):
    missing = Settings(mail_provider="gmail", gmail_token=tmp_path / "nope.json")
    assert provider_is_ready(missing) is False
    present = Settings(mail_provider="gmail", gmail_token=tmp_path / "ok.json")
    present.gmail_token.write_text("{}")
    assert provider_is_ready(present) is True
