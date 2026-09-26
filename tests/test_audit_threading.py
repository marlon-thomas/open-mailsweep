import threading
from pathlib import Path

from openmailsweep.audit import AuditLog
from openmailsweep.models import Decision, EmailMessage


def test_audit_log_can_be_written_from_worker_thread(tmp_path: Path):
    audit = AuditLog(tmp_path / "audit.db")
    msg = EmailMessage(
        id="m1", thread_id="t1", subject="hello", sender="a@example.com",
        sender_address="a@example.com", body="", snippet="", labels=[], headers={}
    )
    errors = []

    def write():
        try:
            audit.record(msg, Decision("review", "test", False, None))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=write)
    thread.start()
    thread.join()
    assert errors == []
    assert audit.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
