import sqlite3
from pathlib import Path

from openmailsweep.audit import AuditLog


def test_existing_audit_db_gets_unsubscribe_columns(tmp_path: Path):
    db = tmp_path / "audit.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE decisions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          message_id TEXT NOT NULL,
          action TEXT NOT NULL,
          reason TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    AuditLog(db)
    conn = sqlite3.connect(db)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    conn.close()

    assert "unsubscribe_method" in columns
    assert "unsubscribe_status" in columns
