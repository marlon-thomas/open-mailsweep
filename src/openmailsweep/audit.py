from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import Decision, EmailMessage, UnsubscribePlan, UnsubscribeResult


def _json_default(value):
    """Make NumPy/Torch scalar-like values safe for SQLite JSON logging."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


class AuditLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              message_id TEXT NOT NULL,
              thread_id TEXT,
              sender TEXT,
              subject TEXT,
              action TEXT NOT NULL,
              reason TEXT NOT NULL,
              category TEXT,
              confidence REAL,
              spam_probability REAL,
              bulk_probability REAL,
              useful_probability REAL,
              importance_score REAL,
              raw_json TEXT,
              unsubscribe_method TEXT,
              unsubscribe_auto_eligible INTEGER,
              unsubscribe_target_host TEXT,
              unsubscribe_status TEXT,
              unsubscribe_detail TEXT,
              unsubscribe_http_status INTEGER
            )
        """)
        self._ensure_columns()
        self.conn.commit()

    def _ensure_columns(self) -> None:
        """Upgrade existing audit databases without deleting prior history."""
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(decisions)")}
        columns = {
            "unsubscribe_method": "TEXT",
            "unsubscribe_auto_eligible": "INTEGER",
            "unsubscribe_target_host": "TEXT",
            "unsubscribe_status": "TEXT",
            "unsubscribe_detail": "TEXT",
            "unsubscribe_http_status": "INTEGER",
        }
        for name, sql_type in columns.items():
            if name not in existing:
                self.conn.execute(f"ALTER TABLE decisions ADD COLUMN {name} {sql_type}")

    def record(
        self,
        msg: EmailMessage,
        decision: Decision,
        unsubscribe_plan: UnsubscribePlan | None = None,
        unsubscribe_result: UnsubscribeResult | None = None,
    ) -> None:
        c = decision.classification
        plan = unsubscribe_plan
        result = unsubscribe_result
        with self._lock:
            self.conn.execute(
                """INSERT INTO decisions (
                    created_at,message_id,thread_id,sender,subject,action,reason,category,confidence,
                    spam_probability,bulk_probability,useful_probability,importance_score,raw_json,
                    unsubscribe_method,unsubscribe_auto_eligible,unsubscribe_target_host,
                    unsubscribe_status,unsubscribe_detail,unsubscribe_http_status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(), msg.id, msg.thread_id, msg.sender, msg.subject,
                    decision.action, decision.reason,
                    c.category if c else None, c.confidence if c else None,
                    c.spam_probability if c else None, c.bulk_probability if c else None,
                    c.useful_probability if c else None, c.importance_score if c else None,
                    json.dumps(c.raw, default=_json_default) if c else None,
                    plan.method if plan else None,
                    int(plan.auto_eligible) if plan else None,
                    plan.target_host if plan else None,
                    result.status if result else None,
                    result.detail if result else None,
                    result.http_status if result else None,
                ),
            )
            self.conn.commit()
