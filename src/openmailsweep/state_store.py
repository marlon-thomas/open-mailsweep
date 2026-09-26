from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import Classification, EmailMessage, UnsubscribePlan

ACTIONS = {"keep", "archive", "read_later", "trash", "unsubscribe_trash"}
RULE_SCOPES = {"list", "sender", "domain"}
QUEUE_STATUSES = {"discovered", "classifying", "pending", "actionable", "processing", "done", "failed"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_list_id(value: str | None) -> str:
    value = (value or "").strip().lower()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value


def sender_domain(address: str | None) -> str:
    address = (address or "").strip().lower()
    return address.rsplit("@", 1)[-1] if "@" in address else ""


def extract_message_features(msg: EmailMessage, plan: UnsubscribePlan | None = None) -> dict[str, Any]:
    """Snapshot Gmail/header/unsubscribe signals used as classifier features.

    Stored beside each queue row so the training dataset can be rebuilt after a
    restart without re-fetching Gmail, and so bootstrap examples created before
    the model existed are reproducible.
    """
    headers = msg.headers or {}
    precedence = (headers.get("precedence") or "").strip().lower()
    auto_submitted = (headers.get("auto-submitted") or "").strip().lower()
    return {
        "labels": sorted(msg.labels or []),
        "list_id": normalize_list_id(headers.get("list-id")),
        "has_list_unsubscribe": bool(headers.get("list-unsubscribe")),
        "one_click": bool(plan and plan.one_click_auto_eligible) or "one-click" in (headers.get("list-unsubscribe-post") or "").lower(),
        "has_mailto": bool(plan and plan.mailto_url) or "mailto:" in (headers.get("list-unsubscribe") or "").lower(),
        "precedence_bulk": precedence in {"bulk", "list"},
        "auto_submitted": bool(auto_submitted) and auto_submitted != "no",
    }


class StateStore:
    """Persistent queue, learned-rule, and UI state backed by SQLite.

    A fresh connection is opened for each operation so the FastAPI request
    threads, scanner thread, and concurrent action workers can safely share the database.
    WAL mode keeps short reads from blocking the background workers.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS queue_items (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message_id TEXT NOT NULL UNIQUE,
                  thread_id TEXT,
                  sender TEXT,
                  sender_address TEXT,
                  sender_domain TEXT,
                  subject TEXT,
                  snippet TEXT,
                  list_id TEXT,
                  status TEXT NOT NULL DEFAULT 'discovered',
                  question TEXT,
                  proposed_action TEXT,
                  desired_action TEXT,
                  final_action TEXT,
                  decision_source TEXT,
                  decision_reason TEXT,
                  protected_reason TEXT,
                  safety_signals_json TEXT,
                  category TEXT,
                  confidence REAL,
                  spam_probability REAL,
                  bulk_probability REAL,
                  useful_probability REAL,
                  importance_score REAL,
                  unsubscribe_method TEXT,
                  unsubscribe_auto_eligible INTEGER NOT NULL DEFAULT 0,
                  unsubscribe_target_host TEXT,
                  rule_id INTEGER,
                  discovered_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  started_at TEXT,
                  completed_at TEXT,
                  error TEXT,
                  action_detail TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_queue_status_id
                  ON queue_items(status, id);
                CREATE INDEX IF NOT EXISTS idx_queue_sender
                  ON queue_items(sender_address);
                CREATE INDEX IF NOT EXISTS idx_queue_domain
                  ON queue_items(sender_domain);
                CREATE INDEX IF NOT EXISTS idx_queue_list
                  ON queue_items(list_id);

                CREATE TABLE IF NOT EXISTS rules (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  scope_type TEXT NOT NULL,
                  scope_value TEXT NOT NULL,
                  action TEXT NOT NULL,
                  source_message_id TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(scope_type, scope_value)
                );

                CREATE INDEX IF NOT EXISTS idx_rules_scope
                  ON rules(scope_type, scope_value);

                CREATE TABLE IF NOT EXISTS rule_history (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  rule_id INTEGER,
                  changed_at TEXT NOT NULL,
                  scope_type TEXT NOT NULL,
                  scope_value TEXT NOT NULL,
                  old_action TEXT,
                  new_action TEXT,
                  note TEXT
                );

                CREATE TABLE IF NOT EXISTS unsubscribe_registry (
                  dedupe_key TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  method TEXT,
                  detail TEXT,
                  source_message_id TEXT,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_state (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(queue_items)")}
        if "features_json" not in existing:
            conn.execute("ALTER TABLE queue_items ADD COLUMN features_json TEXT")
        if "body_excerpt" not in existing:
            conn.execute("ALTER TABLE queue_items ADD COLUMN body_excerpt TEXT")

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def has_message(self, message_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM queue_items WHERE message_id=?", (message_id,)
            ).fetchone() is not None

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            return self._dict(conn.execute("SELECT * FROM queue_items WHERE id=?", (item_id,)).fetchone())

    def get_item_by_message(self, message_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            return self._dict(
                conn.execute("SELECT * FROM queue_items WHERE message_id=?", (message_id,)).fetchone()
            )

    def set_body_excerpt(self, item_id: int, excerpt: str) -> None:
        """Persist a sanitised body excerpt (kept strictly local, in ./data).

        Captured once when the classifier stage fetches the full message so the
        training dataset can learn the *content* of confirmed decisions even if
        the user answers later, after the live message is no longer in hand.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE queue_items SET body_excerpt=?, updated_at=? WHERE id=? AND body_excerpt IS NULL",
                (excerpt[:600], utcnow(), item_id),
            )

    def add_discovered(
        self,
        msg: EmailMessage,
        plan: UnsubscribePlan,
        *,
        protected_reason: str | None = None,
        safety_signals: list[str] | None = None,
    ) -> int:
        now = utcnow()
        list_id = normalize_list_id(msg.headers.get("list-id"))
        domain = sender_domain(msg.sender_address)
        features_json = json.dumps(extract_message_features(msg, plan))
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO queue_items (
                    message_id,thread_id,sender,sender_address,sender_domain,subject,snippet,list_id,
                    status,protected_reason,safety_signals_json,unsubscribe_method,
                    unsubscribe_auto_eligible,unsubscribe_target_host,features_json,discovered_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    msg.id,
                    msg.thread_id,
                    msg.sender,
                    msg.sender_address,
                    domain,
                    msg.subject,
                    msg.snippet,
                    list_id,
                    "discovered",
                    protected_reason,
                    json.dumps(safety_signals or []),
                    plan.method,
                    int(plan.auto_eligible),
                    plan.target_host,
                    features_json,
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT id FROM queue_items WHERE message_id=?", (msg.id,)).fetchone()
            assert row is not None
            return int(row[0])

    def _classification_values(self, classification: Classification | None) -> tuple[Any, ...]:
        if classification is None:
            return (None, None, None, None, None, None)
        return (
            classification.category,
            classification.confidence,
            classification.spam_probability,
            classification.bulk_probability,
            classification.useful_probability,
            classification.importance_score,
        )

    def mark_protected(self, item_id: int, reason: str) -> None:
        now = utcnow()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status='done', desired_action='keep', final_action='protected',
                    decision_source='hard-protection', decision_reason=?, protected_reason=?,
                    completed_at=?, updated_at=?, error=NULL
                WHERE id=? AND status NOT IN ('done')
                """,
                (reason, reason, now, now, item_id),
            )

    def mark_model_keep(
        self,
        item_id: int,
        classification: Classification,
        reason: str,
        safety_signals: list[str] | None = None,
    ) -> None:
        now = utcnow()
        cv = self._classification_values(classification)
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status='done', desired_action='keep', final_action='keep',
                    decision_source='model-safe', decision_reason=?, safety_signals_json=?,
                    category=?, confidence=?, spam_probability=?, bulk_probability=?, useful_probability=?, importance_score=?,
                    completed_at=?, updated_at=?, error=NULL
                WHERE id=? AND status IN ('discovered','classifying')
                """,
                (reason, json.dumps(safety_signals or []), *cv, now, now, item_id),
            )

    def mark_pending(
        self,
        item_id: int,
        classification: Classification | None,
        *,
        reason: str,
        question: str,
        proposed_action: str | None,
        safety_signals: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        cv = self._classification_values(classification)
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status='pending', question=?, proposed_action=?, desired_action=NULL,
                    decision_source='model-review', decision_reason=?, safety_signals_json=?,
                    category=?, confidence=?, spam_probability=?, bulk_probability=?, useful_probability=?, importance_score=?,
                    updated_at=?, error=?
                WHERE id=? AND status IN ('discovered','classifying')
                """,
                (
                    question,
                    proposed_action,
                    reason,
                    json.dumps(safety_signals or []),
                    *cv,
                    utcnow(),
                    error,
                    item_id,
                ),
            )

    def set_actionable(
        self,
        item_id: int,
        action: str,
        *,
        source: str,
        reason: str,
        rule_id: int | None = None,
    ) -> None:
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status='actionable', desired_action=?, decision_source=?, decision_reason=?,
                    rule_id=?, updated_at=?, error=NULL
                WHERE id=? AND status NOT IN ('processing','done')
                """,
                (action, source, reason, rule_id, utcnow(), item_id),
            )

    def _scope_value_for_row(self, row: sqlite3.Row | dict[str, Any], scope_type: str) -> str:
        if scope_type == "list":
            value = row["list_id"]
        elif scope_type == "sender":
            value = row["sender_address"]
        elif scope_type == "domain":
            value = row["sender_domain"]
        else:
            raise ValueError(f"Unsupported rule scope: {scope_type}")
        value = (value or "").strip().lower()
        if not value:
            raise ValueError(f"This message has no usable {scope_type} scope")
        return value

    @staticmethod
    def _scope_where(scope_type: str) -> str:
        return {
            "list": "list_id=?",
            "sender": "sender_address=?",
            "domain": "sender_domain=?",
        }[scope_type]

    def upsert_rule(
        self,
        scope_type: str,
        scope_value: str,
        action: str,
        *,
        source_message_id: str | None = None,
        note: str = "user decision",
    ) -> int:
        if scope_type not in RULE_SCOPES:
            raise ValueError(f"Unsupported rule scope: {scope_type}")
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        if scope_type == "domain" and action == "unsubscribe_trash":
            raise ValueError("Unsubscribe + clean cannot be applied to an entire domain; use mailing list or sender scope")
        scope_value = (scope_value or "").strip().lower()
        if not scope_value:
            raise ValueError("Rule scope value is empty")

        now = utcnow()
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM rules WHERE scope_type=? AND scope_value=?",
                (scope_type, scope_value),
            ).fetchone()
            if existing is None:
                cur = conn.execute(
                    """
                    INSERT INTO rules(scope_type,scope_value,action,source_message_id,created_at,updated_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (scope_type, scope_value, action, source_message_id, now, now),
                )
                rule_id = int(cur.lastrowid)
                old_action = None
            else:
                rule_id = int(existing["id"])
                old_action = existing["action"]
                conn.execute(
                    "UPDATE rules SET action=?, source_message_id=COALESCE(?,source_message_id), updated_at=? WHERE id=?",
                    (action, source_message_id, now, rule_id),
                )
            conn.execute(
                """
                INSERT INTO rule_history(rule_id,changed_at,scope_type,scope_value,old_action,new_action,note)
                VALUES(?,?,?,?,?,?,?)
                """,
                (rule_id, now, scope_type, scope_value, old_action, action, note),
            )
        return rule_id

    def answer_pending(self, item_id: int, action: str, scope_type: str) -> dict[str, Any]:
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        if scope_type not in {*RULE_SCOPES, "message"}:
            raise ValueError(f"Unsupported decision scope: {scope_type}")

        with self._conn() as conn:
            row = conn.execute("SELECT * FROM queue_items WHERE id=?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(f"Queue item {item_id} does not exist")
        if row["status"] != "pending":
            raise ValueError("This pending item has already been answered or actioned")

        if scope_type == "message":
            self.set_actionable(
                item_id,
                action,
                source="user-message",
                reason="explicit user choice for this message",
            )
            return {"rule_id": None, "released": 1, "scope_type": "message", "scope_value": row["message_id"]}

        scope_value = self._scope_value_for_row(row, scope_type)
        rule_id = self.upsert_rule(
            scope_type,
            scope_value,
            action,
            source_message_id=row["message_id"],
            note="created/updated from pending queue",
        )
        released = self.apply_rule_to_open_items(rule_id)
        return {"rule_id": rule_id, "released": released, "scope_type": scope_type, "scope_value": scope_value}

    def apply_rule_to_open_items(self, rule_id: int) -> int:
        with self._conn() as conn:
            rule = conn.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone()
            if rule is None:
                raise KeyError(f"Rule {rule_id} does not exist")
            where = self._scope_where(rule["scope_type"])
            now = utcnow()
            cur = conn.execute(
                f"""
                UPDATE queue_items
                SET status='actionable', desired_action=?, decision_source='learned-rule',
                    decision_reason=?, rule_id=?, updated_at=?, error=NULL
                WHERE {where} AND status IN ('pending','actionable','discovered','classifying')
                """,
                (
                    rule["action"],
                    f"learned {rule['scope_type']} rule",
                    rule_id,
                    now,
                    rule["scope_value"],
                ),
            )
            return int(cur.rowcount)


    def apply_all_rules_to_open_items(self) -> int:
        """Route every open message already covered by a learned rule.

        This is used at service startup so a pre-existing backlog does not sit in
        the classifier queue waiting on the model even though the user has already
        taught OpenMailSweep what to do with that source. Rule precedence matches
        :meth:`matching_rule_for_message`: list, then sender, then domain.
        """
        with self._conn() as conn:
            rules = conn.execute("SELECT * FROM rules").fetchall()
            by_list = {}
            by_sender = {}
            by_domain = {}
            for rule in rules:
                target = {
                    "list": by_list,
                    "sender": by_sender,
                    "domain": by_domain,
                }[rule["scope_type"]]
                target[rule["scope_value"]] = rule

            rows = conn.execute(
                """
                SELECT id, list_id, sender_address, sender_domain
                FROM queue_items
                WHERE status IN ('discovered','classifying','pending')
                ORDER BY id
                """
            ).fetchall()
            now = utcnow()
            released = 0
            for row in rows:
                rule = None
                list_id = (row["list_id"] or "").strip().lower()
                sender = (row["sender_address"] or "").strip().lower()
                domain = (row["sender_domain"] or "").strip().lower()
                if list_id:
                    rule = by_list.get(list_id)
                if rule is None and sender:
                    rule = by_sender.get(sender)
                if rule is None and domain:
                    rule = by_domain.get(domain)
                if rule is None:
                    continue
                conn.execute(
                    """
                    UPDATE queue_items
                    SET status='actionable', desired_action=?, decision_source='learned-rule',
                        decision_reason=?, rule_id=?, updated_at=?, error=NULL
                    WHERE id=? AND status IN ('discovered','classifying','pending')
                    """,
                    (
                        rule["action"],
                        f"learned {rule['scope_type']} rule",
                        int(rule["id"]),
                        now,
                        int(row["id"]),
                    ),
                )
                released += 1
            return released

    def rule_scopes_for(self, list_id: str | None, sender_address: str | None) -> list[str]:
        """Learned-rule scopes that exist for this source (classifier feature)."""
        list_id = normalize_list_id(list_id)
        address = (sender_address or "").strip().lower()
        domain = sender_domain(address)
        scopes: list[str] = []
        with self._conn() as conn:
            if list_id and conn.execute(
                "SELECT 1 FROM rules WHERE scope_type='list' AND scope_value=?", (list_id,)
            ).fetchone():
                scopes.append("list")
            if address and conn.execute(
                "SELECT 1 FROM rules WHERE scope_type='sender' AND scope_value=?", (address,)
            ).fetchone():
                scopes.append("sender")
            if domain and conn.execute(
                "SELECT 1 FROM rules WHERE scope_type='domain' AND scope_value=?", (domain,)
            ).fetchone():
                scopes.append("domain")
        return scopes

    def list_confirmed_user_decisions(self) -> list[dict[str, Any]]:
        """Completed messages decided explicitly by the user (bootstrap truth).

        Autonomous model outputs and hard protections are deliberately excluded:
        only confirmed user intent may train the classifier.
        """
        placeholders = ",".join("?" for _ in ACTIONS)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM queue_items
                WHERE decision_source='user-message' AND status='done'
                  AND final_action IN ({placeholders})
                ORDER BY id ASC
                """,
                tuple(sorted(ACTIONS)),
            ).fetchall()
        return [dict(r) for r in rows]

    def matching_rule_for_message(self, msg: EmailMessage) -> dict[str, Any] | None:
        list_id = normalize_list_id(msg.headers.get("list-id"))
        address = (msg.sender_address or "").strip().lower()
        domain = sender_domain(address)
        with self._conn() as conn:
            # Most specific rule wins.
            if list_id:
                row = conn.execute(
                    "SELECT * FROM rules WHERE scope_type='list' AND scope_value=?", (list_id,)
                ).fetchone()
                if row:
                    return dict(row)
            if address:
                row = conn.execute(
                    "SELECT * FROM rules WHERE scope_type='sender' AND scope_value=?", (address,)
                ).fetchone()
                if row:
                    return dict(row)
            if domain:
                row = conn.execute(
                    "SELECT * FROM rules WHERE scope_type='domain' AND scope_value=?", (domain,)
                ).fetchone()
                if row:
                    return dict(row)
        return None

    def claim_discovered_batch(self, limit: int = 16) -> list[dict[str, Any]]:
        """Atomically claim the oldest unclassified messages for the classifier worker."""
        limit = max(1, int(limit))
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id FROM queue_items WHERE status='discovered' ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            now = utcnow()
            conn.execute(
                f"UPDATE queue_items SET status='classifying', updated_at=?, error=NULL WHERE id IN ({placeholders}) AND status='discovered'",
                (now, *ids),
            )
            claimed = conn.execute(
                f"SELECT * FROM queue_items WHERE id IN ({placeholders}) AND status='classifying' ORDER BY id ASC",
                ids,
            ).fetchall()
            return [dict(r) for r in claimed]

    def return_to_discovered(self, item_id: int, error: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE queue_items SET status='discovered', updated_at=?, error=? WHERE id=? AND status='classifying'",
                (utcnow(), error, item_id),
            )

    def list_pending(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM queue_items WHERE status='pending' ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_queue(self, statuses: tuple[str, ...], limit: int = 100) -> list[dict[str, Any]]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM queue_items WHERE status IN ({placeholders}) ORDER BY id ASC LIMIT ?",
                (*statuses, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM queue_items
                WHERE status IN ('done','failed')
                ORDER BY COALESCE(completed_at,updated_at) DESC, id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def claim_next_action(self) -> dict[str, Any] | None:
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM queue_items WHERE status='actionable' ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = utcnow()
            conn.execute(
                "UPDATE queue_items SET status='processing', started_at=?, updated_at=?, error=NULL WHERE id=? AND status='actionable'",
                (now, now, row["id"]),
            )
            claimed = conn.execute("SELECT * FROM queue_items WHERE id=?", (row["id"],)).fetchone()
            return dict(claimed) if claimed else None

    def finish_action(self, item_id: int, final_action: str, detail: str = "") -> None:
        now = utcnow()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status='done', final_action=?, action_detail=?, completed_at=?, updated_at=?, error=NULL
                WHERE id=?
                """,
                (final_action, detail, now, now, item_id),
            )

    def fail_action(self, item_id: int, error: str, *, retry: bool = False) -> None:
        status = "actionable" if retry else "failed"
        with self._conn() as conn:
            conn.execute(
                "UPDATE queue_items SET status=?, error=?, updated_at=? WHERE id=?",
                (status, error[:2000], utcnow(), item_id),
            )

    def recover_processing(self) -> int:
        """Recover work interrupted by a container/process restart."""
        with self._conn() as conn:
            now = utcnow()
            actions = conn.execute(
                """
                UPDATE queue_items
                SET status='actionable', updated_at=?, error='Recovered after process restart'
                WHERE status='processing'
                """,
                (now,),
            ).rowcount
            classifications = conn.execute(
                """
                UPDATE queue_items
                SET status='discovered', updated_at=?, error='Recovered interrupted classification'
                WHERE status='classifying'
                """,
                (now,),
            ).rowcount
            quota_failed_actions = conn.execute(
                """
                UPDATE queue_items
                SET status='actionable', updated_at=?, error='Recovered transient Gmail quota failure'
                WHERE status='failed'
                  AND (lower(COALESCE(error,'')) LIKE '%ratelimitexceeded%'
                       OR lower(COALESCE(error,'')) LIKE '%quota exceeded%'
                       OR lower(COALESCE(error,'')) LIKE '%gmail rate limit%')
                """,
                (now,),
            ).rowcount
            return int(actions) + int(classifications) + int(quota_failed_actions)

    def list_rules(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM rules ORDER BY updated_at DESC, id DESC").fetchall()
            return [dict(r) for r in rows]

    def data_revision(self) -> str:
        """Fingerprint of all confirmed user intent (rules + explicit decisions).

        The local classifier persists this alongside its weights; a mismatch at
        startup means the model is stale and must be rebuilt before unknown
        messages are classified (spec 8.1/27).
        """
        with self._conn() as conn:
            rules = conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(MAX(id),0) AS m, COALESCE(SUM(id),0) AS s,"
                " COALESCE(SUM(LENGTH(action)),0) AS a FROM rules"
            ).fetchone()
            history = conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(MAX(id),0) AS m FROM rule_history"
            ).fetchone()
            decisions = conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(MAX(id),0) AS m FROM queue_items"
                " WHERE decision_source='user-message' AND status='done' AND final_action IS NOT NULL"
            ).fetchone()
        return (
            f"r{rules['c']}-{rules['m']}-{rules['s']}-{rules['a']}"
            f"-h{history['c']}-{history['m']}"
            f"-d{decisions['c']}-{decisions['m']}"
        )

    def get_rule(self, rule_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            return self._dict(conn.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone())

    def update_rule(self, rule_id: int, action: str) -> dict[str, Any]:
        rule = self.get_rule(rule_id)
        if rule is None:
            raise KeyError(f"Rule {rule_id} does not exist")
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        if rule["scope_type"] == "domain" and action == "unsubscribe_trash":
            raise ValueError("Unsubscribe + clean cannot be applied to an entire domain")
        self.upsert_rule(
            rule["scope_type"],
            rule["scope_value"],
            action,
            source_message_id=rule.get("source_message_id"),
            note="rule changed from Rules screen",
        )
        released = self.apply_rule_to_open_items(rule_id)
        updated = self.get_rule(rule_id) or {}
        updated["released"] = released
        return updated

    def delete_rule(self, rule_id: int) -> int:
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rule = conn.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone()
            if rule is None:
                raise KeyError(f"Rule {rule_id} does not exist")
            # Any not-yet-processed mail governed by this rule becomes pending
            # again. Already processing/completed actions are never reversed.
            cur = conn.execute(
                """
                UPDATE queue_items
                SET status='pending', desired_action=NULL, rule_id=NULL,
                    decision_source='rule-deleted', decision_reason='learned rule deleted; needs a new choice',
                    question='This learned rule was removed. What should OpenMailSweep do with this message?',
                    updated_at=?
                WHERE rule_id=? AND status='actionable'
                """,
                (utcnow(), rule_id),
            )
            conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
            conn.execute(
                """
                INSERT INTO rule_history(rule_id,changed_at,scope_type,scope_value,old_action,new_action,note)
                VALUES(?,?,?,?,?,?,?)
                """,
                (rule_id, utcnow(), rule["scope_type"], rule["scope_value"], rule["action"], None, "rule deleted"),
            )
            return int(cur.rowcount)

    def list_rule_history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM rule_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]


    def get_unsubscribe_registry(self, dedupe_key: str) -> dict[str, Any] | None:
        key = (dedupe_key or "").strip().lower()
        if not key:
            return None
        with self._conn() as conn:
            return self._dict(
                conn.execute(
                    "SELECT * FROM unsubscribe_registry WHERE dedupe_key=?", (key,)
                ).fetchone()
            )

    def record_unsubscribe_registry(
        self,
        dedupe_key: str,
        status: str,
        method: str,
        detail: str,
        *,
        source_message_id: str | None = None,
    ) -> None:
        key = (dedupe_key or "").strip().lower()
        if not key:
            return
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO unsubscribe_registry(
                    dedupe_key,status,method,detail,source_message_id,updated_at
                ) VALUES(?,?,?,?,?,?)
                ON CONFLICT(dedupe_key) DO UPDATE SET
                    status=excluded.status, method=excluded.method, detail=excluded.detail,
                    source_message_id=excluded.source_message_id, updated_at=excluded.updated_at
                """,
                (key, status, method, detail[:2000], source_message_id, utcnow()),
            )

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS c FROM queue_items GROUP BY status").fetchall()
            stats = {r["status"]: int(r["c"]) for r in rows}
            stats["rules"] = int(conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0])
            stats["protected"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE status='done' AND final_action='protected'"
                ).fetchone()[0]
            )
            return stats

    def set_app_state(self, key: str, value: str) -> None:
        now = utcnow()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO app_state(key,value,updated_at) VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )

    def get_app_state(self, key: str, default: str | None = None) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
            return row[0] if row else default
