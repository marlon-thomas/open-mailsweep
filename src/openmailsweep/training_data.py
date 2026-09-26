from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .state_store import ACTIONS, normalize_list_id, sender_domain, utcnow

TRAINING_SOURCES = {"user-message", "pending-answer", "rule-bootstrap", "rule-change"}


@dataclass
class TrainingExample:
    message_id: str
    label: str
    sender_address: str
    subject: str = ""
    snippet: str = ""
    list_id: str = ""
    sender: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    history: dict[str, int] = field(default_factory=dict)
    weight: float = 1.0
    corrected: bool = False


class TrainingDataset:
    """Confirmed user intent stored as reproducible classifier training rows.

    Only explicit user decisions and learned rules seed this dataset. Autonomous
    model outputs are never treated as ground truth unless the user confirmed
    them through a Pending answer. Sanitised feature sources (sender, subject,
    snippet, Gmail/header flags) are stored instead of full message bodies.
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_examples (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message_id TEXT NOT NULL UNIQUE,
                  label TEXT NOT NULL,
                  sender TEXT,
                  sender_address TEXT,
                  sender_domain TEXT,
                  subject TEXT,
                  snippet TEXT,
                  list_id TEXT,
                  features_json TEXT,
                  weight REAL NOT NULL DEFAULT 1.0,
                  corrected INTEGER NOT NULL DEFAULT 0,
                  source TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_sender ON training_examples(sender_address)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_list ON training_examples(list_id)"
            )

    def upsert_example(
        self,
        *,
        message_id: str,
        label: str,
        sender: str = "",
        sender_address: str = "",
        subject: str = "",
        snippet: str = "",
        list_id: str | None = None,
        features: dict[str, Any] | None = None,
        source: str = "pending-answer",
        weight: float = 1.0,
    ) -> bool:
        """Insert or update one confirmed example.

        Returns True when the stored label changed (a correction). A correction
        is up-weighted per the spec: explicit user corrections matter more than
        the original example that produced the wrong prediction.
        """
        if label not in ACTIONS:
            raise ValueError(f"Unsupported training label: {label}")
        address = (sender_address or "").strip().lower()
        now = utcnow()
        list_id = normalize_list_id(list_id)
        payload = json.dumps(features or {})
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT label FROM training_examples WHERE message_id=?", (message_id,)
            ).fetchone()
            corrected = existing is not None and existing["label"] != label
            effective_weight = max(2.0, weight * 2.0) if corrected else weight
            conn.execute(
                """
                INSERT INTO training_examples(
                    message_id,label,sender,sender_address,sender_domain,subject,snippet,list_id,
                    features_json,weight,corrected,source,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(message_id) DO UPDATE SET
                    label=excluded.label, sender=excluded.sender, sender_address=excluded.sender_address,
                    sender_domain=excluded.sender_domain, subject=excluded.subject, snippet=excluded.snippet,
                    list_id=excluded.list_id, features_json=excluded.features_json,
                    weight=excluded.weight, corrected=excluded.corrected, source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (
                    message_id, label, sender, address, sender_domain(address),
                    subject, snippet, list_id, payload, effective_weight,
                    int(corrected), source, now, now,
                ),
            )
        return corrected

    def remove_example(self, message_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM training_examples WHERE message_id=?", (message_id,))

    def prune_rule_bootstrap(self, live_source_message_ids: set[str]) -> int:
        """Drop derived examples whose learned rule no longer exists."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT message_id FROM training_examples WHERE source='rule-bootstrap'"
            ).fetchall()
            stale = [r["message_id"] for r in rows if r["message_id"] not in live_source_message_ids]
            for message_id in stale:
                conn.execute("DELETE FROM training_examples WHERE message_id=?", (message_id,))
        return len(stale)

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0])

    def class_counts(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT label, COUNT(*) AS c FROM training_examples GROUP BY label"
            ).fetchall()
        return {r["label"]: int(r["c"]) for r in rows}

    def revision(self) -> str:
        """Cheap fingerprint of dataset content for stale-model detection."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(MAX(id),0) AS m, COALESCE(SUM(id),0) AS s,"
                " COALESCE(SUM(LENGTH(label)),0) AS l FROM training_examples"
            ).fetchone()
        return f"{row['c']}:{row['m']}:{row['s']}:{row['l']}"

    def history_for(
        self,
        *,
        sender_address: str = "",
        list_id: str = "",
        exclude_message_id: str | None = None,
    ) -> dict[str, int]:
        """Previous confirmed decisions for this sender/list, bucketed per action.

        ``exclude_message_id`` prevents self-label leakage when building the
        training matrix (an example must not see its own decision as history).
        """
        address = (sender_address or "").strip().lower()
        list_id = normalize_list_id(list_id)
        counts: dict[str, int] = {}
        with self._conn() as conn:
            if address:
                for row in conn.execute(
                    "SELECT label, COUNT(*) AS c FROM training_examples"
                    " WHERE sender_address=? AND (? IS NULL OR message_id != ?) GROUP BY label",
                    (address, exclude_message_id, exclude_message_id),
                ):
                    counts[f"hist_sender_{row['label']}"] = counts.get(f"hist_sender_{row['label']}", 0) + int(row["c"])
            if list_id:
                for row in conn.execute(
                    "SELECT label, COUNT(*) AS c FROM training_examples"
                    " WHERE list_id=? AND (? IS NULL OR message_id != ?) GROUP BY label",
                    (list_id, exclude_message_id, exclude_message_id),
                ):
                    counts[f"hist_list_{row['label']}"] = counts.get(f"hist_list_{row['label']}", 0) + int(row["c"])
        return counts

    def all_examples(self) -> list[TrainingExample]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM training_examples ORDER BY id ASC").fetchall()
        return [self._from_row(r) for r in rows]

    def _from_row(self, row: sqlite3.Row) -> TrainingExample:
        try:
            features = json.loads(row["features_json"] or "{}")
        except Exception:
            features = {}
        return TrainingExample(
            message_id=row["message_id"],
            label=row["label"],
            sender=row["sender"] or "",
            sender_address=row["sender_address"] or "",
            subject=row["subject"] or "",
            snippet=row["snippet"] or "",
            list_id=row["list_id"] or "",
            features=features,
            weight=float(row["weight"] or 1.0),
            corrected=bool(row["corrected"]),
        )


def _example_from_queue_row(row: dict[str, Any], label: str, source: str) -> dict[str, Any] | None:
    try:
        features = json.loads(row.get("features_json") or "{}")
    except Exception:
        features = {}
    if not features:
        # Pre-feature rows: salvage what the unsubscribe columns still record.
        features = {
            "labels": [],
            "one_click": bool(row.get("unsubscribe_auto_eligible")),
            "has_mailto": False,
            "has_list_unsubscribe": False,
            "precedence_bulk": False,
            "auto_submitted": False,
        }
    if not (row.get("subject") or row.get("snippet") or row.get("sender_address")):
        return None  # no usable text: rule stays an exact rule, never an invented example
    features["body_excerpt"] = features.get("body_excerpt") or (row.get("body_excerpt") or "")
    return {
        "message_id": row["message_id"],
        "label": label,
        "sender": row.get("sender") or "",
        "sender_address": row.get("sender_address") or "",
        "subject": row.get("subject") or "",
        "snippet": row.get("snippet") or "",
        "list_id": row.get("list_id") or "",
        "features": features,
        "source": source,
    }


def bootstrap_from_history(store, dataset: TrainingDataset, *, include_rules: bool = True) -> int:
    """Seed the dataset from confirmed learned rules and user decisions.

    Learned rules contribute their originating example (never synthesized
    text). Pending answers made per-message contribute directly. Returns the
    number of examples written.
    """
    written = 0
    seen: set[str] = set()
    if include_rules:
        for rule in store.list_rules():
            source_message_id = rule.get("source_message_id")
            if not source_message_id:
                continue
            row = store.get_item_by_message(source_message_id)
            if not row:
                continue
            payload = _example_from_queue_row(row, rule["action"], "rule-bootstrap")
            if payload:
                dataset.upsert_example(**payload)
                written += 1
                seen.add(rule["source_message_id"])
    for row in store.list_confirmed_user_decisions():
        if row["message_id"] in seen:
            continue
        payload = _example_from_queue_row(row, row["final_action"], "user-message")
        if payload:
            dataset.upsert_example(**payload)
            written += 1
    return written
