from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmailMessage:
    id: str
    thread_id: str
    subject: str
    sender: str
    sender_address: str
    body: str
    snippet: str
    labels: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Classification:
    category: str
    confidence: float
    spam_probability: float
    bulk_probability: float
    useful_probability: float
    importance_score: float
    raw: dict[str, Any]


@dataclass
class Decision:
    action: str
    reason: str
    protected: bool
    classification: Classification | None = None


@dataclass
class UnsubscribePlan:
    method: str
    https_url: str | None = None
    mailto_url: str | None = None
    target_host: str | None = None
    one_click: bool = False
    dkim_authenticated: bool = False
    one_click_auto_eligible: bool = False
    mailto_authenticated: bool = False
    mailto_auto_eligible: bool = False
    auto_eligible: bool = False
    reason: str = ""


@dataclass
class UnsubscribeResult:
    status: str
    method: str
    detail: str = ""
    target_host: str | None = None
    http_status: int | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
