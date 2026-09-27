from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ClassificationPolicy(BaseModel):
    auto_junk_confidence: float = 0.985
    review_confidence: float = 0.80
    max_body_chars: int = 1200

    # Small language-model checkpoints can report uncalibrated confidence
    # values. These gates make contradictory outputs review-only instead of
    # allowing a single probability to drive cleanup.
    contradiction_spam_threshold: float = 0.90
    contradiction_useful_threshold: float = 0.70
    contradiction_importance_threshold: float = 2.0
    auto_bulk_probability: float = 0.90

    # Obvious career/job-alert subjects can skip the slow local model. This is
    # only a routing hint to Pending (never an automatic Gmail mutation).
    career_subject_terms: list[str] = Field(default_factory=lambda: [
        "job alert",
        "job alerts",
        "job recommendations",
        "jobs you may be interested in",
        "new jobs for",
        "roles matching",
        "career opportunities",
    ])


class ProtectedPolicy(BaseModel):
    sender_domains: list[str] = Field(default_factory=list)
    sender_addresses: list[str] = Field(default_factory=list)

    # Conservative UK defaults. These are intentionally protection rules, not
    # trust assertions: official-looking mail should be kept for human review.
    official_domain_suffixes: list[str] = Field(default_factory=lambda: [
        "gov.uk",
        "nhs.uk",
        "police.uk",
    ])

    # Strong safety/security/legal/financial subject signals.
    subject_terms: list[str] = Field(default_factory=lambda: [
        "password",
        "verification code",
        "security alert",
        "suspicious sign-in",
        "invoice",
        "receipt",
        "statement",
        "payment",
        "tax",
        "payroll",
        "payslip",
        "contract",
        "legal",
    ])

    # Transactional/service/official messages seen in the first mailbox audit.
    # These are protected before the model runs, both for safety and speed.
    transactional_subject_terms: list[str] = Field(default_factory=lambda: [
        "order confirmation",
        "booking confirmation",
        "reservation confirmation",
        "appointment confirmation",
        "payment confirmation",
        "payment received",
        "refund",
        "charged",
        "charging at",
        "thanks for charging",
        "delivered",
        "delivery update",
        "out for delivery",
        "dispatched",
        "shipped",
        "tracking number",
        "account activity",
        "account data",
        "review your details",
        "policy evidence",
        "case reference",
        "application reference",
        "planning advice",
        "planning application",
        "performance report",
    ])

    # These two are soft safety signals in v0.9: they block automatic cleanup
    # but no longer skip classification or force the message to Protected.
    protect_gmail_personal: bool = True
    protect_prior_correspondents: bool = True


class UnsubscribePolicy(BaseModel):
    auto_one_click: bool = True
    auto_mailto: bool = True
    require_dkim: bool = True
    request_timeout_seconds: float = 10.0
    trash_if_unavailable: bool = True
    trash_on_failure: bool = False


class LabelsPolicy(BaseModel):
    junk: str = "OpenMailSweep/Junk"
    promotion: str = "OpenMailSweep/Promotion"
    newsletter: str = "OpenMailSweep/Newsletter"
    review: str = "OpenMailSweep/Review"
    protected: str = "OpenMailSweep/Protected"
    read_later: str = "Read Later"
    unsubscribe_review: str = "OpenMailSweep/UnsubscribeReview"


class Policy(BaseModel):
    classification: ClassificationPolicy = ClassificationPolicy()
    protected: ProtectedPolicy = ProtectedPolicy()
    unsubscribe: UnsubscribePolicy = UnsubscribePolicy()
    labels: LabelsPolicy = LabelsPolicy()


class Settings(BaseModel):
    mail_provider: str = "gmail"
    classifier: str = "local"
    classify_batch_size: int = 16
    jev_api_key: str = ""
    jev_model: str = "jev-latest"
    jev_base_url: str = "https://api.typesafe.ai"
    allow_sweep: bool = False
    allow_unsubscribe: bool = False
    gmail_credentials: Path = Path("secrets/credentials.json")
    gmail_token: Path = Path("secrets/token.json")
    yahoo_email: str = ""
    yahoo_app_password: str = ""
    yahoo_imap_host: str = "imap.mail.yahoo.com"
    yahoo_imap_port: int = 993
    yahoo_smtp_host: str = "smtp.mail.yahoo.com"
    yahoo_smtp_port: int = 587
    audit_db: Path = Path("data/audit.db")
    state_db: Path = Path("data/openmailsweep.db")
    policy_file: Path = Path("policy.yaml")
    scan_query: str = "in:inbox"
    fetch_batch_size: int = 100
    scan_interval_seconds: float = 60.0
    action_poll_seconds: float = 1.0
    action_workers: int = 3
    ui_port: int = 8787
    gmail_quota_units_per_minute: int = 3600
    gmail_quota_burst_units: int = 240
    gmail_rate_max_retries: int = 7
    gmail_rate_max_backoff: float = 64.0
    classifier_dir: Path = Path("data/classifier")
    local_device: str = "auto"
    local_min_examples: int = 20
    local_auto_action_min_examples: int = 100
    local_keep_threshold: float = 0.70
    local_read_later_threshold: float = 0.75
    local_archive_threshold: float = 0.85
    local_trash_threshold: float = 0.95
    local_unsubscribe_threshold: float = 0.98
    local_early_confidence: float = 0.97
    local_target_auto_precision: float = 0.97
    local_calibrate_min_support: int = 6
    local_retrain_interval_hours: float = 6.0
    local_retrain_new_examples: int = 100
    local_retrain_on_rule_change: bool = True
    local_bootstrap_from_rules: bool = True
    local_batch_size: int = 256

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("MAIL_PROVIDER", "gmail").strip().lower()
        if provider not in {"gmail", "yahoo"}:
            raise RuntimeError("MAIL_PROVIDER must be either 'gmail' or 'yahoo'")

        yahoo_email = os.getenv("YAHOO_EMAIL", "").strip()
        yahoo_app_password = os.getenv("YAHOO_APP_PASSWORD", "").strip()
        if provider == "yahoo" and not (yahoo_email and yahoo_app_password):
            raise RuntimeError(
                "MAIL_PROVIDER=yahoo requires YAHOO_EMAIL and a YAHOO_APP_PASSWORD "
                "(generate an app password at https://mail.yahoo.com -> Settings -> "
                "Accounts -> Manage app passwords; Yahoo IMAP/SMTP rejects the normal login password)."
            )

        classifier = os.getenv("CLASSIFIER", "local").strip().lower()
        if classifier not in {"local", "jev"}:
            raise RuntimeError("CLASSIFIER must be either 'local' or 'jev'")

        jev_key = os.getenv("JEV_API_KEY", "").strip()
        if classifier == "jev" and not jev_key:
            raise RuntimeError("JEV_API_KEY is required when CLASSIFIER=jev")

        try:
            batch_size = int(os.getenv("OPENMAILSWEEP_CLASSIFY_BATCH_SIZE", "16"))
        except ValueError as exc:
            raise RuntimeError("OPENMAILSWEEP_CLASSIFY_BATCH_SIZE must be an integer") from exc
        if batch_size < 1 or batch_size > 256:
            raise RuntimeError("OPENMAILSWEEP_CLASSIFY_BATCH_SIZE must be between 1 and 256")

        return cls(
            mail_provider=provider,
            classifier=classifier,
            classify_batch_size=batch_size,
            jev_api_key=jev_key,
            jev_model=os.getenv("JEV_MODEL", "jev-latest"),
            jev_base_url=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai").rstrip("/"),
            allow_sweep=os.getenv("ALLOW_SWEEP", "false").lower() in {"1", "true", "yes"},
            allow_unsubscribe=os.getenv("ALLOW_UNSUBSCRIBE", "false").lower() in {"1", "true", "yes"},
            gmail_credentials=Path(os.getenv("GMAIL_CREDENTIALS", "secrets/credentials.json")),
            gmail_token=Path(os.getenv("GMAIL_TOKEN", "secrets/token.json")),
            yahoo_email=yahoo_email,
            yahoo_app_password=yahoo_app_password,
            yahoo_imap_host=os.getenv("YAHOO_IMAP_HOST", "imap.mail.yahoo.com").strip(),
            yahoo_imap_port=max(1, int(os.getenv("YAHOO_IMAP_PORT", "993"))),
            yahoo_smtp_host=os.getenv("YAHOO_SMTP_HOST", "smtp.mail.yahoo.com").strip(),
            yahoo_smtp_port=max(1, int(os.getenv("YAHOO_SMTP_PORT", "587"))),
            audit_db=Path(os.getenv("AUDIT_DB", "data/audit.db")),
            state_db=Path(os.getenv("STATE_DB", "data/openmailsweep.db")),
            policy_file=Path(os.getenv("POLICY_FILE", "policy.yaml")),
            scan_query=os.getenv("OPENMAILSWEEP_SCAN_QUERY", "in:inbox").strip() or "in:inbox",
            fetch_batch_size=max(1, min(500, int(os.getenv("OPENMAILSWEEP_FETCH_BATCH_SIZE", "100")))),
            scan_interval_seconds=max(5.0, float(os.getenv("OPENMAILSWEEP_SCAN_INTERVAL", "60"))),
            action_poll_seconds=max(0.2, float(os.getenv("OPENMAILSWEEP_ACTION_POLL", "1"))),
            action_workers=max(1, min(16, int(os.getenv("OPENMAILSWEEP_ACTION_WORKERS", "3")))),
            ui_port=max(1, min(65535, int(os.getenv("UI_PORT", "8787")))),
            gmail_quota_units_per_minute=max(60, int(os.getenv("GMAIL_QUOTA_UNITS_PER_MINUTE", "3600"))),
            gmail_quota_burst_units=max(40, int(os.getenv("GMAIL_QUOTA_BURST_UNITS", "240"))),
            gmail_rate_max_retries=max(0, int(os.getenv("GMAIL_RATE_MAX_RETRIES", "7"))),
            gmail_rate_max_backoff=max(1.0, float(os.getenv("GMAIL_RATE_MAX_BACKOFF", "64"))),
            classifier_dir=Path(os.getenv("LOCAL_CLASSIFIER_DIR", "data/classifier")),
            local_device=_local_device(),
            local_min_examples=max(0, int(os.getenv("LOCAL_CLASSIFIER_MIN_EXAMPLES", "20"))),
            local_auto_action_min_examples=max(
                1, int(os.getenv("LOCAL_CLASSIFIER_AUTO_ACTION_MIN_EXAMPLES", "100"))
            ),
            local_keep_threshold=_probability("LOCAL_KEEP_THRESHOLD", 0.70),
            local_read_later_threshold=_probability("LOCAL_READ_LATER_THRESHOLD", 0.75),
            local_archive_threshold=_probability("LOCAL_ARCHIVE_THRESHOLD", 0.85),
            local_trash_threshold=_probability("LOCAL_CLEAN_THRESHOLD", 0.95),
            local_unsubscribe_threshold=_probability("LOCAL_UNSUBSCRIBE_THRESHOLD", 0.98),
            local_early_confidence=_probability("LOCAL_CLASSIFIER_EARLY_CONFIDENCE", 0.97),
            local_target_auto_precision=_probability("LOCAL_CLASSIFIER_TARGET_PRECISION", 0.97),
            local_calibrate_min_support=max(3, int(os.getenv("LOCAL_CLASSIFIER_CALIBRATE_MIN_SUPPORT", "6"))),
            local_retrain_interval_hours=_duration_hours("LOCAL_CLASSIFIER_RETRAIN_INTERVAL", 6.0),
            local_retrain_new_examples=max(
                1, int(os.getenv("LOCAL_CLASSIFIER_RETRAIN_NEW_EXAMPLES", "100"))
            ),
            local_retrain_on_rule_change=os.getenv("LOCAL_CLASSIFIER_RETRAIN_ON_RULE_CHANGE", "true").lower()
            in {"1", "true", "yes"},
            local_bootstrap_from_rules=os.getenv("LOCAL_CLASSIFIER_BOOTSTRAP_FROM_RULES", "true").lower()
            in {"1", "true", "yes"},
            local_batch_size=max(1, int(os.getenv("LOCAL_CLASSIFIER_BATCH_SIZE", "256"))),
        )


def load_policy(path: Path) -> Policy:
    if not path.exists():
        return Policy()
    with path.open("r", encoding="utf-8") as fh:
        return Policy.model_validate(yaml.safe_load(fh) or {})


def _probability(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number between 0 and 1") from exc
    return min(0.999, max(0.0, value))


def _duration_hours(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip().lower()
    multipliers = {"ms": 1 / 3_600_000, "s": 1 / 3600, "m": 1 / 60, "h": 1.0, "d": 24.0}
    factor = 1.0
    for suffix, multiplier in multipliers.items():
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)].strip()
            factor = multiplier
            break
    try:
        value = float(raw) * factor
    except ValueError as exc:
        raise RuntimeError(f"{name} must look like '6', '6h', '90m', or '0.5d'") from exc
    return max(0.5, value)


def _local_device() -> str:
    device = os.getenv("LOCAL_CLASSIFIER_DEVICE", "auto").strip().lower() or "auto"
    if device not in {"auto", "cpu", "gpu"}:
        raise RuntimeError("LOCAL_CLASSIFIER_DEVICE must be one of: auto, cpu, gpu")
    return device
