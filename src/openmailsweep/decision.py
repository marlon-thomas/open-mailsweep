from __future__ import annotations

from collections.abc import Sequence

from .config import Policy
from .models import Classification, Decision


def decide(
    classification: Classification | None,
    policy: Policy,
    protected_reason: str | None,
    safety_signals: Sequence[str] | None = None,
) -> Decision:
    if protected_reason == "authentication-failure-review":
        return Decision("review", protected_reason, True, classification)
    if protected_reason:
        return Decision("protected", protected_reason, True, classification)

    if classification is None:
        return Decision("review", "missing classification", False, None)

    c = classification
    auto = policy.classification.auto_junk_confidence
    review = policy.classification.review_confidence
    signals = [s for s in (safety_signals or []) if s]

    # Categories that represent legitimate or context-dependent mail are never
    # auto-cleaned. False negatives here cost storage; false positives could
    # lose correspondence, so err on the side of keeping them.
    if c.category in {"keep", "transactional", "notification", "community"}:
        suffix = f"; safety-signal:{','.join(signals)}" if signals else ""
        return Decision("keep", f"preserve-{c.category}{suffix}", False, c)

    if c.category == "career":
        # Career/job alerts are usually useful but not inbox-critical. Keep them
        # human-trainable so the user can choose Read Later, Keep, Archive, etc.
        return Decision("review", "career-opportunity", False, c)

    if c.category == "suspicious":
        return Decision("review", "security-sensitive classification", False, c)

    # Gmail's Personal categorisation and sent-history are useful evidence, but
    # the first mailbox audit showed they are too broad to be unconditional hard
    # protections. They now block automatic cleanup while still allowing the
    # classifier to score the message and provide useful review telemetry.
    if signals:
        return Decision("review", f"safety-signal:{','.join(signals)}", False, c)

    # Contradictory model outputs are a strong sign that the probabilities are
    # not reliable enough for cleanup.
    cp = policy.classification
    if (
        c.spam_probability >= cp.contradiction_spam_threshold
        and c.useful_probability >= cp.contradiction_useful_threshold
    ):
        return Decision("review", "contradictory-spam-and-useful", False, c)

    if (
        c.spam_probability >= cp.contradiction_spam_threshold
        and c.importance_score >= cp.contradiction_importance_threshold
    ):
        return Decision("review", "contradictory-spam-and-important", False, c)

    if c.category in {"junk", "spam"} and c.useful_probability >= cp.contradiction_useful_threshold:
        return Decision("review", "junk-but-useful", False, c)

    if (
        c.category == "promotion"
        and c.confidence >= auto
        and c.bulk_probability >= cp.auto_bulk_probability
        and c.useful_probability <= 0.20
    ):
        return Decision("promotion", "high-confidence low-value promotion", False, c)

    if (
        c.category == "newsletter"
        and c.confidence >= auto
        and c.bulk_probability >= cp.auto_bulk_probability
    ):
        return Decision("newsletter", "high-confidence newsletter", False, c)

    if (
        c.category in {"junk", "spam"}
        and c.confidence >= auto
        and c.spam_probability >= auto
        and c.importance_score < 1.5
    ):
        return Decision("junk", "high-confidence low-importance junk", False, c)

    if c.confidence < review or 0.35 <= c.spam_probability <= 0.90:
        return Decision("review", "classification uncertainty", False, c)

    return Decision("keep", "not safe to clean automatically", False, c)
