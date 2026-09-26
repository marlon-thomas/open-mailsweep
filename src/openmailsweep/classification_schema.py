from __future__ import annotations

from .models import Classification, EmailMessage


def build_state_and_questions(message: EmailMessage, max_body_chars: int) -> tuple[dict, dict]:
    body = (message.body or message.snippet or "")[:max_body_chars]
    state = {
        "subject": message.subject,
        "from": message.sender,
        "list_unsubscribe": message.headers.get("list-unsubscribe", ""),
        "list_id": message.headers.get("list-id", ""),
        "precedence": message.headers.get("precedence", ""),
        "body": body,
    }
    questions = {
        "category": {
            "type": "choice",
            "instructions": (
                "Classify the email by its primary purpose. Be conservative with direct correspondence, "
                "official/public-authority mail, receipts, account/service notices, deliveries, bookings, "
                "local community messages, and anything the recipient may need to act on."
            ),
            "criteria": {
                "keep": "Legitimate personal or work conversation that should remain available.",
                "transactional": "Receipt, order, payment, charging, booking, delivery, official case/application, or other transaction/service record.",
                "notification": "Legitimate account, security, service, or status notification that is not primarily marketing.",
                "community": "Legitimate neighbourhood, local community, club, church, group, or community alert/discussion.",
                "career": "Job alert, recruiter role digest, vacancy recommendation, professional opportunity, or career-related update that may be useful to review later.",
                "newsletter": "Recurring editorial or content newsletter that is not clearly malicious.",
                "promotion": "Marketing, sales offer, commercial promotion, re-engagement, or product recommendation.",
                "junk": "Low-value unsolicited outreach, repetitive bulk noise, cold sales/recruiting, or mail the recipient is unlikely to need.",
                "spam": "Clearly abusive, deceptive, scam-like, malicious, or obvious spam.",
                "suspicious": "Potential phishing, impersonation, credential theft, or other security risk.",
            },
        },
        "spam": {
            "type": "noul",
            "instructions": (
                "Is this email genuinely spam or unwanted unsolicited junk, rather than legitimate direct, transactional, official, account, community, or useful subscription mail?"
            ),
        },
        "bulk": {
            "type": "noul",
            "instructions": "Is this primarily bulk-distributed email rather than a direct one-to-one or case-specific communication?",
        },
        "useful": {
            "type": "noul",
            "instructions": "Is this email likely to contain information the recipient may reasonably need to keep, refer to, or act on?",
        },
        "importance": {
            "type": "score",
            "instructions": "Rate how important it is to preserve this email.",
            "criteria": [
                "No meaningful value; safe candidate for cleanup",
                "Low value; probably replaceable or ignorable",
                "Potentially useful; should normally remain available",
                "Important personal/work/transactional/account/official information",
                "Critical security, legal, financial, contractual, identity, or official-case information",
            ],
        },
    }
    return state, questions


def parse_classification(raw: dict) -> Classification:
    answers = raw["answers"]
    category = answers["category"]
    return Classification(
        category=category["choice"],
        confidence=float(category["confidence"]),
        spam_probability=float(answers["spam"]["noul"]),
        bulk_probability=float(answers["bulk"]["noul"]),
        useful_probability=float(answers["useful"]["noul"]),
        importance_score=float(answers["importance"]["score"]),
        raw=raw,
    )
