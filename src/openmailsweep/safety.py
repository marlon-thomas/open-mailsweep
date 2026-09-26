from __future__ import annotations

from .config import Policy
from .models import EmailMessage


class SafetyRules:
    def __init__(self, policy: Policy):
        self.policy = policy

    @staticmethod
    def _domain_matches_suffix(domain: str, suffix: str) -> bool:
        domain = domain.lower().strip(".")
        suffix = suffix.lower().lstrip("@.").strip(".")
        return bool(domain and suffix and (domain == suffix or domain.endswith("." + suffix)))

    def protect_reason(
        self,
        message: EmailMessage,
        thread_has_sent: bool = False,
        prior_correspondent: bool = False,
    ) -> str | None:
        """Return a hard protection reason.

        Hard protections skip model inference entirely. Gmail CATEGORY_PERSONAL and
        prior-correspondent history are intentionally *not* hard protections in
        v0.9; they are exposed through ``soft_signals`` instead.
        """
        labels = set(message.labels)
        if "STARRED" in labels:
            return "starred"
        if "IMPORTANT" in labels:
            return "gmail-important"
        if thread_has_sent:
            return "thread-has-sent-reply"

        addr = message.sender_address.lower()
        if addr in {x.lower() for x in self.policy.protected.sender_addresses}:
            return "sender-whitelist"
        domain = addr.rsplit("@", 1)[-1] if "@" in addr else ""
        protected_domains = {x.lower().lstrip("@") for x in self.policy.protected.sender_domains}
        if domain and domain in protected_domains:
            return "domain-whitelist"

        # Official domains take precedence over softer Gmail categorisation.
        for suffix in self.policy.protected.official_domain_suffixes:
            if self._domain_matches_suffix(domain, suffix):
                return f"official-domain:{suffix}"

        subject = message.subject.lower()
        for term in self.policy.protected.subject_terms:
            if term.lower() in subject:
                return f"protected-subject:{term}"

        for term in self.policy.protected.transactional_subject_terms:
            if term.lower() in subject:
                return f"transactional-subject:{term}"

        auth_results = message.headers.get("authentication-results", "").lower()
        if "spf=fail" in auth_results or "dmarc=fail" in auth_results:
            return "authentication-failure-review"
        return None

    def soft_signals(
        self,
        message: EmailMessage,
        prior_correspondent: bool = False,
    ) -> list[str]:
        """Return safety evidence that blocks automatic cleanup but still allows classification."""
        signals: list[str] = []
        labels = set(message.labels)
        if self.policy.protected.protect_gmail_personal and "CATEGORY_PERSONAL" in labels:
            signals.append("gmail-personal")
        if prior_correspondent and self.policy.protected.protect_prior_correspondents:
            signals.append("prior-correspondent")
        return signals
