from __future__ import annotations

import httpx

from .classification_schema import build_state_and_questions, parse_classification
from .models import Classification, EmailMessage


class JevClient:
    def __init__(self, api_key: str, model: str, base_url: str, max_body_chars: int = 1200):
        self.model = model
        self.max_body_chars = max_body_chars
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30.0,
        )

    def classify(self, message: EmailMessage) -> Classification:
        state, questions = build_state_and_questions(message, self.max_body_chars)
        response = self.client.post(
            "/v1/systemone",
            json={"model": self.model, "state": state, "questions": questions},
        )
        response.raise_for_status()
        raw = response.json()
        if isinstance(raw, dict):
            raw = {**raw, "_backend": "jev"}
        return parse_classification(raw)
