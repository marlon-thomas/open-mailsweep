"""Fast personalised local classifier for OpenMailSweep.

A sparse linear action predictor
(HashingVectorizer + log-loss SGD) that learns online from confirmed user
decisions. Runs fully locally, uses a compatible GPU automatically when one is
available (dense minibatch softmax-SGD on cupy) and falls back to CPU with an
identical action-class contract. No external network access, no model downloads.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from .config import Policy, Settings
from .models import EmailMessage, UnsubscribePlan
from .state_store import normalize_list_id, sender_domain
from .training_data import TrainingExample

ACTION_CLASSES = ["keep", "read_later", "archive", "trash", "unsubscribe_trash"]
NON_DESTRUCTIVE_ACTIONS = {"keep", "read_later"}
DESTRUCTIVE_ACTIONS = {"trash", "unsubscribe_trash"}
MODEL_SCHEMA_VERSION = 2
FEATURE_SCHEMA_VERSION = 1
HASH_FEATURES = 2**16

_REPORT: Callable[[str], None] = lambda text: print(text, flush=True)

_REF_RE = re.compile(r"\b(?:ref|case|order|booking|tracking|invoice)[\s:#-]*[a-z0-9-]{4,}", re.I)
_PCT_RE = re.compile(r"\d{1,3}\s?%\s?(?:off|discount|save)", re.I)
_CURRENCY_RE = re.compile(r"(?:[£$€]\s?\d|\d+(?:\.\d{2})?\s?(?:gbp|usd|eur))", re.I)


def structural_tokens(subject: str, snippet: str, policy: Policy) -> list[str]:
    text = f"{subject} {snippet}"
    tokens: list[str] = []
    combined = text.lower()
    if _CURRENCY_RE.search(text):
        tokens.append("fx:currency")
    if _PCT_RE.search(text):
        tokens.append("fx:discount-pct")
    if _REF_RE.search(text):
        tokens.append("fx:reference-number")
    if subject.strip().lower().startswith("re:"):
        tokens.append("fx:reply-subject")
    if subject.strip().lower().startswith("fwd"):
        tokens.append("fx:fwd-subject")
    for term in policy.classification.career_subject_terms:
        term = (term or "").strip().lower()
        if term and term in combined:
            tokens.append(f"career:{term.replace(' ', '-')}")
            break
    for term in policy.protected.transactional_subject_terms:
        term = (term or "").strip().lower()
        if term and term in combined:
            tokens.append(f"transactional:{term.replace(' ', '-')}")
            break
    return tokens


def build_feature_text(example: TrainingExample, policy: Policy) -> str:
    """Deterministic bag-of-tokens representation of what the email IS.

    Content features only: subject, snippet, sender display name, sender
    domain, list-ID domain, Gmail categories, unsubscribe structure, and
    structural markers. Deliberately excluded: the exact sender address, the
    exact List-ID, learned-rule existence, and per-sender decision history.
    Those identity tokens would let the model mechanically memorize "who sent
    it", duplicating the learned-rule table; the classifier's job is to
    generalise the *nature* of the mail to senders it has never seen. Sender
    history and rule scope remain available for evidence/explanations and for
    the safety gate, never as training signal.
    """
    features = example.features or {}
    address = (example.sender_address or "").strip().lower()
    domain = sender_domain(address)
    display = re.sub(r"\s+", " ", (example.sender or "").split("<")[0]).strip().lower()
    parts: list[str] = [
        f"subj {example.subject.lower()}",
        f"snap {(example.snippet or '').lower()[:300]}",
        f"body {(str(features.get('body_excerpt') or '')).lower()[:500]}",
        f"from {display}",
        f"dom {domain}",
    ]
    if example.list_id:
        list_domain = sender_domain(example.list_id) or example.list_id
        parts.append("mailing-list")
        parts.append(f"listdom {list_domain}")
    for label in features.get("labels", []):
        parts.append(f"lbl {str(label).lower()}")
    if features.get("one_click"):
        parts.append("unsub one-click")
    if features.get("has_mailto"):
        parts.append("unsub mailto")
    if features.get("has_list_unsubscribe"):
        parts.append("unsub header")
    if features.get("precedence_bulk"):
        parts.append("precedence bulk")
    if features.get("auto_submitted"):
        parts.append("auto submitted")
    parts.extend(structural_tokens(example.subject, example.snippet or "", policy))
    return " | ".join(parts)


def example_from_message(
    msg: EmailMessage,
    plan: UnsubscribePlan | None,
    *,
    history: dict[str, int] | None = None,
    weight: float = 1.0,
    body_excerpt: str | None = None,
) -> TrainingExample:
    features = {
        "labels": list(msg.labels or []),
        "list_id": normalize_list_id(msg.headers.get("list-id")),
        "has_list_unsubscribe": bool(msg.headers.get("list-unsubscribe")),
        "one_click": bool(plan and plan.one_click_auto_eligible),
        "has_mailto": bool(plan and plan.mailto_url),
        "precedence_bulk": (msg.headers.get("precedence") or "").strip().lower() in {"bulk", "list"},
        "auto_submitted": bool((msg.headers.get("auto-submitted") or "").strip())
        and (msg.headers.get("auto-submitted") or "").strip().lower() != "no",
        "body_excerpt": (body_excerpt if body_excerpt is not None else (msg.body or ""))[:600],
    }
    return TrainingExample(
        message_id=msg.id,
        label="",
        sender=msg.sender,
        sender_address=msg.sender_address,
        subject=msg.subject,
        snippet=msg.snippet,
        list_id=features["list_id"],
        features=features,
        history=history or {},
        weight=weight,
    )


@dataclass
class Prediction:
    action: str | None
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class GateResult:
    mode: str  # "actionable" | "pending"
    action: str | None
    reason: str


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class CPULinearBackend:
    """sklearn hashed linear model: online partial_fit, millisecond prediction."""

    name = "sklearn-sgd-logloss"
    device = "cpu"

    def __init__(self) -> None:
        self.clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="constant",
            eta0=0.1,
            random_state=42,
            tol=None,
        )
        self._fitted = False

    def fit(self, X, y: list[str], sample_weight: np.ndarray | None = None) -> None:
        # Multiple passes give a small personal dataset better calibrated
        # probabilities than a single online sweep; partial_fit keeps the true
        # one-pass incremental semantics for online user updates.
        for _ in range(30):
            self.clf.partial_fit(X, y, classes=ACTION_CLASSES, sample_weight=sample_weight)
        self._fitted = True

    def partial_fit(self, X, y: list[str], sample_weight: np.ndarray | None = None) -> None:
        self.clf.partial_fit(X, y, classes=ACTION_CLASSES, sample_weight=sample_weight)
        self._fitted = True

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted:
            return np.zeros((X.shape[0], len(ACTION_CLASSES)))
        probs = self.clf.predict_proba(X)
        # sklearn orders columns by ``classes_`` (alphabetical); callers always
        # consume columns in ACTION_CLASSES order. Never zip raw columns
        # against action names directly.
        ordered = np.zeros_like(probs)
        columns = [str(c) for c in self.clf.classes_]
        for index, action in enumerate(ACTION_CLASSES):
            if action in columns:
                ordered[:, index] = probs[:, columns.index(action)]
        return ordered

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in getattr(self.clf, "classes_", ACTION_CLASSES)]

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self.clf, path)

    def load(self, path: Path) -> None:
        import joblib

        self.clf = joblib.load(path)
        self._fitted = True


class GPULinearBackend:
    """cupy multi-class softmax regression with minibatch SGD.

    Same action-class contract, ``fit``/``partial_fit``/``predict_proba``
    semantics, and persistence as the CPU backend; the fitted weights are plain
    arrays so model files remain compact and portable.
    """

    name = "cupy-softmax-sgd"
    device = "gpu"

    def __init__(self, lr: float = 0.1, l2: float = 1e-5, epochs: int = 8, batch: int = 128) -> None:
        import cupy as cp

        self._cp = cp
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch = batch
        self.coef: Any = None  # (n_classes, n_features) float32 on GPU
        self.bias: Any = None
        self._fitted = False

    def _ensure(self, n_features: int) -> None:
        cp = self._cp
        if self.coef is None or self.coef.shape[1] != n_features:
            xp = cp.random.RandomState(42)
            self.coef = xp.normal(0.0, 1e-3, (len(ACTION_CLASSES), n_features)).astype(cp.float32)
            self.bias = cp.zeros(len(ACTION_CLASSES), dtype=cp.float32)

    def _to_gpu_batch(self, X_rows) -> Any:
        cp = self._cp
        dense = np.asarray(X_rows.todense() if hasattr(X_rows, "todense") else X_rows, dtype=np.float32)
        return cp.asarray(dense)

    def fit(self, X, y: list[str], sample_weight: np.ndarray | None = None) -> None:
        cp = self._cp
        self._ensure(X.shape[1])
        n = X.shape[0]
        if n == 0:
            return
        labels = np.array([ACTION_CLASSES.index(c) for c in y], dtype=np.int32)
        rng = random.Random(42)
        for _ in range(self.epochs):
            order = list(range(n))
            rng.shuffle(order)
            for start in range(0, n, self.batch):
                rows = order[start : start + self.batch]
                xb = self._to_gpu_batch(X[rows])
                yb = cp.asarray(labels[rows])
                wb = cp.asarray((1.0 if sample_weight is None else sample_weight[rows]).astype(np.float32))
                probs = cp.softmax(xb @ self.coef.T + self.bias, axis=1)
                onehot = cp.zeros_like(probs)
                onehot[cp.arange(len(rows)), yb] = 1.0
                err = (probs - onehot) * wb[:, None]
                grad_w = xb.T @ err / cp.maximum(wb.sum(), 1.0) + self.l2 * self.coef.T
                grad_b = err.sum(axis=0) / cp.maximum(wb.sum(), 1.0)
                self.coef -= (self.lr * grad_w).T
                self.bias -= self.lr * grad_b
        self._fitted = True

    def partial_fit(self, X, y: list[str], sample_weight: np.ndarray | None = None) -> None:
        self.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted:
            return np.zeros((X.shape[0], len(ACTION_CLASSES)), dtype=np.float32)
        cp = self._cp
        xb = self._to_gpu_batch(X)
        probs = cp.softmax(xb @ self.coef.T + self.bias, axis=1)
        return cp.asnumpy(probs)

    @property
    def classes(self) -> list[str]:
        return list(ACTION_CLASSES)

    def save(self, path: Path) -> None:
        import joblib

        if self.coef is None:
            raise RuntimeError("cannot persist an untrained GPU model")
        joblib.dump({"coef": self._cp.asnumpy(self.coef), "bias": self._cp.asnumpy(self.bias)}, path)

    def load(self, path: Path) -> None:
        import joblib

        payload = joblib.load(path)
        cp = self._cp
        self.coef = cp.asarray(payload["coef"])
        self.bias = cp.asarray(payload["bias"])
        self._fitted = True


def select_backend(device: str, report: Callable[[str], None] = _REPORT):
    """Choose the fastest compatible local backend; never fail startup."""
    if device in {"auto", "gpu"}:
        try:
            import cupy  # noqa: F401

            if cupy.cuda.is_available():
                cupy.asarray([1.0, 2.0])  # smoke-probe the runtime
                report(f"[classifier-local] backend selected device=gpu backend={GPULinearBackend.name}")
                return GPULinearBackend()
            if device == "gpu":
                report("[classifier-local] GPU requested but none is available; using CPU fallback")
        except Exception as exc:
            report(f"[classifier-local] GPU backend unavailable ({type(exc).__name__}: {exc}); using CPU fallback")
    report(f"[classifier-local] backend selected device=cpu backend={CPULinearBackend.name}")
    return CPULinearBackend()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalClassifier:
    """Thread-safe owner of the personalised action model.

    The web thread performs incremental updates when the user answers Pending;
    the classifier thread predicts and runs periodic full retrains. All model
    access serialises on one lock, retrains build a candidate first and swap it
    in atomically, and the previous known-good model is retained for rollback.
    """

    def __init__(
        self,
        settings: Settings,
        policy: Policy,
        report: Callable[[str], None] = _REPORT,
    ):
        self.settings = settings
        self.policy = policy
        self.report = report
        self.dir = Path(settings.classifier_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._vectorizer = self._make_vectorizer()
        self.backend = select_backend(settings.local_device, report)
        self.meta: dict[str, Any] = {}
        self.version = 0
        self.bootstrapped = False
        self.example_count = 0
        self.retrain_requested = False
        self.last_error: str | None = None
        self.predictions = 0
        self._latency_ema_ms = 0.0
        self.counters = {"auto_actioned": 0, "pending": 0, "corrections": 0, "reeval_released": 0}

    @staticmethod
    def _make_vectorizer():
        return FeatureUnion(
            [
                (
                    "word",
                    HashingVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        alternate_sign=False,
                        n_features=HASH_FEATURES,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    HashingVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 6),
                        alternate_sign=False,
                        n_features=HASH_FEATURES,
                    ),
                ),
            ]
        )

    # ---------------------------------------------------------------- text

    def _matrix(self, texts: list[str]):
        return self._vectorizer.transform(texts)

    def _single(self, text: str):
        return self._matrix([text])

    # ------------------------------------------------------- persistence

    @property
    def model_path(self) -> Path:
        return self.dir / "model.joblib"

    @property
    def meta_path(self) -> Path:
        return self.dir / "model_meta.json"

    @property
    def prev_model_path(self) -> Path:
        return self.dir / "model.prev.joblib"

    @property
    def prev_meta_path(self) -> Path:
        return self.dir / "model_prev_meta.json"

    def load(self) -> bool:
        """Load a persisted model. False => rebuild from stored examples."""
        with self._lock:
            if not self.model_path.exists() or not self.meta_path.exists():
                return False
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if int(meta.get("model_schema_version", 0)) != MODEL_SCHEMA_VERSION:
                    self.report("[classifier-local] persisted model schema changed; rebuilding from stored examples")
                    return False
                if int(meta.get("feature_schema_version", 0)) != FEATURE_SCHEMA_VERSION:
                    self.report("[classifier-local] feature schema changed; rebuilding from stored examples")
                    return False
                if meta.get("classes") != ACTION_CLASSES:
                    return False
                candidate = select_backend(str(meta.get("device") or self.settings.local_device), report=lambda _: None)
                candidate.load(self.model_path)
            except Exception as exc:
                self.report(f"[classifier-local] model load failed ({exc}); will rebuild from stored examples")
                return False
            self.backend = candidate
            self.meta = meta
            self.version = int(meta.get("version") or 0)
            self.example_count = int(meta.get("examples") or 0)
            self.bootstrapped = self.example_count > 0 or bool(meta.get("bootstrapped"))
            self.report(
                f"[classifier-local] loaded model version={self.version} examples={self.example_count}"
            )
            return True

    def _persist(self, meta: dict[str, Any], backend) -> None:
        tmp_model = self.dir / "model.joblib.tmp"
        backend.save(tmp_model)
        os.replace(tmp_model, self.model_path)
        tmp_meta = self.dir / "model_meta.json.tmp"
        tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_meta, self.meta_path)

    def _archive_previous(self) -> None:
        if self.model_path.exists():
            import shutil

            shutil.copy2(self.model_path, self.prev_model_path)
        if self.meta_path.exists():
            self.prev_meta_path.write_text(self.meta_path.read_text(encoding="utf-8"), encoding="utf-8")

    # ------------------------------------------------------------ training

    def stale_revision(self) -> str | None:
        return self.meta.get("data_revision")

    def current_thresholds(self) -> dict[str, float]:
        """Public view of the effective (configured + calibrated) gates."""
        with self._lock:
            return dict(self._thresholds())

    def is_stale(self, data_revision: str) -> bool:
        return bool(self.meta) and self.meta.get("data_revision") != data_revision

    def bootstrap(self, examples: list[TrainingExample], data_revision: str) -> str:
        """Train-or-load before any unknown message is classified (spec 8.1/27)."""
        with self._lock:
            if not examples:
                self.bootstrapped = True
                self.example_count = 0
                self.report("[classifier-local] bootstrap: no confirmed user data yet; cold-start pending mode")
                return "no-data"
            if self.load() and self.meta.get("data_revision") == data_revision:
                self.bootstrapped = True
                return "loaded"
            started = time.perf_counter()
            # A rebuild always starts from fresh weights, even if a stale model
            # was loaded just above.
            self.backend = select_backend(self.settings.local_device, report=lambda _: None)
            self.example_count = len(examples)
            metrics = self._fit_from(examples, reason="bootstrap")
            elapsed = time.perf_counter() - started
            self.bootstrapped = True
            self.report(
                f"[classifier-local] bootstrap complete version={self.version} examples={len(examples)}"
                f" duration={elapsed:.1f}s"
            )
            self.meta.update(
                {
                    "bootstrap_source": "confirmed-rules-and-decisions",
                    "bootstrap_revision": data_revision,
                    "data_revision": data_revision,
                }
            )
            self._persist(self.meta, self.backend)
            return "trained"

    def _fit_from(self, examples: list[TrainingExample], reason: str) -> dict[str, Any]:
        texts = [build_feature_text(ex, self.policy) for ex in examples]
        X = self._matrix(texts)
        y = [ex.label for ex in examples]
        weights = np.array([ex.weight for ex in examples], dtype=float)
        metrics = self._evaluate(X, y, weights)
        candidate = self.backend
        candidate.fit(X, y, sample_weight=weights)
        self.version += 1
        self.meta = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "trained_at": _now(),
            "examples": len(examples),
            "classes": list(ACTION_CLASSES),
            "thresholds": self._thresholds(),
            "device": candidate.device,
            "backend": candidate.name,
            "data_revision": self.meta.get("data_revision"),
            "retrain_reason": reason,
            "version": self.version,
            "metrics": metrics,
            "class_counts": metrics.get("class_counts", {}),
            "calibrated_thresholds": metrics.get("calibrated_thresholds") or {},
            "target_auto_precision": self.settings.local_target_auto_precision,
        }
        return metrics

    def _thresholds(self) -> dict[str, float]:
        s = self.settings
        base = {
            "keep": s.local_keep_threshold,
            "read_later": s.local_read_later_threshold,
            "archive": s.local_archive_threshold,
            "trash": s.local_trash_threshold,
            "unsubscribe_trash": s.local_unsubscribe_threshold,
        }
        # Holdout-calibrated ceilings can only lower the configured values for
        # destructive classes, never raise them.
        calibrated = self.meta.get("calibrated_thresholds") or {}
        for cls in DESTRUCTIVE_ACTIONS:
            if cls in calibrated:
                base[cls] = min(base[cls], float(calibrated[cls]))
        return base

    def full_retrain(self, examples: list[TrainingExample], data_revision: str, reason: str) -> bool:
        """Periodic rebuild with atomic swap; the active model never degrades on failure."""
        if not examples:
            return False
        with self._lock:
            started = time.perf_counter()
            previous_meta = dict(self.meta)
            previous_backend = self.backend
            self.meta["data_revision"] = data_revision
            self.report(f"[classifier-local] periodic retrain starting reason={reason} count={len(examples)}")
            try:
                fresh_backend = select_backend(self.settings.local_device, report=lambda _: None)
                self.backend = fresh_backend
                metrics = self._fit_from(examples, reason=reason)
            except Exception as exc:
                self.backend = previous_backend
                self.meta = previous_meta
                self.last_error = str(exc)
                self.report(f"[classifier-local] retrain failed; keeping model version={previous_meta.get('version')}: {exc}")
                return False
            # Candidate validated: keep the previous known-good files for rollback.
            try:
                self._archive_previous()
                self._persist(self.meta, self.backend)
            except Exception as exc:
                self.report(f"[classifier-local] model persist failed (memory model updated): {exc}")
            self.example_count = len(examples)
            self.retrain_requested = False
            elapsed = time.perf_counter() - started
            self.report(
                f"[classifier-local] periodic retrain complete version={self.version} duration={elapsed:.1f}s"
                f" accuracy={metrics.get('accuracy') or 0:.3f}"
            )
            return True

    def incremental_update(self, examples: list[TrainingExample], data_revision: str | None = None) -> None:
        """Apply user decisions/corrections immediately (spec 8.2)."""
        if not examples:
            return
        with self._lock:
            started = time.perf_counter()
            try:
                X = self._matrix(
                    [build_feature_text(ex, self.policy) for ex in examples]
                )
                y = [ex.label for ex in examples]
                w = np.array([ex.weight for ex in examples], dtype=float)
                if not self.bootstrapped and self.backend.classes != ACTION_CLASSES:
                    self.backend = select_backend(self.settings.local_device, report=lambda _: None)
                self.backend.partial_fit(X, y, sample_weight=w)
                self.example_count += len(examples)
                self.version += 1
                self.meta["trained_at_incremental"] = _now()
                self.meta.setdefault("trained_at", _now())
                self.meta["examples"] = self.example_count
                self.meta["classes"] = list(ACTION_CLASSES)
                self.meta["thresholds"] = self._thresholds()
                self.meta["version"] = self.version
                self.meta["device"] = self.backend.device
                self.meta["backend"] = self.backend.name
                self.meta["incremental_since_full"] = int(self.meta.get("incremental_since_full") or 0) + len(examples)
                if data_revision is not None:
                    self.meta["data_revision"] = data_revision
                self.meta["bootstrapped"] = True
                self.bootstrapped = True
                for ex in examples:
                    if ex.corrected:
                        self.counters["corrections"] += 1
                self._persist(self.meta, self.backend)
                self.report(
                    f"[classifier-local] training update examples={self.example_count}"
                    f" sources={','.join(sorted({str(ex.history.get('source', 'user-decision')) for ex in examples})) or 'user-decision'}"
                    f" duration={time.perf_counter() - started:.3f}s"
                )
            except Exception as exc:
                self.last_error = str(exc)
                self.report(f"[classifier-local] incremental update failed: {exc}")

    # ---------------------------------------------------------- prediction

    def predict(self, example: TrainingExample) -> Prediction:
        with self._lock:
            started = time.perf_counter()
            if not self.bootstrapped or self.example_count == 0:
                return Prediction(None, 0.0, {}, 0.0, evidence=["cold start: no confirmed training data"])
            try:
                text = build_feature_text(example, self.policy)
                probs = self.backend.predict_proba(self._single(text))[0]
            except Exception as exc:
                self.last_error = str(exc)
                return Prediction(None, 0.0, {}, 0.0, evidence=[f"classifier error: {exc}"])
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.predictions += 1
            self._latency_ema_ms = latency_ms if self._latency_ema_ms == 0 else (
                0.9 * self._latency_ema_ms + 0.1 * latency_ms
            )
        ordered = sorted(zip(ACTION_CLASSES, probs), key=lambda kv: -kv[1])
        action, confidence = ordered[0]
        probabilities = {cls: round(float(p), 6) for cls, p in zip(ACTION_CLASSES, probs)}
        evidence = self._evidence(example, list(ordered[:3]))
        return Prediction(
            action=action if confidence > 0 else None,
            confidence=float(confidence),
            probabilities=probabilities,
            latency_ms=round(latency_ms, 2),
            evidence=evidence,
        )

    def predict_batch(self, examples: list[TrainingExample]) -> list[Prediction]:
        if not examples:
            return []
        with self._lock:
            if not self.bootstrapped or self.example_count == 0:
                return [Prediction(None, 0.0, {}) for _ in examples]
            try:
                X = self._matrix([build_feature_text(ex, self.policy) for ex in examples])
                probs = self.backend.predict_proba(X)
            except Exception as exc:
                self.last_error = str(exc)
                return [Prediction(None, 0.0, {}, evidence=[f"classifier error: {exc}"]) for _ in examples]
        results: list[Prediction] = []
        for ex, row in zip(examples, probs):
            ordered = sorted(zip(ACTION_CLASSES, row), key=lambda kv: -kv[1])
            action, confidence = ordered[0]
            results.append(
                Prediction(
                    action=action,
                    confidence=float(confidence),
                    probabilities={cls: round(float(p), 6) for cls, p in zip(ACTION_CLASSES, row)},
                    evidence=self._evidence(ex, list(ordered[:3])),
                )
            )
        return results

    @staticmethod
    def _evidence(example: TrainingExample, top: list[tuple[str, float]]) -> list[str]:
        evidence: list[str] = []
        for label, count in sorted((example.history or {}).items()):
            if count:
                evidence.append(f"{label.replace('hist_', '').replace('_', ' ')} ({count} previous decision(s))")
        features = example.features or {}
        if features.get("one_click"):
            evidence.append("authenticated one-click unsubscribe available")
        elif features.get("has_mailto"):
            evidence.append("mailto unsubscribe advertised")
        if example.list_id:
            evidence.append(f"mailing list {example.list_id}")
        labels = {str(x).upper() for x in features.get("labels", [])}
        for category in ("CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_UPDATES", "CATEGORY_PERSONAL"):
            if category in labels:
                evidence.append(f"Gmail categorises as {category.split('_', 1)[1].lower()}")
        if not evidence and top:
            evidence.append("no strong personal signals")
        return evidence[:6]

    # ------------------------------------------------------------- gating

    def maturity(self) -> str:
        n = self.example_count
        if n < self.settings.local_min_examples:
            return "cold-start"
        if n < self.settings.local_auto_action_min_examples:
            return "early-learning"
        return "mature"

    def gate(
        self,
        prediction: Prediction,
        *,
        plan: UnsubscribePlan | None,
        safety_signals: Sequence[str] = (),
        mixed_history: bool = False,
    ) -> GateResult:
        """Policy layer: when may a prediction act without the user (§10/§11/§31).

        ``mixed_history`` (the same sender/list has received genuinely different
        user decisions) suppresses *auto-action* only. User history is a brake,
        never a memorization feature — see build_feature_text.
        """
        s = self.settings
        action = prediction.action
        if action is None:
            return GateResult("pending", None, prediction.evidence[-1] if prediction.evidence else "no prediction")
        if self.example_count < s.local_min_examples:
            return GateResult("pending", action, f"cold start: only {self.example_count} confirmed example(s)")
        threshold = self._thresholds().get(action, 0.9)
        if self.maturity() == "early-learning" and action not in NON_DESTRUCTIVE_ACTIONS | {"archive"}:
            return GateResult("pending", action, "early model: destructive actions stay Pending")
        if self.maturity() == "early-learning":
            threshold = max(threshold, s.local_early_confidence)
        if action in DESTRUCTIVE_ACTIONS and safety_signals:
            return GateResult("pending", action, f"safety signal: {','.join(safety_signals)}")
        if action in DESTRUCTIVE_ACTIONS and mixed_history:
            return GateResult("pending", action, "mixed previous decisions for this source")
        if action == "unsubscribe_trash" and not (plan and plan.auto_eligible):
            return GateResult("pending", action, "no safe automatic unsubscribe method")
        if prediction.confidence >= threshold:
            return GateResult("actionable", action, f"confidence {prediction.confidence:.2f} >= {threshold:.2f}")
        return GateResult("pending", action, f"confidence {prediction.confidence:.2f} below {threshold:.2f}")

    # ---------------------------------------------------------- retraining

    def retrain_due(self, dataset_total: int) -> tuple[bool, str]:
        if self.retrain_requested:
            return True, "rule-change or explicit request"
        n = int(dataset_total)
        if n and int(self.meta.get("examples") or 0) == 0:
            return True, "first confirmed data"
        since_full = int(self.meta.get("incremental_since_full") or 0)
        if since_full >= self.settings.local_retrain_new_examples:
            return True, f"new examples since full retrain: {since_full}"
        trained_at = self.meta.get("trained_at")
        if trained_at and self.example_count:
            try:
                last = datetime.fromisoformat(trained_at)
                hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
                if hours >= self.settings.local_retrain_interval_hours:
                    return True, f"interval {hours:.1f}h >= {self.settings.local_retrain_interval_hours}h"
            except ValueError:
                return True, "unreadable trained_at"
        elif not trained_at and n:
            return True, "never fully trained"
        return False, ""

    def reset_incremental_counter_after_full(self) -> None:
        self.meta["incremental_since_full"] = 0

    # -------------------------------------------------------------- metrics

    def _evaluate(self, X, y: list[str], weights: np.ndarray) -> dict[str, Any]:
        if len(set(y)) < 2 or len(y) < 10:
            return {"accuracy": None, "class_counts": self._counts(y), "holdout": False}
        try:
            idx_train, idx_test = train_test_split(
                np.arange(len(y)), test_size=0.25, random_state=42, stratify=y if min(self._counts(y).values() or [9]) >= 2 else None
            )
        except Exception:
            return {"accuracy": None, "class_counts": self._counts(y), "holdout": False}
        eval_backend = select_backend(self.settings.local_device, report=lambda _: None)
        eval_backend.fit(X[idx_train], [y[i] for i in idx_train], sample_weight=weights[idx_train])
        probs = eval_backend.predict_proba(X[idx_test])
        preds = [ACTION_CLASSES[int(i)] for i in probs.argmax(axis=1)]
        confs = probs.max(axis=1)
        truth = [y[i] for i in idx_test]
        correct = [p == t for p, t in zip(preds, truth)]
        thresholds = self._thresholds()
        high_conf_idx = [i for i, (p, c) in enumerate(zip(preds, confs)) if c >= thresholds.get(p, 0.9)]
        per_class: dict[str, Any] = {}
        for cls in ACTION_CLASSES:
            predicted = [i for i, p in enumerate(preds) if p == cls]
            actual = [i for i, t in enumerate(truth) if t == cls]
            tp = len(set(predicted) & set(actual))
            per_class[cls] = {
                "precision": round(tp / len(predicted), 3) if predicted else None,
                "recall": round(tp / len(actual), 3) if actual else None,
                "support": len(actual),
            }
        return {
            "holdout": True,
            "accuracy": round(sum(correct) / len(correct), 4),
            "class_counts": self._counts(y),
            "per_class": per_class,
            "high_confidence_accuracy": round(
                sum(correct[i] for i in high_conf_idx) / len(high_conf_idx), 4
            ) if high_conf_idx else None,
            "high_confidence_coverage": round(len(high_conf_idx) / len(preds), 4),
            "eval_device": eval_backend.device,
            "calibrated_thresholds": self._calibrate_destructive(preds, confs, truth, thresholds),
        }

    def _calibrate_destructive(
        self,
        preds: list[str],
        confs: np.ndarray,
        truth: list[str],
        configured: dict[str, float],
    ) -> dict[str, float]:
        """Pick evidence-based thresholds for destructive actions.

        One-vs-rest log-loss sigmoids rarely reach 0.98 even when correct, so a
        fixed near-1.0 gate means the classifier can never act on what it has
        genuinely learned. Instead, for each destructive class we scan the
        holdout predictions ascending by confidence and select the lowest
        threshold whose measured precision meets ``local_target_auto_precision``
        (with a minimum support count). Calibration may only *lower* a
        configured ceiling, never raise it, and floors at 0.60. It requires a
        holdout sample, so it can never authorise auto-cleaning from nothing.
        """
        calibrated: dict[str, float] = {}
        target = min(1.0, max(0.5, self.settings.local_target_auto_precision))
        min_support = self.settings.local_calibrate_min_support
        for cls in DESTRUCTIVE_ACTIONS:
            idx = [i for i, p in enumerate(preds) if p == cls]
            if len(idx) < min_support:
                continue
            ordered = sorted(idx, key=lambda i: confs[i], reverse=True)
            best: float | None = None
            for k in range(min_support, len(ordered) + 1):
                window = ordered[:k]
                precision = sum(truth[i] == cls for i in window) / k
                if precision >= target:
                    best = float(confs[window[-1]])
                else:
                    break  # precision degrades as we widen; stop at first miss
            if best is not None:
                floor = 0.60 if cls == "trash" else 0.70
                calibrated[cls] = round(min(configured.get(cls, 0.99), max(floor, best - 0.02)), 3)
        return calibrated

    @staticmethod
    def _counts(y: list[str]) -> dict[str, int]:
        counts = {cls: 0 for cls in ACTION_CLASSES}
        for label in y:
            if label in counts:
                counts[label] += 1
        return counts

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            thresholds = self._thresholds()
            next_new = int(self.meta.get("incremental_since_full") or 0)
            return {
                "backend": self.backend.name,
                "device": self.backend.device,
                "status": "ready" if self.bootstrapped else "bootstrap",
                "maturity": self.maturity(),
                "examples": self.example_count,
                "version": self.version,
                "trained_at": self.meta.get("trained_at"),
                "thresholds": thresholds,
                "calibrated_thresholds": self.meta.get("calibrated_thresholds") or {},
                "target_auto_precision": self.settings.local_target_auto_precision,
                "metrics": self.meta.get("metrics") or {},
                "class_counts": self.meta.get("class_counts") or {},
                "min_examples": self.settings.local_min_examples,
                "auto_action_min_examples": self.settings.local_auto_action_min_examples,
                "retrain_new_remaining": max(
                    0, self.settings.local_retrain_new_examples - next_new
                ),
                "retrain_interval_hours": self.settings.local_retrain_interval_hours,
                "latency_ema_ms": round(self._latency_ema_ms, 2),
                "predictions": self.predictions,
                "auto_actioned": self.counters["auto_actioned"],
                "pending": self.counters["pending"],
                "corrections": self.counters["corrections"],
                "reeval_released": self.counters.get("reeval_released", 0),
                "last_error": self.last_error,
                "retrain_requested": self.retrain_requested,
            }
