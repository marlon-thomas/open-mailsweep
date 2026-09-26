"""Tests for the fast local personalised classifier.

The central design rule these tests defend: the classifier learns what an
email IS (content), not merely who it is FROM (identity). Per-source
memorisation is the learned-rule table's job; an example from a sender the
model has never seen must still be classified from its content.
"""

from pathlib import Path

import pytest

from openmailsweep.config import Policy, Settings
from openmailsweep.local_classifier import (
    ACTION_CLASSES,
    LocalClassifier,
    build_feature_text,
    select_backend,
)
from openmailsweep.training_data import TrainingDataset, TrainingExample


def local_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        classifier="local",
        gmail_token=tmp_path / "token.json",
        state_db=tmp_path / "state.db",
        audit_db=tmp_path / "audit.db",
        classifier_dir=tmp_path / "classifier",
        local_min_examples=6,
        local_auto_action_min_examples=6,
        local_keep_threshold=0.55,
        local_read_later_threshold=0.55,
        local_archive_threshold=0.55,
        local_trash_threshold=0.55,
        local_unsubscribe_threshold=0.55,
        local_retrain_new_examples=1000,
        local_retrain_interval_hours=9999.0,
    )
    base.update(overrides)
    return Settings(**base)


def promo_example(mid: str, address: str, display: str, subject: str, snippet: str) -> TrainingExample:
    return TrainingExample(
        message_id=mid,
        label="unsubscribe_trash",
        sender=f"{display} <{address}>",
        sender_address=address,
        subject=subject,
        snippet=snippet,
        list_id="",
        features={
            "labels": ["CATEGORY_PROMOTIONS", "UNREAD"],
            "one_click": True,
            "has_mailto": True,
            "has_list_unsubscribe": True,
            "precedence_bulk": True,
            "auto_submitted": False,
        },
    )


def personal_example(mid: str, address: str, display: str, subject: str, snippet: str) -> TrainingExample:
    return TrainingExample(
        message_id=mid,
        label="keep",
        sender=f"{display} <{address}>",
        sender_address=address,
        subject=subject,
        snippet=snippet,
        list_id="",
        features={"labels": ["CATEGORY_PERSONAL", "IMPORTANT", "INBOX"]},
    )


def read_later_example(mid: str, address: str, display: str, subject: str, snippet: str) -> TrainingExample:
    return TrainingExample(
        message_id=mid,
        label="read_later",
        sender=f"{display} <{address}>",
        sender_address=address,
        subject=subject,
        snippet=snippet,
        list_id="",
        features={"labels": ["CATEGORY_UPDATES", "UNREAD"]},
    )


def archive_example(mid: str, address: str, display: str, subject: str, snippet: str) -> TrainingExample:
    return TrainingExample(
        message_id=mid,
        label="archive",
        sender=f"{display} <{address}>",
        sender_address=address,
        subject=subject,
        snippet=snippet,
        list_id="",
        features={"labels": ["CATEGORY_UPDATES"]},
    )


def trash_example(mid: str, address: str, display: str, subject: str, snippet: str) -> TrainingExample:
    return TrainingExample(
        message_id=mid,
        label="trash",
        sender=f"{display} <{address}>",
        sender_address=address,
        subject=subject,
        snippet=snippet,
        list_id="",
        features={"labels": []},
    )


def training_set() -> list[TrainingExample]:
    # Multiple distinct senders/domains per class: no single identity appears
    # more than once, so a high score on held-out *new* senders proves content
    # generalisation, not memorisation. All five action classes appear because
    # sklearn's log-loss is one-vs-rest: an unseen class sits at p=0.5.
    return [
        promo_example("p1", "deals@brand-a.com", "Brand A Sale", "Flash sale 50% off everything shop now", "limited time offer save 50% discount on all items"),
        promo_example("p2", "news@brand-b.com", "Brand B Newsletter", "Weekly deals newsletter 40% off", "unsubscribe manage preferences bulk promo discount"),
        promo_example("p3", "offers@brand-c.com", "Brand C Offers", "Your exclusive promo offer inside today only", "shop now coupon code discount marketing"),
        promo_example("p4", "promo@brand-d.com", "Brand D Promotions", "Big sale this weekend up to 60% off", "clearance sale best deals opt out preferences"),
        personal_example("k1", "amy@person-a.com", "Amy Whitfield", "Re: dinner on Friday?", "hey are we still on for dinner friday let me know"),
        personal_example("k2", "ben@person-b.com", "Ben Ortiz", "Photos from the wedding", "here are the photos we took at the wedding last weekend"),
        personal_example("k3", "carla@person-c.com", "Carla Mendes", "Fwd: project deadline next week", "can you review my draft before the meeting on tuesday"),
        personal_example("k4", "dan@person-d.com", "Dan Fowler", "Are you coming to the party", "sarah and the kids say hi hoping you can make saturday"),
        read_later_example("r1", "alerts@jobsite-a.com", "JobSite A Alerts", "New jobs for you this week", "12 roles matching your profile applied recently recommended jobs"),
        read_later_example("r2", "careers@jobsite-b.com", "JobSite B Careers", "Job alert: roles matching your saved search", "new vacancies matching your career alert digest"),
        read_later_example("r3", "digest@jobsite-c.com", "JobSite C Digest", "Weekly job recommendations for you", "featured roles recommended jobs career opportunities salary"),
        read_later_example("r4", "jobs@jobsite-d.com", "JobSite D Jobs", "Career opportunities this week only", "recommended jobs matching your profile browse roles"),
        archive_example("a1", "no-reply@tooling-a.io", "Tool A", "Your weekly usage summary", "summary of your usage activity this week generated automatically"),
        archive_example("a2", "support@tooling-b.io", "Tool B", "Monthly account activity summary", "account summary report generated automatically for your records"),
        archive_example("a3", "no-reply@tooling-c.io", "Tool C", "Quarterly usage statistics ready", "statistics summary your usage data export ready"),
        archive_example("a4", "help@tooling-d.io", "Tool D", "Weekly report summary for your team", "summary report generated automatically team usage"),
        trash_example("t1", "winner@lottery-e.biz", "Prize Center", "You have won a free iphone claim now", "click here claim your free prize winner congratulations urgent"),
        trash_example("t2", "singles@dating-f.biz", "Local Singles", "Hot singles want to meet you tonight", "click here photos hot singles nearby message now free"),
        trash_example("t3", "pill@pharma-g.biz", "Cheap Pills", "Buy cheap viagra pills online now", "click here order cheap pills pharmacy discount free shipping"),
        trash_example("t4", "crypto@pump-h.biz", "Crypto Pump", "Get rich quick with this one weird crypto trick", "click here limited spots free money fast riches guaranteed"),
    ]


@pytest.fixture()
def classifier(tmp_path: Path) -> LocalClassifier:
    settings = local_settings(tmp_path)
    model = LocalClassifier(settings, Policy(), report=lambda _t: None)
    state = model.bootstrap(training_set(), "rev-1")
    assert state in {"trained", "loaded"}
    model.meta["data_revision"] = "rev-1"
    return model


def test_unseen_sender_is_classified_from_content(classifier: LocalClassifier):
    """The core anti-memorisation requirement."""
    unseen = promo_example(
        "z9", "sales@never-seen-shop.co.uk", "Never Seen Shop",
        "Weekend flash sale 45% off all orders", "limited time promo discount code shop now",
    )
    prediction = classifier.predict(unseen)
    assert prediction.action == "unsubscribe_trash"

    unseen_person = personal_example(
        "z8", "erin@someone-new.co.uk", "Erin Novak",
        "Re: coffee tomorrow morning?", "would love to catch up over coffee before work",
    )
    assert classifier.predict(unseen_person).action == "keep"


def test_identity_tokens_are_not_part_of_the_model_input():
    ex = promo_example("p1", "deals@brand-a.com", "Brand A Sale", "Weekly sale", "big offers inside")
    ex.list_id = "newsletters@brand-a.com"
    ex.history = {"hist_sender_trash": 5, "hist_list_unsubscribe_trash": 9}
    ex.features["rule_scopes"] = ["list", "sender"]
    ex.features["prior_sent"] = True
    text = build_feature_text(ex, Policy())
    assert "deals@brand-a.com" not in text          # exact sender address excluded
    assert "newsletters@brand-a.com" not in text    # exact list-ID excluded
    assert "hist_sender" not in text                # per-sender history excluded
    assert "rule " not in text                      # learned-rule existence excluded
    assert "prior sent" not in text                 # identity history excluded
    assert "dom brand-a.com" in text                # domain-level generalisation kept
    assert "subj weekly sale" in text               # content features kept


def test_gate_respects_thresholds_and_maturity(classifier: LocalClassifier):
    from openmailsweep.models import UnsubscribePlan

    plan = UnsubscribePlan(method="one_click", one_click=True, one_click_auto_eligible=True, auto_eligible=True, reason="ok")
    unseen = promo_example("z1", "x@shop-now.com", "Shop", "flash sale 50% off limited time", "promo discount shop now coupon")
    prediction = classifier.predict(unseen)
    gate = classifier.gate(prediction, plan=plan)
    assert gate.mode == "actionable"
    assert gate.action == "unsubscribe_trash"

    # No safe unsubscribe method => never auto unsubscribe.
    gate2 = classifier.gate(prediction, plan=UnsubscribePlan(method="none", reason="no method"))
    assert gate2.mode == "pending"
    assert "unsubscribe" in gate2.reason

    # Destructive action + safety signal => pending.
    gate3 = classifier.gate(prediction, plan=plan, safety_signals=["gmail-personal"])
    assert gate3.mode == "pending"

    # Destructive action + mixed history for this source => pending (brake only).
    gate4 = classifier.gate(prediction, plan=plan, mixed_history=True)
    assert gate4.mode == "pending"
    assert "mixed" in gate4.reason


def test_cold_start_never_autocleans(tmp_path: Path):
    settings = local_settings(tmp_path)
    model = LocalClassifier(settings, Policy(), report=lambda _t: None)
    assert model.bootstrap([], "rev-empty") == "no-data"
    from openmailsweep.models import UnsubscribePlan

    plan = UnsubscribePlan(method="one_click", auto_eligible=True, one_click=True, reason="ok")
    prediction = model.predict(training_set()[0])
    gate = model.gate(prediction, plan=plan)
    assert gate.mode == "pending"
    assert gate.action is None


def test_persistence_roundtrip_and_stale_detection(tmp_path: Path):
    settings = local_settings(tmp_path)
    model = LocalClassifier(settings, Policy(), report=lambda _t: None)
    model.bootstrap(training_set(), "rev-1")
    model.meta["data_revision"] = "rev-1"
    model._persist(model.meta, model.backend)

    reloaded = LocalClassifier(settings, Policy(), report=lambda _t: None)
    assert reloaded.load() is True
    assert reloaded.is_stale("rev-1") is False
    assert reloaded.is_stale("rev-2") is True
    unseen = promo_example("z2", "q@totally-new.store", "New Store", "sale 70% off coupon code", "limited time promo")
    assert reloaded.predict(unseen).action == "unsubscribe_trash"


def test_corrupted_model_file_is_not_silently_used(tmp_path: Path):
    settings = local_settings(tmp_path)
    model = LocalClassifier(settings, Policy(), report=lambda _t: None)
    model.bootstrap(training_set(), "rev-1")
    model.meta["data_revision"] = "rev-1"
    model._persist(model.meta, model.backend)
    model.model_path.write_bytes(b"not a model")

    broken = LocalClassifier(settings, Policy(), report=lambda _t: None)
    assert broken.load() is False  # caller must rebuild from stored examples


def test_incremental_update_persists_and_survives_reload(tmp_path: Path):
    settings = local_settings(tmp_path)
    model = LocalClassifier(settings, Policy(), report=lambda _t: None)
    model.bootstrap(training_set()[:8], "rev-1")
    model.meta["data_revision"] = "rev-1"
    model.incremental_update([personal_example("k5", "fiona@person-e.com", "Fiona Grant", "Re: weekend hike?", "are you joining the hike saturday bring water")], data_revision="rev-2")
    assert model.example_count == 9
    reloaded = LocalClassifier(settings, Policy(), report=lambda _t: None)
    assert reloaded.load() is True
    assert reloaded.example_count >= 8


def test_retrain_due_and_atomic_swap_keep_previous_on_failure(classifier: LocalClassifier):
    classifier.meta["incremental_since_full"] = classifier.settings.local_retrain_new_examples
    due, reason = classifier.retrain_due(classifier.example_count)
    assert due and "examples" in reason

    version_before = classifier.version
    files_before = classifier.meta_path.read_text()

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated training failure")

    classifier._fit_from = boom  # type: ignore[method-assign]
    assert classifier.full_retrain(training_set(), "rev-2", "test") is False
    assert classifier.version == version_before
    assert classifier.meta_path.read_text() == files_before


def test_backend_selection_falls_back_to_cpu(monkeypatch, tmp_path: Path):
    import sys

    class FakeCupyUnavailable:
        def __getattr__(self, _name):
            raise ImportError("no gpu")

    monkeypatch.setitem(sys.modules, "cupy", FakeCupyUnavailable())
    backend = select_backend("auto", report=lambda _t: None)
    assert backend.device == "cpu"
    assert backend.classes == ACTION_CLASSES


def test_batch_prediction_matches_single(classifier: LocalClassifier):
    examples = [
        promo_example("b1", "sales@new-co.com", "New Co", "flash sale up to 80% off", "clearance promo discount code"),
        personal_example("b2", "mia@person-f.com", "Mia Stone", "Re: lunch on monday", "let us grab lunch monday near the office"),
    ]
    batch = classifier.predict_batch(examples)
    singles = [classifier.predict(ex) for ex in examples]
    assert [p.action for p in batch] == [p.action for p in singles]
