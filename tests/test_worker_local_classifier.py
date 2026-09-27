"""End-to-end tests for the local classifier inside the worker pipeline.

These pin the architectural promises:
- the model trains on email CONTENT (subject/snippet/body signals), not sender
  identity: a sender never seen before is routed by content;
- bootstrap runs before any unknown message is classified;
- learned rules still bypass the model entirely;
- user answers train the model online;
- a failed retrain retains the previous known-good model.
"""

from pathlib import Path

from openmailsweep.audit import AuditLog
from openmailsweep.config import Policy, Settings
from openmailsweep.models import EmailMessage, UnsubscribePlan
from openmailsweep.state_store import StateStore
from openmailsweep.worker import ClassifyCandidate, OpenMailSweepService


def promo_msg(mid: str, sender_addr: str, list_domain: str) -> EmailMessage:
    display = sender_addr.split("@")[0].title()
    return EmailMessage(
        id=mid,
        thread_id=f"t-{mid}",
        subject=f"Flash sale {display}: 50% off everything shop now",
        sender=f"{display} Deals <{sender_addr}>",
        sender_address=sender_addr,
        body="limited time offer save 50% discount promo code clearance clearance coupon",
        snippet="limited time offer save 50% discount on all items",
        labels=["CATEGORY_PROMOTIONS"],
        headers={
            "list-id": f"<news@{list_domain}>",
            "list-unsubscribe": f"<https://{list_domain}/u>, <mailto:u@{list_domain}>",
            "list-unsubscribe-post": "List-Unsubscribe=One-Click",
            "precedence": "bulk",
        },
    )


def plan() -> UnsubscribePlan:
    return UnsubscribePlan(
        method="one_click",
        https_url="https://example.com/u",
        target_host="example.com",
        one_click=True,
        dkim_authenticated=True,
        one_click_auto_eligible=True,
        auto_eligible=True,
        reason="test",
    )


def make_service(tmp_path: Path) -> OpenMailSweepService:
    settings = Settings(
        classifier="local",
        gmail_token=tmp_path / "missing-token.json",
        state_db=tmp_path / "state.db",
        audit_db=tmp_path / "audit.db",
        classifier_dir=tmp_path / "classifier",
        action_workers=1,
        local_min_examples=4,
        local_auto_action_min_examples=4,
        local_keep_threshold=0.55,
        local_read_later_threshold=0.55,
        local_archive_threshold=0.55,
        local_trash_threshold=0.55,
        local_unsubscribe_threshold=0.55,
        local_retrain_new_examples=10_000,
        local_retrain_interval_hours=9999.0,
    )
    store = StateStore(tmp_path / "state.db")
    audit = AuditLog(tmp_path / "audit.db")
    return OpenMailSweepService(settings, Policy(), store, audit)


def train_from_senders(service: OpenMailSweepService) -> None:
    """Create learned rules from four DISTINCT senders (bootstrap source)."""
    senders = [
        ("boot1", "deals@brand-a.com", "brand-a.com"),
        ("boot2", "news@brand-b.com", "brand-b.com"),
        ("boot3", "offers@brand-c.com", "brand-c.com"),
        ("boot4", "promo@brand-d.com", "brand-d.com"),
    ]
    for mid, address, list_domain in senders:
        msg = promo_msg(mid, address, list_domain)
        item_id = service.store.add_discovered(msg, plan(), safety_signals=None)
        service.store.set_body_excerpt(item_id, msg.body)
        service.store.set_actionable(item_id, "unsubscribe_trash", source="user-message", reason="seed")
        service.store.finish_action(item_id, "unsubscribe_trash", "seeded")
        service.store.upsert_rule("sender", address, "unsubscribe_trash", source_message_id=mid, note="seed")
    service._bootstrap_local_classifier()
    assert service.local is not None and service.local.bootstrapped
    assert service.local.example_count >= 4


def test_content_trained_pipeline_routes_unseen_sender_to_actionable(tmp_path: Path):
    """The core requirement: the model acts on what the email IS.

    Training data contains four senders (brand-a..d.com). The predicted
    message comes from an entirely unseen sender/list domain; routing it to
    Actionable proves the decision came from content, not sender identity.
    """
    service = make_service(tmp_path)
    train_from_senders(service)

    fresh = promo_msg("z-new", "sales@totally-unseen-shop.co.uk", "unseen-shop.co.uk")
    item_id = service.store.add_discovered(fresh, plan())
    claimed = service.store.claim_discovered_batch(16)
    assert [int(r["id"]) for r in claimed] == [item_id]
    candidate = ClassifyCandidate(item_id, fresh, [], plan())
    service._classify_claimed_locally([candidate])

    item = service.store.get_item(item_id)
    assert item["status"] == "actionable"
    assert item["desired_action"] == "unsubscribe_trash"
    assert item["decision_source"] == "local-classifier"
    assert service.local.counters["auto_actioned"] >= 1


def test_identity_tokens_never_enter_the_training_matrix(tmp_path: Path):
    """Static guard: the exact sender address, full List-ID, rule-existence and
    per-sender history tokens must not appear in the model feature text."""
    service = make_service(tmp_path)
    train_from_senders(service)
    from openmailsweep.local_classifier import build_feature_text
    from openmailsweep.training_data import TrainingExample

    ex = TrainingExample(
        message_id="x",
        label="keep",
        sender="Zoe <zoe@exact-test.com>",
        sender_address="zoe@exact-test.com",
        subject="hello there",
        snippet="snippet text",
        list_id="weekly@lists.test.com",
        features={
            "labels": ["INBOX"],
            "rule_scopes": ["sender", "domain"],
            "prior_sent": True,
            "body_excerpt": "body content here",
        },
        history={"hist_sender_keep": 4},
    )
    text = build_feature_text(ex, service.policy)
    assert "zoe@exact-test.com" not in text
    assert "weekly@lists.test.com" not in text
    assert "hist_sender" not in text
    assert "rule sender" not in text
    assert "rule domain" not in text
    assert "prior sent" not in text
    assert "body content here" in text  # content IS a first-class feature


def test_cold_unknowns_go_to_pending_not_actions(tmp_path: Path):
    """No confirmed data => zero auto actions ever (fail safe)."""
    service = make_service(tmp_path)
    service._bootstrap_local_classifier()
    fresh = promo_msg("z1", "deals@brand-z.com", "brand-z.com")
    item_id = service.store.add_discovered(fresh, plan())
    service.store.claim_discovered_batch(16)
    candidate = ClassifyCandidate(item_id, fresh, [], plan())
    service._classify_claimed_locally([candidate])
    item = service.store.get_item(item_id)
    assert item["status"] == "pending"
    assert item["desired_action"] is None


def test_learned_rule_beats_classifier(tmp_path: Path):
    """An exact rule routes with source=learned-rule even when the model would
    predict something else; priority order is never weakened."""
    service = make_service(tmp_path)
    train_from_senders(service)
    msg = promo_msg("rule1", "vip@brand-x.com", "brand-x.com")
    service.store.upsert_rule("sender", "vip@brand-x.com", "keep", source_message_id="rule1")
    item_id = service.store.add_discovered(msg, plan())
    service.store.claim_discovered_batch(16)
    service._classify_claimed_locally([ClassifyCandidate(item_id, msg, [], plan())])
    item = service.store.get_item(item_id)
    assert item["decision_source"] == "learned-rule"
    assert item["desired_action"] == "keep"


def test_user_answer_trains_model_online(tmp_path: Path):
    service = make_service(tmp_path)
    train_from_senders(service)
    before = service.local.example_count

    # A *personal-style* message from a new sender, answered "keep" via the
    # web flow path, must land in the dataset and in the live model.
    msg = EmailMessage(
        id="ans1",
        thread_id="t-ans1",
        subject="Re: are we still meeting sunday",
        sender="Nina <nina@friend-mail.com>",
        sender_address="nina@friend-mail.com",
        body="looking forward to seeing you sunday afternoon",
        snippet="looking forward to seeing you sunday",
        labels=["CATEGORY_PERSONAL"],
        headers={},
    )
    item_id = service.store.add_discovered(msg, UnsubscribePlan(method="none", reason="none"))
    service.store.set_body_excerpt(item_id, msg.body)
    service.store.mark_pending(
        item_id, None, reason="cold", question="q", proposed_action=None
    )
    service.store.answer_pending(item_id, "keep", "sender")
    service.handle_user_answer(item_id, "keep", "sender")

    assert service.training.count() == before + 1
    assert service.local.example_count == before + 1
    # The model must now be able to say something about this content.
    prediction = service.local.predict(
        service._classifier_example(msg, UnsubscribePlan(method="none", reason="none"))
    )
    assert prediction.action is not None


def test_failed_periodic_retrain_keeps_previous_model(tmp_path: Path):
    service = make_service(tmp_path)
    train_from_senders(service)
    version_before = service.local.version
    service.local.retrain_requested = True

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated retrain failure")

    service.local._fit_from = boom  # type: ignore[method-assign]
    service._maybe_periodic_retrain()
    assert service.local.version == version_before
    assert service.local.bootstrapped is True
    # The pipeline still classifies after a failed retrain.
    fresh = promo_msg("post-fail", "new@unseen-domain.io", "unseen-domain.io")
    item_id = service.store.add_discovered(fresh, plan())
    service.store.claim_discovered_batch(16)
    service._classify_claimed_locally([ClassifyCandidate(item_id, fresh, [], plan())])
    assert service.store.get_item(item_id)["status"] in {"actionable", "pending"}

def _make_pending(service, msg, safety_signals=None):
    unsub = UnsubscribePlan(
        method="one_click", one_click=True, one_click_auto_eligible=True,
        auto_eligible=True, reason="ok",
    )
    item_id = service.store.add_discovered(msg, unsub)
    service.store.claim_discovered_batch(16)
    service.store.mark_pending(
        item_id, None, reason="below gate", question="q", proposed_action=None,
        safety_signals=safety_signals or [],
    )
    return item_id


def test_pending_reeval_released_items_rechecked_by_executor(tmp_path: Path):
    """Items released via local-classifier-reeval must pass the deferred hard
    safety re-check before any Gmail mutation (they may have waited days and
    acquired replies/star flags meanwhile)."""
    service = make_service(tmp_path)
    train_from_senders(service)
    msg = promo_msg("z-exec", "sales@unseen-exec.co.uk", "unseen-exec.co.uk")
    item_id = service.store.add_discovered(msg, plan())

    class BlockedGmail:
        def __init__(self):
            self.trashed = []
        def get_message_metadata(self, mid):
            return msg
        def thread_has_sent_message(self, tid):
            return True  # the user replied after the message went Pending
        def trash(self, mid):
            self.trashed.append(mid)

    gmail = BlockedGmail()
    service.store.set_actionable(item_id, "trash", source="local-classifier-reeval", reason="test")
    claimed = service.store.claim_next_action()
    service._execute_action(claimed, gmail, worker_id=1)

    item = service.store.get_item(item_id)
    assert item["status"] == "done" and item["final_action"] == "protected"
    assert "thread-has-sent-reply" in (item["protected_reason"] or "")
    assert gmail.trashed == []


def test_pending_reeval_calibrated_release_and_signal_brake(tmp_path: Path):
    """After a calibrated retrain lowers the gate: identical content releases
    from Pending, while stored safety signals keep theirs blocked."""
    from test_local_classifier import big_training_set

    service = make_service(tmp_path)
    train_from_senders(service)
    for ex in big_training_set():
        service.training.upsert_example(
            message_id=ex.message_id, label=ex.label, sender=ex.sender,
            sender_address=ex.sender_address, subject=ex.subject,
            snippet=ex.snippet, list_id=ex.list_id, features=ex.features,
            source="pending-answer",
        )
    service.local.full_retrain(service._dataset_examples(service.training), "rev-x", "test")
    assert service.local.current_thresholds()["unsubscribe_trash"] < 0.98 or \
           service.local.current_thresholds()["trash"] < 0.95

    clean = promo_msg("z-clean", "bulk@calibrated-sender.com", "calibrated-sender.com")
    clean_id = _make_pending(service, clean)
    signalled = promo_msg("z-sig", "bulk2@calibrated-sender.com", "calibrated-sender.com")
    signalled_id = _make_pending(service, signalled, safety_signals=["prior-correspondent"])

    service._reeval_pending_sweep("test")

    released = service.store.get_item(clean_id)
    assert released["status"] == "actionable"
    assert released["decision_source"] == "local-classifier-reeval"
    assert service.store.get_item(signalled_id)["status"] == "pending"
    assert service.local.counters["reeval_released"] >= 1


def test_pending_reeval_survives_restart_via_startup_trigger(tmp_path: Path):
    """Sweep is wired into _bootstrap_local_classifier, so a fresh process
    re-scores Pending with the loaded model (trigger: startup)."""
    service = make_service(tmp_path)
    train_from_senders(service)
    msg = promo_msg("z-boot", "new@boot-seen.co.uk", "boot-seen.co.uk")
    item_id = _make_pending(service, msg)

    # Second "process": new service over the same persisted files.
    from openmailsweep.audit import AuditLog
    from openmailsweep.state_store import StateStore
    service2 = OpenMailSweepService(
        service.settings, Policy(), StateStore(tmp_path / "state.db"), AuditLog(tmp_path / "audit.db")
    )
    service2._bootstrap_local_classifier()
    assert service2.local is not None and service2.local.bootstrapped
    # The startup sweep ran inside bootstrap and released the item under this
    # fixture's deliberately low 0.55 test gate.
    item = service2.store.get_item(item_id)
    assert item["status"] == "actionable"
    assert item["decision_source"] == "local-classifier-reeval"
    assert "startup" in (item["decision_reason"] or "")


class GoneGmail:
    """Body fetch raises like Gmail 404 / Yahoo LookupError for deleted messages."""
    def __init__(self, exc):
        self.exc = exc
        self.trashed = []
    def get_message(self, mid):
        raise self.exc
    def get_message_metadata(self, mid):
        raise self.exc


def _claim_pending_gone(service, mid="gone1"):
    msg = promo_msg(mid, "old@deleted-mailing.com", "deleted-mailing.com")
    item_id = service.store.add_discovered(msg, plan())
    return item_id


def test_gone_gmail_404_marks_failed_instead_of_requeuing(tmp_path: Path):
    class FakeHttpError404(RuntimeError):
        def __init__(self):
            super().__init__(
                "<HttpError 404 when requesting ... returned \"Requested entity was not found.\". "
                "Details: \"[{'message': 'Requested entity was not found.', 'reason': 'notFound'}]\">"
            )
            self.resp = SimpleNamespace(status=404)

    from types import SimpleNamespace

    service = make_service(tmp_path)
    train_from_senders(service)
    http404 = FakeHttpError404()
    item_id = _claim_pending_gone(service)
    service._classifier_gmail = GoneGmail(http404)
    claimed = service.store.claim_discovered_batch(16)
    service._classify_claimed(claimed)

    item = service.store.get_item(item_id)
    assert item["status"] == "failed"
    assert "not found" in (item["error"] or "").lower()


def test_gone_yahoo_lookup_error_marks_failed(tmp_path: Path):
    service = make_service(tmp_path)
    train_from_senders(service)
    item_id = _claim_pending_gone(service, mid="gone2")
    service._classifier_gmail = GoneGmail(LookupError("message not found"))
    claimed = service.store.claim_discovered_batch(16)
    service._classify_claimed(claimed)
    assert service.store.get_item(item_id)["status"] == "failed"


def test_transient_fetch_error_still_requeues(tmp_path: Path):
    service = make_service(tmp_path)
    train_from_senders(service)
    item_id = _claim_pending_gone(service, mid="gone3")
    service._classifier_gmail = GoneGmail(ConnectionError("temporary network blip"))
    claimed = service.store.claim_discovered_batch(16)
    service._classify_claimed(claimed)
    assert service.store.get_item(item_id)["status"] == "discovered"  # retried later
