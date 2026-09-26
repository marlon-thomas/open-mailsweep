from pathlib import Path

from openmailsweep.models import Classification, EmailMessage, UnsubscribePlan
from openmailsweep.state_store import StateStore


def msg(mid: str, sender: str, *, list_id: str = "", subject: str = "Offer") -> EmailMessage:
    address = sender.lower()
    return EmailMessage(
        id=mid,
        thread_id=f"t-{mid}",
        subject=subject,
        sender=f"Sender <{address}>",
        sender_address=address,
        body="body",
        snippet="preview",
        labels=[],
        headers={"list-id": list_id} if list_id else {},
    )


def plan(auto=True):
    return UnsubscribePlan(
        method="one_click" if auto else "none",
        auto_eligible=auto,
        one_click=auto,
        target_host="example.com" if auto else None,
        reason="test",
    )


def classification():
    return Classification("newsletter", 0.4, 0.8, 0.95, 0.2, 0.5, {})


def make_pending(store: StateStore, m: EmailMessage) -> int:
    item_id = store.add_discovered(m, plan())
    store.mark_pending(
        item_id,
        classification(),
        reason="uncertain",
        question="What should OpenMailSweep do?",
        proposed_action="unsubscribe_trash",
    )
    return item_id


def test_pending_fifo_answer_drops_top_and_next_floats_up(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    first = make_pending(store, msg("m1", "one@example.com"))
    second = make_pending(store, msg("m2", "two@example.com"))

    assert [x["id"] for x in store.list_pending()] == [first, second]
    store.answer_pending(first, "trash", "message")

    pending = store.list_pending()
    assert [x["id"] for x in pending] == [second]
    assert store.get_item(first)["status"] == "actionable"


def test_sender_rule_releases_all_matching_pending(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    one = make_pending(store, msg("m1", "offers@example.com"))
    two = make_pending(store, msg("m2", "offers@example.com"))
    other = make_pending(store, msg("m3", "other@example.com"))

    result = store.answer_pending(one, "trash", "sender")

    assert result["released"] == 2
    assert store.get_item(one)["status"] == "actionable"
    assert store.get_item(two)["status"] == "actionable"
    assert store.get_item(other)["status"] == "pending"


def test_list_rule_is_more_specific_than_sender_and_domain(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    store.upsert_rule("domain", "example.com", "archive")
    store.upsert_rule("sender", "offers@example.com", "trash")
    store.upsert_rule("list", "weekly.example.com", "keep")

    matched = store.matching_rule_for_message(msg("m1", "offers@example.com", list_id="<weekly.example.com>"))
    assert matched["scope_type"] == "list"
    assert matched["action"] == "keep"


def test_rule_update_changes_unprocessed_matching_action(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    one = make_pending(store, msg("m1", "offers@example.com"))
    answer = store.answer_pending(one, "trash", "sender")
    rule_id = answer["rule_id"]
    two = make_pending(store, msg("m2", "offers@example.com"))

    updated = store.update_rule(rule_id, "archive")

    assert updated["action"] == "archive"
    assert store.get_item(one)["desired_action"] == "archive"
    assert store.get_item(two)["desired_action"] == "archive"
    assert store.get_item(two)["status"] == "actionable"


def test_claim_action_is_fifo_and_marks_processing(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    first = make_pending(store, msg("m1", "a@example.com"))
    second = make_pending(store, msg("m2", "b@example.com"))
    store.answer_pending(first, "archive", "message")
    store.answer_pending(second, "trash", "message")

    claimed = store.claim_next_action()
    assert claimed["id"] == first
    assert claimed["status"] == "processing"


def test_domain_rule_cannot_unsubscribe_whole_domain(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    try:
        store.upsert_rule("domain", "example.com", "unsubscribe_trash")
    except ValueError as exc:
        assert "entire domain" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_classifier_claims_discovered_in_fifo_batches(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    ids = [store.add_discovered(msg(f"m{i}", f"u{i}@example.com"), plan()) for i in range(1, 4)]

    claimed = store.claim_discovered_batch(2)

    assert [x["id"] for x in claimed] == ids[:2]
    assert store.get_item(ids[0])["status"] == "classifying"
    assert store.get_item(ids[1])["status"] == "classifying"
    assert store.get_item(ids[2])["status"] == "discovered"


def test_new_rule_can_release_message_while_it_is_classifying(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    item_id = store.add_discovered(msg("m1", "offers@example.com"), plan())
    claimed = store.claim_discovered_batch(1)
    assert claimed[0]["status"] == "classifying"

    rule_id = store.upsert_rule("sender", "offers@example.com", "trash")
    released = store.apply_rule_to_open_items(rule_id)

    assert released == 1
    assert store.get_item(item_id)["status"] == "actionable"
    assert store.get_item(item_id)["desired_action"] == "trash"


def test_recover_processing_restores_classifier_and_action_work(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    classify_id = store.add_discovered(msg("m1", "a@example.com"), plan())
    store.claim_discovered_batch(1)
    action_id = make_pending(store, msg("m2", "b@example.com"))
    store.answer_pending(action_id, "trash", "message")
    store.claim_next_action()

    recovered = store.recover_processing()

    assert recovered == 2
    assert store.get_item(classify_id)["status"] == "discovered"
    assert store.get_item(action_id)["status"] == "actionable"


def test_recover_processing_requeues_old_gmail_quota_action_failures(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    item = store.add_discovered(msg("quota1", "offers@example.com"), plan())
    store.set_actionable(item, "trash", source="test", reason="test")
    claimed = store.claim_next_action()
    assert claimed is not None
    store.fail_action(item, "HttpError 403 rateLimitExceeded: Quota exceeded")
    assert store.get_item(item)["status"] == "failed"

    recovered = store.recover_processing()
    assert recovered == 1
    assert store.get_item(item)["status"] == "actionable"


def test_read_later_is_a_learnable_action(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    first = make_pending(store, msg("career1", "jobs@example.com", list_id="jobs.example.com"))
    result = store.answer_pending(first, "read_later", "list")
    assert result["released"] == 1
    item = store.get_item(first)
    assert item["status"] == "actionable"
    assert item["desired_action"] == "read_later"
    rule = store.matching_rule_for_message(msg("career2", "jobs@example.com", list_id="jobs.example.com"))
    assert rule["action"] == "read_later"


def test_apply_all_rules_routes_existing_backlog_before_classifier(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    matched = store.add_discovered(msg("m1", "jobs@example.com", list_id="jobs.example.com"), plan())
    unknown = store.add_discovered(msg("m2", "other@example.com"), plan())
    store.upsert_rule("list", "jobs.example.com", "read_later")

    released = store.apply_all_rules_to_open_items()

    assert released == 1
    assert store.get_item(matched)["status"] == "actionable"
    assert store.get_item(matched)["desired_action"] == "read_later"
    assert store.get_item(unknown)["status"] == "discovered"


def test_apply_all_rules_respects_list_over_sender_over_domain(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    item = store.add_discovered(msg("m1", "offers@example.com", list_id="weekly.example.com"), plan())
    store.upsert_rule("domain", "example.com", "archive")
    store.upsert_rule("sender", "offers@example.com", "trash")
    store.upsert_rule("list", "weekly.example.com", "read_later")

    assert store.apply_all_rules_to_open_items() == 1
    row = store.get_item(item)
    assert row["desired_action"] == "read_later"
    assert row["decision_source"] == "learned-rule"


def test_unsubscribe_registry_persists_outcome(tmp_path):
    store = StateStore(tmp_path / "state.db")
    assert store.get_unsubscribe_registry("list:news.example") is None
    store.record_unsubscribe_registry(
        "list:news.example", "success", "one_click", "accepted", source_message_id="m1"
    )
    row = store.get_unsubscribe_registry("LIST:NEWS.EXAMPLE")
    assert row["status"] == "success"
    assert row["method"] == "one_click"
    assert row["source_message_id"] == "m1"
