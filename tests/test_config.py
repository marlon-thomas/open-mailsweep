import pytest

from openmailsweep.config import Settings


def test_local_classifier_is_default_without_api_key(monkeypatch):
    monkeypatch.delenv("CLASSIFIER", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    s = Settings.from_env()
    assert s.classifier == "local"
    assert s.jev_api_key == ""


def test_jev_requires_key(monkeypatch):
    monkeypatch.setenv("CLASSIFIER", "jev")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        Settings.from_env()


def test_unsubscribe_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_UNSUBSCRIBE", raising=False)
    assert Settings.from_env().allow_unsubscribe is False


def test_unsubscribe_can_be_enabled(monkeypatch):
    monkeypatch.setenv("ALLOW_UNSUBSCRIBE", "true")
    assert Settings.from_env().allow_unsubscribe is True


def test_local_device_and_batch_defaults(monkeypatch):
    monkeypatch.delenv("LOCAL_CLASSIFIER_DEVICE", raising=False)
    monkeypatch.delenv("OPENMAILSWEEP_CLASSIFY_BATCH_SIZE", raising=False)
    s = Settings.from_env()
    assert s.local_device == "auto"
    assert s.classify_batch_size == 16


def test_classify_batch_size_from_env(monkeypatch):
    monkeypatch.setenv("OPENMAILSWEEP_CLASSIFY_BATCH_SIZE", "32")
    assert Settings.from_env().classify_batch_size == 32


def test_continuous_intake_defaults_have_no_message_count_limit(monkeypatch):
    monkeypatch.delenv("OPENMAILSWEEP_SCAN_QUERY", raising=False)
    monkeypatch.delenv("OPENMAILSWEEP_FETCH_BATCH_SIZE", raising=False)
    s = Settings.from_env()
    assert s.scan_query == "in:inbox"
    assert s.fetch_batch_size == 100
    assert not hasattr(s, "scan_limit")


def test_fetch_batch_size_is_capped_to_gmail_max(monkeypatch):
    monkeypatch.setenv("OPENMAILSWEEP_FETCH_BATCH_SIZE", "9999")
    assert Settings.from_env().fetch_batch_size == 500


def test_gmail_quota_defaults_leave_headroom(monkeypatch):
    monkeypatch.delenv("GMAIL_QUOTA_UNITS_PER_MINUTE", raising=False)
    monkeypatch.delenv("GMAIL_QUOTA_BURST_UNITS", raising=False)
    s = Settings.from_env()
    assert s.gmail_quota_units_per_minute == 3600
    assert s.gmail_quota_burst_units == 240


def test_gmail_quota_can_be_tuned(monkeypatch):
    monkeypatch.setenv("GMAIL_QUOTA_UNITS_PER_MINUTE", "3000")
    monkeypatch.setenv("GMAIL_QUOTA_BURST_UNITS", "120")
    s = Settings.from_env()
    assert s.gmail_quota_units_per_minute == 3000
    assert s.gmail_quota_burst_units == 120


def test_read_later_label_default():
    from openmailsweep.config import Policy
    assert Policy().labels.read_later == "Read Later"


def test_action_workers_default_and_override(monkeypatch):
    monkeypatch.delenv("OPENMAILSWEEP_ACTION_WORKERS", raising=False)
    assert Settings.from_env().action_workers == 3
    monkeypatch.setenv("OPENMAILSWEEP_ACTION_WORKERS", "5")
    assert Settings.from_env().action_workers == 5
