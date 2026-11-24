"""Unit tests for deterministic experiment cache and security checks (M2.6)."""

import pytest

from delivery.experiments.cache import ResultCache, compute_cache_key


def test_cache_key_determinism():
    cfg1 = {"alpha": 1, "beta": "test"}
    cfg2 = {"beta": "test", "alpha": 1}  # different dict order

    k1 = compute_cache_key(cfg1, seed=42, run_mode="quick")
    k2 = compute_cache_key(cfg2, seed=42, run_mode="quick")
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hex


def test_cache_set_get_and_fresh(tmp_path):
    cache = ResultCache(cache_dir=tmp_path)
    key = "test_key_123"
    data = {"metric": 42.5, "status": "success"}

    assert cache.has(key) is False
    assert cache.get(key) is None

    cache.set(key, data)
    assert cache.has(key) is True
    retrieved = cache.get(key)
    assert retrieved == data

    # Fresh mode bypasses read
    fresh_cache = ResultCache(cache_dir=tmp_path, fresh=True)
    assert fresh_cache.has(key) is False
    assert fresh_cache.get(key) is None


def test_cache_security_rejection_of_secrets(tmp_path):
    cache = ResultCache(cache_dir=tmp_path)
    bad_payload = {"user": "alice", "api_key": "secret_abc_123"}

    with pytest.raises(ValueError, match="Security invariant violated"):
        cache.set("unsafe_key", bad_payload)
