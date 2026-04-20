"""Tests for the LabelCache."""

from __future__ import annotations

from pathlib import Path

from ddqn_router.labeler.cache import LabelCache


def test_cache_hit_same_version(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.jsonl")
    cache.store("hello", "gpt-test", "v1", [0, 1])
    assert cache.lookup("hello", "gpt-test", "v1") == [0, 1]


def test_cache_miss_different_version(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.jsonl")
    cache.store("hello", "gpt-test", "v1", [0, 1])
    assert cache.lookup("hello", "gpt-test", "v2") is None


def test_cache_miss_different_model(tmp_path: Path) -> None:
    cache = LabelCache(tmp_path / "cache.jsonl")
    cache.store("hello", "gpt-test", "v1", [0, 1])
    assert cache.lookup("hello", "gpt-other", "v1") is None


def test_cache_reload_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    first = LabelCache(path)
    first.store("hello", "gpt-test", "v1", [2])
    second = LabelCache(path)
    assert second.lookup("hello", "gpt-test", "v1") == [2]


def test_cache_handles_corruption(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text('not-json\n{"cache_key": "abc", broken\n')
    cache = LabelCache(path)
    assert cache.lookup("hello", "gpt-test", "v1") is None
    cache.store("hello", "gpt-test", "v1", [1])
    assert cache.lookup("hello", "gpt-test", "v1") == [1]
