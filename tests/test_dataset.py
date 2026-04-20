"""Tests for dataset loading and splitting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ddqn_router.dataset.dataset import load_tasks
from ddqn_router.dataset.splitter import stratified_split


def test_load_tasks(tiny_dataset: Path, tiny_tasks: list[dict]) -> None:
    loaded = load_tasks(tiny_dataset)
    assert len(loaded) == len(tiny_tasks)
    assert loaded[0]["text"] == tiny_tasks[0]["text"]
    assert loaded[0]["required_agents"] == tiny_tasks[0]["required_agents"]


def test_load_tasks_rejects_missing_fields(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"text": "no required_agents"}) + "\n")
    with pytest.raises(ValueError, match="required_agents"):
        load_tasks(p)


def test_split_preserves_size_distribution(tiny_tasks: list[dict]) -> None:
    # Boost dataset size per bucket so stratification has material to work with
    big = tiny_tasks * 5
    train, val, test = stratified_split(big, 0.7, 0.15, 0.15, seed=42)
    assert len(train) + len(val) + len(test) == len(big)

    size_dist_before = Counter(len(t["required_agents"]) for t in big)
    size_dist_train = Counter(len(t["required_agents"]) for t in train)
    for size, cnt in size_dist_before.items():
        assert size_dist_train.get(size, 0) >= 1, (
            f"stratification missed size bucket {size}: train={size_dist_train}"
        )


def test_split_rejects_bad_ratios(tiny_tasks: list[dict]) -> None:
    with pytest.raises(ValueError):
        stratified_split(tiny_tasks, 0.8, 0.15, 0.15)
