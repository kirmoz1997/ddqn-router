"""Stratified dataset splitting by set size."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ddqn_router.dataset.dataset import Task, save_tasks


def stratified_split(
    tasks: list[Task],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Task], list[Task], list[Task]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"
        )

    rng = random.Random(seed)
    buckets: dict[int, list[Task]] = defaultdict(list)
    for t in tasks:
        buckets[len(t["required_agents"])].append(t)

    train: list[Task] = []
    val: list[Task] = []
    test: list[Task] = []

    for _size, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, round(n * train_ratio))
        n_val = max(0, round(n * val_ratio))
        n_test = max(0, n - n_train - n_val)

        if n <= 2:
            n_val = 0
            n_test = n - n_train

        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    return train, val, test


def split_and_save(
    tasks: list[Task],
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[int, int, int]:
    train, val, test = stratified_split(tasks, train_ratio, val_ratio, test_ratio, seed)
    out = Path(output_dir)
    save_tasks(train, out / "train.jsonl")
    save_tasks(val, out / "val.jsonl")
    save_tasks(test, out / "test.jsonl")
    return len(train), len(val), len(test)
