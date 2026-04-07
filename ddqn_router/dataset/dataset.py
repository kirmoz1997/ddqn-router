"""Load, validate, and inspect tasks.jsonl datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TypedDict


class Task(TypedDict):
    id: str
    text: str
    required_agents: list[int]


def load_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
            if "text" not in obj:
                raise ValueError(f"Missing 'text' field on line {i}")
            if "required_agents" not in obj:
                raise ValueError(f"Missing 'required_agents' field on line {i}")
            tasks.append(
                Task(
                    id=obj.get("id", f"ex_{i:04d}"),
                    text=obj["text"],
                    required_agents=obj["required_agents"],
                )
            )
    return tasks


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


def compute_stats(tasks: list[Task], num_agents: int | None = None) -> dict:
    agent_counter: Counter[int] = Counter()
    set_sizes: list[int] = []
    for t in tasks:
        for a in t["required_agents"]:
            agent_counter[a] += 1
        set_sizes.append(len(t["required_agents"]))

    size_dist: Counter[int] = Counter(set_sizes)

    return {
        "total_examples": len(tasks),
        "agent_frequency": dict(sorted(agent_counter.items())),
        "set_size_distribution": dict(sorted(size_dist.items())),
        "mean_set_size": sum(set_sizes) / len(set_sizes) if set_sizes else 0.0,
        "min_set_size": min(set_sizes) if set_sizes else 0,
        "max_set_size": max(set_sizes) if set_sizes else 0,
    }


def print_stats(tasks: list[Task], num_agents: int | None = None) -> None:
    stats = compute_stats(tasks, num_agents)
    print(f"\n  Dataset Statistics")
    print(f"  {'─' * 40}")
    print(f"  Total examples: {stats['total_examples']}")
    print(
        f"  Set size: mean={stats['mean_set_size']:.2f}, "
        f"min={stats['min_set_size']}, max={stats['max_set_size']}"
    )
    print(f"\n  Agent frequency:")
    for aid, cnt in stats["agent_frequency"].items():
        pct = 100 * cnt / stats["total_examples"] if stats["total_examples"] else 0
        print(f"    Agent {aid}: {cnt} ({pct:.1f}%)")
    print(f"\n  Set size distribution:")
    for size, cnt in stats["set_size_distribution"].items():
        pct = 100 * cnt / stats["total_examples"] if stats["total_examples"] else 0
        print(f"    Size {size}: {cnt} ({pct:.1f}%)")
    print()
