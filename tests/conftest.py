"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import AgentDef, RouterConfig


@pytest.fixture
def sample_agents() -> list[AgentDef]:
    return [
        AgentDef(id=0, name="Billing", description="billing invoices payments subscriptions"),
        AgentDef(id=1, name="Technical", description="bugs errors integration api issues"),
        AgentDef(id=2, name="Account", description="account settings passwords permissions"),
    ]


@pytest.fixture
def sample_registry(sample_agents: list[AgentDef]) -> AgentRegistry:
    return AgentRegistry(sample_agents)


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "agents": [
            {"id": 0, "name": "Billing", "description": "billing invoices payments"},
            {"id": 1, "name": "Technical", "description": "bugs errors integration"},
            {"id": 2, "name": "Account", "description": "account settings permissions"},
        ],
        "training": {
            "total_steps": 200,
            "batch_size": 8,
            "min_replay_size": 16,
            "epsilon_decay_steps": 100,
            "tfidf_max_features": 64,
            "hidden_layers": [32],
            "val_eval_freq": 50,
            "max_steps_per_episode": 5,
            "target_update_freq": 20,
            "save_best": False,
            "seed": 1,
        },
        "labeler": {"model": "test", "base_url": "https://api.test/v1"},
        "dataset": {
            "input": "./data/tasks.jsonl",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "output_dir": "./artifacts/",
    }


@pytest.fixture
def sample_config(sample_config_dict: dict) -> RouterConfig:
    return RouterConfig.model_validate(sample_config_dict)


@pytest.fixture
def tiny_tasks() -> list[dict]:
    """12 tiny examples across 3 agents with varied set sizes."""
    return [
        {"id": "ex_01", "text": "my invoice was charged twice", "required_agents": [0]},
        {"id": "ex_02", "text": "please refund my payment", "required_agents": [0]},
        {"id": "ex_03", "text": "cancel my subscription", "required_agents": [0]},
        {"id": "ex_04", "text": "API returns 500 error", "required_agents": [1]},
        {"id": "ex_05", "text": "integration bug in webhook", "required_agents": [1]},
        {"id": "ex_06", "text": "reset my password", "required_agents": [2]},
        {"id": "ex_07", "text": "change account email", "required_agents": [2]},
        {"id": "ex_08", "text": "billing error and cannot login", "required_agents": [0, 2]},
        {"id": "ex_09", "text": "refund plus password issue", "required_agents": [0, 2]},
        {"id": "ex_10", "text": "api broken and invoice wrong", "required_agents": [0, 1]},
        {"id": "ex_11", "text": "bug on login page", "required_agents": [1, 2]},
        {"id": "ex_12", "text": "full audit billing api account", "required_agents": [0, 1, 2]},
    ]


@pytest.fixture
def tiny_dataset(tmp_path: Path, tiny_tasks: list[dict]) -> Path:
    """Write tiny tasks to a jsonl file and return the path."""
    p = tmp_path / "tasks.jsonl"
    with open(p, "w") as f:
        for t in tiny_tasks:
            f.write(json.dumps(t) + "\n")
    return p


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run a short training session once per test session and cache artifacts."""
    import shutil

    cache_dir = tmp_path_factory.mktemp("trained_artifacts", numbered=False)
    artifacts = cache_dir / "artifacts"
    data_dir = cache_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    tasks = (
        [
            {"id": f"ex_{i:02d}", "text": f"billing invoice payment {i}", "required_agents": [0]}
            for i in range(8)
        ]
        + [
            {"id": f"ex_{i:02d}", "text": f"api bug error {i}", "required_agents": [1]}
            for i in range(8, 16)
        ]
        + [
            {"id": f"ex_{i:02d}", "text": f"account password {i}", "required_agents": [2]}
            for i in range(16, 24)
        ]
        + [
            {
                "id": f"ex_{i:02d}",
                "text": f"billing and account issue {i}",
                "required_agents": [0, 2],
            }
            for i in range(24, 30)
        ]
    )

    for split_name in ("train", "val", "test"):
        path = data_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

    cfg_dict = {
        "agents": [
            {"id": 0, "name": "Billing", "description": "billing invoice payment"},
            {"id": 1, "name": "Technical", "description": "api bug error"},
            {"id": 2, "name": "Account", "description": "account password"},
        ],
        "training": {
            "total_steps": 300,
            "batch_size": 8,
            "min_replay_size": 16,
            "epsilon_decay_steps": 150,
            "tfidf_max_features": 64,
            "hidden_layers": [32],
            "val_eval_freq": 100,
            "max_steps_per_episode": 4,
            "target_update_freq": 25,
            "save_best": True,
            "checkpoint_freq": 200,
            "seed": 0,
        },
        "dataset": {
            "input": str(data_dir / "tasks.jsonl"),
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "output_dir": str(data_dir),
        },
        "output_dir": str(artifacts),
    }

    from ddqn_router.config import RouterConfig
    from ddqn_router.rl.ddqn_agent import train as run_training

    cfg = RouterConfig.model_validate(cfg_dict)
    run_training(cfg)

    yield artifacts

    shutil.rmtree(cache_dir, ignore_errors=True)
