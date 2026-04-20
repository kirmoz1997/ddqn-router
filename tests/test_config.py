"""Tests for RouterConfig and its sub-configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ddqn_router.config import RouterConfig


def test_valid_config_loads(sample_config_dict: dict) -> None:
    cfg = RouterConfig.model_validate(sample_config_dict)
    assert len(cfg.agents) == 3
    assert cfg.training.total_steps == 200
    assert (
        cfg.dataset.train_ratio + cfg.dataset.val_ratio + cfg.dataset.test_ratio
        == pytest.approx(1.0)
    )


def test_ratios_must_sum_to_one(sample_config_dict: dict) -> None:
    sample_config_dict["dataset"]["train_ratio"] = 0.8
    sample_config_dict["dataset"]["val_ratio"] = 0.15
    sample_config_dict["dataset"]["test_ratio"] = 0.15
    with pytest.raises(ValidationError):
        RouterConfig.model_validate(sample_config_dict)


def test_duplicate_agent_ids_rejected(sample_config_dict: dict) -> None:
    sample_config_dict["agents"] = [
        {"id": 0, "name": "A", "description": "a"},
        {"id": 0, "name": "B", "description": "b"},
    ]
    with pytest.raises(ValidationError):
        RouterConfig.model_validate(sample_config_dict)


def test_negative_step_cost_rejected(sample_config_dict: dict) -> None:
    sample_config_dict["training"]["step_cost"] = -0.1
    with pytest.raises(ValidationError):
        RouterConfig.model_validate(sample_config_dict)


def test_negative_ratio_rejected(sample_config_dict: dict) -> None:
    sample_config_dict["dataset"]["train_ratio"] = -0.1
    sample_config_dict["dataset"]["val_ratio"] = 0.55
    sample_config_dict["dataset"]["test_ratio"] = 0.55
    with pytest.raises(ValidationError):
        RouterConfig.model_validate(sample_config_dict)
