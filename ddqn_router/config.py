"""Single source of truth for all configuration defaults and schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class AgentDef(BaseModel):
    id: int
    name: str
    description: str


class TrainingConfig(BaseModel):
    total_steps: int = 200_000
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    target_update_freq: int = 500
    replay_buffer_size: int = 50_000
    min_replay_size: int = 1000
    reward_mode: Literal["jaccard", "stochastic"] = "jaccard"
    step_cost: float = 0.05
    hidden_layers: list[int] = Field(default_factory=lambda: [256, 128])
    tfidf_max_features: int = 5000
    action_masking: bool = True
    seed: int = 42
    val_eval_freq: int = 5000
    save_best: bool = True
    max_steps_per_episode: int = 20
    checkpoint_freq: int = 10_000

    @model_validator(mode="after")
    def _check_non_negative(self) -> TrainingConfig:
        if self.step_cost < 0:
            raise ValueError(f"training.step_cost must be >= 0, got {self.step_cost}")
        if self.total_steps <= 0:
            raise ValueError(f"training.total_steps must be > 0, got {self.total_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"training.batch_size must be > 0, got {self.batch_size}")
        return self


class LabelerConfig(BaseModel):
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    input: str = ""
    output: str = "./data/tasks.jsonl"
    min_agents: int = 2
    max_agents: int | None = None
    prompt_template: str | None = None
    prompt_version: str = "v1"
    batch_size: int = 1
    cache: str = "./cache/label_cache.jsonl"
    fallback_strategy: Literal["skip", "keyword", "all-agents"] = "keyword"


class DatasetConfig(BaseModel):
    input: str = "./data/tasks.jsonl"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    output_dir: str = "./data/"

    @model_validator(mode="after")
    def _check_ratios(self) -> DatasetConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"dataset.train_ratio + val_ratio + test_ratio must sum to 1.0, got {total:.4f}"
            )
        for name, val in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if val < 0:
                raise ValueError(f"dataset.{name} must be >= 0, got {val}")
        return self


class RouterConfig(BaseModel):
    agents: list[AgentDef] = Field(default_factory=list)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    labeler: LabelerConfig = Field(default_factory=LabelerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    output_dir: str = "./artifacts/"

    @model_validator(mode="after")
    def _check_agents(self) -> RouterConfig:
        if self.agents:
            seen: set[int] = set()
            for agent in self.agents:
                if agent.id in seen:
                    raise ValueError(f"Duplicate agent id: {agent.id}")
                seen.add(agent.id)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> RouterConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})

    def to_dict(self) -> dict:
        return self.model_dump()
