"""Single source of truth for all configuration defaults and schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


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


class RouterConfig(BaseModel):
    agents: list[AgentDef] = Field(default_factory=list)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    labeler: LabelerConfig = Field(default_factory=LabelerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    output_dir: str = "./artifacts/"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RouterConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})

    def to_dict(self) -> dict:
        return self.model_dump()
