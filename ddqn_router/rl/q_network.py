"""MLP Q-network for Double DQN routing."""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Input:  TF-IDF vector (tfidf_dim) concatenated with agent selection mask (num_agents).
    Output: Q-values for each action (num_agents + 1, where last is STOP).
    """

    def __init__(
        self,
        tfidf_dim: int,
        num_agents: int,
        hidden_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128]

        self.tfidf_dim = tfidf_dim
        self.num_agents = num_agents
        num_actions = num_agents + 1  # agents + STOP

        input_dim = tfidf_dim + num_agents
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
