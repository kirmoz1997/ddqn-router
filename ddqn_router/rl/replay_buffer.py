"""Uniform replay buffer for experience storage."""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    mask: np.ndarray  # action mask at next_state


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
