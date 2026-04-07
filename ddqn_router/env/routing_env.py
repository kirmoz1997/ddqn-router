"""Custom MDP environment for multi-agent set routing (no gymnasium dependency)."""

from __future__ import annotations

import numpy as np

from ddqn_router.rl.reward import compute_reward


class RoutingEnv:
    """
    State: concatenation of TF-IDF vector + binary mask of selected agents.
    Actions: 0..N-1 = select agent, N = STOP.
    Episode ends when: STOP selected, all agents selected, or max_steps reached.
    """

    def __init__(
        self,
        num_agents: int,
        reward_mode: str = "jaccard",
        step_cost: float = 0.05,
        max_steps: int = 20,
        action_masking: bool = True,
    ) -> None:
        self.num_agents = num_agents
        self.stop_action = num_agents
        self.num_actions = num_agents + 1
        self.reward_mode = reward_mode
        self.step_cost = step_cost
        self.max_steps = max_steps
        self.action_masking = action_masking

        self._tfidf_vec: np.ndarray | None = None
        self._target_agents: set[int] = set()
        self._selected: set[int] = set()
        self._steps: int = 0

    def reset(
        self, tfidf_vec: np.ndarray, target_agents: list[int]
    ) -> np.ndarray:
        self._tfidf_vec = tfidf_vec
        self._target_agents = set(target_agents)
        self._selected = set()
        self._steps = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        mask = np.zeros(self.num_agents, dtype=np.float32)
        for a in self._selected:
            mask[a] = 1.0
        return np.concatenate([self._tfidf_vec, mask])

    def get_action_mask(self) -> np.ndarray:
        """Returns a boolean mask: True = action is valid."""
        mask = np.ones(self.num_actions, dtype=bool)
        if self.action_masking:
            for a in self._selected:
                mask[a] = False
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Returns (next_state, reward, done)."""
        self._steps += 1

        if action == self.stop_action:
            reward = compute_reward(
                self._selected,
                self._target_agents,
                self.step_cost,
                self.reward_mode,
                is_terminal=True,
            )
            return self._get_state(), reward, True

        self._selected.add(action)

        if len(self._selected) >= self.num_agents or self._steps >= self.max_steps:
            reward = compute_reward(
                self._selected,
                self._target_agents,
                self.step_cost,
                self.reward_mode,
                is_terminal=True,
            )
            return self._get_state(), reward, True

        reward = compute_reward(
            self._selected,
            self._target_agents,
            self.step_cost,
            self.reward_mode,
            is_terminal=False,
        )
        return self._get_state(), reward, False

    @property
    def selected_agents(self) -> set[int]:
        return set(self._selected)
