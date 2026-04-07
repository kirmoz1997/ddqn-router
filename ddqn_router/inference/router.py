"""DDQNRouter — main inference API with graceful degradation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import AgentDef, RouterConfig
from ddqn_router.rl.q_network import QNetwork
from ddqn_router.rl.state_encoder import StateEncoder


class RouteResult(NamedTuple):
    agents: list[int]
    agent_names: list[str]
    confidence: float
    steps: int


class RouterNotInitializedError(Exception):
    """Raised when no trained model or fallback is available."""

    def __init__(self) -> None:
        super().__init__(
            "No router artifacts found. Run 'ddqn-router train' to train a DDQN model, "
            "or 'ddqn-router baseline' to fit a supervised fallback. "
            "See README.md for the full quickstart guide."
        )


class DDQNRouter:
    """Inference router with DDQN primary and supervised fallback."""

    def __init__(
        self,
        q_net: QNetwork | None,
        encoder: StateEncoder,
        registry: AgentRegistry,
        config: RouterConfig,
        supervised: object | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._q_net = q_net
        self._encoder = encoder
        self._registry = registry
        self._config = config
        self._supervised = supervised
        self._device = device or torch.device("cpu")
        self._action_masking = config.training.action_masking

    @classmethod
    def load(cls, artifacts_path: str | Path) -> DDQNRouter:
        path = Path(artifacts_path)
        config_path = path / "config_used.json"
        if not config_path.exists():
            raise RouterNotInitializedError()

        with open(config_path) as f:
            config = RouterConfig.model_validate(json.load(f))

        registry = AgentRegistry(config.agents)
        num_agents = registry.num_agents
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_path = path / "encoder.joblib"
        model_path = path / "model.pt"
        supervised_path = path / "supervised.joblib"

        # Try DDQN first
        if model_path.exists() and encoder_path.exists():
            encoder = StateEncoder.load(encoder_path)
            q_net = QNetwork(
                tfidf_dim=encoder.dim,
                num_agents=num_agents,
                hidden_layers=config.training.hidden_layers,
            ).to(device)
            q_net.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            q_net.eval()
            return cls(q_net, encoder, registry, config, device=device)

        # Fall back to supervised
        if supervised_path.exists():
            from ddqn_router.baselines.supervised_router import SupervisedRouter

            supervised = SupervisedRouter.load(supervised_path, registry)
            encoder = StateEncoder(config.training.tfidf_max_features)
            return cls(None, encoder, registry, config, supervised=supervised, device=device)

        raise RouterNotInitializedError()

    def route(self, query: str) -> RouteResult:
        if self._supervised is not None and self._q_net is None:
            return self._route_supervised(query)
        return self._route_ddqn(query)

    def _route_ddqn(self, query: str) -> RouteResult:
        assert self._q_net is not None
        tfidf_vec = self._encoder.transform(query)
        num_agents = self._registry.num_agents
        selected: list[int] = []
        mask = np.zeros(num_agents, dtype=np.float32)
        steps = 0
        q_values_list: list[np.ndarray] = []

        while True:
            state = np.concatenate([tfidf_vec, mask])
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self._device
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = self._q_net(state_t).squeeze(0).cpu().numpy()

            q_values_list.append(q_values.copy())

            action_mask = np.ones(num_agents + 1, dtype=bool)
            if self._action_masking:
                for a in selected:
                    action_mask[a] = False

            masked_q = q_values.copy()
            masked_q[~action_mask] = float("-inf")
            action = int(np.argmax(masked_q))
            steps += 1

            if action == num_agents:  # STOP
                break
            selected.append(action)
            mask[action] = 1.0

            if len(selected) >= num_agents or steps >= self._config.training.max_steps_per_episode:
                break

        confidence = self._compute_confidence(q_values_list)
        agent_names = [self._registry.get_by_id(a).name for a in selected]
        return RouteResult(
            agents=selected,
            agent_names=agent_names,
            confidence=confidence,
            steps=steps,
        )

    def _route_supervised(self, query: str) -> RouteResult:
        from ddqn_router.dataset.dataset import Task

        task: Task = {"id": "inference", "text": query, "required_agents": []}
        preds = self._supervised.predict([task])  # type: ignore[union-attr]
        selected = sorted(preds[0])
        agent_names = [self._registry.get_by_id(a).name for a in selected]
        return RouteResult(
            agents=selected,
            agent_names=agent_names,
            confidence=0.5,
            steps=1,
        )

    def _compute_confidence(self, q_values_list: list[np.ndarray]) -> float:
        if not q_values_list:
            return 0.0
        last_q = q_values_list[-1]
        valid = last_q[last_q > float("-inf")]
        if len(valid) < 2:
            return 1.0
        max_q = float(np.max(valid))
        min_q = float(np.min(valid))
        mean_q = float(np.mean(valid))
        return float(np.clip((max_q - mean_q) / (max_q - min_q + 1e-8), 0.0, 1.0))

    def route_batch(self, queries: list[str]) -> list[RouteResult]:
        return [self.route(q) for q in queries]

    def explain(self, query: str) -> None:
        """Print step-by-step Q-values and selected agents."""
        if self._q_net is None:
            print("  explain() requires a DDQN model (not available in supervised fallback)")
            return

        tfidf_vec = self._encoder.transform(query)
        num_agents = self._registry.num_agents
        selected: list[int] = []
        mask = np.zeros(num_agents, dtype=np.float32)

        agent_names = self._registry.names()
        header = ["Step", "Selected"] + agent_names + ["STOP"]
        col_widths = [max(6, len(h)) for h in header]

        print(f"\n  Query: \"{query}\"")
        print(f"  {'─' * (sum(col_widths) + len(col_widths) * 3)}")
        fmt_header = "  ".join(h.center(w) for h, w in zip(header, col_widths))
        print(f"  {fmt_header}")
        print(f"  {'─' * (sum(col_widths) + len(col_widths) * 3)}")

        step = 0
        while True:
            state = np.concatenate([tfidf_vec, mask])
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self._device
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = self._q_net(state_t).squeeze(0).cpu().numpy()

            action_mask = np.ones(num_agents + 1, dtype=bool)
            if self._action_masking:
                for a in selected:
                    action_mask[a] = False

            masked_q = q_values.copy()
            masked_q[~action_mask] = float("-inf")
            action = int(np.argmax(masked_q))
            step += 1

            if action == num_agents:
                action_name = "STOP"
            else:
                action_name = agent_names[action]

            row_vals = [str(step), action_name]
            for i in range(num_agents + 1):
                if action_mask[i]:
                    row_vals.append(f"{q_values[i]:.3f}")
                else:
                    row_vals.append("  --  ")
            row = "  ".join(v.center(w) for v, w in zip(row_vals, col_widths))
            print(f"  {row}")

            if action == num_agents:
                break
            selected.append(action)
            mask[action] = 1.0
            if len(selected) >= num_agents or step >= self._config.training.max_steps_per_episode:
                break

        print(f"\n  Final selection: {[agent_names[a] for a in selected]}")
        print()
