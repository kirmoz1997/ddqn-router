"""DDQNRouter — main inference API for routing queries to agent subsets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import RouterConfig
from ddqn_router.rl.q_network import QNetwork
from ddqn_router.rl.state_encoder import StateEncoder


@dataclass(frozen=True)
class StepTrace:
    """Per-step structural trace of a routing decision."""

    step_index: int
    q_values: dict[int, float]
    selected_action: int
    stop_selected: bool
    masked_agents: list[int]


@dataclass
class RouteResult:
    """Result of routing a single query.

    Attributes:
        agents: Selected agent IDs.
        agent_names: Names of selected agents, in selection order.
        confidence: Scalar in [0, 1] derived from last-step Q-value dispersion.
        steps: Number of routing steps taken.
        steps_trace: Optional per-step trace (populated by ``route_verbose``).
    """

    agents: list[int]
    agent_names: list[str]
    confidence: float
    steps: int
    steps_trace: list[StepTrace] | None = field(default=None)


class RouterNotTrainedError(Exception):
    """Raised when no trained DDQN artifacts are found."""

    def __init__(self, path: str = "") -> None:
        hint = f" in '{path}'" if path else ""
        super().__init__(
            f"No trained DDQN router found{hint}. "
            "To get started:\n"
            "  1. Define your agents in a YAML config\n"
            "  2. Label your dataset:  ddqn-router label --config config.yaml --input queries.txt\n"
            "  3. Split the dataset:   ddqn-router dataset split --input data/tasks.jsonl\n"
            "  4. Train the router:    ddqn-router train --config config.yaml\n"
            "  5. Load and use:        DDQNRouter.load('./artifacts/')"
        )


class DDQNRouter:
    """Load a trained DDQN model and route queries to optimal agent subsets.

    Usage::

        router = DDQNRouter.load("./artifacts/")
        result = router.route("fix the bug in my API")
        print(result.agents, result.agent_names, result.confidence)
    """

    def __init__(
        self,
        q_net: QNetwork,
        encoder: StateEncoder,
        registry: AgentRegistry,
        config: RouterConfig,
        device: torch.device | None = None,
    ) -> None:
        self._q_net = q_net
        self._encoder = encoder
        self._registry = registry
        self._config = config
        self._device = device or torch.device("cpu")
        self._action_masking = config.training.action_masking
        self._max_steps = config.training.max_steps_per_episode

    @classmethod
    def load(cls, artifacts_path: str | Path) -> DDQNRouter:
        """Load a trained router from an artifacts directory.

        The directory must contain ``model.pt``, ``encoder.joblib``, and
        ``config_used.json`` (all produced by ``ddqn-router train``).

        Raises:
            RouterNotTrainedError: If any required artifact is missing.
        """
        path = Path(artifacts_path)
        config_path = path / "config_used.json"
        model_path = path / "model.pt"
        encoder_path = path / "encoder.joblib"

        for required in (config_path, model_path, encoder_path):
            if not required.exists():
                raise RouterNotTrainedError(str(path))

        with open(config_path) as f:
            config = RouterConfig.model_validate(json.load(f))

        registry = AgentRegistry(config.agents)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = StateEncoder.load(encoder_path)
        q_net = QNetwork(
            tfidf_dim=encoder.dim,
            num_agents=registry.num_agents,
            hidden_layers=config.training.hidden_layers,
        ).to(device)
        q_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        q_net.eval()
        return cls(q_net, encoder, registry, config, device=device)

    @property
    def agents(self) -> list[dict]:
        """Return the list of configured agents."""
        return [
            {"id": a.id, "name": a.name, "description": a.description}
            for a in self._registry.all_agents()
        ]

    def _rollout(
        self, query: str, collect_trace: bool = False
    ) -> tuple[list[int], list[np.ndarray], list[StepTrace] | None]:
        """Run the greedy routing episode. Returns selected agents, q-value history, optional trace."""
        tfidf_vec = self._encoder.transform(query)
        num_agents = self._registry.num_agents
        selected: list[int] = []
        mask = np.zeros(num_agents, dtype=np.float32)
        steps = 0
        q_values_list: list[np.ndarray] = []
        trace: list[StepTrace] | None = [] if collect_trace else None

        while True:
            state = np.concatenate([tfidf_vec, mask])
            state_t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)

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

            if trace is not None:
                q_dict = {i: float(q_values[i]) for i in range(num_agents + 1) if action_mask[i]}
                masked_agent_ids = [i for i in range(num_agents) if not action_mask[i]]
                trace.append(
                    StepTrace(
                        step_index=steps - 1,
                        q_values=q_dict,
                        selected_action=action,
                        stop_selected=(action == num_agents),
                        masked_agents=masked_agent_ids,
                    )
                )

            if action == num_agents:
                break
            selected.append(action)
            mask[action] = 1.0

            if len(selected) >= num_agents or steps >= self._max_steps:
                break

        return selected, q_values_list, trace

    def route(self, query: str) -> RouteResult:
        """Route a single query to the optimal subset of agents.

        Returns a ``RouteResult`` with the selected agent IDs, names,
        a confidence score in [0, 1], and the number of routing steps.
        """
        selected, q_values_list, _ = self._rollout(query, collect_trace=False)
        confidence = self._compute_confidence(q_values_list)
        agent_names = [self._registry.get_by_id(a).name for a in selected]
        return RouteResult(
            agents=selected,
            agent_names=agent_names,
            confidence=confidence,
            steps=len(q_values_list),
        )

    def route_batch(self, queries: list[str]) -> list[RouteResult]:
        """Route multiple queries. Returns a list of ``RouteResult``."""
        return [self.route(q) for q in queries]

    def route_verbose(self, query: str) -> RouteResult:
        """Route a query and populate ``steps_trace`` with structural per-step info."""
        selected, q_values_list, trace = self._rollout(query, collect_trace=True)
        confidence = self._compute_confidence(q_values_list)
        agent_names = [self._registry.get_by_id(a).name for a in selected]
        return RouteResult(
            agents=selected,
            agent_names=agent_names,
            confidence=confidence,
            steps=len(q_values_list),
            steps_trace=trace,
        )

    def explain(self, query: str) -> None:
        """Print a step-by-step breakdown of routing decisions.

        Shows Q-values for every agent and the STOP action at each step,
        along with which agent was selected.
        """
        num_agents = self._registry.num_agents
        agent_names = self._registry.names()
        header = ["Step", "Selected"] + agent_names + ["STOP"]
        col_widths = [max(6, len(h)) for h in header]

        print(f'\n  Query: "{query}"')
        print(f"  {'─' * (sum(col_widths) + len(col_widths) * 3)}")
        fmt_header = "  ".join(h.center(w) for h, w in zip(header, col_widths, strict=False))
        print(f"  {fmt_header}")
        print(f"  {'─' * (sum(col_widths) + len(col_widths) * 3)}")

        result = self.route_verbose(query)
        assert result.steps_trace is not None

        for step_trace in result.steps_trace:
            step_num = step_trace.step_index + 1
            action = step_trace.selected_action
            action_name = "STOP" if step_trace.stop_selected else agent_names[action]

            row_vals = [str(step_num), action_name]
            for i in range(num_agents + 1):
                if i in step_trace.q_values:
                    row_vals.append(f"{step_trace.q_values[i]:.3f}")
                else:
                    row_vals.append("  --  ")
            row = "  ".join(v.center(w) for v, w in zip(row_vals, col_widths, strict=False))
            print(f"  {row}")

        print(f"\n  Result: {[agent_names[a] for a in result.agents]}")
        print()

    def _compute_confidence(self, q_values_list: list[np.ndarray]) -> float:
        """Confidence = (max_q - mean_q) / (max_q - min_q + 1e-8), clipped to [0, 1].

        Computed over the valid (non-masked) Q-values at the **last** routing step.
        Returns 0.0 if no steps were taken, 1.0 if fewer than 2 actions remain valid.
        """
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
