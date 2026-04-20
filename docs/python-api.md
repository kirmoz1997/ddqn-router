# Python API

## `DDQNRouter`

```python
from ddqn_router import DDQNRouter, RouteResult, StepTrace

router = DDQNRouter.load("./artifacts/")
```

### `DDQNRouter.load(artifacts_path) -> DDQNRouter`

Loads a trained router. `artifacts_path` must contain `model.pt`,
`encoder.joblib`, and `config_used.json`. Raises `RouterNotTrainedError`
if any file is missing.

### `router.route(query: str) -> RouteResult`

Runs a greedy routing episode; returns a `RouteResult`.

### `router.route_batch(queries: list[str]) -> list[RouteResult]`

Sugar for `[router.route(q) for q in queries]`. Same semantics.

### `router.route_verbose(query: str) -> RouteResult`

Same as `route`, but also populates `RouteResult.steps_trace` with a list of
`StepTrace` entries — one per routing step.

### `router.explain(query: str) -> None`

Pretty-prints a step-by-step Q-value table. Output format is stable across
versions.

### `router.agents -> list[dict]`

Configured agents: `[{"id": ..., "name": ..., "description": ...}, …]`.

## `RouteResult`

```python
@dataclass
class RouteResult:
    agents: list[int]           # selected agent IDs, in selection order
    agent_names: list[str]      # parallel list of agent names
    confidence: float           # scalar in [0, 1] (see below)
    steps: int                  # number of routing steps taken
    steps_trace: list[StepTrace] | None = None  # populated by route_verbose
```

## `StepTrace`

```python
@dataclass(frozen=True)
class StepTrace:
    step_index: int                   # 0-based index of this step
    q_values: dict[int, float]        # Q-value per valid (non-masked) action
    selected_action: int              # chosen action id (num_agents == STOP)
    stop_selected: bool               # True if the STOP action was taken
    masked_agents: list[int]          # agent IDs masked out at this step
```

## How `confidence` is computed

`confidence` is a scalar in `[0, 1]` derived from the **last** routing step's
Q-value distribution:

```python
# ddqn_router/inference/router.py (line 253)
confidence = clip(
    (max_q - mean_q) / (max_q - min_q + 1e-8),
    0.0, 1.0,
)
```

where `max_q`, `min_q`, and `mean_q` are computed over the Q-values of the
valid (non-masked) actions at the final step of the routing episode. Returns
`0.0` if no steps were taken and `1.0` if fewer than two actions remain valid
(i.e. nothing to be uncertain between).

Interpretation:

- Close to **1.0** means the argmax Q-value stood out sharply from the rest
  of the valid options at the stopping step — the policy was decisive.
- Close to **0.0** means the valid Q-values were clustered tightly around
  the mean — several candidates were roughly equally good (or bad).

This is **not** a probability and it is not calibrated. It reflects Q-value
dispersion only; two models with identical accuracy may report different
confidence scales because Q-values depend on the reward structure. If you
need calibrated probabilities, post-hoc methods (Platt / isotonic) are out of
scope for this library — see the "Out of scope" section in the
[productionization plan](https://github.com/kirmoz1997/ddqn-router).

## Exceptions

- `RouterNotTrainedError` — raised by `DDQNRouter.load()` when any artifact
  is missing. The message includes actionable next steps.
