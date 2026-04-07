# ddqn-router

A lightweight Python library that trains a Double DQN agent to route user queries to the optimal *subset* of specialized agents in a multi-agent system.

You define your agents, label a dataset with an LLM, train the router, and get a fast inference model that selects the right combination of agents for any input query — no LLM needed at inference time.

## Why ddqn-router?

| Problem | Solution |
|---|---|
| LLM-based routing is slow and expensive at scale | DDQN inference is a single forward pass (~1ms) |
| Hard-coded rules break when agents change | Train a new router in minutes on your own data |
| Most routers pick *one* agent | ddqn-router selects *subsets* (multi-agent routing) |
| Need a PhD in RL to set it up | Config-driven, CLI-first, sensible defaults from published research |

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Define your agents

Create a `config.yaml`:

```yaml
agents:
  - id: 0
    name: "Billing Agent"
    description: "Handles billing, invoices, payments, subscriptions"
  - id: 1
    name: "Technical Agent"
    description: "Handles bugs, errors, API integration issues"
  - id: 2
    name: "Account Agent"
    description: "Handles account settings, passwords, permissions"

labeler:
  model: "gpt-4o-mini"

dataset:
  input: "./data/tasks.jsonl"

output_dir: "./artifacts/"
```

### 3. Label your data

Write your raw queries to a text file (one per line), then label them with an LLM:

```bash
export DDQN_ROUTER_API_KEY=sk-...

ddqn-router label \
  --config config.yaml \
  --input queries.txt \
  --output data/tasks.jsonl
```

The labeler calls any OpenAI-compatible API (OpenAI, DeepSeek, Ollama, etc.) and produces a JSONL dataset where each line maps a query to the agents required to handle it:

```json
{"id": "ex_a1b2c3d4", "text": "...", "required_agents": [0, 2]}
```

### 4. Split and train

```bash
ddqn-router dataset split --input data/tasks.jsonl
ddqn-router train --config config.yaml
```

Training outputs are saved to `./artifacts/`:

| File | Content |
|---|---|
| `model.pt` | Trained Q-network weights |
| `encoder.joblib` | Fitted TF-IDF encoder |
| `config_used.json` | Exact config snapshot for reproducibility |
| `metrics_val_best.json` | Best validation metrics |
| `metrics_test.json` | Final test set metrics |
| `training_log.jsonl` | Step-by-step training log |

### 5. Use the router

```python
from ddqn_router import DDQNRouter

router = DDQNRouter.load("./artifacts/")

result = router.route("my invoice was charged twice")
print(result.agents)       # [0, 2]
print(result.agent_names)  # ["Billing Agent", "Account Agent"]
print(result.confidence)   # 0.87
print(result.steps)        # 3

# Batch routing
results = router.route_batch(["query one", "query two"])

# See what the model is thinking
router.explain("debug the webhook integration")
```

---

## CLI Reference

### `ddqn-router label`

Label raw queries with required agents using an LLM.

```
ddqn-router label --config CONFIG [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `--config` | Path to YAML config (required) | — |
| `--input` | Path to raw texts file | from config |
| `--output` | Output tasks.jsonl path | `./data/tasks.jsonl` |
| `--model` | LLM model string | `gpt-4o-mini` |
| `--base-url` | API base URL | `https://api.openai.com/v1` |
| `--api-key` | API key (or `DDQN_ROUTER_API_KEY` env var) | — |
| `--min-agents` | Min agents per example | `2` |
| `--max-agents` | Max agents per example | all |
| `--prompt-template` | Custom Jinja2 prompt file | built-in |
| `--batch-size` | Examples per API call | `1` |
| `--cache` | Cache file path | `./cache/label_cache.jsonl` |
| `--fallback-strategy` | On LLM parse failure: `skip` / `keyword` / `all-agents` | `keyword` |

The labeler uses raw `httpx` — no OpenAI SDK required. Any provider that exposes `POST /chat/completions` works: OpenAI, Azure, DeepSeek, Anthropic via proxy, local Ollama, vLLM, etc.

### `ddqn-router dataset stats`

```
ddqn-router dataset stats --input data/tasks.jsonl
```

Prints total examples, per-agent frequency, and set size distribution.

### `ddqn-router dataset split`

```
ddqn-router dataset split --input data/tasks.jsonl \
  [--train 0.7] [--val 0.15] [--test 0.15] [--output-dir data/]
```

Stratified split by set size into `train.jsonl`, `val.jsonl`, `test.jsonl`.

### `ddqn-router train`

```
ddqn-router train --config config.yaml [--output-dir ./artifacts/]
```

Trains the DDQN routing agent. Prints live progress with step count, epsilon, loss, and validation Jaccard.

---

## Python API

```python
from ddqn_router import DDQNRouter, RouteResult, RouterNotTrainedError
```

### `DDQNRouter.load(artifacts_path) -> DDQNRouter`

Load a trained router. Raises `RouterNotTrainedError` with setup instructions if artifacts are missing.

### `router.route(query) -> RouteResult`

Route a single query. Returns:

```python
RouteResult(
    agents=[0, 2],              # selected agent IDs
    agent_names=["Billing Agent", "Account Agent"],
    confidence=0.87,            # 0.0 to 1.0
    steps=3,                    # routing steps taken
)
```

### `router.route_batch(queries) -> list[RouteResult]`

Route multiple queries at once.

### `router.explain(query) -> None`

Print a step-by-step table showing Q-values for every agent at each routing step — useful for debugging and understanding model behavior.

### `router.agents -> list[dict]`

Returns the list of configured agents with `id`, `name`, and `description`.

---

## Configuration Reference

All parameters live in a single YAML file. Defaults come from the best research configuration and work well out of the box.

### Agents

| Field | Type | Description |
|---|---|---|
| `agents[].id` | int | Unique agent ID (0-indexed) |
| `agents[].name` | str | Human-readable name |
| `agents[].description` | str | What this agent handles (used by labeler and TF-IDF) |

### Labeler

| Field | Type | Default | Description |
|---|---|---|---|
| `labeler.model` | str | `gpt-4o-mini` | LLM model string |
| `labeler.base_url` | str | `https://api.openai.com/v1` | API base URL |
| `labeler.api_key` | str | `""` | API key (prefer `DDQN_ROUTER_API_KEY` env var) |
| `labeler.input` | str | `""` | Path to raw texts |
| `labeler.output` | str | `./data/tasks.jsonl` | Output labeled dataset |
| `labeler.min_agents` | int | `2` | Min agents per example |
| `labeler.max_agents` | int\|null | `null` | Max agents (null = no limit) |
| `labeler.prompt_template` | str\|null | `null` | Custom Jinja2 prompt path |
| `labeler.prompt_version` | str | `v1` | Version tag for cache invalidation |
| `labeler.batch_size` | int | `1` | Examples per API call |
| `labeler.cache` | str | `./cache/label_cache.jsonl` | Cache file path |
| `labeler.fallback_strategy` | str | `keyword` | Fallback on LLM parse failure |

### Dataset

| Field | Type | Default | Description |
|---|---|---|---|
| `dataset.input` | str | `./data/tasks.jsonl` | Path to labeled dataset |
| `dataset.train_ratio` | float | `0.7` | Train split ratio |
| `dataset.val_ratio` | float | `0.15` | Validation split ratio |
| `dataset.test_ratio` | float | `0.15` | Test split ratio |
| `dataset.output_dir` | str | `./data/` | Where to save split files |

### Training

| Field | Type | Default | Description |
|---|---|---|---|
| `training.total_steps` | int | `200000` | Total training steps |
| `training.batch_size` | int | `64` | Replay sample batch size |
| `training.learning_rate` | float | `0.001` | Adam optimizer learning rate |
| `training.gamma` | float | `0.99` | Discount factor |
| `training.epsilon_start` | float | `1.0` | Initial exploration rate |
| `training.epsilon_end` | float | `0.05` | Final exploration rate |
| `training.epsilon_decay_steps` | int | `100000` | Steps over which epsilon decays |
| `training.target_update_freq` | int | `500` | Steps between target network syncs |
| `training.replay_buffer_size` | int | `50000` | Replay buffer capacity |
| `training.min_replay_size` | int | `1000` | Min buffer fill before training starts |
| `training.reward_mode` | str | `jaccard` | `"jaccard"` or `"stochastic"` |
| `training.step_cost` | float | `0.05` | Per-agent selection penalty |
| `training.hidden_layers` | list[int] | `[256, 128]` | Q-network hidden layer sizes |
| `training.tfidf_max_features` | int | `5000` | TF-IDF vocabulary limit |
| `training.action_masking` | bool | `true` | Mask already-selected agents |
| `training.seed` | int | `42` | Random seed for reproducibility |
| `training.val_eval_freq` | int | `5000` | Steps between validation evaluations |
| `training.save_best` | bool | `true` | Save best checkpoint by val Jaccard |
| `training.max_steps_per_episode` | int | `20` | Max routing steps per episode |

### Output

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | str | `./artifacts/` | Where to save trained model artifacts |

---

## How It Works

1. **State**: Each query is encoded as a TF-IDF vector concatenated with a binary mask of already-selected agents.
2. **Actions**: The agent can select any not-yet-selected agent, or choose STOP (action ID = N) to finish.
3. **Reward**: Jaccard similarity between selected agents and the ground-truth set, minus a small step cost per selection.
4. **Action masking**: Already-selected agents get Q-value = -inf, preventing redundant picks and speeding up training.
5. **Double DQN**: Reduces Q-value overestimation — the online network selects actions, the target network evaluates them.

At inference, the trained model runs a greedy forward pass in ~1ms, selecting agents one by one until it triggers STOP.

---

## Project Structure

```
ddqn_router/
├── __init__.py              # DDQNRouter, RouteResult, RouterNotTrainedError
├── cli.py                   # Typer CLI (label, dataset, train)
├── config.py                # Pydantic config schema with all defaults
├── agents.py                # AgentRegistry
├── labeler/
│   ├── labeler.py           # LLMLabeler (httpx-based, any provider)
│   ├── prompt_template.j2   # Default recall-biased prompt
│   └── cache.py             # JSONL cache (SHA256-keyed)
├── dataset/
│   ├── dataset.py           # Load / validate / stats for tasks.jsonl
│   └── splitter.py          # Stratified train/val/test split
├── env/
│   └── routing_env.py       # Custom MDP environment (no gymnasium)
├── rl/
│   ├── q_network.py         # MLP Q-network (PyTorch)
│   ├── state_encoder.py     # TF-IDF encoder wrapper
│   ├── replay_buffer.py     # Uniform replay buffer
│   ├── ddqn_agent.py        # Double DQN training loop
│   └── reward.py            # Jaccard + stochastic reward
├── eval/
│   └── evaluator.py         # Metrics (Jaccard, P/R/F1, bucketed)
├── inference/
│   └── router.py            # DDQNRouter.load() + route() + explain()
└── serve/
    └── app.py               # Optional FastAPI server (pip install ddqn-router[serve])
```

---

## Research Background

Based on ["Multi-Agent Set Routing with Double DQN"](https://github.com/kirmoz1997/dqn_routing_research). All default hyperparameters in this library come directly from the best-performing experiment in the research (iteration 9: `reward_mode=jaccard`, `step_cost=0.05`, `action_masking=true`, `gamma=0.99`, hidden layers `[256, 128]`, 200k training steps).
