# ddqn-router

Double DQN-based router for multi-agent systems. Routes user queries to an optimal *subset* of specialized agents using reinforcement learning.

Based on the research ["Multi-Agent Set Routing with Double DQN"](https://github.com/kirmoz1997/dqn_routing_research), which demonstrated that a Double DQN agent with TF-IDF state encoding, Jaccard-based reward, action masking, and step cost outperforms random, rule-based, supervised, and LLM-based routing baselines.

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Define agents

Create `router_config.yaml` (see `examples/example_config.yaml` for all options):

```yaml
agents:
  - id: 0
    name: "Billing Agent"
    description: "Handles billing questions, invoices, payment issues"
  - id: 1
    name: "Technical Agent"
    description: "Handles technical bugs, errors, integration issues"
  - id: 2
    name: "Account Agent"
    description: "Handles account settings, password resets, permissions"
```

### 3. Label your data

Prepare raw queries (one per line), then label with an LLM:

```bash
export DDQN_ROUTER_API_KEY=sk-...
ddqn-router label --config router_config.yaml --input raw_texts.txt
```

### 4. Split and train

```bash
ddqn-router dataset split --input ./data/tasks.jsonl
ddqn-router train --config router_config.yaml
```

### 5. Use the router

```python
from ddqn_router import DDQNRouter

router = DDQNRouter.load("./artifacts/")
result = router.route("fix the bug in my pandas script")
print(result.agents, result.agent_names, result.confidence)
```

---

## CLI Reference

### `ddqn-router label`

Label raw texts with required agents using an LLM.

```
ddqn-router label --config CONFIG [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config (required) | — |
| `--input` | Path to raw texts file | from config |
| `--output` | Output tasks.jsonl path | `./data/tasks.jsonl` |
| `--model` | LLM model string | `gpt-4o-mini` |
| `--base-url` | API base URL | `https://api.openai.com/v1` |
| `--api-key` | API key (or `DDQN_ROUTER_API_KEY` env) | — |
| `--min-agents` | Minimum agents per example | `2` |
| `--max-agents` | Maximum agents per example | all |
| `--prompt-template` | Path to custom Jinja2 prompt | built-in |
| `--batch-size` | Examples per API call | `1` |
| `--cache` | Cache file path | `./cache/label_cache.jsonl` |
| `--fallback-strategy` | On parse failure: `skip`/`keyword`/`all-agents` | `keyword` |

### `ddqn-router dataset stats`

Print dataset statistics.

```
ddqn-router dataset stats --input tasks.jsonl
```

### `ddqn-router dataset split`

Stratified split by set size.

```
ddqn-router dataset split --input tasks.jsonl [--train 0.7] [--val 0.15] [--test 0.15]
```

### `ddqn-router train`

Train the DDQN routing agent.

```
ddqn-router train --config router_config.yaml [--output-dir ./artifacts/]
```

### `ddqn-router baseline`

Run all baselines and print a comparison table.

```
ddqn-router baseline --config router_config.yaml [--skip-llm]
```

---

## Python API

```python
from ddqn_router import DDQNRouter, RouteResult

# Load trained router (falls back to supervised if DDQN not found)
router = DDQNRouter.load("./artifacts/")

# Single query
result: RouteResult = router.route("I need help with billing")
result.agents        # [0, 2]
result.agent_names   # ["Billing Agent", "Account Agent"]
result.confidence    # 0.84
result.steps         # 2

# Batch
results = router.route_batch(["query1", "query2"])

# Explain routing decisions (Q-values at each step)
router.explain("fix the bug in my code")
```

---

## Config Parameters

All parameters live in a single YAML file. Defaults are from the best research configuration.

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| `agents[]` | `id` | int | — | Agent ID (0-indexed) |
| `agents[]` | `name` | str | — | Human-readable name |
| `agents[]` | `description` | str | — | What this agent handles |
| `training` | `total_steps` | int | `200000` | Total training steps |
| `training` | `batch_size` | int | `64` | Replay sample batch size |
| `training` | `learning_rate` | float | `0.001` | Adam LR |
| `training` | `gamma` | float | `0.99` | Discount factor |
| `training` | `epsilon_start` | float | `1.0` | Initial exploration rate |
| `training` | `epsilon_end` | float | `0.05` | Final exploration rate |
| `training` | `epsilon_decay_steps` | int | `100000` | Epsilon linear decay steps |
| `training` | `target_update_freq` | int | `500` | Target net sync frequency |
| `training` | `replay_buffer_size` | int | `50000` | Replay buffer capacity |
| `training` | `min_replay_size` | int | `1000` | Min buffer fill before training |
| `training` | `reward_mode` | str | `jaccard` | `jaccard` or `stochastic` |
| `training` | `step_cost` | float | `0.05` | Per-selection penalty |
| `training` | `hidden_layers` | list[int] | `[256, 128]` | Q-network layer sizes |
| `training` | `tfidf_max_features` | int | `5000` | TF-IDF vocabulary limit |
| `training` | `action_masking` | bool | `true` | Mask selected agents in Q-values |
| `training` | `seed` | int | `42` | Random seed |
| `training` | `val_eval_freq` | int | `5000` | Validation eval frequency |
| `training` | `save_best` | bool | `true` | Save best val checkpoint |
| `training` | `max_steps_per_episode` | int | `20` | Max steps per episode |
| `labeler` | `model` | str | `gpt-4o-mini` | LLM model string |
| `labeler` | `base_url` | str | OpenAI URL | API base URL |
| `labeler` | `fallback_strategy` | str | `keyword` | Parse failure handling |
| `dataset` | `train_ratio` | float | `0.7` | Train split ratio |
| `dataset` | `val_ratio` | float | `0.15` | Validation split ratio |
| `dataset` | `test_ratio` | float | `0.15` | Test split ratio |
| root | `output_dir` | str | `./artifacts/` | Artifact output directory |

---

## Architecture

- **No gymnasium dependency** — custom MDP environment
- **No OpenAI SDK** — raw `httpx` for any compatible provider
- **STOP action** = agent ID N (for N agents)
- **Action masking** prevents redundant agent selection
- **Graceful degradation**: DDQN → supervised fallback → clear error with setup instructions
- **Single config** as source of truth — all defaults in one Pydantic model
