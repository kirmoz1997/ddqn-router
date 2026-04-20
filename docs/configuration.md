# Configuration

`RouterConfig` is the single source of truth for every tunable knob. Load
from YAML or instantiate directly.

```python
from ddqn_router.config import RouterConfig
cfg = RouterConfig.from_yaml("config.yaml")
```

## Top-level

| Field | Default | Description |
| --- | --- | --- |
| `agents` | `[]` | List of agent definitions (see below). Must be non-empty for training/serving. |
| `training` | — | `TrainingConfig` block. |
| `labeler` | — | `LabelerConfig` block. |
| `dataset` | — | `DatasetConfig` block. |
| `output_dir` | `./artifacts/` | Where to write `model.pt`, `encoder.joblib`, checkpoints. |

## `agents`

Each entry must have a unique `id` and `name`. IDs must be **contiguous**
starting from `0` (e.g. `0, 1, 2` — not `0, 2, 3`). Duplicate IDs are
rejected at load time.

```yaml
agents:
  - id: 0
    name: "Billing"
    description: "invoice payment refund subscription"
```

## `training` (`TrainingConfig`)

| Field | Default | Description |
| --- | --- | --- |
| `total_steps` | `200000` | Total environment steps. |
| `batch_size` | `64` | Mini-batch size for the Q-network update. |
| `learning_rate` | `0.001` | Adam learning rate. |
| `gamma` | `0.99` | Discount factor. |
| `epsilon_start` / `epsilon_end` | `1.0` / `0.05` | Linear ε decay. |
| `epsilon_decay_steps` | `100000` | Steps over which ε decays. |
| `target_update_freq` | `500` | Sync target ← online every N steps. |
| `replay_buffer_size` | `50000` | Capacity of the uniform replay buffer. |
| `min_replay_size` | `1000` | Steps to warm-up before starting gradient updates. |
| `reward_mode` | `"jaccard"` | `"jaccard"` or `"stochastic"`. |
| `step_cost` | `0.05` | Per-selection penalty. Must be ≥ 0. |
| `hidden_layers` | `[256, 128]` | MLP hidden dims for the Q-network. |
| `tfidf_max_features` | `5000` | Vocabulary cap for the TF-IDF encoder. |
| `action_masking` | `true` | Disallow re-selecting an already-chosen agent. |
| `seed` | `42` | Python / numpy / torch seed. |
| `val_eval_freq` | `5000` | Validation cadence in steps. |
| `save_best` | `true` | Persist the best val-Jaccard checkpoint. |
| `max_steps_per_episode` | `20` | Hard cap on episode length. |
| `checkpoint_freq` | `10000` | Training-resume checkpoint cadence. |

## `labeler` (`LabelerConfig`)

| Field | Default | Description |
| --- | --- | --- |
| `model` | `"gpt-4o-mini"` | Any OpenAI-compatible model name. |
| `base_url` | `"https://api.openai.com/v1"` | Swap for DeepSeek, Ollama, etc. |
| `api_key` | `""` | Or set `DDQN_ROUTER_API_KEY`. |
| `input` | `""` | Path to raw texts (jsonl or one-per-line). |
| `output` | `./data/tasks.jsonl` | Labeled output path. |
| `min_agents` | `2` | Minimum agents per example. |
| `max_agents` | `null` | Upper cap (default: no cap). |
| `prompt_template` | `null` | Override path to a Jinja2 template. |
| `prompt_version` | `"v1"` | Cache key component — bump to invalidate cache. |
| `batch_size` | `1` | Requests per batch (currently serialized). |
| `cache` | `./cache/label_cache.jsonl` | Append-only JSONL cache. |
| `fallback_strategy` | `"keyword"` | `skip` / `keyword` / `all-agents`. |

## `dataset` (`DatasetConfig`)

| Field | Default | Description |
| --- | --- | --- |
| `input` | `./data/tasks.jsonl` | Labeled dataset. |
| `train_ratio` / `val_ratio` / `test_ratio` | `0.7 / 0.15 / 0.15` | Must sum to 1.0. |
| `output_dir` | `./data/` | Where split files land. |

## Validation

Loading a config runs Pydantic validators that enforce:

- `train_ratio + val_ratio + test_ratio == 1.0` (within 1e-6);
- every `agent.id` is unique;
- `training.step_cost >= 0`;
- `training.total_steps > 0`;
- `training.batch_size > 0`.

Invalid configs raise `pydantic.ValidationError`.
