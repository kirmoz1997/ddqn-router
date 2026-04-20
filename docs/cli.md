# CLI reference

All subcommands are accessible via the `ddqn-router` entry point.

```
$ ddqn-router --help
```

## `ddqn-router init`

Scaffold a new project directory.

| Flag | Default | Description |
| --- | --- | --- |
| `--path` | `.` | Target directory. |
| `--force` | `false` | Overwrite existing files. |

Produces `config.yaml`, `data/queries.example.txt`, `.gitignore`, `README.md`.

## `ddqn-router label`

Label raw queries with an LLM. Required: `--config`.

Key flags: `--input`, `--output`, `--model`, `--base-url`, `--api-key`,
`--min-agents`, `--max-agents`, `--prompt-template`, `--cache`,
`--fallback-strategy {skip|keyword|all-agents}`.

Shows a live `rich` progress bar with cache-hit rate and estimated prompt
cost (for known models).

## `ddqn-router dataset stats`

`--input path/to/tasks.jsonl` — prints agent frequency and set-size distribution.

## `ddqn-router dataset split`

`--input`, `--train 0.7`, `--val 0.15`, `--test 0.15`, `--output-dir`, `--seed 42`.

Stratified by required-set size.

## `ddqn-router train`

Train the Double DQN router.

| Flag | Description |
| --- | --- |
| `--config` | Path to `router_config.yaml` (required). |
| `--output-dir` | Override `output_dir` from config. |
| `--resume` | Resume from a checkpoint directory (e.g. `./artifacts/checkpoint/`). |
| `--save-replay` | Persist the replay buffer alongside each checkpoint. |

Checkpoints are written every `training.checkpoint_freq` steps (default
`10000`).

## `ddqn-router eval`

Evaluate a trained router on a labeled JSONL.

| Flag | Description |
| --- | --- |
| `--artifacts` | Directory containing `model.pt` etc. (default `./artifacts/`). |
| `--input` | Path to labeled JSONL (e.g. `test.jsonl`). Required. |
| `--output` | Optional path to dump metrics JSON. |

Exit codes: `0` success, `2` missing artifacts, `1` dataset errors.

## `ddqn-router serve`

Start the FastAPI server (requires `[serve]` extras).

| Flag | Default | Description |
| --- | --- | --- |
| `--artifacts` | `./artifacts/` | Trained model directory. |
| `--host` | `0.0.0.0` | Bind host. |
| `--port` | `8000` | Bind port. |
| `--cors` | *(off)* | Comma-separated allowed origins, or `*`. |

Exposes `POST /route`, `POST /route/batch`, `GET /agents`, `GET /health`.
