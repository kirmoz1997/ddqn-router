# Troubleshooting

## `RouterNotTrainedError: No trained DDQN router found in …`

The artifacts directory is missing `model.pt`, `encoder.joblib`, or
`config_used.json`. Run `ddqn-router train --config config.yaml` and check
that `output_dir` matches the path you're loading from.

## `ValidationError: dataset.train_ratio + val_ratio + test_ratio must sum to 1.0`

Your config has ratios that don't add up. `0.7 / 0.15 / 0.15` is the default.

## `ValueError: Duplicate agent id: X`

Two agents share the same `id`. Ids must be unique and contiguous starting
from 0.

## `ValueError: Agent ids must be contiguous without gaps`

You skipped an id (e.g. `0, 2, 3`). Renumber so ids are `0..N-1`.

## Training is slow / CPU pegged

- Lower `training.total_steps` and `tfidf_max_features` for quick iteration.
- Check `training.min_replay_size` — warm-up steps don't update weights.
- GPU is supported automatically (`torch.cuda.is_available()`), but TF-IDF
  preprocessing is CPU-only and tends to dominate small runs.

## CUDA out of memory

Reduce `training.batch_size` or `hidden_layers`, or run on CPU:

```bash
CUDA_VISIBLE_DEVICES="" ddqn-router train --config config.yaml
```

## Labeler API errors

- Check `DDQN_ROUTER_API_KEY` / `labeler.api_key`.
- The labeler has three fallback strategies: `skip`, `keyword`,
  `all-agents`. Pick one that's appropriate when the LLM returns malformed
  output.
- Responses and errors are **not** cached — re-runs will retry.

## Encoding mismatch at inference

The encoder must match the model. If you mix `encoder.joblib` from one run
with `model.pt` from another, shape errors are raised. Always load the
`artifacts/` directory produced by a single training run.

## `/route` returns 500

The server crashed during startup if artifacts are missing — check the
uvicorn logs. If the server is up and a single request 500s, re-check the
request schema: `{"query": "<string>"}`.

## Resume didn't pick up where I left off

`--resume` expects the **checkpoint directory** (`./artifacts/checkpoint/`),
not the artifacts root. Confirm `step.json` exists inside the directory.
