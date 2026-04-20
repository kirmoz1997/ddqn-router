# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-04-20

### Added
- `ddqn-router init` — scaffolds `config.yaml`, `data/queries.example.txt`,
  `.gitignore`, and a stub `README.md` in a new project directory.
- `ddqn-router eval` — evaluates a trained router on a labeled JSONL dataset
  (`--artifacts`, `--input`, optional `--output metrics.json`).
- `ddqn-router train --resume <checkpoint-dir>` — resumes training from a saved
  checkpoint. Online + target Q-nets, Adam state, RNG state, and the step/epsilon
  metadata are restored. Opt-in `--save-replay` persists the replay buffer.
- `training.checkpoint_freq` config field (default `10000`) controls checkpoint
  cadence. Checkpoints land in `<output_dir>/checkpoint/`.
- `DDQNRouter.route_verbose(query)` + `StepTrace` + `RouteResult.steps_trace`
  expose per-step Q-values, selected action, and masked agents structurally.
  `router.explain(query)` now reuses this trace while keeping stdout output
  byte-identical to prior versions.
- `rich`-based progress panels for `label` (progress bar + cache hit rate +
  estimated cost) and `train` (live training panel with ETA).
- PyPI / Python / License / CI / downloads / docs badges in README.
- `py.typed` marker and `Typing :: Typed` classifier.
- `docs/` mkdocs-material site (quickstart, CLI/API/config reference,
  deployment, troubleshooting, confidence semantics). Auto-deployed from
  `main` via `.github/workflows/docs.yml`.
- `Dockerfile`, `docker-compose.yml`, and `.github/workflows/docker.yml`
  publishing multi-arch `ghcr.io/kirmoz1997/ddqn-router` images on each
  GitHub Release.
- `tests/` suite with fixtures and ~70%+ line coverage outside `rl/`
  (pytest + `fastapi.testclient` for serve tests).
- `.github/workflows/ci.yml` matrix: Python 3.10/3.11/3.12 × Ubuntu/macOS.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue & PR templates,
  `.pre-commit-config.yaml`, `.github/dependabot.yml`.
- `examples/01_customer_support/` and `examples/02_it_helpdesk/` reproducible
  end-to-end demos plus an `examples/README.md` index.
- `[project.optional-dependencies] docs = […]`.
- `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`,
  `[tool.setuptools.package-data]` sections in `pyproject.toml`.

### Changed
- Validated `RouterConfig`: dataset ratios must sum to `1.0`; duplicate agent
  IDs are now rejected at parse time; `step_cost` must be `>= 0`.
- `RouteResult` is now a `dataclass` (was `NamedTuple`) so it can carry an
  optional `steps_trace` field. All existing attribute access
  (`result.agents`, `.agent_names`, `.confidence`, `.steps`) is preserved.

### Fixed
- `AgentRegistry` now rejects duplicate IDs / names and non-contiguous ID
  ranges instead of silently overwriting them.

## [0.2.1] - 2026-04-07

### Fixed
- Removed empty email from `authors` in `pyproject.toml` to satisfy PyPI
  metadata validation during trusted publishing.

## [0.2.0] - 2026-04-07

### Added
- PyPI publishing setup: `pip install ddqn-router` now works.
- `examples/quickstart.ipynb`: end-to-end runnable demo notebook
  with a built-in labeled dataset (no API key required to start).
- "Open in Colab" badge in README pointing to the quickstart notebook.
- `CHANGELOG.md`.

## [0.1.0] - 2026-04-07

### Added
- Core DDQN routing library.
- CLI: `label`, `dataset stats`, `dataset split`, `train`, `serve`.
- Python inference API: `DDQNRouter.load()`, `route()`, `route_batch()`,
  `explain()`.
- FastAPI serve mode (optional `[serve]` extra).
