# Contributing to ddqn-router

Thanks for your interest in improving `ddqn-router`! This guide walks through
local setup, the PR workflow, and how releases are cut.

## Development setup

```bash
git clone https://github.com/kirmoz1997/ddqn-router
cd ddqn-router
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve]"
pre-commit install
pytest
```

The `[dev]` extras pull in `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`,
`pandas`, and `matplotlib`. `[serve]` adds FastAPI + uvicorn.

## Branching and PRs

- Branch from `main`. Name branches by purpose:
  `feat/<topic>`, `fix/<issue>`, `docs/<area>`, `chore/<cleanup>`.
- Keep PRs focused and ideally under ~500 changed lines (excluding lock/test
  fixture files).
- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `ci:`, `test:`, `build:`, `chore:`, `refactor:`.
- Before opening a PR, run `ruff check . && ruff format --check . && pytest -q
  && mypy ddqn_router/`.

## Running checks locally

```bash
ruff check .
ruff format --check .
mypy ddqn_router/
pytest -q
pytest --cov=ddqn_router --cov-report=term
```

You can reproduce CI locally with [`act`](https://github.com/nektos/act):

```bash
act -j test
```

## Code style

- Target `py310+` syntax (the package still declares `requires-python >= 3.9`,
  but ruff/mypy are configured for `py310`).
- 100-col line length, enforced by `ruff format`.
- Public API must be typed; CI runs `mypy --strict_optional`.
- No ad-hoc scripts — new functionality should land as CLI subcommands or
  library APIs.

## Tests

- Tests live under `tests/` and mirror the package layout.
- Fixtures in `tests/conftest.py` include a tiny in-memory dataset and a
  session-scoped `trained_artifacts` fixture (runs ~300 training steps once
  per test session).
- Keep unit tests fast (<1s each); use fixtures for shared state.
- Mark slow/integration tests with `@pytest.mark.slow` if added.

## Release process

Releases are automated by the `publish.yml` workflow:

1. Update `CHANGELOG.md` — move `Unreleased` entries into a new versioned
   section and set today's date.
2. Bump `version` in `pyproject.toml` and `__version__` in
   `ddqn_router/__init__.py`.
3. Commit on `main`, tag, and push:
   ```bash
   git commit -am "chore: release v0.3.0"
   git tag v0.3.0
   git push origin main --tags
   ```
4. The `Publish to PyPI` workflow runs on tag push and uploads to PyPI via
   [Trusted Publishing (OIDC)](https://docs.pypi.org/trusted-publishers/).
   Configure a Trusted Publisher for this repo on pypi.org once; no API token
   needs to live in GitHub secrets.
5. Draft a GitHub Release from the tag; the `Docker` workflow will then build
   multi-arch images and push them to `ghcr.io/kirmoz1997/ddqn-router`.

### First-time PyPI setup (fallback)

If Trusted Publishing isn't configured yet, add `PYPI_API_TOKEN` to repo
secrets and uncomment the password line in `.github/workflows/publish.yml`.

## Reporting issues

Use the GitHub issue templates under **Issues → New**. For **security**
issues, see [`SECURITY.md`](SECURITY.md) — do **not** open a public issue.
