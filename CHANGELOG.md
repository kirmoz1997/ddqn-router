# Changelog

## [0.2.0] - 2026-04-07

### Added
- PyPI publishing setup: `pip install ddqn-router` now works
- `examples/quickstart.ipynb`: end-to-end runnable demo notebook
  with a built-in labeled dataset (no API key required to start)
- "Open in Colab" badge in README pointing to the quickstart notebook
- `CHANGELOG.md`

## [0.1.0] - initial release

- Core DDQN routing library
- CLI: label, dataset, train, serve
- Python inference API: DDQNRouter.load(), route(), explain()
- FastAPI serve mode (optional extra)
