# ddqn-router

A lightweight Python library that trains a Double DQN agent to route user
queries to the optimal **subset** of specialized agents in a multi-agent
system.

- Fast: sub-millisecond inference on CPU — no LLM needed at serving time.
- Practical: define agents in YAML, label a small dataset with an LLM, train,
  serve. All via a single `ddqn-router` CLI or a typed Python API.
- Typed, tested, and production-ready.

## Where to go next

- [Quickstart](quickstart.md) — install and train your first router in ~10 minutes.
- [CLI reference](cli.md) — every subcommand and its flags.
- [Python API](python-api.md) — `DDQNRouter`, `RouteResult`, `StepTrace`.
- [Configuration](configuration.md) — every YAML field and its default.
- [Deployment](deployment.md) — Docker, compose, reverse-proxy, production checklist.
- [Troubleshooting](troubleshooting.md) — common errors.

## Install

```bash
pip install ddqn-router
pip install "ddqn-router[serve]"   # for the FastAPI server
```
