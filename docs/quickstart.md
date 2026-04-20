# Quickstart

Go from an empty folder to a running `/route` endpoint in ~10 minutes.

## 1. Install

```bash
pip install "ddqn-router[serve]"
```

## 2. Scaffold a project

```bash
ddqn-router init --path my-router
cd my-router
```

This creates:

```
my-router/
├── config.yaml
├── data/queries.example.txt
├── .gitignore
└── README.md
```

## 3. Label your data (or skip to use the shipped examples)

```bash
export DDQN_ROUTER_API_KEY=sk-...

ddqn-router label \
  --config config.yaml \
  --input data/queries.example.txt \
  --output data/tasks.jsonl
```

The labeler calls any OpenAI-compatible endpoint (OpenAI, DeepSeek, Ollama, …)
and caches each result so re-runs are cheap. Fallbacks (`keyword`,
`all-agents`, `skip`) handle malformed LLM responses.

## 4. Split and train

```bash
ddqn-router dataset split --input data/tasks.jsonl
ddqn-router train --config config.yaml
```

Artifacts land in `./artifacts/`: `model.pt`, `encoder.joblib`,
`config_used.json`, `metrics_test.json`, `training_log.jsonl`, and a
rolling `checkpoint/` directory.

## 5. Use the router

=== "Python"

    ```python
    from ddqn_router import DDQNRouter

    router = DDQNRouter.load("./artifacts/")
    result = router.route("my invoice was charged twice")
    print(result.agents, result.agent_names, result.confidence)
    ```

=== "CLI evaluation"

    ```bash
    ddqn-router eval --artifacts ./artifacts/ --input data/test.jsonl
    ```

=== "HTTP serving"

    ```bash
    ddqn-router serve --artifacts ./artifacts/ --port 8000
    curl -s -X POST http://localhost:8000/route \
      -H 'content-type: application/json' \
      -d '{"query":"my invoice was charged twice"}' | jq
    ```
