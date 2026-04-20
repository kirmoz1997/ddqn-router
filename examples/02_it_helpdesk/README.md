# IT helpdesk tier-1 triage (5 agents)

Routes IT helpdesk tickets to Hardware, Network, Access, Software, or Security.

**Time to run on CPU:** ~5 minutes.

## Steps

```bash
cd examples/02_it_helpdesk

python make_dataset.py
ddqn-router dataset split --input data/tasks.jsonl --output-dir data/
ddqn-router train --config config.yaml
ddqn-router eval --artifacts ./artifacts/ --input data/test.jsonl
```

## Try it

```python
from ddqn_router import DDQNRouter
router = DDQNRouter.load("./artifacts/")
print(router.route("vpn drops every few minutes, reset my sso password too").agent_names)
```
