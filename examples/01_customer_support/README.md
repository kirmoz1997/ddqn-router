# Customer support triage (5 agents)

End-to-end demo routing support queries to the right combination of five
specialized agents: Billing, Technical, Account, Shipping, General.

**Time to run on CPU:** ~5 minutes.

## Steps

```bash
cd examples/01_customer_support

# 1. Generate 200 pre-labeled synthetic queries (no LLM needed — deterministic)
python make_dataset.py

# 2. Split into train/val/test
ddqn-router dataset split --input data/tasks.jsonl --output-dir data/

# 3. Train (10k steps, CPU-friendly)
ddqn-router train --config config.yaml

# 4. Evaluate on the held-out test set
ddqn-router eval --artifacts ./artifacts/ --input data/test.jsonl
```

## Using the trained router

```python
from ddqn_router import DDQNRouter

router = DDQNRouter.load("./artifacts/")

print(router.route("package never arrived, refund shipping please").agent_names)
# -> ['Shipping', 'Billing']

router.explain("I cannot login and my last invoice looks wrong")
# -> prints a step-by-step Q-value table
```

## Reset / re-run

```bash
rm -rf data/*.jsonl artifacts/ cache/
python make_dataset.py
```
