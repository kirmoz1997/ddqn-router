"""Generate a 200-example synthetic labeled dataset for the customer-support demo.

Deterministic, rule-based — no LLM calls. Run once before training:

    python make_dataset.py

Writes ./data/tasks.jsonl with already-labeled examples so you can skip the
`ddqn-router label` step in this demo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

AGENTS = {
    "BILL": 0,
    "TECH": 1,
    "ACCT": 2,
    "SHIP": 3,
    "GEN": 4,
}

TEMPLATES: list[tuple[list[int], list[str]]] = [
    (
        [AGENTS["BILL"]],
        [
            "I was charged twice for my subscription this month",
            "Please refund my last payment",
            "Why was my invoice higher than last month",
            "Cancel my auto-renewal for next cycle",
            "My credit card got charged without authorization",
        ],
    ),
    (
        [AGENTS["TECH"]],
        [
            "The API returns a 500 error on every request",
            "Webhook integration is broken since yesterday",
            "Latency is really high when I call /search",
            "SDK crashes on import in Python 3.12",
            "Rate limit headers seem wrong",
        ],
    ),
    (
        [AGENTS["ACCT"]],
        [
            "Reset my password, I forgot it",
            "Change the email address on my account",
            "I cannot log in after 2FA reset",
            "My user role is missing admin permissions",
            "Delete my account per GDPR request",
        ],
    ),
    (
        [AGENTS["SHIP"]],
        [
            "Where is my shipment, tracking hasn't updated",
            "Package marked delivered but I didn't receive it",
            "Change the delivery address for order 12345",
            "Delivery is late by three days",
            "Tracking number is invalid",
        ],
    ),
    (
        [AGENTS["GEN"]],
        [
            "What are your business hours",
            "Do you have a referral program",
            "Is there a free tier available",
            "Where is your documentation",
            "I just want some general information",
        ],
    ),
    # Multi-agent combinations
    (
        [AGENTS["BILL"], AGENTS["ACCT"]],
        [
            "I cannot login and my last invoice looks wrong",
            "Refund the duplicate charge and also unlock my account",
            "Billing error plus password reset request",
        ],
    ),
    (
        [AGENTS["BILL"], AGENTS["TECH"]],
        [
            "API broken and I was billed for the failed calls",
            "Webhook outage during billing cycle, need a refund",
            "Integration error caused duplicate charges",
        ],
    ),
    (
        [AGENTS["TECH"], AGENTS["ACCT"]],
        [
            "Cannot login, seems like an API error",
            "2FA reset is failing with a 500",
            "Permissions update didn't propagate after the deploy",
        ],
    ),
    (
        [AGENTS["SHIP"], AGENTS["BILL"]],
        [
            "Package never arrived, please refund shipping",
            "Wrong shipping fee charged on the invoice",
        ],
    ),
    (
        [AGENTS["BILL"], AGENTS["TECH"], AGENTS["ACCT"]],
        [
            "Full audit please: billing, API errors, and account access issues",
        ],
    ),
]


def main(n_examples: int = 200, seed: int = 7) -> None:
    rng = random.Random(seed)
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tasks.jsonl"

    flat: list[tuple[list[int], str]] = []
    for agents, texts in TEMPLATES:
        for t in texts:
            flat.append((agents, t))

    with open(out_path, "w") as f:
        for i in range(n_examples):
            agents, text = rng.choice(flat)
            noise_suffixes = [
                "",
                " please",
                " asap",
                " (urgent)",
                f" - customer #{rng.randint(1000, 9999)}",
            ]
            text_out = text + rng.choice(noise_suffixes)
            f.write(
                json.dumps(
                    {
                        "id": f"cs_{i:04d}",
                        "text": text_out,
                        "required_agents": sorted(agents),
                    }
                )
                + "\n"
            )

    print(f"Wrote {n_examples} examples -> {out_path}")


if __name__ == "__main__":
    main()
