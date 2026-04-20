"""Generate a 200-example IT helpdesk dataset. Run once:

python make_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

AGENTS = {"HW": 0, "NET": 1, "AX": 2, "SW": 3, "SEC": 4}

TEMPLATES: list[tuple[list[int], list[str]]] = [
    (
        [AGENTS["HW"]],
        [
            "My laptop will not turn on this morning",
            "The external monitor has no signal",
            "Printer is offline in all applications",
            "Keyboard keys are sticking",
            "Mouse keeps disconnecting",
        ],
    ),
    (
        [AGENTS["NET"]],
        [
            "VPN client cannot connect from home",
            "Wifi keeps dropping every few minutes",
            "DNS resolution fails for internal sites",
            "Proxy is blocking a required domain",
            "Firewall rule request for port 5432",
        ],
    ),
    (
        [AGENTS["AX"]],
        [
            "Please reset my SSO password",
            "MFA device is lost, need to re-enroll",
            "Locked out of my account after vacation",
            "Need admin permissions on the data share",
            "Account shows disabled after rename",
        ],
    ),
    (
        [AGENTS["SW"]],
        [
            "Excel crashes when opening large files",
            "Install latest version of Slack please",
            "License expired for design software",
            "Application update failed mid-way",
            "Office 365 keeps prompting to activate",
        ],
    ),
    (
        [AGENTS["SEC"]],
        [
            "Received a suspicious email asking for credentials",
            "Antivirus flagged a file I did not download",
            "Noticed unusual login alerts overnight",
            "Possible phishing link in a calendar invite",
            "Laptop may have been compromised at conference wifi",
        ],
    ),
    # Multi-agent
    (
        [AGENTS["NET"], AGENTS["AX"]],
        [
            "Cannot reach internal site and SSO won't let me in",
            "VPN connects but SSO rejects the token",
        ],
    ),
    (
        [AGENTS["SW"], AGENTS["AX"]],
        [
            "Application keeps logging me out after password reset",
            "Software won't activate with my SSO credentials",
        ],
    ),
    (
        [AGENTS["HW"], AGENTS["NET"]],
        [
            "Laptop dock ethernet port dead, no network",
            "Wifi adapter driver seems broken after windows update",
        ],
    ),
    (
        [AGENTS["SEC"], AGENTS["AX"]],
        [
            "Suspicious login from another country, please lock my account",
            "I clicked a phishing link, reset my credentials please",
        ],
    ),
    (
        [AGENTS["HW"], AGENTS["SW"], AGENTS["NET"]],
        [
            "New laptop setup: needs software, network configuration, VPN",
        ],
    ),
]


def main(n_examples: int = 200, seed: int = 13) -> None:
    rng = random.Random(seed)
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tasks.jsonl"

    flat: list[tuple[list[int], str]] = []
    for agents, texts in TEMPLATES:
        for t in texts:
            flat.append((agents, t))

    suffixes = [
        "",
        " - urgent",
        " (ticket reopened)",
        " thanks",
        f" - user #{rng.randint(100, 999)}",
    ]

    with open(out_path, "w") as f:
        for i in range(n_examples):
            agents, text = rng.choice(flat)
            text_out = text + rng.choice(suffixes)
            f.write(
                json.dumps(
                    {
                        "id": f"it_{i:04d}",
                        "text": text_out,
                        "required_agents": sorted(agents),
                    }
                )
                + "\n"
            )

    print(f"Wrote {n_examples} examples -> {out_path}")


if __name__ == "__main__":
    main()
