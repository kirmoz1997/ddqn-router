# Examples

Reproducible end-to-end demos of `ddqn-router`.

| Example | Agents | Time to run (CPU) | Description |
| --- | --- | --- | --- |
| [`01_customer_support/`](01_customer_support/) | 5 | ~5 min | Customer-support triage across Billing, Technical, Account, Shipping, General. |
| [`02_it_helpdesk/`](02_it_helpdesk/) | 5 | ~5 min | IT tier-1 helpdesk triage across Hardware, Network, Access, Software, Security. |
| [`colab_quickstart.ipynb`](colab_quickstart.ipynb) | 3 | ~5 min | End-to-end notebook — works on a free Colab CPU with no API key required. |
| [`example_config.yaml`](example_config.yaml) | — | — | Annotated config reference. |
| [`raw_texts.txt`](raw_texts.txt) | — | — | Legacy sample queries (kept for the Quickstart in the README). |

Each subdirectory has its own `README.md` with step-by-step commands. All
examples ship a small generator script instead of a large pre-labeled
`queries.jsonl` to keep the repo lean — run `python make_dataset.py` inside
the directory first.
