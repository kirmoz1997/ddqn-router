"""CLI entry point for ddqn-router."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="ddqn-router",
    help="Double DQN-based router for multi-agent systems.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Dataset utilities.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


# ── label ──────────────────────────────────────────────────────────────────────

@app.command()
def label(
    config: str = typer.Option(..., "--config", help="Path to router_config.yaml"),
    input: Optional[str] = typer.Option(None, "--input", help="Path to raw texts"),
    output: Optional[str] = typer.Option(None, "--output", help="Output tasks.jsonl path"),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model string"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    min_agents: Optional[int] = typer.Option(None, "--min-agents"),
    max_agents: Optional[int] = typer.Option(None, "--max-agents"),
    prompt_template: Optional[str] = typer.Option(None, "--prompt-template"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    cache: Optional[str] = typer.Option(None, "--cache"),
    fallback_strategy: Optional[str] = typer.Option(None, "--fallback-strategy"),
) -> None:
    """Label raw texts with required agents using an LLM."""
    from ddqn_router.agents import AgentRegistry
    from ddqn_router.config import RouterConfig
    from ddqn_router.labeler.labeler import LLMLabeler

    cfg = RouterConfig.from_yaml(config)

    if input is not None:
        cfg.labeler.input = input
    if output is not None:
        cfg.labeler.output = output
    if model is not None:
        cfg.labeler.model = model
    if base_url is not None:
        cfg.labeler.base_url = base_url
    if api_key is not None:
        cfg.labeler.api_key = api_key
    elif not cfg.labeler.api_key:
        cfg.labeler.api_key = os.environ.get("DDQN_ROUTER_API_KEY", "")
    if min_agents is not None:
        cfg.labeler.min_agents = min_agents
    if max_agents is not None:
        cfg.labeler.max_agents = max_agents
    if prompt_template is not None:
        cfg.labeler.prompt_template = prompt_template
    if batch_size is not None:
        cfg.labeler.batch_size = batch_size
    if cache is not None:
        cfg.labeler.cache = cache
    if fallback_strategy is not None:
        cfg.labeler.fallback_strategy = fallback_strategy  # type: ignore[assignment]

    if not cfg.labeler.input:
        typer.echo("Error: --input is required (path to raw texts)", err=True)
        raise typer.Exit(1)

    registry = AgentRegistry(cfg.agents)
    labeler = LLMLabeler(cfg.labeler, registry)

    typer.echo(f"Labeling {cfg.labeler.input} → {cfg.labeler.output}")
    try:
        count = labeler.label_file(cfg.labeler.input, cfg.labeler.output)
    finally:
        labeler.close()
    typer.echo(f"Labeled {count} examples → {cfg.labeler.output}")


# ── dataset ────────────────────────────────────────────────────────────────────

@dataset_app.command("stats")
def dataset_stats(
    input: str = typer.Option(..., "--input", help="Path to tasks.jsonl"),
) -> None:
    """Print dataset statistics."""
    from ddqn_router.dataset.dataset import load_tasks, print_stats

    tasks = load_tasks(input)
    print_stats(tasks)


@dataset_app.command("split")
def dataset_split(
    input: str = typer.Option(..., "--input", help="Path to tasks.jsonl"),
    train: float = typer.Option(0.7, "--train", help="Train ratio"),
    val: float = typer.Option(0.15, "--val", help="Validation ratio"),
    test: float = typer.Option(0.15, "--test", help="Test ratio"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Output directory"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Split dataset into train/val/test with stratification by set size."""
    from ddqn_router.dataset.dataset import load_tasks
    from ddqn_router.dataset.splitter import split_and_save

    tasks = load_tasks(input)
    out = output_dir or str(Path(input).parent)
    n_train, n_val, n_test = split_and_save(tasks, out, train, val, test, seed)
    typer.echo(f"Split {len(tasks)} examples → train={n_train}, val={n_val}, test={n_test}")
    typer.echo(f"Saved to {out}/")


# ── train ──────────────────────────────────────────────────────────────────────

@app.command()
def train(
    config: str = typer.Option(..., "--config", help="Path to router_config.yaml"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Artifacts output dir"),
) -> None:
    """Train the DDQN routing agent."""
    from ddqn_router.config import RouterConfig
    from ddqn_router.rl.ddqn_agent import train as run_training

    cfg = RouterConfig.from_yaml(config)
    if output_dir is not None:
        cfg.output_dir = output_dir

    run_training(cfg)


# ── baseline ───────────────────────────────────────────────────────────────────

@app.command()
def baseline(
    config: str = typer.Option(..., "--config", help="Path to router_config.yaml"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Artifacts output dir"),
    skip_llm: bool = typer.Option(False, "--skip-llm", help="Skip LLM baseline"),
) -> None:
    """Run all baselines on the test split and print a comparison table."""
    from ddqn_router.agents import AgentRegistry
    from ddqn_router.baselines.random_router import random_route
    from ddqn_router.baselines.rule_router import rule_route
    from ddqn_router.baselines.supervised_router import SupervisedRouter
    from ddqn_router.config import RouterConfig
    from ddqn_router.dataset.dataset import load_tasks
    from ddqn_router.eval.evaluator import evaluate_routing

    cfg = RouterConfig.from_yaml(config)
    if output_dir is not None:
        cfg.output_dir = output_dir

    out_path = Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    registry = AgentRegistry(cfg.agents)
    data_dir = Path(cfg.dataset.input).parent
    train_tasks = load_tasks(data_dir / "train.jsonl")
    test_tasks = load_tasks(data_dir / "test.jsonl")
    test_targets = [set(t["required_agents"]) for t in test_tasks]

    results: dict[str, dict] = {}

    # Random
    preds = random_route(test_tasks, registry, cfg.training.seed)
    results["Random"] = evaluate_routing(preds, test_targets)

    # Rule-based
    preds = rule_route(test_tasks, registry)
    results["Rule-based"] = evaluate_routing(preds, test_targets)

    # Supervised
    sup = SupervisedRouter(registry, cfg.training.tfidf_max_features)
    sup.fit(train_tasks)
    sup.save(out_path / "supervised.joblib")
    preds = sup.predict(test_tasks)
    results["Supervised"] = evaluate_routing(preds, test_targets)

    # LLM (optional)
    if not skip_llm and cfg.labeler.api_key:
        from ddqn_router.baselines.llm_router import llm_route

        preds = llm_route(test_tasks, registry, cfg.labeler)
        results["LLM Router"] = evaluate_routing(preds, test_targets)

    # Print comparison table
    _print_comparison(results)

    with open(out_path / "baselines_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    typer.echo(f"\nSaved to {out_path / 'baselines_summary.json'}")


def _print_comparison(results: dict[str, dict]) -> None:
    metrics_keys = [
        "mean_jaccard", "success_rate", "exact_match_rate",
        "mean_precision", "mean_recall", "mean_f1",
    ]
    headers = ["Method"] + [k.replace("mean_", "").replace("_", " ").title() for k in metrics_keys]
    col_w = [16] + [12] * len(metrics_keys)

    print(f"\n  Baseline Comparison")
    print(f"  {'─' * (sum(col_w) + len(col_w) * 2)}")
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(f"  {header_row}")
    print(f"  {'─' * (sum(col_w) + len(col_w) * 2)}")

    for name, metrics in results.items():
        vals = [name] + [f"{metrics.get(k, 0):.4f}" for k in metrics_keys]
        row = "  ".join(v.ljust(w) for v, w in zip(vals, col_w))
        print(f"  {row}")
    print()


if __name__ == "__main__":
    app()
