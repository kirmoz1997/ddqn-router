"""CLI entry point for ddqn-router."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

app = typer.Typer(
    name="ddqn-router",
    help="Double DQN-based router for multi-agent systems.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Dataset utilities.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


@app.command()
def label(
    config: str = typer.Option(..., "--config", help="Path to router_config.yaml"),
    input: str | None = typer.Option(None, "--input", help="Path to raw texts"),
    output: str | None = typer.Option(None, "--output", help="Output tasks.jsonl path"),
    model: str | None = typer.Option(None, "--model", help="LLM model string"),
    base_url: str | None = typer.Option(None, "--base-url", help="API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key"),
    min_agents: int | None = typer.Option(None, "--min-agents"),
    max_agents: int | None = typer.Option(None, "--max-agents"),
    prompt_template: str | None = typer.Option(None, "--prompt-template"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    cache: str | None = typer.Option(None, "--cache"),
    fallback_strategy: str | None = typer.Option(None, "--fallback-strategy"),
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

    typer.echo(f"Labeling {cfg.labeler.input} -> {cfg.labeler.output}")
    try:
        count = labeler.label_file(cfg.labeler.input, cfg.labeler.output)
    finally:
        labeler.close()
    typer.echo(f"Labeled {count} examples -> {cfg.labeler.output}")


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
    output_dir: str | None = typer.Option(None, "--output-dir", help="Output directory"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Split dataset into train/val/test with stratification by set size."""
    from ddqn_router.dataset.dataset import load_tasks
    from ddqn_router.dataset.splitter import split_and_save

    tasks = load_tasks(input)
    out = output_dir or str(Path(input).parent)
    n_train, n_val, n_test = split_and_save(tasks, out, train, val, test, seed)
    typer.echo(f"Split {len(tasks)} examples -> train={n_train}, val={n_val}, test={n_test}")
    typer.echo(f"Saved to {out}/")


@app.command()
def train(
    config: str = typer.Option(..., "--config", help="Path to router_config.yaml"),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Artifacts output dir"),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume from checkpoint directory (e.g. ./artifacts/checkpoint/)"
    ),
    save_replay: bool = typer.Option(
        False, "--save-replay", help="Also persist replay buffer on each checkpoint"
    ),
) -> None:
    """Train the DDQN routing agent."""
    from ddqn_router.config import RouterConfig
    from ddqn_router.rl.ddqn_agent import train as run_training

    cfg = RouterConfig.from_yaml(config)
    if output_dir is not None:
        cfg.output_dir = output_dir

    run_training(cfg, resume_from=resume, save_replay=save_replay)


@app.command()
def serve(
    artifacts: str = typer.Option(
        "./artifacts/", "--artifacts", help="Path to trained model artifacts"
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    cors: str | None = typer.Option(
        None, "--cors", help="Allowed CORS origins (comma-separated, or '*' for all)"
    ),
) -> None:
    """Start a FastAPI server for routing inference.

    Requires the 'serve' extras: pip install ddqn-router[serve]
    """
    try:
        import uvicorn
    except ImportError as err:
        typer.echo(
            "Error: uvicorn is required. Install with: pip install ddqn-router[serve]",
            err=True,
        )
        raise typer.Exit(1) from err

    from ddqn_router.serve.app import create_app

    cors_origins = None
    if cors is not None:
        cors_origins = [o.strip() for o in cors.split(",")]

    application = create_app(artifacts, cors_origins=cors_origins)
    typer.echo(f"Starting ddqn-router server on {host}:{port}")
    typer.echo(f"Artifacts: {artifacts}")
    if cors_origins:
        typer.echo(f"CORS: {cors_origins}")
    uvicorn.run(application, host=host, port=port)


@app.command()
def init(
    path: str = typer.Option(".", "--path", help="Target directory for the scaffold"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
) -> None:
    """Scaffold a new ddqn-router project in the given directory."""
    from ddqn_router.scaffold import write_scaffold

    target = Path(path).resolve()
    results = write_scaffold(target, force=force)

    created = sum(1 for _, action in results if action == "created")
    skipped = sum(1 for _, action in results if action == "skipped")
    for p, action in results:
        typer.echo(f"  {action:>8}  {p.relative_to(target) if p.is_relative_to(target) else p}")

    typer.echo(f"\n{created} files created, {skipped} skipped in {target}")
    typer.echo("\nNext steps:")
    typer.echo(f"  cd {target}")
    typer.echo("  ddqn-router label --config config.yaml --input data/queries.example.txt")
    typer.echo("  ddqn-router dataset split --input data/tasks.jsonl")
    typer.echo("  ddqn-router train --config config.yaml")


@app.command(name="eval")
def eval_cmd(
    artifacts: str = typer.Option(
        "./artifacts/", "--artifacts", help="Path to trained model artifacts"
    ),
    input: str = typer.Option(..., "--input", help="Path to a labeled test.jsonl"),
    output: str | None = typer.Option(None, "--output", help="Write metrics JSON here"),
) -> None:
    """Evaluate a trained router on a labeled dataset."""
    from ddqn_router.dataset.dataset import load_tasks
    from ddqn_router.eval.evaluator import evaluate_routing, print_metrics
    from ddqn_router.inference.router import DDQNRouter, RouterNotTrainedError

    try:
        router = DDQNRouter.load(artifacts)
    except RouterNotTrainedError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(2) from err

    try:
        tasks = load_tasks(input)
    except Exception as err:
        typer.echo(f"Dataset error: {err}", err=True)
        raise typer.Exit(1) from err

    preds: list[set[int]] = []
    targets: list[set[int]] = []
    for t in tasks:
        result = router.route(t["text"])
        preds.append(set(result.agents))
        targets.append(set(t["required_agents"]))

    metrics = evaluate_routing(preds, targets)
    print_metrics(metrics, label=f"Evaluation on {input}")

    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)
        typer.echo(f"\nMetrics written to {output}")


if __name__ == "__main__":
    app()
