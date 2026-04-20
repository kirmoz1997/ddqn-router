"""Tests for the Typer CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ddqn_router.cli import app

runner = CliRunner()


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ddqn-router" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_subcommand_help() -> None:
    for subcmd in ("label", "train", "serve", "init", "eval"):
        result = runner.invoke(app, [subcmd, "--help"])
        assert result.exit_code == 0, f"{subcmd} --help failed: {result.stdout}"


def test_dataset_stats_on_fixture(tiny_dataset: Path) -> None:
    result = runner.invoke(app, ["dataset", "stats", "--input", str(tiny_dataset)])
    assert result.exit_code == 0
    assert "Total examples" in result.stdout
    assert "12" in result.stdout


def test_init_creates_scaffold(tmp_path: Path) -> None:
    target = tmp_path / "project"
    result = runner.invoke(app, ["init", "--path", str(target)])
    assert result.exit_code == 0, result.stdout
    assert (target / "config.yaml").exists()
    assert (target / "data" / "queries.example.txt").exists()
    assert (target / ".gitignore").exists()
