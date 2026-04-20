"""Tests for LLMLabeler response parsing and fallbacks."""

from __future__ import annotations

from pathlib import Path

import httpx

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import AgentDef, LabelerConfig
from ddqn_router.labeler.labeler import LLMLabeler


def _make_labeler(
    tmp_path: Path, fallback: str, *, raw_response: str | None, raise_error: bool = False
) -> LLMLabeler:
    cfg = LabelerConfig(
        model="gpt-test",
        base_url="https://api.test/v1",
        api_key="x",
        cache=str(tmp_path / "cache.jsonl"),
        min_agents=1,
        fallback_strategy=fallback,  # type: ignore[arg-type]
    )
    registry = AgentRegistry(
        [
            AgentDef(id=0, name="A", description="billing invoice payment refund"),
            AgentDef(id=1, name="B", description="bug error api integration"),
            AgentDef(id=2, name="C", description="account password settings"),
        ]
    )
    labeler = LLMLabeler(cfg, registry)

    def fake_call(prompt: str) -> str:
        if raise_error:
            raise httpx.HTTPError("boom")
        assert raw_response is not None
        return raw_response

    labeler._call_llm = fake_call  # type: ignore[assignment]
    return labeler


def test_valid_response_parses(tmp_path: Path) -> None:
    labeler = _make_labeler(tmp_path, "skip", raw_response="Agents: [0, 2]")
    result = labeler.label_one("my invoice billing issue")
    assert result is not None
    assert result["required_agents"] == [0, 2]
    labeler.close()


def test_malformed_response_skip(tmp_path: Path) -> None:
    labeler = _make_labeler(tmp_path, "skip", raw_response="no brackets here")
    result = labeler.label_one("random text")
    assert result is None
    labeler.close()


def test_malformed_response_keyword_fallback(tmp_path: Path) -> None:
    labeler = _make_labeler(tmp_path, "keyword", raw_response="no brackets")
    result = labeler.label_one("billing invoice question")
    assert result is not None
    assert 0 in result["required_agents"]
    labeler.close()


def test_malformed_response_all_agents_fallback(tmp_path: Path) -> None:
    labeler = _make_labeler(tmp_path, "all-agents", raw_response="garbage")
    result = labeler.label_one("whatever")
    assert result is not None
    assert sorted(result["required_agents"]) == [0, 1, 2]
    labeler.close()


def test_http_error_triggers_fallback(tmp_path: Path) -> None:
    labeler = _make_labeler(tmp_path, "all-agents", raw_response=None, raise_error=True)
    result = labeler.label_one("anything")
    assert result is not None
    assert sorted(result["required_agents"]) == [0, 1, 2]
    labeler.close()
