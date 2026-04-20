"""LLM-based labeler using httpx for OpenAI-compatible APIs."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import httpx
import jinja2

from ddqn_router.agents import AgentRegistry
from ddqn_router.config import LabelerConfig
from ddqn_router.labeler.cache import LabelCache

_DEFAULT_TEMPLATE = Path(__file__).parent / "prompt_template.j2"

# USD per 1K prompt tokens (rough, input-side only; unknown models -> "n/a").
_MODEL_PRICES: dict[str, float] = {
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.0025,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.0005,
    "deepseek-chat": 0.00014,
    "claude-3-haiku-20240307": 0.00025,
    "claude-3-5-sonnet-20240620": 0.003,
}


class LLMLabeler:
    def __init__(self, config: LabelerConfig, registry: AgentRegistry) -> None:
        self.config = config
        self.registry = registry
        self.cache = LabelCache(config.cache)
        template_path = config.prompt_template or str(_DEFAULT_TEMPLATE)
        with open(template_path) as f:
            self._template = jinja2.Template(f.read())
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def _render_prompt(self, text: str) -> str:
        max_agents = self.config.max_agents
        if max_agents is None:
            max_agents = self.registry.num_agents
        return self._template.render(
            agents=self.registry.all_agents(),
            text=text,
            min_agents=self.config.min_agents,
            max_agents=max_agents if self.config.max_agents is not None else None,
        )

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _parse_response(self, raw: str) -> list[int] | None:
        match = re.search(r"\[[\d\s,]*\]", raw)
        if not match:
            return None
        try:
            ids = json.loads(match.group())
            valid_ids = self.registry.ids()
            return [i for i in ids if i in valid_ids]
        except (json.JSONDecodeError, TypeError):
            return None

    def _keyword_fallback(self, text: str) -> list[int]:
        text_lower = text.lower()
        selected: list[int] = []
        for agent in self.registry.all_agents():
            keywords = agent.description.lower().split()
            if any(kw in text_lower for kw in keywords if len(kw) > 3):
                selected.append(agent.id)
        if len(selected) < self.config.min_agents:
            selected = self.registry.ids()[: self.config.min_agents]
        return selected

    def _apply_fallback(self, text: str) -> list[int] | None:
        strategy = self.config.fallback_strategy
        if strategy == "skip":
            return None
        elif strategy == "keyword":
            return self._keyword_fallback(text)
        elif strategy == "all-agents":
            return self.registry.ids()
        return None

    def label_one(self, text: str) -> dict | None:
        cached = self.cache.lookup(text, self.config.model, self.config.prompt_version)
        if cached is not None:
            return {
                "id": f"ex_{uuid.uuid4().hex[:8]}",
                "text": text,
                "required_agents": cached,
            }

        prompt = self._render_prompt(text)
        try:
            raw = self._call_llm(prompt)
            agents = self._parse_response(raw)
        except Exception:
            agents = None

        if agents is None:
            agents = self._apply_fallback(text)
        if agents is None:
            return None

        self.cache.store(text, self.config.model, self.config.prompt_version, agents)
        return {
            "id": f"ex_{uuid.uuid4().hex[:8]}",
            "text": text,
            "required_agents": agents,
        }

    def label_file(self, input_path: str, output_path: str) -> int:
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        texts: list[str] = []
        with open(in_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    texts.append(obj["text"])
                except (json.JSONDecodeError, KeyError):
                    texts.append(line)

        price_per_1k = _MODEL_PRICES.get(self.config.model)
        cache_hits = 0
        est_cost = 0.0
        count = 0

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[cache_rate]}"),
            TextColumn("[yellow]{task.fields[cost]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress, open(out_path, "w") as out:
            task_id = progress.add_task(
                "Labeling", total=len(texts), cache_rate="cache=0%", cost="cost=$0.000"
            )
            for idx, text in enumerate(texts, start=1):
                before_hits = cache_hits
                result, was_cache_hit, tokens = self._label_one_with_stats(text)
                if was_cache_hit:
                    cache_hits = before_hits + 1
                elif price_per_1k is not None:
                    est_cost += price_per_1k * (tokens / 1000.0)

                if result is not None:
                    out.write(json.dumps(result) + "\n")
                    count += 1

                cache_rate = f"cache={100 * cache_hits / idx:.0f}%"
                cost_label = f"cost=${est_cost:.3f}" if price_per_1k is not None else "cost=n/a"
                progress.update(task_id, advance=1, cache_rate=cache_rate, cost=cost_label)
        return count

    def _label_one_with_stats(self, text: str) -> tuple[dict | None, bool, int]:
        """Like ``label_one`` but also returns (was_cache_hit, approx_prompt_tokens)."""
        cached = self.cache.lookup(text, self.config.model, self.config.prompt_version)
        if cached is not None:
            import uuid

            return (
                {
                    "id": f"ex_{uuid.uuid4().hex[:8]}",
                    "text": text,
                    "required_agents": cached,
                },
                True,
                0,
            )

        prompt = self._render_prompt(text)
        approx_tokens = max(1, len(prompt) // 4)
        try:
            raw = self._call_llm(prompt)
            agents = self._parse_response(raw)
        except Exception:
            agents = None

        if agents is None:
            agents = self._apply_fallback(text)
        if agents is None:
            return None, False, approx_tokens

        self.cache.store(text, self.config.model, self.config.prompt_version, agents)
        import uuid

        return (
            {
                "id": f"ex_{uuid.uuid4().hex[:8]}",
                "text": text,
                "required_agents": agents,
            },
            False,
            approx_tokens,
        )

    def close(self) -> None:
        self._client.close()
