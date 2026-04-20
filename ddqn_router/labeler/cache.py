"""JSONL append-only cache keyed by SHA256(text + model + prompt_version)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _make_key(text: str, model: str, prompt_version: str) -> str:
    raw = f"{text}|{model}|{prompt_version}"
    return hashlib.sha256(raw.encode()).hexdigest()


class LabelCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._store: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._store[entry["cache_key"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue

    def lookup(self, text: str, model: str, prompt_version: str) -> list[int] | None:
        key = _make_key(text, model, prompt_version)
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.get("model") != model or entry.get("prompt_version") != prompt_version:
            return None
        return entry.get("required_agents")

    def store(
        self,
        text: str,
        model: str,
        prompt_version: str,
        required_agents: list[int],
    ) -> None:
        key = _make_key(text, model, prompt_version)
        entry = {
            "cache_key": key,
            "model": model,
            "prompt_version": prompt_version,
            "required_agents": required_agents,
        }
        self._store[key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
