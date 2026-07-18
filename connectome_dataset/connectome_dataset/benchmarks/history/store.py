"""Append-only JSONL history store (the derived cross-cutting index)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

_REPO = Path(__file__).parents[3]
HISTORY_PATH = _REPO / "results" / "history.jsonl"


def append(records: Iterable[dict[str, Any]], path: Path = HISTORY_PATH) -> int:
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return len(records)


def load(path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
