"""JSONL traces: one record per engine step. Evaluation reads only these files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: Path | None):
        self.path = path
        self.records: list[dict[str, Any]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def log(self, **rec: Any) -> None:
        rec = {"i": len(self.records), **rec}
        self.records.append(rec)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
