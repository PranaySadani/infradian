"""Machine-comparable results schema. Every run writes one of these; the leaderboard is built by
reading them. The `is_primary` flag on each metric block is the multiplicity guard (plan §6.2):
exactly the two pre-registered endpoints carry it, everything else is exploratory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class MetricBlock(BaseModel):
    task: str  # e.g. "T2-R", "T2-P@0", "T1-pdg", "T3", "T2-A"
    stratum: str = "all"  # "all" | "regular" | "irregular" | ...
    metric: str  # e.g. "mae_days", "SoC", "macro_f1", "pr_auc", "spearman"
    value: float
    ci_lo: float | None = None
    ci_hi: float | None = None
    n: int
    is_primary: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class RunResults(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_name: str
    tier: str  # "C" | "B"
    model: str
    dataset: dict[str, Any] = Field(default_factory=dict)
    splits: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=list)
    git_commit: str = ""
    integrity_status: str = "OK"
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    metrics: list[MetricBlock] = Field(default_factory=list)
    wall_clock_s: float = 0.0
    notes: str = ""

    def add(self, block: MetricBlock) -> None:
        self.metrics.append(block)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True))
        return path


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
