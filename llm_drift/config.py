from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

DEFAULT_CONFIG_PATH = "llm-drift.yaml"


@dataclass
class Config:
    store_path: str = ".llm-drift/baselines.db"
    drift_threshold: float = 0.15
    embedding_model: str = "all-MiniLM-L6-v2"
    alerts: List[dict] = field(default_factory=lambda: [{"type": "stdout"}])

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        """Load llm-drift.yaml, falling back to defaults if it doesn't exist."""
        p = Path(path)
        if not p.exists():
            return cls()

        data = yaml.safe_load(p.read_text()) or {}
        store = data.get("store") or {}
        thresholds = data.get("thresholds") or {}
        embedding = data.get("embedding") or {}
        alerts = data.get("alerts")
        if not alerts:
            alerts = [{"type": "stdout"}]

        return cls(
            store_path=store.get("path", cls.store_path),
            drift_threshold=thresholds.get("drift_score", cls.drift_threshold),
            embedding_model=embedding.get("model", cls.embedding_model),
            alerts=alerts,
        )
