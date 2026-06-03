from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm_drift.fingerprint import Fingerprint


@dataclass
class ProbeResult:
    probe_id: str
    drift_score: float
    semantic: float
    structural: float
    assertion_regression: float


@dataclass
class DriftResult:
    drift_score: float
    drifted: bool
    probe_results: List[ProbeResult] = field(default_factory=list)

    def report(self) -> str:
        status = "DRIFTED" if self.drifted else "OK"
        lines = [f"Drift score: {self.drift_score:.3f} ({status})"]
        for pr in self.probe_results:
            lines.append(f"  {pr.probe_id}: {pr.drift_score:.3f}")
        return "\n".join(lines)


def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return max(0.0, 1.0 - dot / (mag_a * mag_b))


def _structural_distance(baseline: Fingerprint, current: Fingerprint) -> float:
    format_changed = float(baseline.format != current.format)
    denom = max(baseline.token_count, current.token_count, 1)
    length_ratio = abs(baseline.token_count - current.token_count) / denom
    return min(format_changed * 0.6 + length_ratio * 0.4, 1.0)


def _assertion_regression(baseline: Fingerprint, current: Fingerprint) -> float:
    b = {r["expression"]: r["passed"] for r in baseline.assertion_results}
    c = {r["expression"]: r["passed"] for r in current.assertion_results}
    shared = set(b) & set(c)
    if not shared:
        return 0.0
    regressions = sum(1 for k in shared if b[k] and not c[k])
    return regressions / len(shared)


@dataclass
class DriftScorer:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.5,
        "structural": 0.25,
        "assertion": 0.25,
    })
    threshold: float = 0.15

    def score(
        self,
        baselines: List[Fingerprint],
        currents: List[Fingerprint],
        probe_ids: Optional[List[str]] = None,
    ) -> DriftResult:
        ids = probe_ids or [str(i) for i in range(len(baselines))]
        w = self.weights
        probe_results = []
        for probe_id, b, c in zip(ids, baselines, currents):
            semantic = _cosine_distance(b.embedding, c.embedding)
            structural = _structural_distance(b, c)
            assertion = _assertion_regression(b, c)
            score = (
                w.get("semantic", 0.5) * semantic
                + w.get("structural", 0.25) * structural
                + w.get("assertion", 0.25) * assertion
            )
            probe_results.append(ProbeResult(
                probe_id=probe_id,
                drift_score=round(min(score, 1.0), 4),
                semantic=round(semantic, 4),
                structural=round(structural, 4),
                assertion_regression=round(assertion, 4),
            ))

        suite_score = sum(pr.drift_score for pr in probe_results) / max(len(probe_results), 1)
        return DriftResult(
            drift_score=round(suite_score, 4),
            drifted=suite_score > self.threshold,
            probe_results=probe_results,
        )
