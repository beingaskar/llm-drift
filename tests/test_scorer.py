import hashlib
import math
import random
from typing import List

import pytest

from llm_drift.fingerprint import Fingerprint
from llm_drift.scorer import DriftScorer, DriftResult


def _vec(seed: int, dim: int = 8) -> List[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]


def _fp(seed: int = 0, fmt: str = "plain", assertions: list = None, tokens: int = 10) -> Fingerprint:
    return Fingerprint(
        embedding=_vec(seed),
        token_count=tokens,
        format=fmt,
        assertion_results=assertions or [],
    )


scorer = DriftScorer()


def test_score_identical_fingerprints_is_zero():
    fp = _fp(seed=42)
    result = scorer.score([fp], [fp])
    assert result.drift_score == 0.0


def test_score_is_between_zero_and_one():
    b = _fp(seed=1)
    c = _fp(seed=99)
    result = scorer.score([b], [c])
    assert 0.0 <= result.drift_score <= 1.0


def test_score_semantic_weight_zero_ignores_embedding():
    s = DriftScorer(weights={"semantic": 0.0, "structural": 0.0, "assertion": 0.0})
    b = _fp(seed=1)
    c = _fp(seed=99)  # completely different embedding
    result = s.score([b], [c])
    assert result.drift_score == 0.0


def test_score_assertion_regression_increases_score():
    b = _fp(assertions=[{"expression": "output.is_valid_json()", "passed": True}])
    c = _fp(assertions=[{"expression": "output.is_valid_json()", "passed": False}])
    semantic_only = DriftScorer(weights={"semantic": 1.0, "structural": 0.0, "assertion": 0.0})
    full = DriftScorer(weights={"semantic": 0.5, "structural": 0.25, "assertion": 0.25})
    r_sem = semantic_only.score([b], [c])
    r_full = full.score([b], [c])
    assert r_full.drift_score > r_sem.drift_score


def test_score_format_change_increases_structural_signal():
    b = _fp(seed=0, fmt="json")
    c = _fp(seed=0, fmt="plain")  # same embedding, different format
    result = scorer.score([b], [c])
    assert result.probe_results[0].structural > 0.0


def test_score_drifted_flag_respects_threshold():
    custom = DriftScorer(threshold=0.5)
    b = _fp(seed=1)
    c = _fp(seed=99)
    result = custom.score([b], [c])
    assert result.drifted == (result.drift_score > 0.5)


def test_score_probe_ids_used_in_results():
    b = _fp()
    c = _fp()
    result = scorer.score([b, b], [c, c], probe_ids=["probe-a", "probe-b"])
    assert result.probe_results[0].probe_id == "probe-a"
    assert result.probe_results[1].probe_id == "probe-b"


def test_score_multiple_probes_averaged():
    b1, b2 = _fp(seed=1), _fp(seed=2)
    c1, c2 = _fp(seed=1), _fp(seed=99)  # first identical, second drifted
    result = scorer.score([b1, b2], [c1, c2])
    assert result.probe_results[0].drift_score < result.probe_results[1].drift_score
    assert result.drift_score == pytest.approx(
        (result.probe_results[0].drift_score + result.probe_results[1].drift_score) / 2,
        abs=1e-3,
    )


def test_report_contains_score_and_status():
    b = _fp(seed=1)
    c = _fp(seed=99)
    result = scorer.score([b], [c])
    report = result.report()
    assert "Drift score" in report
    assert ("DRIFTED" in report) or ("OK" in report)
