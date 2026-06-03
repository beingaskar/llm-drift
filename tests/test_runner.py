import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import List

import pytest

from llm_drift.fingerprint import Fingerprint
from llm_drift.models import Probe, ProbeSuite
from llm_drift.runner import NoBaselineError, SuiteResult, SuiteRunner
from llm_drift.store import SQLiteStore


class ConstantEmbeddingModel:
    def encode(self, text: str) -> List[float]:
        return [0.1] * 8


def _suite(n_probes: int = 3) -> ProbeSuite:
    return ProbeSuite(
        name="test-suite",
        model="gpt-4o",
        probes=[Probe(id=f"p{i}", prompt=f"prompt {i}") for i in range(n_probes)],
    )


def _adapter(response: str = "ok") -> MagicMock:
    m = MagicMock()
    m.call = AsyncMock(return_value=response)
    return m


async def test_runner_calls_provider_once_per_probe(tmp_path: Path):
    suite = _suite(3)
    adapter = _adapter()
    store = SQLiteStore(tmp_path / "db")
    runner = SuiteRunner(suite, adapter, store, ConstantEmbeddingModel())
    await runner.capture_baseline()
    assert adapter.call.call_count == 3


async def test_runner_returns_suite_result(tmp_path: Path):
    suite = _suite(2)
    store = SQLiteStore(tmp_path / "db")
    runner = SuiteRunner(suite, _adapter(), store, ConstantEmbeddingModel())
    await runner.capture_baseline()
    result = await runner.run()
    assert isinstance(result, SuiteResult)
    assert result.suite_name == "test-suite"
    assert isinstance(result.drift_score, float)
    assert len(result.drift_result.probe_results) == 2


async def test_runner_no_baseline_raises_error(tmp_path: Path):
    suite = _suite()
    store = SQLiteStore(tmp_path / "db")
    runner = SuiteRunner(suite, _adapter(), store, ConstantEmbeddingModel())
    with pytest.raises(NoBaselineError) as exc_info:
        await runner.run()
    assert "test-suite" in str(exc_info.value)


async def test_runner_capture_baseline_persists_to_store(tmp_path: Path):
    suite = _suite()
    store = SQLiteStore(tmp_path / "db")
    runner = SuiteRunner(suite, _adapter(), store, ConstantEmbeddingModel())
    await runner.capture_baseline()
    runs = await store.list_runs("test-suite")
    assert len(runs) == 1


async def test_runner_run_persists_result_to_store(tmp_path: Path):
    suite = _suite()
    store = SQLiteStore(tmp_path / "db")
    runner = SuiteRunner(suite, _adapter(), store, ConstantEmbeddingModel())
    await runner.capture_baseline()
    result = await runner.run()
    results = await store.list_results("test-suite")
    assert len(results) == 1
    assert results[0]["drift_score"] == pytest.approx(result.drift_score, abs=1e-4)


async def test_runner_concurrency_limit_respected(tmp_path: Path):
    max_concurrent = 0
    current = 0

    async def slow_call(prompt: str) -> str:
        nonlocal max_concurrent, current
        current += 1
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)
        current -= 1
        return "output"

    suite = _suite(10)
    adapter = MagicMock()
    adapter.call = slow_call
    store = SQLiteStore(tmp_path / "db")
    runner = SuiteRunner(suite, adapter, store, ConstantEmbeddingModel(), concurrency=2)
    await runner.capture_baseline()
    assert max_concurrent <= 2
