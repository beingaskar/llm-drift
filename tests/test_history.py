from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from llm_drift.cli import cli
from llm_drift.fingerprint import Fingerprint
from llm_drift.models import Probe, ProbeSuite
from llm_drift.runner import SuiteResult, SuiteRunner
from llm_drift.scorer import DriftResult
from llm_drift.store import SQLiteStore

cli_runner = CliRunner()


def _fp(raw: str = "hello", probe_id: str = "p1") -> Fingerprint:
    return Fingerprint(embedding=[0.1] * 8, token_count=1, format="plain", raw_output=raw, probe_id=probe_id)


class ConstantModel:
    def encode(self, text: str):
        return [0.1] * 8


def _suite() -> ProbeSuite:
    return ProbeSuite(
        name="history-suite",
        model="gpt-4o",
        probes=[Probe(id="p1", prompt="Say hello.")],
    )


# ---------------------------------------------------------------------------
# Persistence after run()
# ---------------------------------------------------------------------------

async def test_run_result_persisted_after_run(tmp_path: Path):
    from unittest.mock import MagicMock
    store = SQLiteStore(tmp_path / "db")
    adapter = MagicMock()
    adapter.call = AsyncMock(return_value="hello world")
    runner = SuiteRunner(_suite(), adapter, store, ConstantModel())
    await runner.capture_baseline()
    result = await runner.run()

    results = await store.list_results("history-suite")
    assert len(results) == 1
    assert abs(results[0]["drift_score"] - result.drift_score) < 1e-4


# ---------------------------------------------------------------------------
# Report command
# ---------------------------------------------------------------------------

def test_report_shows_all_runs_for_suite():
    fake = [
        {"run_id": f"id-{i}", "created_at": f"2026-06-0{i+1}T10:00:00", "drift_score": 0.01 * i, "drifted": False}
        for i in range(5)
    ]
    with patch.object(SQLiteStore, "list_results", new=AsyncMock(return_value=fake)):
        result = cli_runner.invoke(cli, ["report", "--suite", "history-suite"])

    # All 5 run IDs appear in the output
    for i in range(5):
        assert f"id-{i}" in result.output
    assert result.exit_code == 0


def test_report_no_runs_prints_helpful_message():
    with patch.object(SQLiteStore, "list_results", new=AsyncMock(return_value=[])):
        result = cli_runner.invoke(cli, ["report", "--suite", "empty-suite"])
    assert "No runs found" in result.output
    assert "baseline" in result.output.lower()
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Diff command
# ---------------------------------------------------------------------------

def test_diff_shows_baseline_vs_latest(tmp_path: Path):
    fps = [_fp("the baseline output text", probe_id="p1")]
    run_outputs = [{"probe_id": "p1", "raw_output": "the current output text"}]
    with patch.object(SQLiteStore, "load_latest", new=AsyncMock(return_value=fps)), \
         patch.object(SQLiteStore, "load_run_outputs", new=AsyncMock(return_value=run_outputs)):
        result = cli_runner.invoke(cli, ["diff", "--suite", "history-suite"])
    assert "the baseline output text" in result.output
    assert "the current output text" in result.output
    assert result.exit_code == 0


def test_diff_no_baseline_exits_nonzero():
    with patch.object(SQLiteStore, "load_latest", new=AsyncMock(return_value=None)), \
         patch.object(SQLiteStore, "load_run_outputs", new=AsyncMock(return_value=None)):
        result = cli_runner.invoke(cli, ["diff", "--suite", "empty-suite"])
    assert result.exit_code == 1
