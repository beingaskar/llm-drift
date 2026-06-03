from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from llm_drift.cli import cli
from llm_drift.runner import SuiteResult
from llm_drift.scorer import DriftResult
from llm_drift.store import SQLiteStore

runner = CliRunner()

VALID_SUITE_YAML = """\
name: test-suite
model: gpt-4o
provider: openai
probes:
  - id: p1
    prompt: "Say hello."
"""


def _suite_result(drift_score: float, drifted: bool = None) -> SuiteResult:
    if drifted is None:
        drifted = drift_score > 0.15
    return SuiteResult(
        suite_name="test-suite",
        run_id="run-abc",
        drift_score=drift_score,
        drifted=drifted,
        drift_result=DriftResult(drift_score=drift_score, drifted=drifted),
    )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def test_cli_init_creates_config_file(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        result = runner.invoke(cli, ["init"])
        assert (Path(td) / "llm-drift.yaml").exists()
    assert result.exit_code == 0


def test_cli_init_creates_example_probe(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        runner.invoke(cli, ["init"])
        assert (Path(td) / "probes" / "example.yaml").exists()


def test_cli_init_skips_existing_files(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(cli, ["init"])
        result = runner.invoke(cli, ["init"])
    assert "already exists" in result.output
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_cli_run_exits_nonzero_on_drift(tmp_path: Path):
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(VALID_SUITE_YAML)

    with patch("llm_drift.cli._do_run", new=AsyncMock(return_value=_suite_result(0.40, drifted=True))):
        result = runner.invoke(cli, ["run", "--suite", str(suite_file), "--fail-on-drift"])

    assert result.exit_code == 1
    assert "DRIFTED" in result.output


def test_cli_run_exits_zero_on_no_drift(tmp_path: Path):
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(VALID_SUITE_YAML)

    with patch("llm_drift.cli._do_run", new=AsyncMock(return_value=_suite_result(0.05, drifted=False))):
        result = runner.invoke(cli, ["run", "--suite", str(suite_file), "--fail-on-drift"])

    assert result.exit_code == 0
    assert "OK" in result.output


def test_cli_run_unknown_suite_prints_error(tmp_path: Path):
    result = runner.invoke(cli, ["run", "--suite", str(tmp_path / "nonexistent.yaml")])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "Error" in result.output


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_cli_report_no_runs_prints_helpful_message():
    with patch.object(SQLiteStore, "list_results", new=AsyncMock(return_value=[])):
        result = runner.invoke(cli, ["report", "--suite", "nonexistent"])
    assert "No runs found" in result.output
    assert result.exit_code == 0


def test_cli_report_shows_runs():
    fake_results = [
        {"run_id": "abc", "created_at": "2026-06-03T10:00:00", "drift_score": 0.05, "drifted": False},
        {"run_id": "def", "created_at": "2026-06-02T10:00:00", "drift_score": 0.20, "drifted": True},
    ]
    with patch.object(SQLiteStore, "list_results", new=AsyncMock(return_value=fake_results)):
        result = runner.invoke(cli, ["report", "--suite", "test-suite"])
    assert "abc" in result.output
    assert "DRIFTED" in result.output
    assert result.exit_code == 0
