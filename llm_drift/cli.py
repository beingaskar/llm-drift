from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from llm_drift.models import ProbeSuite
from llm_drift.runner import SuiteResult

_EXAMPLE_CONFIG = """\
store:
  backend: sqlite
  path: .llm-drift/baselines.db

thresholds:
  drift_score: 0.15

alerts:
  - type: stdout
"""

_EXAMPLE_PROBE = """\
name: example-suite
model: gpt-4o
provider: openai
probes:
  - id: hello
    prompt: "Say hello in one sentence."
    assertions:
      - "output.min_length(5)"
"""


async def _do_run(suite: ProbeSuite) -> SuiteResult:
    """Separated so tests can patch this."""
    from llm_drift.store import SQLiteStore
    from llm_drift.runner import SuiteRunner
    from llm_drift.fingerprint import SentenceTransformerModel
    from llm_drift.adapters import OpenAIAdapter, AnthropicAdapter

    store = SQLiteStore()
    embedding_model = SentenceTransformerModel()

    if suite.provider == "anthropic":
        adapter = AnthropicAdapter(model=suite.model)
    else:
        adapter = OpenAIAdapter(model=suite.model)

    runner = SuiteRunner(suite, adapter, store, embedding_model)
    return await runner.run()


async def _do_baseline(suite: ProbeSuite) -> str:
    """Separated so tests can patch this."""
    from llm_drift.store import SQLiteStore
    from llm_drift.runner import SuiteRunner
    from llm_drift.fingerprint import SentenceTransformerModel
    from llm_drift.adapters import OpenAIAdapter, AnthropicAdapter

    store = SQLiteStore()
    embedding_model = SentenceTransformerModel()

    if suite.provider == "anthropic":
        adapter = AnthropicAdapter(model=suite.model)
    else:
        adapter = OpenAIAdapter(model=suite.model)

    runner = SuiteRunner(suite, adapter, store, embedding_model)
    return await runner.capture_baseline()


@click.group()
def cli():
    """llm-drift — detect when your LLM's behavior has silently changed."""


@cli.command()
def init():
    """Scaffold llm-drift.yaml and an example probe file."""
    config_path = Path("llm-drift.yaml")
    probe_dir = Path("probes")
    probe_path = probe_dir / "example.yaml"

    if config_path.exists():
        click.echo(f"{config_path} already exists, skipping.")
    else:
        config_path.write_text(_EXAMPLE_CONFIG)
        click.echo(f"Created {config_path}")

    probe_dir.mkdir(exist_ok=True)
    if probe_path.exists():
        click.echo(f"{probe_path} already exists, skipping.")
    else:
        probe_path.write_text(_EXAMPLE_PROBE)
        click.echo(f"Created {probe_path}")

    click.echo(
        "\nNext: edit probes/example.yaml, then run:\n"
        "  llm-drift baseline --suite probes/example.yaml"
    )


@cli.command()
@click.option("--suite", required=True, help="Path to suite YAML file.")
def baseline(suite: str):
    """Capture a baseline for a probe suite."""
    try:
        suite_obj = ProbeSuite.from_yaml(suite)
    except FileNotFoundError:
        click.echo(f"Error: suite file not found: {suite}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading suite: {e}", err=True)
        sys.exit(1)

    click.echo(f"Capturing baseline for '{suite_obj.name}'...")
    try:
        run_id = asyncio.run(_do_baseline(suite_obj))
        click.echo(f"Baseline captured. run_id={run_id}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("run")
@click.option("--suite", required=True, help="Path to suite YAML file.")
@click.option("--fail-on-drift", is_flag=True, default=False)
def run_cmd(suite: str, fail_on_drift: bool):
    """Run drift detection against the stored baseline."""
    try:
        suite_obj = ProbeSuite.from_yaml(suite)
    except FileNotFoundError:
        click.echo(f"Error: suite file not found: {suite}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading suite: {e}", err=True)
        sys.exit(1)

    try:
        result = asyncio.run(_do_run(suite_obj))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    status = "DRIFTED" if result.drifted else "OK"
    click.echo(f"Suite: {result.suite_name}  score: {result.drift_score:.3f}  [{status}]")

    if fail_on_drift and result.drifted:
        sys.exit(1)


@cli.command()
@click.option("--suite", required=True, help="Suite name (not file path).")
def report(suite: str):
    """Show drift history for a suite."""
    from llm_drift.store import SQLiteStore

    async def _fetch():
        store = SQLiteStore()
        return await store.list_results(suite)

    results = asyncio.run(_fetch())
    if not results:
        click.echo(f"No runs found for '{suite}'. Run 'llm-drift baseline' first.")
        return

    click.echo(f"{'Run ID':<38}  {'Date':<27}  {'Score':>7}  {'Status'}")
    click.echo("-" * 85)
    for r in results:
        status = "DRIFTED" if r["drifted"] else "OK"
        click.echo(f"{r['run_id']:<38}  {r['created_at']:<27}  {r['drift_score']:>7.3f}  {status}")


@cli.command()
@click.option("--suite", required=True, help="Suite name.")
@click.option("--run", "run_id", default=None, help="Run ID to diff against baseline (default: latest).")
def diff(suite: str, run_id: Optional[str]):
    """Show raw output diff between baseline and a run."""
    from llm_drift.store import SQLiteStore

    async def _fetch():
        store = SQLiteStore()
        baseline_fps = await store.load_latest(suite)
        return baseline_fps

    fps = asyncio.run(_fetch())
    if not fps:
        click.echo(f"No baseline found for '{suite}'.", err=True)
        sys.exit(1)

    for i, fp in enumerate(fps):
        click.echo(f"\n--- probe {i} baseline ---")
        click.echo(fp.raw_output or "(no raw output stored)")


def main():
    cli()
