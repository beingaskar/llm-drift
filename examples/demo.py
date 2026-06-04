"""End-to-end llm-drift demo — no API key or model download required.

Simulates a "silent model update" by swapping the adapter's responses between
runs, then shows a GOOD case (no drift) and a BAD case (drift detected + alert).

Run with:  python examples/demo.py
"""
import asyncio
import hashlib
import re
from pathlib import Path

from llm_drift.models import Probe, ProbeSuite
from llm_drift.store import SQLiteStore
from llm_drift.runner import SuiteRunner
from llm_drift.scorer import DriftScorer
from llm_drift.alerts import AlertDispatcher, StdoutAlertBackend

DB_PATH = "/tmp/llm-drift-demo/baselines.db"
THRESHOLD = 0.15


class FakeEmbeddingModel:
    """Deterministic bag-of-words hashing vector — stands in for sentence-transformers.

    Same text -> identical vector (distance 0). Different wording -> different vector.
    """
    DIM = 64

    def encode(self, text: str):
        vec = [0.0] * self.DIM
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.DIM] += 1.0
        return vec


class ScriptedAdapter:
    """A fake LLM whose answers we control — lets us simulate a silent update."""
    def __init__(self, responses: dict):
        self.responses = responses

    async def call(self, prompt: str) -> str:
        return self.responses[prompt]


# --- The suite under watch -------------------------------------------------
P1 = "Extract invoice fields as JSON: 'Invoice #1234, total $99.00'"
P2 = "Greet the user in one short sentence."

suite = ProbeSuite(
    name="demo-suite",
    model="fake-model",
    probes=[
        Probe(id="extract-json", prompt=P1, assertions=[
            "output.is_valid_json()",
            'output.contains("1234")',
        ]),
        Probe(id="greeting", prompt=P2, assertions=["output.min_length(5)"]),
    ],
)

# Behavior at baseline time (the "known-good" model).
GOOD_RESPONSES = {
    P1: '{"invoice_number": "1234", "total": "$99.00"}',
    P2: "Hello! How can I help you today?",
}

# Behavior after a silent provider update: JSON probe now returns prose,
# greeting is unchanged.
DRIFTED_RESPONSES = {
    P1: "Sure! The invoice number is 1234 and the total amount is $99.00.",
    P2: "Hello! How can I help you today?",
}


def banner(title: str):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def show(result):
    status = "DRIFTED ❌" if result.drifted else "OK ✓"
    print(f"\nSuite drift score: {result.drift_score:.3f}  (threshold {THRESHOLD})  ->  {status}")
    print("Per-probe breakdown:")
    for pr in result.drift_result.probe_results:
        print(
            f"  {pr.probe_id:<14} score={pr.drift_score:.3f}  "
            f"semantic={pr.semantic:.3f} structural={pr.structural:.3f} "
            f"assertion={pr.assertion_regression:.3f}"
        )


async def main():
    # Fresh store for a clean demo.
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(DB_PATH).unlink(missing_ok=True)

    store = SQLiteStore(DB_PATH)
    embed = FakeEmbeddingModel()
    scorer = DriftScorer(threshold=THRESHOLD)
    dispatcher = AlertDispatcher([StdoutAlertBackend()], threshold=THRESHOLD)

    # 1) Capture baseline against the known-good model.
    banner("STEP 1 — Capture baseline (model behaving normally)")
    adapter = ScriptedAdapter(GOOD_RESPONSES)
    runner = SuiteRunner(suite, adapter, store, embed, scorer=scorer)
    run_id = await runner.capture_baseline(strict=True)
    print(f"Baseline stored. run_id={run_id}")
    print("Baseline assertions all passed (strict capture succeeded).")

    # 2) GOOD CASE — model unchanged.
    banner("STEP 2 — GOOD case: run again, model unchanged")
    adapter.responses = GOOD_RESPONSES
    result = await runner.run()
    show(result)
    await dispatcher.dispatch(result)  # below threshold → no alert
    print("(no alert fired)" if not result.drifted else "")

    # 3) BAD CASE — silent update changes the JSON probe's behavior.
    banner("STEP 3 — BAD case: provider silently updated the model")
    adapter.responses = DRIFTED_RESPONSES
    result = await runner.run()
    show(result)
    print("\nAlert dispatch:")
    await dispatcher.dispatch(result)  # above threshold → alert fires


if __name__ == "__main__":
    asyncio.run(main())
