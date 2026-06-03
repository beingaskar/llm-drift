from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import List, Optional

from llm_drift.adapters import ProviderAdapter
from llm_drift.assertions import AssertionRunner
from llm_drift.fingerprint import EmbeddingModel, Fingerprint, fingerprint
from llm_drift.models import Probe, ProbeSuite
from llm_drift.scorer import DriftResult, DriftScorer
from llm_drift.store import BaselineStore


class NoBaselineError(Exception):
    pass


class BaselineAssertionError(Exception):
    """Raised when assertions fail at baseline capture time."""
    pass


@dataclass
class SuiteResult:
    suite_name: str
    run_id: str
    drift_score: float
    drifted: bool
    drift_result: DriftResult


class SuiteRunner:
    def __init__(
        self,
        suite: ProbeSuite,
        adapter: ProviderAdapter,
        store: BaselineStore,
        embedding_model: EmbeddingModel,
        concurrency: int = 5,
        scorer: Optional[DriftScorer] = None,
    ):
        self.suite = suite
        self.adapter = adapter
        self.store = store
        self.embedding_model = embedding_model
        self.concurrency = concurrency
        self.scorer = scorer or DriftScorer()
        self._assertion_runner = AssertionRunner()

    async def _run_probe(self, probe: Probe, sem: asyncio.Semaphore) -> Fingerprint:
        async with sem:
            output = await self.adapter.call(probe.prompt)
        fp = fingerprint(output, self.embedding_model)
        results = self._assertion_runner.run(output, probe.assertions)
        fp.assertion_results = [{"expression": r.expression, "passed": r.passed} for r in results]
        return fp

    async def _run_all(self) -> List[Fingerprint]:
        sem = asyncio.Semaphore(self.concurrency)
        return list(await asyncio.gather(*[self._run_probe(p, sem) for p in self.suite.probes]))

    async def capture_baseline(self, strict: bool = True) -> str:
        """Capture baseline fingerprints.

        Args:
            strict: If True (default), raise BaselineAssertionError when any
                    assertion fails at capture time — prevents storing a broken
                    baseline that would score future correct runs as regressions.
        """
        fingerprints = await self._run_all()

        if strict:
            failures = []
            for probe, fp in zip(self.suite.probes, fingerprints):
                for r in fp.assertion_results:
                    if not r["passed"]:
                        failures.append(f"  [{probe.id}] {r['expression']}")
            if failures:
                raise BaselineAssertionError(
                    "Baseline capture aborted — assertions failed on the current model output.\n"
                    "Fix the assertions or re-run once the model output meets your expectations:\n"
                    + "\n".join(failures)
                )

        return await self.store.save(self.suite.name, fingerprints)

    async def run(self) -> SuiteResult:
        baselines = await self.store.load_latest(self.suite.name)
        if baselines is None:
            raise NoBaselineError(
                f"No baseline found for suite {self.suite.name!r}. Run capture_baseline() first."
            )
        currents = await self._run_all()
        probe_ids = [p.id for p in self.suite.probes]
        drift_result = self.scorer.score(baselines, currents, probe_ids=probe_ids)
        run_id = str(uuid.uuid4())
        await self.store.save_result(
            self.suite.name, run_id, drift_result.drift_score, drift_result.drifted
        )
        return SuiteResult(
            suite_name=self.suite.name,
            run_id=run_id,
            drift_score=drift_result.drift_score,
            drifted=drift_result.drifted,
            drift_result=drift_result,
        )
