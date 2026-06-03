from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import httpx

from llm_drift.runner import SuiteResult

logger = logging.getLogger(__name__)


@runtime_checkable
class AlertBackend(Protocol):
    async def alert(self, result: SuiteResult) -> None:
        ...


@dataclass
class SlackAlertBackend:
    webhook_url: str

    async def alert(self, result: SuiteResult) -> None:
        text = (
            f":warning: *llm-drift*: `{result.suite_name}` drifted "
            f"(score: `{result.drift_score:.3f}`)"
        )
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json={"text": text})


@dataclass
class WebhookAlertBackend:
    url: str

    async def alert(self, result: SuiteResult) -> None:
        payload = {
            "suite_name": result.suite_name,
            "drift_score": result.drift_score,
            "drifted": result.drifted,
        }
        async with httpx.AsyncClient() as client:
            await client.post(self.url, json=payload)


class AlertDispatcher:
    def __init__(self, backends: List[AlertBackend], threshold: float = 0.15):
        self.backends = backends
        self.threshold = threshold

    async def dispatch(self, result: SuiteResult) -> None:
        if result.drift_score <= self.threshold:
            return
        for backend in self.backends:
            try:
                await backend.alert(result)
            except Exception:
                logger.exception("Alert backend %r failed silently", backend)
