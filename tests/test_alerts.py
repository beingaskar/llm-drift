from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from llm_drift.alerts import AlertDispatcher, AlertBackend, SlackAlertBackend, WebhookAlertBackend
from llm_drift.runner import SuiteResult
from llm_drift.scorer import DriftResult


def _result(drift_score: float, drifted: bool = None) -> SuiteResult:
    if drifted is None:
        drifted = drift_score > 0.15
    return SuiteResult(
        suite_name="my-suite",
        run_id="run-1",
        drift_score=drift_score,
        drifted=drifted,
        drift_result=DriftResult(drift_score=drift_score, drifted=drifted),
    )


def _mock_backend() -> MagicMock:
    b = MagicMock(spec=AlertBackend)
    b.alert = AsyncMock()
    return b


async def test_dispatcher_fires_when_score_exceeds_threshold():
    backend = _mock_backend()
    dispatcher = AlertDispatcher([backend], threshold=0.15)
    await dispatcher.dispatch(_result(0.20))
    backend.alert.assert_called_once()


async def test_dispatcher_silent_when_score_below_threshold():
    backend = _mock_backend()
    dispatcher = AlertDispatcher([backend], threshold=0.15)
    await dispatcher.dispatch(_result(0.10))
    backend.alert.assert_not_called()


async def test_dispatcher_silent_at_exact_threshold():
    backend = _mock_backend()
    dispatcher = AlertDispatcher([backend], threshold=0.15)
    await dispatcher.dispatch(_result(0.15))
    backend.alert.assert_not_called()


async def test_dispatcher_calls_all_backends():
    b1, b2 = _mock_backend(), _mock_backend()
    dispatcher = AlertDispatcher([b1, b2], threshold=0.10)
    await dispatcher.dispatch(_result(0.20))
    b1.alert.assert_called_once()
    b2.alert.assert_called_once()


async def test_dispatcher_continues_after_backend_failure():
    b1, b2 = _mock_backend(), _mock_backend()
    b1.alert = AsyncMock(side_effect=RuntimeError("boom"))
    dispatcher = AlertDispatcher([b1, b2], threshold=0.10)
    await dispatcher.dispatch(_result(0.20))  # must not raise
    b2.alert.assert_called_once()


async def test_slack_backend_posts_to_webhook_url():
    result = _result(0.30, drifted=True)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llm_drift.alerts.httpx.AsyncClient", return_value=mock_client):
        backend = SlackAlertBackend(webhook_url="https://hooks.slack.com/test")
        await backend.alert(result)

    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0] == "https://hooks.slack.com/test"
    assert "text" in call_kwargs[1]["json"]
    assert "my-suite" in call_kwargs[1]["json"]["text"]


async def test_webhook_backend_posts_structured_payload():
    result = _result(0.25, drifted=True)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llm_drift.alerts.httpx.AsyncClient", return_value=mock_client):
        backend = WebhookAlertBackend(url="https://example.com/alert")
        await backend.alert(result)

    payload = mock_client.post.call_args[1]["json"]
    assert payload["suite_name"] == "my-suite"
    assert payload["drift_score"] == pytest.approx(0.25)
    assert payload["drifted"] is True


def test_custom_backend_satisfies_protocol():
    class MyBackend:
        async def alert(self, result: SuiteResult) -> None:
            pass

    assert isinstance(MyBackend(), AlertBackend)
