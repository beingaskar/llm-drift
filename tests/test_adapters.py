import pytest
from unittest.mock import AsyncMock, MagicMock

from llm_drift.adapters import AnthropicAdapter, OpenAIAdapter, ProviderAdapter, ProviderError


def _mock_openai_client(content: str = "Hello!") -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _mock_anthropic_client(text: str = "Hello!") -> MagicMock:
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


async def test_openai_adapter_returns_string():
    adapter = OpenAIAdapter(model="gpt-4o", client=_mock_openai_client("Hello!"))
    result = await adapter.call("Say hello.")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_anthropic_adapter_returns_string():
    adapter = AnthropicAdapter(model="claude-sonnet-4-6", client=_mock_anthropic_client("Hi there!"))
    result = await adapter.call("Say hello.")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_adapter_propagates_api_error_openai():
    error = Exception("Internal Server Error")
    error.status_code = 500
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=error)

    with pytest.raises(ProviderError) as exc_info:
        await OpenAIAdapter(model="gpt-4o", client=client).call("Hello")

    assert exc_info.value.status_code == 500


async def test_adapter_propagates_api_error_anthropic():
    error = Exception("Internal Server Error")
    error.status_code = 500
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=error)

    with pytest.raises(ProviderError) as exc_info:
        await AnthropicAdapter(model="claude-sonnet-4-6", client=client).call("Hello")

    assert exc_info.value.status_code == 500


def test_custom_adapter_satisfies_protocol():
    class MyAdapter:
        async def call(self, prompt: str) -> str:
            return "custom"

    assert isinstance(MyAdapter(), ProviderAdapter)


def test_class_missing_call_does_not_satisfy_protocol():
    class BadAdapter:
        pass

    assert not isinstance(BadAdapter(), ProviderAdapter)
