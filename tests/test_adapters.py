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
    block.type = "text"
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


async def test_openai_adapter_passes_temperature_zero():
    client = _mock_openai_client("hi")
    await OpenAIAdapter(model="gpt-4o", client=client).call("prompt")
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["temperature"] == 0.0


async def test_openai_adapter_raises_on_none_content():
    message = MagicMock()
    message.content = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    with pytest.raises(ProviderError):
        await OpenAIAdapter(model="gpt-4o", client=client).call("prompt")


async def test_anthropic_adapter_raises_on_no_text_block():
    tool_block = MagicMock()
    tool_block.type = "tool_use"   # not text
    response = MagicMock()
    response.content = [tool_block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    with pytest.raises(ProviderError):
        await AnthropicAdapter(model="claude-sonnet-4-6", client=client).call("prompt")


def test_custom_adapter_satisfies_protocol():
    class MyAdapter:
        async def call(self, prompt: str) -> str:
            return "custom"

    assert isinstance(MyAdapter(), ProviderAdapter)


def test_class_missing_call_does_not_satisfy_protocol():
    class BadAdapter:
        pass

    assert not isinstance(BadAdapter(), ProviderAdapter)
