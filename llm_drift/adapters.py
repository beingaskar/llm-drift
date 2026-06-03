from __future__ import annotations

from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@runtime_checkable
class ProviderAdapter(Protocol):
    async def call(self, prompt: str) -> str:
        ...


class OpenAIAdapter:
    # temperature defaults to 0.0: drift detection needs the most deterministic
    # output the model can give, so sampling variance doesn't masquerade as drift.
    def __init__(self, model: str, client=None, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package is required: pip install llm-drift[openai]")
        return self._client

    async def call(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
        except Exception as e:
            raise ProviderError(str(e), status_code=getattr(e, "status_code", None)) from e
        if content is None:
            raise ProviderError("OpenAI returned no text content (possibly a tool/function call response).")
        return content


class AnthropicAdapter:
    def __init__(self, model: str, client=None, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic()
            except ImportError:
                raise ImportError("anthropic package is required: pip install llm-drift[anthropic]")
        return self._client

    async def call(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        except Exception as e:
            raise ProviderError(str(e), status_code=getattr(e, "status_code", None)) from e
        if not blocks:
            raise ProviderError("Anthropic returned no text content (possibly a tool-use response).")
        return "".join(blocks)
