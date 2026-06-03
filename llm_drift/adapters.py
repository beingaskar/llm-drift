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
    def __init__(self, model: str, client=None):
        self.model = model
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
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ProviderError(str(e), status_code=getattr(e, "status_code", None)) from e


class AnthropicAdapter:
    def __init__(self, model: str, client=None):
        self.model = model
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
            )
            return response.content[0].text
        except Exception as e:
            raise ProviderError(str(e), status_code=getattr(e, "status_code", None)) from e
