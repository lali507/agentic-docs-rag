"""A LlamaIndex LLM over Cohere's current Chat API.

The llama-index-llms-cohere package sends complete() and acomplete() to /v1/generate, the older
Generate API that Cohere has deprecated. Selectors and prompts in this project rely on completion
calls, so this thin class sends everything to /v2/chat instead.
"""
from __future__ import annotations

from typing import Any, Optional

import cohere
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field, PrivateAttr

_CONTEXT_WINDOW = 128_000
_MAX_OUTPUT_TOKENS = 4_000


def _text_of(response: Any) -> str:
    content = response.message.content or []
    return "".join(part.text for part in content if getattr(part, "type", "") == "text")


class CohereChatLLM(CustomLLM):
    model: str = Field(description="Cohere chat model name")
    temperature: float = 0.0
    _client: cohere.ClientV2 = PrivateAttr()
    _aclient: cohere.AsyncClientV2 = PrivateAttr()

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = cohere.ClientV2(api_key=api_key, base_url=base_url)
        self._aclient = cohere.AsyncClientV2(api_key=api_key, base_url=base_url)

    @classmethod
    def class_name(cls) -> str:
        return "CohereChatLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=_CONTEXT_WINDOW,
            num_output=_MAX_OUTPUT_TOKENS,
            model_name=self.model,
        )

    def _request(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": _MAX_OUTPUT_TOKENS,
        }

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=_text_of(self._client.chat(**self._request(prompt))))

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = await self._aclient.chat(**self._request(prompt))
        return CompletionResponse(text=_text_of(response))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        text = self.complete(prompt, formatted=True).text

        def single_chunk():
            yield CompletionResponse(text=text, delta=text)

        return single_chunk()
