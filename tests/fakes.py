"""Offline stand-ins for Cohere, so that the tests need no key and no network."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM
from pydantic import Field


class FakeLLM(CustomLLM):
    """Answers through a handler, and records every prompt it receives."""

    handler: Callable[[str], str] = Field(exclude=True)
    prompts: list[str] = Field(default_factory=list, exclude=True)

    @property
    def metadata(self):
        from llama_index.core.base.llms.types import LLMMetadata

        return LLMMetadata(context_window=100000, num_output=2000, model_name="fake")

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text=self.handler(prompt))

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        raise NotImplementedError


class HashEmbedding(BaseEmbedding):
    """A bag of hashed words: similar text gets a similar vector, deterministically."""

    size: int = 1024

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in re.findall(r"\w+", text.casefold()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)
