"""Getting validated objects out of an LLM, and coping with free-tier rate limits."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Awaitable, Callable, Type, TypeVar

from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)
ResultT = TypeVar("ResultT")

_PROMPT = PromptTemplate("{prompt}")
_RATE_LIMIT_ATTEMPTS = 4
_RATE_LIMIT_WAIT_SECONDS = 15.0


class LLMOutputError(ValueError):
    """The LLM did not return data matching the expected schema."""


def is_rate_limit_error(error: BaseException) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "http_status", None)
    return status == 429 or type(error).__name__ == "TooManyRequestsError"


async def with_rate_limit_retry(
    call: Callable[[], Awaitable[ResultT]], wait_seconds: float = _RATE_LIMIT_WAIT_SECONDS
) -> ResultT:
    for attempt in range(1, _RATE_LIMIT_ATTEMPTS + 1):
        try:
            return await call()
        except Exception as error:  # the SDK's error types differ between versions
            if not is_rate_limit_error(error) or attempt == _RATE_LIMIT_ATTEMPTS:
                raise
            await asyncio.sleep(wait_seconds * attempt)
    raise AssertionError("unreachable")


def parse_json_model(text: str, model: Type[ModelT]) -> ModelT:
    """Parse a model from LLM text that may be wrapped in a code fence or commentary."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMOutputError("no JSON object in the LLM response")
    try:
        return model.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as error:
        raise LLMOutputError(str(error)) from error


async def predict_model(llm: LLM, model: Type[ModelT], prompt: str) -> ModelT:
    """Use LlamaIndex structured prediction, falling back to a plain JSON completion."""

    async def structured() -> ModelT:
        return await llm.astructured_predict(model, _PROMPT, prompt=prompt)

    try:
        return await with_rate_limit_retry(structured)
    except Exception as error:
        if is_rate_limit_error(error):
            raise

    schema = json.dumps(model.model_json_schema(), ensure_ascii=False)
    fallback_prompt = (
        f"{prompt}\n\nReturn only one JSON object matching this JSON schema, with no "
        f"commentary:\n{schema}"
    )

    async def plain() -> str:
        return (await llm.acomplete(fallback_prompt)).text

    return parse_json_model(await with_rate_limit_retry(plain), model)
