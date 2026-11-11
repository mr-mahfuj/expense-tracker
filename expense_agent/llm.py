from dataclasses import dataclass
from typing import Any, Protocol

import ollama

Messages = list[dict[str, str]]


class LLMError(Exception):
    """Raised when the model backend cannot be reached or returns an error."""


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_k: int
    top_p: float
    num_predict: int


class LLMClient(Protocol):
    """Everything the agent needs from a model backend.

    To swap Ollama for another backend, implement this one method and change
    the wiring in cli.py. Nothing else in the project talks to the model.
    """

    model: str

    def chat(
        self,
        messages: Messages,
        sampling: Sampling,
        json_schema: dict[str, Any] | None = None,
    ) -> str: ...


class OllamaClient:
    def __init__(self, model: str, host: str, num_ctx: int, keep_alive: str):
        self.model = model
        self._client = ollama.Client(host=host)
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive

    def chat(
        self,
        messages: Messages,
        sampling: Sampling,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                format=json_schema,
                options={
                    "temperature": sampling.temperature,
                    "top_k": sampling.top_k,
                    "top_p": sampling.top_p,
                    "num_predict": sampling.num_predict,
                    "num_ctx": self._num_ctx,
                },
                keep_alive=self._keep_alive,
            )
        except Exception as exc:  # network, missing model, server error: all mean "LLM unavailable"
            raise LLMError(f"Ollama call failed: {exc}") from exc
        return response.message.content or ""
