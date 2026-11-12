"""The orchestrator: the only component that talks to the LLM about *what to do next*
and the only one that connects the tools.

Flow per question:
    LLM picks an action -> orchestrator runs the tool -> result goes back to the LLM -> repeat

Why the orchestrator fills in tool arguments itself: small models paraphrase or
drop details when they copy text (pasted data, long SQL). So the LLM only
chooses *which* tool runs next. The orchestrator supplies the original question
to generate_sql and the generated SQL to execute_sql.
"""

import json
import uuid
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cache import SqlCache
from .db import DatabaseError
from .llm import LLMClient, LLMError, Messages, Sampling
from .logger import JsonlLogger
from .relevance import REFUSAL, RequestRejected, check_relevance
from .tools import SqlExecutorTool, SqlGeneratorTool

FINAL = "final"

ROUTER_SYSTEM_PROMPT = """You control an expense-tracker agent. Choose the next action.

Actions:
{actions}
- {final}: stop and answer, once the SQL has run successfully.

Normal order: {generate} -> {execute} -> {final}.
If {execute} reports an error, choose {generate} again so the SQL gets fixed.
You do not write SQL or arguments yourself; the controller supplies them.
Reply with JSON only: {{"action": "<name>", "reason": "<short reason>"}}"""

ANSWER_SYSTEM_PROMPT = """You answer questions about the user's personal expenses.
Use ONLY the query result you are given. Never invent numbers.
Answer in one or two short sentences. If the result is empty, say no matching expenses were found."""


@dataclass
class AgentResult:
    ok: bool
    answer: str
    sql: str | None
    run_id: str
    steps: int
    from_cache: bool


@dataclass
class _RunState:
    question: str
    last_sql: str | None = None
    last_key: str | None = None
    last_from_cache: bool = False
    last_result: dict[str, Any] | None = None
    last_error: str | None = None
    retries: int = 0
    bypass_cache: bool = False
    use_state_machine_next: bool = False


def _describe_result(result: dict[str, Any], max_rows: int) -> str:
    if not result["columns"]:
        return f"OK. {result['rowcount']} row(s) affected."
    shown = result["rows"][:max_rows]
    text = f"columns={result['columns']} rows={json.dumps(shown, ensure_ascii=False)}"
    hidden = len(result["rows"]) - len(shown)
    if hidden > 0 or result["truncated"]:
        text += " (more rows exist but are not shown)"
    return text


def _format_table(result: dict[str, Any]) -> str:
    lines = [" | ".join(result["columns"])]
    lines += [" | ".join(str(value) for value in row) for row in result["rows"]]
    return "\n".join(lines)


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient,
        generator: SqlGeneratorTool,
        executor: SqlExecutorTool,
        cache: SqlCache,
        logger: JsonlLogger,
        router_sampling: Sampling,
        answer_sampling: Sampling,
        max_steps: int,
        max_sql_retries: int,
        max_rows_to_llm: int,
    ):
        self._llm = llm
        self._generator = generator
        self._executor = executor
        self._cache = cache
        self._logger = logger
        self._router_sampling = router_sampling
        self._answer_sampling = answer_sampling
        self._max_steps = max_steps
        self._max_sql_retries = max_sql_retries
        self._max_rows_to_llm = max_rows_to_llm

        self._actions = [generator.name, executor.name, FINAL]
        self._router_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": self._actions},
                "reason": {"type": "string"},
            },
            "required": ["action"],
        }
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in (generator, executor))
        self._router_prompt = ROUTER_SYSTEM_PROMPT.format(
            actions=tool_lines, final=FINAL, generate=generator.name, execute=executor.name
        )

    def run(self, question: str, on_token: Callable[[str], None] | None = None) -> AgentResult:
        """Run a request, optionally delivering final answer text incrementally."""
        run_id = uuid.uuid4().hex[:8]
        self._logger.log("question", run_id, question=question)
        state = _RunState(question=question)
        messages: Messages = [
            {"role": "system", "content": self._router_prompt},
            {"role": "user", "content": f"User request: {question}"},
        ]

        try:
            check_relevance(self._llm, question, self._router_sampling)
            self._logger.log("request_allowed", run_id, model=self._llm.model)
            for step in range(1, self._max_steps + 1):
                action, reason = self._decide(messages, state, run_id, step)
                messages.append(
                    {"role": "assistant", "content": json.dumps({"action": action, "reason": reason})}
                )

                if action == FINAL:
                    if state.last_result is None:
                        observation = self._reject_choice(
                            state, "No query has run successfully yet. Generate and execute SQL first."
                        )
                    else:
                        return self._finish(state, run_id, step, on_token)
                elif action == self._generator.name:
                    observation = self._do_generate(state, run_id)
                    if re.fullmatch(r"SELECT\s+'UNSUPPORTED'", state.last_sql or "", re.I):
                        raise RequestRejected(REFUSAL)
                else:
                    observation = self._do_execute(state, run_id)
                    if state.retries > self._max_sql_retries:
                        return self._fail(state, run_id, step, f"SQL kept failing: {state.last_error}")

                messages.append({"role": "user", "content": f"Observation: {observation}"})

            if state.last_result is not None:
                return self._finish(state, run_id, self._max_steps, on_token)
            return self._fail(state, run_id, self._max_steps, "Ran out of steps before finishing.")
        except RequestRejected as exc:
            self._logger.log("request_rejected", run_id, reason=str(exc))
            return self._fail(state, run_id, 0, str(exc))
        except (LLMError, DatabaseError) as exc:
            return self._fail(state, run_id, 0, str(exc))

    # --- decisions -------------------------------------------------------

    def _decide(self, messages: Messages, state: _RunState, run_id: str, step: int) -> tuple[str, str]:
        if state.use_state_machine_next:
            # The model's last choice was impossible. Take the one valid next
            # step instead of asking again, which prevents a loop of bad picks.
            state.use_state_machine_next = False
            action = self._default_action(state)
            self._logger.log("decision", run_id, step=step, action=action, source="state_machine")
            return action, "model choice was invalid; using the only valid next step"

        for _ in range(2):
            raw = self._llm.chat(messages, self._router_sampling, self._router_schema)
            try:
                decision = json.loads(raw)
                action = decision["action"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if action in self._actions:
                reason = str(decision.get("reason", ""))
                self._logger.log("decision", run_id, step=step, action=action, reason=reason, source="llm")
                return action, reason

        action = self._default_action(state)
        self._logger.log("decision", run_id, step=step, action=action, source="fallback_after_bad_json")
        return action, "model returned no usable decision"

    def _default_action(self, state: _RunState) -> str:
        if state.last_sql is None or state.last_error is not None:
            return self._generator.name
        if state.last_result is None:
            return self._executor.name
        return FINAL

    def _reject_choice(self, state: _RunState, message: str) -> str:
        state.use_state_machine_next = True
        return f"ERROR: {message}"

    # --- tool calls ------------------------------------------------------

    def _do_generate(self, state: _RunState, run_id: str) -> str:
        output = self._generator.run(
            state.question,
            run_id,
            previous_sql=state.last_sql if state.last_error else None,
            error=state.last_error,
            use_cache=not state.bypass_cache,
        )
        state.last_sql = output["sql"]
        state.last_key = output["cache_key"]
        state.last_from_cache = output["from_cache"]
        state.last_result = None
        state.last_error = None
        return f"SQL generated: {state.last_sql}"

    def _do_execute(self, state: _RunState, run_id: str) -> str:
        if state.last_sql is None:
            return self._reject_choice(state, "There is no SQL yet. Call generate_sql first.")

        result = self._executor.run(state.last_sql, run_id)
        if result["ok"]:
            state.last_result = result
            self._remember(state)
            return _describe_result(result, self._max_rows_to_llm)

        state.retries += 1
        state.last_error = result["error"]
        state.bypass_cache = True
        if state.last_from_cache and state.last_key:
            self._cache.delete(state.last_key)  # a cached statement that fails is a bad entry
        return f"ERROR: {state.last_error}. Call generate_sql again to fix it."

    def _remember(self, state: _RunState) -> None:
        """Cache SQL only after it has actually worked, so bad SQL is never reused."""
        if state.last_from_cache or not state.last_key or not state.last_sql:
            return
        if "UNSUPPORTED" in state.last_sql.upper():
            return
        self._cache.put(state.last_key, state.question, state.last_sql)

    # --- finishing -------------------------------------------------------

    def _finish(
        self, state: _RunState, run_id: str, steps: int,
        on_token: Callable[[str], None] | None = None,
    ) -> AgentResult:
        answer = self._compose_answer(state, on_token)
        self._logger.log("answer", run_id, answer=answer, sql=state.last_sql, steps=steps)
        return AgentResult(True, answer, state.last_sql, run_id, steps, state.last_from_cache)

    def _fail(self, state: _RunState, run_id: str, steps: int, message: str) -> AgentResult:
        self._logger.log("error", run_id, message=message, sql=state.last_sql, steps=steps)
        return AgentResult(False, message, state.last_sql, run_id, steps, state.last_from_cache)

    def _compose_answer(self, state: _RunState, on_token: Callable[[str], None] | None = None) -> str:
        result = state.last_result
        if result is None:
            return ""
        if not result["columns"]:
            text = f"Saved. {result['rowcount']} row(s) added."
            if on_token:
                on_token(text)
            return text

        messages: Messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {state.question}\n"
                    f"SQL: {state.last_sql}\n"
                    f"Result: {_describe_result(result, self._max_rows_to_llm)}"
                ),
            },
        ]
        if on_token is not None:
            chunks: list[str] = []
            try:
                for chunk in self._llm.chat_stream(messages, self._answer_sampling):
                    chunks.append(chunk)
                    on_token(chunk)
            except LLMError as exc:
                if chunks:
                    raise LLMError("Answer stream interrupted; response is incomplete. Please retry.") from exc
            text = "".join(chunks)
            if text.strip():
                return text
            fallback = _format_table(result)
            on_token(fallback)
            return fallback

        try:
            text = self._llm.chat(messages, self._answer_sampling).strip()
        except LLMError:
            text = ""
        return text or _format_table(result)
