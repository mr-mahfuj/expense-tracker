import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import relevance
from .cache import JsonCache, NullCache, ResponseCache, build_key
from .clock import Clock
from .config import ConfigError, Settings, load_settings
from .db import PostgresDatabase
from .llm import OllamaClient
from .logger import JsonlLogger
from .orchestrator import ANSWER_SYSTEM_PROMPT, AgentResult, Orchestrator
from .tools import SQL_SYSTEM_PROMPT, SqlExecutorTool, SqlGeneratorTool

EXIT_WORDS = {"exit", "quit", ":q"}


def build_agent(settings: Settings) -> Orchestrator:
    """Wire everything together. This is the only place that names concrete backends."""
    clock = Clock(settings.timezone)
    logger = JsonlLogger(settings.log_path, clock)
    llm = OllamaClient(
        model=settings.ollama_model,
        host=settings.ollama_host,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )
    db = PostgresDatabase(
        dsn=settings.database_url,
        schema=settings.db_schema,
        sample_rows=settings.schema_sample_rows,
        statement_timeout_ms=settings.statement_timeout_ms,
        max_rows=settings.max_result_rows,
        timezone=settings.timezone,
    )
    cache = (
        JsonCache(settings.cache_path, settings.cache_max_entries, clock)
        if settings.cache_enabled
        else NullCache()
    )
    generator = SqlGeneratorTool(llm, db, cache, logger, clock, settings.sql_sampling)
    executor = SqlExecutorTool(db, logger)
    agent = Orchestrator(
        llm=llm,
        generator=generator,
        executor=executor,
        cache=cache,
        logger=logger,
        router_sampling=settings.router_sampling,
        answer_sampling=settings.answer_sampling,
        max_steps=settings.max_steps,
        max_sql_retries=settings.max_sql_retries,
        max_rows_to_llm=settings.max_rows_to_llm,
    )
    # Hash configuration locally: cache hits need no schema/database inspection.
    # Do not persist credentials; only this one-way namespace hash is used.
    configuration = asdict(settings)
    for name in ("cache_enabled", "log_path", "cache_max_entries"):
        configuration.pop(name)
    namespace = build_key(json.dumps(configuration, sort_keys=True, default=str),
                          agent._router_prompt, SQL_SYSTEM_PROMPT, ANSWER_SYSTEM_PROMPT,
                          relevance.SCOPE_PROMPT, json.dumps(relevance.SCOPE_SCHEMA, sort_keys=True))
    response_path = settings.cache_path.with_name(settings.cache_path.stem + "_responses.json")
    agent.configure_response_cache(
        ResponseCache(response_path, settings.cache_max_entries, clock), namespace,
        enabled=settings.cache_enabled,
    )
    return agent


def _print_result(result: AgentResult, answer_printed: bool = False) -> None:
    if not answer_printed:
        print(result.answer)
    if result.sql:
        print(f"\n  SQL: {result.sql}")
    cache_note = ", cache hit" if result.from_cache else ""
    print(f"  [run {result.run_id}, {result.steps} step(s){cache_note}]\n")


def _run_request(agent: Orchestrator, question: str, stream: bool) -> AgentResult:
    printed = False

    def emit(chunk: str) -> None:
        nonlocal printed
        if chunk:
            print(chunk, end="", flush=True)
            printed = True

    result = agent.run(question, on_token=emit) if stream else agent.run(question)
    if printed:
        print()
    # Failures after partial output must still show the error message.
    _print_result(result, answer_printed=printed and result.ok)
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talk to your expense tracker.")
    parser.add_argument("question", nargs="*", help="one-shot request; omit for interactive mode")
    parser.add_argument("--file", type=Path, help="read the request from a text file (good for pasted data)")
    parser.add_argument("--no-cache", action="store_true", help="skip SQL and response caches for this run")
    parser.add_argument("--no-stream", action="store_true", help="print the complete answer at once")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if args.no_cache:
        settings = replace(settings, cache_enabled=False)

    agent = build_agent(settings)

    one_shot = args.file.read_text(encoding="utf-8").strip() if args.file else " ".join(args.question)
    if one_shot:
        result = _run_request(agent, one_shot, stream=not args.no_stream)
        return 0 if result.ok else 1

    print(f"Expense agent ({settings.ollama_model}). Type a request, or 'exit'.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            return 0
        _run_request(agent, line, stream=not args.no_stream)
