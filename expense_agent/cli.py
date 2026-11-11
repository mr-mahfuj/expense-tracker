import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .cache import JsonCache, NullCache
from .clock import Clock
from .config import ConfigError, Settings, load_settings
from .db import PostgresDatabase
from .llm import OllamaClient
from .logger import JsonlLogger
from .orchestrator import AgentResult, Orchestrator
from .tools import SqlExecutorTool, SqlGeneratorTool

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
    return Orchestrator(
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


def _print_result(result: AgentResult) -> None:
    print(result.answer)
    if result.sql:
        print(f"\n  SQL: {result.sql}")
    cache_note = ", cache hit" if result.from_cache else ""
    print(f"  [run {result.run_id}, {result.steps} step(s){cache_note}]\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talk to your expense tracker.")
    parser.add_argument("question", nargs="*", help="one-shot request; omit for interactive mode")
    parser.add_argument("--file", type=Path, help="read the request from a text file (good for pasted data)")
    parser.add_argument("--no-cache", action="store_true", help="skip the SQL cache for this run")
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
        result = agent.run(one_shot)
        _print_result(result)
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
        _print_result(agent.run(line))
