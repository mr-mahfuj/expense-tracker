import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .llm import Sampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    timezone: str
    ollama_host: str
    ollama_model: str
    ollama_num_ctx: int
    ollama_keep_alive: str
    router_sampling: Sampling
    sql_sampling: Sampling
    answer_sampling: Sampling
    database_url: str
    db_schema: str
    schema_sample_rows: int
    statement_timeout_ms: int
    max_result_rows: int
    max_rows_to_llm: int
    max_steps: int
    max_sql_retries: int
    cache_enabled: bool
    cache_path: Path
    cache_max_entries: int
    log_path: Path


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    path = Path(_str(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sampling(prefix: str, temperature: float, top_k: int, top_p: float, num_predict: int) -> Sampling:
    return Sampling(
        temperature=_float(f"{prefix}_TEMPERATURE", temperature),
        top_k=_int(f"{prefix}_TOP_K", top_k),
        top_p=_float(f"{prefix}_TOP_P", top_p),
        num_predict=_int(f"{prefix}_NUM_PREDICT", num_predict),
    )


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ConfigError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")

    return Settings(
        timezone=_str("APP_TIMEZONE", "Asia/Dhaka"),
        ollama_host=_str("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=_str("OLLAMA_MODEL", "qwen2.5-coder:7b"),
        ollama_num_ctx=_int("OLLAMA_NUM_CTX", 4096),
        ollama_keep_alive=_str("OLLAMA_KEEP_ALIVE", "30m"),
        # Router picks the next tool: deterministic. SQL writer: near-deterministic.
        # Answer writer: a little freedom so the wording is natural.
        router_sampling=_sampling("ROUTER", 0.0, 20, 0.8, 64),
        sql_sampling=_sampling("SQL", 0.1, 40, 0.9, 512),
        answer_sampling=_sampling("ANSWER", 0.5, 40, 0.9, 200),
        database_url=database_url,
        db_schema=_str("DB_SCHEMA", "public"),
        schema_sample_rows=_int("SCHEMA_SAMPLE_ROWS", 3),
        statement_timeout_ms=_int("STATEMENT_TIMEOUT_MS", 5000),
        max_result_rows=_int("MAX_RESULT_ROWS", 100),
        max_rows_to_llm=_int("MAX_ROWS_TO_LLM", 20),
        max_steps=_int("MAX_STEPS", 10),
        max_sql_retries=_int("MAX_SQL_RETRIES", 2),
        cache_enabled=_bool("CACHE_ENABLED", True),
        cache_path=_path("CACHE_PATH", "data/sql_cache.json"),
        cache_max_entries=_int("CACHE_MAX_ENTRIES", 500),
        log_path=_path("LOG_PATH", "logs/agent_log.jsonl"),
    )
