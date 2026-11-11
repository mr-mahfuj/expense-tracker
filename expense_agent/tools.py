"""The two tools. They never import or call each other.

Only the orchestrator passes data from one to the other: it takes the SQL that
generate_sql returned and hands it to execute_sql.
"""

import time
from typing import Any

from . import guards
from .cache import SqlCache, build_key, normalize_question
from .clock import Clock
from .db import Database, DatabaseError
from .llm import LLMClient, Sampling
from .logger import JsonlLogger

# The examples name the 'expenses' table from sql/02_schema.sql. If you change
# the table, change the examples with it; small models copy them closely.
SQL_SYSTEM_PROMPT = """You write PostgreSQL for a personal expense tracker.
Return exactly ONE SQL statement and nothing else: no markdown, no explanation.

Today is {today} (timezone {timezone}).

Rules:
- Only SELECT or INSERT. Never UPDATE, DELETE, DROP, ALTER or TRUNCATE.
- Use only the tables and columns in the schema below. Never invent names.
- "last N days" means spent_on >= CURRENT_DATE - N. "today" is CURRENT_DATE and "yesterday" is CURRENT_DATE - 1. "this month" starts at date_trunc('month', CURRENT_DATE).
- To add several expenses, write ONE INSERT with several VALUES rows. Leave out id and created_at.
- Amounts are plain numbers without currency symbols. category must be one of the values listed in the schema comments.
- If the request cannot be answered from this schema, return exactly: SELECT 'UNSUPPORTED'

Schema:
{schema}

Examples:
Request: how much did I spend on food in the last 10 days
SQL: SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE category = 'food' AND spent_on >= CURRENT_DATE - 10

Request: spending per category this month
SQL: SELECT category, SUM(amount) AS total FROM expenses WHERE spent_on >= date_trunc('month', CURRENT_DATE) GROUP BY category ORDER BY total DESC

Request: yesterday lunch 180, rickshaw 60
SQL: INSERT INTO expenses (spent_on, amount, category, note) VALUES (CURRENT_DATE - 1, 180, 'food', 'lunch'), (CURRENT_DATE - 1, 60, 'transport', 'rickshaw')
"""

RETRY_SUFFIX = """

Your previous SQL failed.
SQL: {previous_sql}
Error: {error}
Write a corrected statement."""


class SqlGeneratorTool:
    name = "generate_sql"
    description = "Write one PostgreSQL statement (SELECT or INSERT) for the user's request, using the live database schema."

    def __init__(
        self,
        llm: LLMClient,
        db: Database,
        cache: SqlCache,
        logger: JsonlLogger,
        clock: Clock,
        sampling: Sampling,
    ):
        self._llm = llm
        self._db = db
        self._cache = cache
        self._logger = logger
        self._clock = clock
        self._sampling = sampling

    def run(
        self,
        question: str,
        run_id: str,
        previous_sql: str | None = None,
        error: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        # The schema is read from the database on every call, so tables created
        # since the last question are visible. It is also part of the cache key,
        # so a schema change invalidates old entries by itself.
        schema = self._db.get_schema()
        key = build_key(
            normalize_question(question), schema, self._llm.model, self._clock.today().isoformat()
        )

        if use_cache:
            cached_sql = self._cache.get(key)
            if cached_sql:
                self._logger.log(
                    "sql_generated", run_id, question=question, sql=cached_sql, from_cache=True
                )
                return {"sql": cached_sql, "cache_key": key, "from_cache": True}

        system_prompt = SQL_SYSTEM_PROMPT.format(
            today=self._clock.today().isoformat(),
            timezone=self._clock.tz_name,
            schema=schema,
        )
        user_prompt = f"Request: {question}"
        if error:
            user_prompt += RETRY_SUFFIX.format(previous_sql=previous_sql, error=error)

        started = time.perf_counter()
        raw = self._llm.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            self._sampling,
        )
        sql = guards.extract_sql(raw)
        self._logger.log(
            "sql_generated",
            run_id,
            question=question,
            sql=sql,
            raw_output=raw,
            from_cache=False,
            retry=error is not None,
            model=self._llm.model,
            temperature=self._sampling.temperature,
            top_k=self._sampling.top_k,
            top_p=self._sampling.top_p,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return {"sql": sql, "cache_key": key, "from_cache": False}


class SqlExecutorTool:
    name = "execute_sql"
    description = "Run the most recently generated SQL on Postgres and return the rows or an error."

    def __init__(self, db: Database, logger: JsonlLogger):
        self._db = db
        self._logger = logger

    def run(self, sql: str, run_id: str) -> dict[str, Any]:
        try:
            checked = guards.validate(sql)
        except guards.SqlRejected as exc:
            self._logger.log("sql_rejected", run_id, sql=sql, reason=str(exc))
            return {"ok": False, "error": f"Rejected by SQL guard: {exc}"}

        started = time.perf_counter()
        try:
            result = self._db.run_sql(checked.sql, read_only=checked.kind == "SELECT")
        except DatabaseError as exc:
            self._logger.log(
                "sql_executed",
                run_id,
                sql=checked.sql,
                kind=checked.kind,
                ok=False,
                error=str(exc),
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
            return {"ok": False, "error": str(exc)}

        self._logger.log(
            "sql_executed",
            run_id,
            sql=checked.sql,
            kind=checked.kind,
            ok=True,
            rowcount=result.rowcount,
            truncated=result.truncated,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return {
            "ok": True,
            "kind": checked.kind,
            "columns": result.columns,
            "rows": result.rows,
            "rowcount": result.rowcount,
            "truncated": result.truncated,
        }
