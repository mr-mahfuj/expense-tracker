from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from psycopg import sql as pgsql


class DatabaseError(Exception):
    """Backend-neutral database error, so callers never import psycopg."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    rowcount: int
    truncated: bool


class Database(Protocol):
    """What the tools need from a database. Swap Postgres by implementing these two methods."""

    def get_schema(self) -> str: ...

    def run_sql(self, sql: str, read_only: bool) -> QueryResult: ...


_COLUMNS_QUERY = """
    SELECT c.relname,
           a.attname,
           format_type(a.atttypid, a.atttypmod),
           pg_catalog.col_description(a.attrelid, a.attnum)
    FROM pg_catalog.pg_attribute a
    JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = %s AND c.relkind = 'r' AND a.attnum > 0 AND NOT a.attisdropped
    ORDER BY c.relname, a.attnum
"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


class PostgresDatabase:
    def __init__(
        self,
        dsn: str,
        schema: str,
        sample_rows: int,
        statement_timeout_ms: int,
        max_rows: int,
        timezone: str,
    ):
        self._dsn = dsn
        self._schema = schema
        self._sample_rows = sample_rows
        self._max_rows = max_rows
        # Timeout and timezone are set per connection. The timezone matters:
        # without it CURRENT_DATE follows the server clock, which can be a day
        # off from the user's date around midnight.
        self._options = f"-c statement_timeout={statement_timeout_ms} -c timezone={timezone}"

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._dsn, options=self._options)
        except psycopg.Error as exc:
            raise DatabaseError(f"Cannot connect to Postgres: {exc}") from exc

    def get_schema(self) -> str:
        """Describe the live tables as text for the prompt: columns, types, comments, sample rows."""
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(_COLUMNS_QUERY, (self._schema,))
                tables: dict[str, list[str]] = {}
                for table, column, data_type, comment in cur.fetchall():
                    line = f"  {column} {data_type}"
                    if comment:
                        line += f"  -- {comment}"
                    tables.setdefault(table, []).append(line)

                if not tables:
                    return "(no tables exist yet)"

                parts = []
                for table, column_lines in tables.items():
                    parts.append(f"TABLE {table} (\n" + "\n".join(column_lines) + "\n)")
                    parts.append(self._sample_block(cur, table))
                return "\n".join(part for part in parts if part)
        except psycopg.Error as exc:
            raise DatabaseError(f"Could not read the schema: {exc}") from exc

    def _sample_block(self, cur: psycopg.Cursor, table: str) -> str:
        if self._sample_rows <= 0:
            return ""
        query = pgsql.SQL("SELECT * FROM {}.{} ORDER BY 1 DESC LIMIT {}").format(
            pgsql.Identifier(self._schema), pgsql.Identifier(table), pgsql.Literal(self._sample_rows)
        )
        cur.execute(query)
        columns = [d.name for d in cur.description or []]
        rows = cur.fetchall()
        if not rows:
            return ""
        lines = [f"-- sample rows from {table} ({', '.join(columns)}):"]
        lines += [f"-- {[_jsonable(v) for v in row]}" for row in rows]
        return "\n".join(lines)

    def run_sql(self, sql: str, read_only: bool) -> QueryResult:
        try:
            with self._connect() as conn:
                # Must be set before the first statement. A read-only transaction
                # means even a SELECT that smuggles in a write cannot change data.
                conn.read_only = read_only
                with conn.cursor() as cur:
                    cur.execute(sql)  # no params, so '%' in LIKE patterns is safe
                    rowcount = cur.rowcount
                    if cur.description is None:
                        return QueryResult([], [], rowcount, False)
                    columns = [d.name for d in cur.description]
                    fetched = cur.fetchmany(self._max_rows + 1)
        except psycopg.Error as exc:
            message = (exc.diag.message_primary or str(exc)).strip()
            raise DatabaseError(message) from exc

        truncated = len(fetched) > self._max_rows
        rows = [[_jsonable(v) for v in row] for row in fetched[: self._max_rows]]
        return QueryResult(columns, rows, rowcount, truncated)
