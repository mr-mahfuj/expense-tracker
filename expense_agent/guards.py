"""Clean up model output and refuse anything that is not one plain SELECT or INSERT.

The database user is also limited to SELECT/INSERT (see sql/02_schema.sql), so
this is the first of two layers, not the only one.
"""

import re
from dataclasses import dataclass


class SqlRejected(Exception):
    pass


@dataclass(frozen=True)
class ValidatedSql:
    sql: str
    kind: str  # "SELECT" (includes WITH ... SELECT) or "INSERT"


# Strings, quoted identifiers and comments, matched together in one pass so
# that a '--' inside a string is not mistaken for a comment (and vice versa).
_LITERALS_AND_COMMENTS = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/", re.S)
_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.S)
_SQL_START = re.compile(r"\b(WITH|SELECT|INSERT)\b", re.I)
_FORBIDDEN_WORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|COPY|CALL|DO|EXECUTE|VACUUM|SET|LOCK)\b",
    re.I,
)
_FORBIDDEN_NAMES = re.compile(r"\b(pg_\w+|information_schema|lo_\w+|dblink\w*|set_config)\b", re.I)
_INTO = re.compile(r"\bINTO\b", re.I)


def _mask(sql: str) -> str:
    """Replace literals and comments with 'x' of equal length, so positions still line up."""
    return _LITERALS_AND_COMMENTS.sub(lambda m: "x" * len(m.group(0)), sql)


def extract_sql(raw: str) -> str:
    """Pull one bare SQL statement out of whatever the model returned.

    Small models often wrap SQL in markdown fences, add 'Here is the query:' in
    front, or explain it afterwards. All of that is removed here.
    """
    text = raw.strip()
    fenced = _FENCE.search(text)
    text = fenced.group(1).strip() if fenced else text.replace("```", "")

    start = _SQL_START.search(text)
    if start:
        text = text[start.start():]

    end = _mask(text).find(";")
    if end != -1:
        text = text[:end]
    return text.strip()


def validate(sql: str) -> ValidatedSql:
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise SqlRejected("empty statement")

    masked = _mask(cleaned)
    if ";" in masked:
        raise SqlRejected("only one statement is allowed")
    if "'" in masked or '"' in masked or "/*" in masked:
        raise SqlRejected("unbalanced quote or comment")
    if "$" in masked:
        raise SqlRejected("dollar quoting and parameters are not allowed")

    first_word = masked.split(None, 1)[0].upper()
    if first_word in ("SELECT", "WITH"):
        kind = "SELECT"
        if _INTO.search(masked):
            raise SqlRejected("SELECT ... INTO is not allowed")
    elif first_word == "INSERT":
        kind = "INSERT"
    else:
        raise SqlRejected(f"only SELECT and INSERT are allowed, got {first_word}")

    forbidden = _FORBIDDEN_WORDS.search(masked) or _FORBIDDEN_NAMES.search(masked)
    if forbidden:
        raise SqlRejected(f"forbidden keyword or function: {forbidden.group(0)}")

    return ValidatedSql(sql=cleaned, kind=kind)
