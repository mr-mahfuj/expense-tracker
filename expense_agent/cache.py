import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from .clock import Clock


class SqlCache(Protocol):
    def get(self, key: str) -> str | None: ...

    def put(self, key: str, question: str, sql: str) -> None: ...

    def delete(self, key: str) -> None: ...


def build_key(*parts: str) -> str:
    """Hash the inputs that decide the SQL: question, schema, model, date."""
    payload = json.dumps(parts, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_question(text: str) -> str:
    return " ".join(text.split())


class NullCache:
    """Used when caching is switched off, so callers never branch on it."""

    def get(self, key: str) -> str | None:
        return None

    def put(self, key: str, question: str, sql: str) -> None:
        pass

    def delete(self, key: str) -> None:
        pass


class JsonCache:
    """Caches generated SQL (never query results, which change as data is added)."""

    def __init__(self, path: Path, max_entries: int, clock: Clock):
        self._path = path
        self._max_entries = max_entries
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> str | None:
        entry = self._load().get(key)
        return entry["sql"] if entry else None

    def put(self, key: str, question: str, sql: str) -> None:
        entries = self._load()
        entries[key] = {
            "question": question,
            "sql": sql,
            "created_at": self._clock.now().isoformat(timespec="seconds"),
        }
        if len(entries) > self._max_entries:
            oldest_first = sorted(entries, key=lambda k: entries[k]["created_at"])
            for stale_key in oldest_first[: len(entries) - self._max_entries]:
                del entries[stale_key]
        self._save(entries)

    def delete(self, key: str) -> None:
        entries = self._load()
        if entries.pop(key, None) is not None:
            self._save(entries)

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open(encoding="utf-8") as handle:
                return json.load(handle).get("entries", {})
        except (json.JSONDecodeError, AttributeError):
            return {}  # a corrupt cache is just an empty cache

    def _save(self, entries: dict[str, Any]) -> None:
        # Write to a temp file, then rename: readers never see a half-written file.
        with tempfile.NamedTemporaryFile(
            "w", dir=self._path.parent, suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            json.dump({"entries": entries}, handle, ensure_ascii=False, indent=2)
            temp_name = handle.name
        os.replace(temp_name, self._path)
