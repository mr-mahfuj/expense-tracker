import json
from pathlib import Path
from typing import Any

from .clock import Clock


class JsonlLogger:
    """Append-only JSON log: one JSON object per line, each with a timestamp.

    Append-only means a crash can never corrupt earlier entries, and the file
    can be filtered with jq or read line by line.
    """

    def __init__(self, path: Path, clock: Clock):
        self._path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, run_id: str | None = None, **fields: Any) -> None:
        record = {
            "ts": self._clock.now().isoformat(timespec="milliseconds"),
            "run_id": run_id,
            "event": event,
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
