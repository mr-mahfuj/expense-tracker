from datetime import date, datetime
from zoneinfo import ZoneInfo


class Clock:
    """Single source of 'now', so the prompt, the DB session and the logs agree on the date."""

    def __init__(self, tz_name: str):
        self.tz_name = tz_name
        self._tz = ZoneInfo(tz_name)

    def now(self) -> datetime:
        return datetime.now(self._tz)

    def today(self) -> date:
        return self.now().date()
