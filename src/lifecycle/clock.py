"""
Centralized Timezone-Aware Clock Abstraction for Railway-AI Lifecycle Management.

Ensures deterministic, timezone-aware time across all lifecycle operations,
preventing scattered datetime.now() calls and naive datetime bugs.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, Union


def to_aware_datetime(
    val: Union[datetime, str, int, float],
    default_tz: timezone = timezone.utc,
) -> datetime:
    """
    Normalizes a value into a timezone-aware datetime object.

    - Aware datetime: returned as-is (preserves timezone).
    - Naive datetime: normalized to default_tz (default: UTC).
    - ISO format string: parsed. If naive, normalized to default_tz.
    - Timestamp (int/float): converted to timezone-aware datetime.
    - Invalid types/strings: raises ValueError.
    """
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=default_tz)
        return val

    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(float(val), tz=default_tz)

    if isinstance(val, str):
        val_clean = val.strip()
        if not val_clean:
            raise ValueError("Datetime string cannot be empty")
        try:
            # Handle trailing 'Z' if present
            if val_clean.endswith("Z"):
                val_clean = val_clean[:-1] + "+00:00"
            dt = datetime.fromisoformat(val_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=default_tz)
            return dt
        except Exception as e:
            raise ValueError(f"Unable to parse datetime string '{val}': {e}") from e

    raise ValueError(f"Unsupported datetime type: {type(val)} ({val})")


class Clock(ABC):
    """Abstract interface for system clock."""

    @abstractmethod
    def now(self) -> datetime:
        """Returns the current timezone-aware timestamp."""
        pass


class SystemClock(Clock):
    """Production clock obtaining live system time with timezone awareness."""

    def __init__(self, tz: timezone = timezone.utc):
        self.tz = tz

    def now(self) -> datetime:
        return datetime.now(self.tz)


class MockClock(Clock):
    """
    Deterministic clock for testing and simulation.
    Allows exact time setting and advancing.
    """

    def __init__(self, initial_time: Optional[Union[datetime, str]] = None, tz: timezone = timezone.utc):
        self.tz = tz
        if initial_time is None:
            self._current_time = datetime(2026, 9, 20, 8, 0, 0, tzinfo=self.tz)
        else:
            self._current_time = to_aware_datetime(initial_time, default_tz=self.tz)

    def now(self) -> datetime:
        return self._current_time

    def set_time(self, new_time: Union[datetime, str, int, float]):
        """Explicitly sets the clock time."""
        self._current_time = to_aware_datetime(new_time, default_tz=self.tz)

    def advance(self, duration: Union[timedelta, int, float]):
        """
        Advances the clock by a timedelta or number of seconds/minutes.
        If int/float is passed, it represents seconds.
        """
        if isinstance(duration, (int, float)):
            delta = timedelta(seconds=float(duration))
        elif isinstance(duration, timedelta):
            delta = duration
        else:
            raise TypeError(f"duration must be timedelta, int, or float, got {type(duration)}")
        self._current_time += delta


# Global default clock instance
_DEFAULT_CLOCK: Clock = SystemClock()


def get_default_clock() -> Clock:
    """Returns the current process-wide default clock."""
    return _DEFAULT_CLOCK


def set_default_clock(clock: Clock):
    """Sets the process-wide default clock."""
    global _DEFAULT_CLOCK
    if not isinstance(clock, Clock):
        raise TypeError("clock must implement the Clock interface")
    _DEFAULT_CLOCK = clock
