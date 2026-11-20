"""Local earthquake catalog archive service.

The application talks to a *catalog source* through a small, well-defined
interface.  The bundled ``MockCatalogSource`` simulates the regional network's
paged-by-date endpoint (max 20 events per response).  A real deployment can
swap in another source with the same two methods.
"""

from __future__ import annotations

import abc
import datetime as dt
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Iterable, Sequence


PAGE_SIZE = 20
MAX_SPAN_DAYS = 31
UTC = dt.timezone.utc


class SourceError(Exception):
    """Raised when the upstream catalog cannot be queried."""


class StaleDayError(Exception):
    """Raised when a write targets a day whose membership changed.

    A reschedule that removes (and possibly re-adds) a day invalidates every
    collection request and backfill form that was issued before it, even when
    the day is part of the saved range again.  The day's ``epoch`` is what
    carries that information.
    """


@dataclass(frozen=True)
class Event:
    event_id: str
    utc_time: str  # canonical "YYYY-MM-DD HH:MM:SS"
    magnitude: float

    def as_tuple(self) -> tuple[str, str, float]:
        return (self.event_id, self.utc_time, self.magnitude)


def canonical_time(value: str) -> str:
    """Normalise a UTC issuance time to ``YYYY-MM-DD HH:MM:SS``."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"无法解析 UTC 时刻: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def parse_day(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"日期必须为 YYYY-MM-DD: {value!r}") from exc


class CatalogSource(abc.ABC):
    @abc.abstractmethod
    def fetch_page(self, start: dt.date, end: dt.date) -> list[Event]:
        """Return the events with start <= UTC date <= end, capped at 20.

        The upstream has no paging: a full response means "there may be more
        inside this window", which the collector resolves by narrowing dates.
        """

    def page_size(self) -> int:
        return PAGE_SIZE


class MockCatalogSource(CatalogSource):
    """Reads a deterministic JSON catalog standing in for the live endpoint.

    File layout::

        {"events": [{"id": "...", "time": "2024-01-02T03:04:05Z",
                     "magnitude": 3.2}, ...]}

    The mock is intentionally strict: it only serves windows of at most 31
    days, returns at most 20 events, and reports an error for a configured list
    of dates so retry logic can be exercised.
    """

    def __init__(self, path: str | os.PathLike[str], fail_dates: Iterable[str] = ()):
        self.path = os.fspath(path)
        self.fail_dates = set(fail_dates)
        self._lock = threading.Lock()
        self._events: list[Event] | None = None
        self._mtime: float | None = None

    def _load(self) -> list[Event]:
        mtime = os.stat(self.path).st_mtime
        with self._lock:
            if self._events is None or mtime != self._mtime:
                with open(self.path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                events = []
                for raw in payload.get("events", []):
                    events.append(
                        Event(
                            event_id=str(raw["id"]).strip(),
                            utc_time=canonical_time(raw["time"]),
                            magnitude=float(raw["magnitude"]),
                        )
                    )
                events.sort(key=lambda e: (e.utc_time, e.event_id))
                self._events = events
                self._mtime = mtime
            return self._events

    def fetch_page(self, start: dt.date, end: dt.date) -> list[Event]:
        if end < start:
            raise SourceError("日期区间无效")
        if (end - start).days + 1 > MAX_SPAN_DAYS:
            raise SourceError(f"日期区间不得超过 {MAX_SPAN_DAYS} 天")
        for day in self.fail_dates:
            d = dt.date.fromisoformat(day)
            if start <= d <= end:
                raise SourceError(f"模拟接口故障: {day}")
        events = [
            e
            for e in self._load()
            if start <= dt.date.fromisoformat(e.utc_time[:10]) <= end
        ]
        return events[:PAGE_SIZE]


def source_from_env() -> CatalogSource:
    path = os.environ.get("CATALOG_FILE", "")
    if not path:
        for candidate in ("data/catalog.json", "seed/catalog.json"):
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            path = "data/catalog.json"
    if not os.path.exists(path):
        raise RuntimeError(
            f"找不到模拟目录数据文件 {path}；用 CATALOG_FILE 指定，"
            "或运行 scripts/generate_catalog.py 生成。"
        )
    fail = [d.strip() for d in os.environ.get("MOCK_FAIL_DATES", "").split(",") if d.strip()]
    return MockCatalogSource(path, fail_dates=fail)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    epoch INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS day_status (
    run_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    status TEXT NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    checked_at TEXT,
    last_error TEXT,
    epoch INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, day),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS events (
    run_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    event_id TEXT NOT NULL,
    utc_time TEXT NOT NULL,
    magnitude REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (run_id, event_id),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(run_id, day);
"""

# Columns added after the first release; applied to databases created before
# the rescheduling feature existed.
_ADDED_COLUMNS = {
    "runs": ("epoch", "INTEGER NOT NULL DEFAULT 0"),
    "day_status": ("epoch", "INTEGER NOT NULL DEFAULT 0"),
}

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"   # day is fully archived
STATUS_BLOCKED = "blocked"     # day returned a full page; needs CSV backfill
STATUS_FAILED = "failed"       # request failed; retry later

ALLOWED_TRANSITIONS = {
    STATUS_PENDING: {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED},
    STATUS_FAILED: {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED},
    STATUS_BLOCKED: {STATUS_COMPLETE, STATUS_BLOCKED},
    STATUS_COMPLETE: set(),
}


@dataclass(frozen=True)
class RescheduleResult:
    """Outcome of a saved reschedule, expressed in UTC days."""

    start: dt.date
    end: dt.date
    epoch: int
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removed or self.added)


@dataclass(frozen=True)
class RunSnapshot:
    """A single saved state: range, per-day progress and events together."""

    run: sqlite3.Row
    days: list[sqlite3.Row]
    events: list[sqlite3.Row]

    @property
    def ready(self) -> bool:
        return bool(self.days) and all(
            row["status"] == STATUS_COMPLETE for row in self.days
        )

    def counts(self) -> dict[str, int]:
        result = {
            STATUS_PENDING: 0,
            STATUS_COMPLETE: 0,
            STATUS_BLOCKED: 0,
            STATUS_FAILED: 0,
        }
        for row in self.days:
            result[row["status"]] = result.get(row["status"], 0) + 1
        return result


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA)
            for table, (column, decl) in _ADDED_COLUMNS.items():
                existing = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table})")
                }
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            conn.commit()

    # -- runs ---------------------------------------------------------

    def create_run(self, start: dt.date, end: dt.date) -> int:
        with self._lock, self.connect() as conn:
            run_id = self._create_run(conn, start, end)
            conn.commit()
            return run_id

    @staticmethod
    def _create_run(conn: sqlite3.Connection, start: dt.date, end: dt.date) -> int:
        now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = conn.execute(
            "INSERT INTO runs(start_date, end_date, created_at, epoch) VALUES (?,?,?,0)",
            (start.isoformat(), end.isoformat(), now),
        )
        run_id = int(cur.lastrowid)
        day = start
        while day <= end:
            conn.execute(
                "INSERT INTO day_status(run_id, day, status, epoch) VALUES (?,?,?,0)",
                (run_id, day.isoformat(), STATUS_PENDING),
            )
            day += dt.timedelta(days=1)
        return run_id

    def reschedule_run(
        self, run_id: int, new_start: dt.date, new_end: dt.date
    ) -> RescheduleResult:
        """Atomically change a run's UTC range, still at most 31 consecutive days.

        * Days that stay inside the range keep their status, progress, events
          and epoch, so requests/forms issued before the reschedule remain
          valid for them.
        * Days that leave the range lose their status rows and events in the
          same transaction, freeing their event IDs.
        * New or re-added days start as ``pending`` with a fresh epoch, so no
          request or form created before the reschedule can ever write to them
          -- even if a day with the same date had been removed earlier.

        Any error rolls the whole transaction back; the previously saved range,
        progress and events remain exactly as they were.
        """
        if new_end < new_start:
            raise ValueError("结束日期不能早于开始日期")
        span = (new_end - new_start).days + 1
        if span > MAX_SPAN_DAYS:
            raise ValueError(f"一次最多选择 {MAX_SPAN_DAYS} 个连续日")
        with self._lock, self.connect() as conn:
            run = conn.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            old_rows = conn.execute(
                "SELECT day, epoch FROM day_status WHERE run_id=? ORDER BY day",
                (run_id,),
            ).fetchall()
            old_epochs = {row["day"]: row["epoch"] for row in old_rows}

            new_days: list[str] = []
            day = new_start
            while day <= new_end:
                new_days.append(day.isoformat())
                day += dt.timedelta(days=1)
            new_set = set(new_days)
            removed = sorted(d for d in old_epochs if d not in new_set)
            added = sorted(d for d in new_days if d not in old_epochs)
            retained = sorted(d for d in new_days if d in old_epochs)

            try:
                for day_text in removed:
                    conn.execute(
                        "DELETE FROM events WHERE run_id=? AND day=?",
                        (run_id, day_text),
                    )
                    conn.execute(
                        "DELETE FROM day_status WHERE run_id=? AND day=?",
                        (run_id, day_text),
                    )
                new_epoch = int(run["epoch"]) + 1 if removed or added else int(run["epoch"])
                for day_text in added:
                    conn.execute(
                        "INSERT INTO day_status(run_id, day, status, epoch) "
                        "VALUES (?,?,?,?)",
                        (run_id, day_text, STATUS_PENDING, new_epoch),
                    )
                conn.execute(
                    "UPDATE runs SET start_date=?, end_date=?, epoch=? WHERE id=?",
                    (new_start.isoformat(), new_end.isoformat(), new_epoch, run_id),
                )
            except Exception:
                conn.rollback()
                raise
            conn.commit()
        return RescheduleResult(
            start=new_start,
            end=new_end,
            epoch=new_epoch,
            removed=removed,
            added=added,
            retained=retained,
        )

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def runs(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT r.*, "
                "(SELECT COUNT(*) FROM day_status d WHERE d.run_id=r.id AND d.status='complete') AS complete_days, "
                "(SELECT COUNT(*) FROM day_status d WHERE d.run_id=r.id AND d.status='blocked') AS blocked_days "
                "FROM runs r ORDER BY r.id DESC"
            ).fetchall()

    def days(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM day_status WHERE run_id=? ORDER BY day", (run_id,)
            ).fetchall()

    def events(self, run_id: int, day: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as conn:
            if day is None:
                return conn.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY utc_time, event_id",
                    (run_id,),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM events WHERE run_id=? AND day=? ORDER BY utc_time, event_id",
                (run_id, day),
            ).fetchall()

    def run_snapshot(self, run_id: int) -> RunSnapshot | None:
        """Read range, progress and events from one committed state.

        All three views are fetched inside a single transaction so the task
        page (and every CSV download) reflects the same saved state even when a
        reschedule lands between requests.
        """
        with self.connect() as conn:
            run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                return None
            days = conn.execute(
                "SELECT * FROM day_status WHERE run_id=? ORDER BY day", (run_id,)
            ).fetchall()
            events = conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY utc_time, event_id",
                (run_id,),
            ).fetchall()
            return RunSnapshot(run=run, days=days, events=events)

    # -- progress -----------------------------------------------------

    def mark(
        self,
        run_id: int,
        day: str,
        status: str,
        last_error: str | None = None,
        epoch: int | None = None,
    ) -> None:
        if status not in {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED}:
            raise ValueError(status)
        with self._lock, self.connect() as conn:
            self._mark(conn, run_id, day, status, last_error, epoch)
            conn.commit()

    @staticmethod
    def _day_row(
        conn: sqlite3.Connection, run_id: int, day: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT status, epoch FROM day_status WHERE run_id=? AND day=?",
            (run_id, day),
        ).fetchone()

    @staticmethod
    def _check_epoch(row: sqlite3.Row | None, epoch: int | None, day: str) -> None:
        if row is None:
            raise KeyError(day)
        if epoch is not None and int(row["epoch"]) != int(epoch):
            raise StaleDayError(day)

    @staticmethod
    def _mark(
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        status: str,
        last_error: str | None = None,
        epoch: int | None = None,
    ) -> None:
        row = Store._day_row(conn, run_id, day)
        Store._check_epoch(row, epoch, day)
        current = row["status"]
        if current not in ALLOWED_TRANSITIONS or status not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"状态不允许从 {current} 变为 {status}")
        now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND day=?", (run_id, day)
        ).fetchone()[0]
        conn.execute(
            "UPDATE day_status SET status=?, event_count=?, checked_at=?, last_error=? "
            "WHERE run_id=? AND day=?",
            (status, count, now, last_error, run_id, day),
        )

    def complete_days(
        self,
        run_id: int,
        by_day: dict[str, Sequence[Event]],
        source: str = "api",
        epochs: dict[str, int] | None = None,
    ) -> list[str]:
        """Persist a whole exhausted range and mark its days complete at once.

        The write is transactional: any conflict (for example the same event ID
        showing up on two days) rolls the range back so no day is half-recorded.
        Days that were removed by a reschedule, or whose membership epoch no
        longer matches the request that fetched this data, are skipped rather
        than written.  Returns the days actually marked complete.
        """
        epochs = epochs or {}
        with self._lock, self.connect() as conn:
            written: list[str] = []
            try:
                for day, events in by_day.items():
                    row = self._day_row(conn, run_id, day)
                    if row is None:
                        continue
                    expected = epochs.get(day)
                    if expected is not None and int(row["epoch"]) != int(expected):
                        continue
                    for e in events:
                        if e.utc_time[:10] != day:
                            raise ValueError(f"事件 {e.event_id} 不在 {day}")
                        try:
                            conn.execute(
                                "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                                "VALUES (?,?,?,?,?,?)",
                                (run_id, day, e.event_id, e.utc_time, e.magnitude, source),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"事件编号冲突: {e.event_id}") from exc
                    self._mark(conn, run_id, day, STATUS_COMPLETE)
                    written.append(day)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
            return written

    def record_blocked_day(
        self,
        run_id: int,
        day: str,
        events: Sequence[Event],
        epoch: int | None = None,
    ) -> None:
        """Store the first (full) page for a day that cannot be exhausted."""
        with self._lock, self.connect() as conn:
            row = self._day_row(conn, run_id, day)
            self._check_epoch(row, epoch, day)
            try:
                for e in events:
                    try:
                        conn.execute(
                            "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                            "VALUES (?,?,?,?,?,?)",
                            (run_id, day, e.event_id, e.utc_time, e.magnitude, "api"),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"事件编号冲突: {e.event_id}") from exc
                self._mark(conn, run_id, day, STATUS_BLOCKED)
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    @staticmethod
    def _replace(
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        events: Sequence[Event],
        source: str,
    ) -> None:
        conn.execute("DELETE FROM events WHERE run_id=? AND day=?", (run_id, day))
        for e in events:
            try:
                conn.execute(
                    "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, day, e.event_id, e.utc_time, e.magnitude, source),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"事件编号已被其他日期使用: {e.event_id}") from exc

    def backfill_day(
        self,
        run_id: int,
        day: str,
        events: Sequence[Event],
        verify=None,
        epoch: int | None = None,
    ) -> None:
        """Atomically replace a blocked day's events and mark it complete.

        ``verify`` receives the already-collected events and may raise to abort;
        the whole operation rolls back in that case, preserving prior progress.
        ``epoch`` must match the day's current membership epoch, otherwise the
        upload belongs to a form that predates a reschedule which removed the
        day (even if it has since been re-added) and is rejected untouched.
        """
        with self._lock, self.connect() as conn:
            row = self._day_row(conn, run_id, day)
            self._check_epoch(row, epoch, day)
            if row["status"] != STATUS_BLOCKED:
                raise ValueError(f"{day} 不是待补档日期")
            existing = [
                Event(r["event_id"], r["utc_time"], r["magnitude"])
                for r in conn.execute(
                    "SELECT event_id, utc_time, magnitude FROM events "
                    "WHERE run_id=? AND day=? ORDER BY utc_time, event_id",
                    (run_id, day),
                )
            ]
            if verify is not None:
                verify(existing)
            self._replace(conn, run_id, day, events, "csv")
            self._mark(conn, run_id, day, STATUS_COMPLETE)
            conn.commit()

    def is_ready(self, run_id: int) -> bool:
        with self.connect() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM day_status WHERE run_id=? AND status!='complete'",
                (run_id,),
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM day_status WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            return total > 0 and bad == 0
