"""
Local store for downloaded market data: Parquet files plus a catalog.

Layout under the data directory:

    catalog.sqlite                                 what is held, and a log of every change
    binance/candles/1m/BTC-USDT/2024-01.parquet    one file per month below 1h
    binance/candles/1h/BTC-USDT/2024.parquet       one file per year from 1h up
    binance/funding/BTC-USDT/2024.parquet          funding payments, per year

The catalog (SQLite, part of Python) has three tables:

    coverage   time ranges already downloaded for each dataset. A range is
               recorded even when the exchange had no data in it (before a
               listing, for example), so it is never asked for again.
    files      every Parquet file the store knows: rows, first and last time,
               size and modification time, and where it came from
    log        one row per event: a download, a file imported, a file found
               missing or changed, a file rejected

Before a dataset is used, its folder is compared with the catalog
(`_reconcile`). A file deleted by hand takes its period out of coverage, so
it is downloaded again. A file changed or put there by hand is checked and,
if valid, its bars count as held; if not, it is renamed to *.invalid and
the period is downloaded as usual.

Parquet files cannot be appended to, so saved rows are kept in memory and
written in batches: each affected file is read, merged with the new rows
(a newer row for the same time replaces the older one), and replaced
atomically. The catalog is updated in one transaction after the files are
written, so it never records data that is not on disk.

The directory is ~/.overfitting/data unless the OVERFITTING_DATA_DIR
environment variable or a `data_dir` argument says otherwise. It is outside
the package and the virtual environment, so it survives reinstalls and is
shared by every project on the machine.
"""
from __future__ import annotations

import os
import re
import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CANDLE_COLUMNS = ("time", "open", "high", "low", "close", "volume")
FUNDING_COLUMNS = ("time", "rate")
CATALOG = "catalog.sqlite"
FLUSH_ROWS = 500_000            # write to disk once this many rows are waiting
MONTHLY_BELOW_MS = 3_600_000    # timeframes shorter than 1h get one file per month

SCHEMA = """
CREATE TABLE IF NOT EXISTS coverage (
    exchange TEXT NOT NULL, dataset TEXT NOT NULL, symbol TEXT NOT NULL,
    start INTEGER NOT NULL, "end" INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS coverage_key ON coverage (exchange, dataset, symbol);
CREATE TABLE IF NOT EXISTS files (
    exchange TEXT NOT NULL, dataset TEXT NOT NULL, symbol TEXT NOT NULL, name TEXT NOT NULL,
    rows INTEGER NOT NULL, first_time INTEGER, last_time INTEGER,
    size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, source TEXT NOT NULL,
    PRIMARY KEY (exchange, dataset, symbol, name)
);
CREATE TABLE IF NOT EXISTS log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, event TEXT NOT NULL,
    exchange TEXT NOT NULL, dataset TEXT NOT NULL, symbol TEXT NOT NULL,
    start INTEGER, "end" INTEGER, rows INTEGER, files TEXT, version TEXT, note TEXT
);
"""


def default_dir() -> Path:
    return Path(os.environ.get("OVERFITTING_DATA_DIR") or Path.home() / ".overfitting" / "data")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("overfitting")
    except Exception:
        return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, non-overlapping ranges; touching ranges are joined."""
    out: list[list[int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _subtract(ranges: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    out = []
    for a, b in ranges:
        if b <= start or a >= end:
            out.append((a, b))
            continue
        if a < start:
            out.append((a, start))
        if b > end:
            out.append((end, b))
    return out


class Store:
    def __init__(self, data_dir: str | os.PathLike | None = None):
        import pyarrow
        import pyarrow.parquet
        self._pa, self._pq = pyarrow, pyarrow.parquet

        self.root = Path(data_dir) if data_dir is not None else default_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / CATALOG)
        self._db.executescript(SCHEMA)
        self._version = _version()

        self._rows: dict[tuple, list] = defaultdict(list)     # dataset key -> pending row arrays
        self._pending_rows = 0
        self._marked: dict[tuple, list] = defaultdict(list)   # dataset key -> ranges marked since flush
        self._coverage: dict[tuple, list] = {}                # dataset key -> ranges, incl. pending
        self._reconciled: set[tuple] = set()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Also on errors: whatever was downloaded before the failure is kept.
        self.close()

    # ---------------------------------------------------------- datasets ---

    def _folder(self, exchange: str, dataset: str, symbol: str) -> Path:
        folder = symbol.replace("/", "-")
        if dataset == "funding":
            return self.root / exchange / "funding" / folder
        kind, _, timeframe = dataset.partition(":")
        if kind != "candles" or not timeframe:
            raise ValueError(f"Unknown dataset {dataset!r}.")
        return self.root / exchange / "candles" / timeframe / folder

    @staticmethod
    def _step(dataset: str) -> int | None:
        """Bar length in ms, or None for funding (irregular times)."""
        if dataset == "funding":
            return None
        from overfitting.market.exchanges import TIMEFRAMES
        return TIMEFRAMES[dataset.partition(":")[2]]

    def _monthly(self, dataset: str) -> bool:
        step = self._step(dataset)
        return step is not None and step < MONTHLY_BELOW_MS

    @staticmethod
    def _columns(dataset: str) -> tuple:
        return FUNDING_COLUMNS if dataset == "funding" else CANDLE_COLUMNS

    @staticmethod
    def _stems(times: np.ndarray, monthly: bool) -> np.ndarray:
        """File name stem for each time: '2024-01' (monthly) or '2024' (yearly)."""
        unit = "M" if monthly else "Y"
        return times.astype("datetime64[ms]").astype(f"datetime64[{unit}]").astype(str)

    @staticmethod
    def _period(stem: str, monthly: bool) -> tuple[int, int] | None:
        """[start, end) in ms of the month or year a file covers, or None for a bad name."""
        if not re.fullmatch(r"\d{4}-\d{2}" if monthly else r"\d{4}", stem):
            return None
        unit = "M" if monthly else "Y"
        try:
            begin = np.datetime64(stem, unit)
        except ValueError:
            return None
        to_ms = lambda d: int(d.astype("datetime64[ms]").astype(np.int64))
        return to_ms(begin), to_ms(begin + 1)

    # ---------------------------------------------------------- coverage ---

    def covered(self, exchange: str, dataset: str, symbol: str) -> list[tuple[int, int]]:
        """Downloaded ranges [start, end), sorted and non-overlapping."""
        key = (exchange, dataset, symbol)
        self._reconcile_once(key)
        return list(self._coverage_of(key))

    def _coverage_of(self, key: tuple) -> list[tuple[int, int]]:
        if key not in self._coverage:
            rows = self._db.execute(
                'SELECT start, "end" FROM coverage WHERE exchange=? AND dataset=? AND symbol=? '
                "ORDER BY start", key).fetchall()
            self._coverage[key] = [(int(a), int(b)) for a, b in rows]
        return self._coverage[key]

    def missing(self, exchange: str, dataset: str, symbol: str,
                start: int, end: int) -> list[tuple[int, int]]:
        """The parts of [start, end) not downloaded yet."""
        gaps, cursor = [], start
        for a, b in self.covered(exchange, dataset, symbol):
            if b <= cursor:
                continue
            if a >= end:
                break
            if a > cursor:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < end:
            gaps.append((cursor, end))
        return gaps

    def _set_coverage(self, key: tuple, ranges: list[tuple[int, int]]) -> None:
        """Replace a dataset's coverage, in memory and in the catalog (caller commits)."""
        self._coverage[key] = _merge(ranges)
        self._db.execute("DELETE FROM coverage WHERE exchange=? AND dataset=? AND symbol=?", key)
        self._db.executemany(
            'INSERT INTO coverage (exchange, dataset, symbol, start, "end") VALUES (?, ?, ?, ?, ?)',
            [(*key, a, b) for a, b in self._coverage[key]])

    def _log(self, key: tuple, event: str, start=None, end=None, rows=None, files=None, note=None):
        self._db.execute(
            'INSERT INTO log (at, event, exchange, dataset, symbol, start, "end", rows, files, '
            "version, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), event, *key, start, end, rows, files, self._version, note))

    # ------------------------------------------------------- reconciling ---

    def _reconcile_once(self, key: tuple) -> None:
        if key not in self._reconciled:
            self._reconciled.add(key)
            self._reconcile(key)

    def _reconcile(self, key: tuple) -> None:
        """Bring the catalog in line with the files actually in the dataset's folder."""
        folder = self._folder(*key)
        on_disk = {p.name: p for p in folder.glob("*.parquet")} if folder.exists() else {}
        known = {name: (size, mtime) for name, size, mtime in self._db.execute(
            "SELECT name, size, mtime_ns FROM files WHERE exchange=? AND dataset=? AND symbol=?", key)}
        monthly = self._monthly(key[1])
        changed = False

        for name, (size, mtime) in known.items():
            path = on_disk.get(name)
            if path is not None:
                stat = path.stat()
                if stat.st_size == size and stat.st_mtime_ns == mtime:
                    continue
            # Deleted, or changed by something other than this store.
            start, end = self._period(Path(name).stem, monthly)
            self._set_coverage(key, _subtract(self._coverage_of(key), start, end))
            self._db.execute("DELETE FROM files WHERE exchange=? AND dataset=? AND symbol=? AND name=?",
                             (*key, name))
            if path is None:
                self._log(key, "missing", start, end, files=name,
                          note="file deleted; its period will be downloaded again")
            else:
                self._adopt(key, path, "changed")
            changed = True

        for name, path in on_disk.items():
            if name not in known:
                self._adopt(key, path, "import")
                changed = True
        if changed:
            self._db.commit()

    def _adopt(self, key: tuple, path: Path, event: str) -> None:
        """
        Take in a Parquet file the store did not write: check it, rewrite it in the
        store's format and count its bars as held. A file that fails the checks is
        renamed to *.invalid and never read.
        """
        monthly = self._monthly(key[1])
        columns = self._columns(key[1])
        period = self._period(path.stem, monthly)
        rows, problem = None, None
        if period is None:
            problem = f"the name must be {'YYYY-MM' if monthly else 'YYYY'}.parquet"
        else:
            try:
                rows = self._read_foreign(path, columns)
            except Exception as error:      # unreadable, wrong columns, wrong types
                problem = f"it cannot be read as {', '.join(columns)} ({error})"
            else:
                problem = self._check(rows, key[1], period)

        if problem:
            invalid = path.with_name(path.name + ".invalid")
            os.replace(path, invalid)
            self._log(key, "rejected", files=path.name, note=problem)
            warnings.warn(f"{path}: {problem}. Renamed to {invalid.name}; that period will be "
                          "downloaded instead.", stacklevel=4)
            return

        _, last = np.unique(rows[::-1, 0], return_index=True)
        rows = rows[::-1][last]                                # sorted, last row per time
        self._write_file(path, rows, columns)
        self._record_file(key, path, rows, source="imported" if event == "import" else "changed")
        step = self._step(key[1]) or 1
        start, end = int(rows[0, 0]), min(int(rows[-1, 0]) + step, period[1])
        self._set_coverage(key, self._coverage_of(key) + [(start, end)])
        self._log(key, event, start, end, rows=len(rows), files=path.name,
                  note="file added outside the store" if event == "import" else
                       "file changed outside the store; its contents were checked and kept")

    def _read_foreign(self, path: Path, columns: tuple) -> np.ndarray:
        table = self._pq.read_table(path, columns=list(columns))
        out = []
        for name in columns:
            column = table.column(name)
            if name == "time" and self._pa.types.is_timestamp(column.type):
                column = column.cast(self._pa.timestamp("ms", tz=column.type.tz)).cast(self._pa.int64())
            out.append(column.to_numpy(zero_copy_only=False).astype(float))
        return np.column_stack(out) if out[0].size else np.empty((0, len(columns)))

    def _check(self, rows: np.ndarray, dataset: str, period: tuple[int, int]) -> str | None:
        """Why a foreign file cannot be used, or None if it can."""
        if not len(rows):
            return "it has no rows"
        if not np.isfinite(rows).all():
            return "it has missing or infinite values"
        times = rows[:, 0]
        if times.min() < period[0] or times.max() >= period[1]:
            return "it has rows outside the period its name gives"
        step = self._step(dataset)
        if step is not None:
            if (times % step).any():
                return f"its times are not on {dataset.partition(':')[2]} bar boundaries"
            ohlc = rows[:, 1:5]
            if (ohlc[:, 1] < ohlc.max(axis=1)).any() or (ohlc[:, 2] > ohlc.min(axis=1)).any():
                return "some bars have high below or low above the other prices"
        return None

    # ------------------------------------------------------------- write ---

    def _save(self, key: tuple, rows: list[list]) -> None:
        self._reconcile_once(key)
        if rows:
            width = len(self._columns(key[1]))
            self._rows[key].append(np.asarray(rows, dtype=float).reshape(-1, width))
            self._pending_rows += len(rows)

    def _mark(self, key: tuple, start: int, end: int) -> None:
        self._reconcile_once(key)
        self._marked[key].append((start, end))
        self._coverage[key] = _merge(self._coverage_of(key) + [(start, end)])
        if self._pending_rows >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        """Write waiting rows to their files, then record them in the catalog."""
        written: dict[tuple, list[str]] = {}
        counts: dict[tuple, int] = {}
        for key, chunks in self._rows.items():
            rows = np.concatenate(chunks)
            counts[key] = len(rows)
            written[key] = self._write(key, rows)
        self._rows.clear()
        self._pending_rows = 0

        for key in set(written) | set(self._marked):
            ranges = self._marked.get(key, [])
            if ranges:
                self._set_coverage(key, self._coverage_of(key))
            self._log(key, "download",
                      min((a for a, _ in ranges), default=None), max((b for _, b in ranges), default=None),
                      rows=counts.get(key, 0), files=", ".join(written.get(key, [])) or None)
        self._marked.clear()
        self._db.commit()

    def _write(self, key: tuple, rows: np.ndarray) -> list[str]:
        """Merge rows into their period files; returns the file names written."""
        folder = self._folder(*key)
        folder.mkdir(parents=True, exist_ok=True)
        columns = self._columns(key[1])
        stems = self._stems(rows[:, 0].astype(np.int64), self._monthly(key[1]))
        names = []
        for stem in np.unique(stems):
            new = rows[stems == stem]
            path = folder / f"{stem}.parquet"
            if path.exists():
                new = np.concatenate([self._read_file(path, columns), new])
            _, last = np.unique(new[::-1, 0], return_index=True)
            new = new[::-1][last]                              # sorted, newest row per time
            self._write_file(path, new, columns)
            self._record_file(key, path, new, source="download")
            names.append(path.name)
        return names

    def _write_file(self, path: Path, rows: np.ndarray, columns: tuple) -> None:
        arrays = {"time": self._pa.array(rows[:, 0].astype(np.int64))}
        arrays.update({name: self._pa.array(rows[:, j]) for j, name in enumerate(columns) if j})
        tmp = path.with_name(path.name + ".tmp")
        self._pq.write_table(self._pa.table(arrays), tmp, compression="zstd")
        os.replace(tmp, path)

    def _record_file(self, key: tuple, path: Path, rows: np.ndarray, source: str) -> None:
        stat = path.stat()
        self._db.execute(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*key, path.name, len(rows), int(rows[0, 0]), int(rows[-1, 0]),
             stat.st_size, stat.st_mtime_ns, source))

    # -------------------------------------------------------------- read ---

    def _read_file(self, path: Path, columns: tuple) -> np.ndarray:
        table = self._pq.read_table(path, columns=list(columns))
        return np.column_stack([table.column(c).to_numpy().astype(float) for c in columns])

    def _read(self, key: tuple, start: int, end: int) -> np.ndarray:
        self.flush()
        self._reconcile_once(key)
        columns = self._columns(key[1])
        folder = self._folder(*key)
        if not folder.exists():
            return np.empty((0, len(columns)))
        first, last = self._stems(np.array([start, end - 1], dtype=np.int64), self._monthly(key[1]))
        names = [name for (name,) in self._db.execute(
            "SELECT name FROM files WHERE exchange=? AND dataset=? AND symbol=? ORDER BY name", key)]
        parts = [self._read_file(folder / name, columns) for name in names
                 if first <= Path(name).stem <= last]
        if not parts:
            return np.empty((0, len(columns)))
        rows = np.concatenate(parts)
        return rows[(rows[:, 0] >= start) & (rows[:, 0] < end)]

    # ----------------------------------------------------------- candles ---

    def save_candles(self, exchange: str, symbol: str, timeframe: str, rows: list[list]) -> None:
        self._save((exchange, f"candles:{timeframe}", symbol), rows)

    def mark_candles(self, exchange: str, symbol: str, timeframe: str, start: int, end: int) -> None:
        """Record [start, end) as downloaded (written to the catalog with the rows)."""
        self._mark((exchange, f"candles:{timeframe}", symbol), start, end)

    def candles(self, exchange: str, symbol: str, timeframe: str,
                start: int, end: int) -> np.ndarray:
        """Rows (time, open, high, low, close, volume) with start <= time < end, oldest first."""
        return self._read((exchange, f"candles:{timeframe}", symbol), start, end)

    # ----------------------------------------------------------- funding ---

    def save_funding(self, exchange: str, symbol: str, rows: list[list]) -> None:
        self._save((exchange, "funding", symbol), rows)

    def mark_funding(self, exchange: str, symbol: str, start: int, end: int) -> None:
        self._mark((exchange, "funding", symbol), start, end)

    def funding(self, exchange: str, symbol: str, start: int, end: int) -> np.ndarray:
        """Rows (time, rate) with start <= time < end, oldest first."""
        return self._read((exchange, "funding", symbol), start, end)

    # --------------------------------------------------------- inventory ---

    def _all_keys(self) -> list[tuple]:
        keys = set(self._db.execute("SELECT DISTINCT exchange, dataset, symbol FROM coverage"))
        keys |= set(self._db.execute("SELECT DISTINCT exchange, dataset, symbol FROM files"))
        for path in self.root.glob("*/candles/*/*/*.parquet"):
            exchange, _, timeframe, folder = path.relative_to(self.root).parts[:4]
            keys.add((exchange, f"candles:{timeframe}", folder.replace("-", "/", 1)))
        for path in self.root.glob("*/funding/*/*.parquet"):
            exchange, _, folder = path.relative_to(self.root).parts[:3]
            keys.add((exchange, "funding", folder.replace("-", "/", 1)))
        return sorted(keys)

    def inventory(self):
        """What is held: one row per exchange, dataset and symbol."""
        import pandas as pd
        self.flush()
        rows = []
        for key in self._all_keys():
            self._reconcile_once(key)
            ranges = self._coverage_of(key)
            stats = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(rows), 0), COALESCE(SUM(size), 0) FROM files "
                "WHERE exchange=? AND dataset=? AND symbol=?", key).fetchone()
            if ranges or stats[0]:
                rows.append((*key, ranges[0][0] if ranges else None, ranges[-1][1] if ranges else None,
                             len(ranges), stats[1], stats[0], stats[2] / 1e6))
        frame = pd.DataFrame(rows, columns=["exchange", "dataset", "symbol", "start", "end",
                                            "ranges", "rows", "files", "size_mb"])
        for column in ("start", "end"):
            frame[column] = pd.to_datetime(frame[column], unit="ms")
        return frame

    def history(self):
        """The log, newest first: every download, import, missing, changed or rejected file."""
        import pandas as pd
        self.flush()
        frame = pd.read_sql_query(
            'SELECT id, at, event, exchange, dataset, symbol, start, "end", rows, files, version, note '
            "FROM log ORDER BY id DESC", self._db)
        for column in ("start", "end"):
            frame[column] = pd.to_datetime(frame[column], unit="ms")
        frame["at"] = pd.to_datetime(frame["at"])
        frame["rows"] = frame["rows"].astype("Int64")        # whole numbers, blank when not counted
        return frame
