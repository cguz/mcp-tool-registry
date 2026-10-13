"""
On-disk cache of daily ERA5 data, one file per ERA5 grid cell and year.

Why: the free Open-Meteo API weights a request by its time span (a request
longer than two weeks counts as several calls), so a 35-year baseline costs
about 1,000 calls of a 10,000/day budget. Past ERA5 years never change once
final, so each cell is downloaded once and afterwards only the recent years
are refreshed.

The download itself is injected (``fetch``), which keeps this module free of
network code and testable offline.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from typing import Any

from climate import VARIABLES, Series

CELL_DEG = 0.25  # ERA5 grid spacing
# ERA5T (preliminary) data is replaced by final ERA5 within about 2-3 months,
# so a year is only cached for good once fetched after 1 April of the next year.
FINAL_AFTER_MONTH = 4

# fetch(lat, lon, start, end) -> (daily, grid_cell); daily is Open-Meteo's
# {"time": [...], "<variable>": [...]} block and grid_cell describes the cell.
Fetcher = Callable[[float, float, date, date], Awaitable[tuple[dict[str, list], dict[str, Any]]]]


def snap_to_cell(lat: float, lon: float) -> tuple[float, float]:
    """Centre of the ERA5 0.25° cell containing the point."""
    # "+ 0.0" turns -0.0 into 0.0 so both map to the same cache folder.
    return round(lat / CELL_DEG) * CELL_DEG + 0.0, round(lon / CELL_DEG) * CELL_DEG + 0.0


class ERA5Store:
    def __init__(
        self,
        cache_dir: Path,
        fetch: Fetcher,
        recent_ttl_s: float = 6 * 3600,
        today: Callable[[], date] = date.today,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cache_dir = cache_dir
        self.fetch = fetch
        self.recent_ttl_s = recent_ttl_s
        self.today = today
        self.clock = clock

    def _path(self, cell: tuple[float, float], year: int) -> Path:
        return self.cache_dir / f"{cell[0]:+07.2f}_{cell[1]:+08.2f}" / f"{year}.json"

    def _load(self, cell: tuple[float, float], year: int) -> dict | None:
        path = self._path(cell, year)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        final = date.fromisoformat(entry["fetched_on"]) >= date(year + 1, FINAL_AFTER_MONTH, 1)
        if final or self.clock() - entry["fetched_at"] < self.recent_ttl_s:
            return entry
        return None

    def _save(self, cell: tuple[float, float], year: int, daily: dict[str, list], grid_cell: dict) -> dict:
        entry = {
            "fetched_at": self.clock(),
            "fetched_on": self.today().isoformat(),
            "grid_cell": grid_cell,
            "daily": daily,
        }
        path = self._path(cell, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry))
        tmp.replace(path)
        return entry

    async def series(self, lat: float, lon: float, first_year: int) -> tuple[Series, dict[str, Any]]:
        """Daily series from 1 January of first_year to today for the point's cell."""
        cell = snap_to_cell(lat, lon)
        today = self.today()
        years = list(range(first_year, today.year + 1))
        entries: dict[int, dict] = {}
        missing: list[int] = []
        for year in years:
            entry = self._load(cell, year)
            if entry is None:
                missing.append(year)
            else:
                entries[year] = entry

        # One request per run of consecutive missing years.
        runs: list[list[int]] = []
        for year in missing:
            if runs and runs[-1][-1] == year - 1:
                runs[-1].append(year)
            else:
                runs.append([year])
        for run in runs:
            start = date(run[0], 1, 1)
            end = min(date(run[-1], 12, 31), today)
            daily, grid_cell = await self.fetch(cell[0], cell[1], start, end)
            per_year: dict[int, dict[str, list]] = {y: {"time": [], **{v: [] for v in VARIABLES}} for y in run}
            for i, day in enumerate(daily["time"]):
                bucket = per_year.get(int(day[:4]))
                if bucket is None:
                    continue
                bucket["time"].append(day)
                for var in VARIABLES:
                    bucket[var].append(daily[var][i])
            for year in run:
                entries[year] = self._save(cell, year, per_year[year], grid_cell)

        series: Series = {}
        for year in years:
            daily = entries[year]["daily"]
            for i, day in enumerate(daily["time"]):
                series[date.fromisoformat(day)] = {var: daily[var][i] for var in VARIABLES}

        fetched = [entries[y]["fetched_at"] for y in years]
        meta = {
            "grid_cell": entries[years[-1]]["grid_cell"],
            "cache": {
                "years_downloaded_now": missing,
                "years_from_cache": len(years) - len(missing),
            },
            "data_retrieved_between": [
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(fetched))),
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(fetched))),
            ],
        }
        return series, meta
