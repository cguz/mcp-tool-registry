"""
Climate Risk — MCP Server
=========================
Climate anomalies and drought indices for any point on Earth, computed from
the ERA5 reanalysis (ECMWF / Copernicus Climate Change Service) through the
Open-Meteo Historical Weather API. No API key required.

Tools:
    climate_anomaly — temperature and precipitation of a period compared with
                      the same calendar window in a 30-year baseline
    drought_index   — SPI and SPEI (1–24 months) plus recent hot days, with a
                      normalized risk level

Every response carries a ``provenance`` block (dataset, model, grid cell,
request URL, retrieval time, method) so an agent can cite its sources.

Usage:
    python server.py                               # HTTP transport (default)
    python server.py --transport stdio             # stdio transport
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import climate
import httpx
from dotenv import load_dotenv
from era5_store import ERA5Store, snap_to_cell
from mcp.server.fastmcp import FastMCP

_SERVER_DIR = Path(__file__).resolve().parent
load_dotenv(_SERVER_DIR / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("climate-risk-mcp")

SERVER_NAME = "Climate Risk"
SERVER_VERSION = "0.1.0"

PUBLIC_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CUSTOMER_ARCHIVE_URL = "https://customer-archive-api.open-meteo.com/v1/archive"
# Optional paid-plan key. Without it the free, non-commercial tier is used.
OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY", "").strip()
CACHE_DIR = Path(os.getenv("CLIMATE_RISK_CACHE_DIR") or Path(tempfile.gettempdir()) / "climate_risk_cache")
ERA5_MODEL = "era5"
TIMEOUT = 120
RATE_LIMIT_WAIT_S = 61
DEFAULT_BASELINE = (1991, 2020)  # WMO climatological standard normal
MIN_BASELINE_YEARS = 10
FIRST_ERA5_YEAR = 1940

mcp = FastMCP(SERVER_NAME, host="0.0.0.0", port=8000, stateless_http=True)


# ─── Data access ─────────────────────────────────────────────────────────────


def _params(lat: float, lon: float, start: date, end: date) -> dict[str, str]:
    return {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(climate.VARIABLES),
        "models": ERA5_MODEL,
        "timezone": "UTC",
    }


async def _download(lat: float, lon: float, start: date, end: date) -> tuple[dict[str, list], dict[str, Any]]:
    """One Open-Meteo archive request; waits once if the per-minute limit is hit."""
    params = _params(lat, lon, start, end)
    url = PUBLIC_ARCHIVE_URL
    if OPEN_METEO_API_KEY:
        url, params = CUSTOMER_ARCHIVE_URL, {**params, "apikey": OPEN_METEO_API_KEY}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in (1, 2):
            started = time.monotonic()
            resp = await client.get(url, params=params)
            if resp.status_code == 429 and "minutely" in resp.text.lower() and attempt == 1:
                logger.warning("Open-Meteo per-minute limit hit; retrying in %ss", RATE_LIMIT_WAIT_S)
                await asyncio.sleep(RATE_LIMIT_WAIT_S)
                continue
            break
    if resp.status_code == 429:
        raise RuntimeError(
            "Open-Meteo rate limit reached (free tier: 10,000 weighted calls/day). "
            f"Try again later or set OPEN_METEO_API_KEY. Detail: {resp.text[:200]}"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo returned {resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    logger.info("Downloaded %s..%s for cell (%.2f, %.2f) in %.1fs", start, end, lat, lon, time.monotonic() - started)
    grid_cell = {
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "elevation_m": payload.get("elevation"),
    }
    return payload["daily"], grid_cell


def _utc_today() -> date:
    """Open-Meteo rejects dates after the current UTC day, whatever the local time zone."""
    return datetime.now(UTC).date()


_store = ERA5Store(CACHE_DIR, _download, today=_utc_today)


async def _load(lat: float, lon: float, first_year: int) -> tuple[climate.Series, dict[str, Any]]:
    """ERA5 daily series for the location's grid cell plus provenance metadata."""
    series, meta = await _store.series(lat, lon, first_year)
    cell = snap_to_cell(lat, lon)
    # Reproducible public URL for the whole range; never includes the API key.
    query = urlencode(_params(*cell, date(first_year, 1, 1), _utc_today()))
    return series, {"request_url": f"{PUBLIC_ARCHIVE_URL}?{query}", "era5_cell_centre": list(cell), **meta}


def _baseline(start_year: int, end_year: int) -> range:
    if start_year < FIRST_ERA5_YEAR or end_year < start_year:
        raise ValueError(f"invalid baseline {start_year}-{end_year} (ERA5 starts in {FIRST_ERA5_YEAR})")
    if end_year - start_year + 1 < MIN_BASELINE_YEARS:
        raise ValueError(f"baseline must span at least {MIN_BASELINE_YEARS} years")
    return range(start_year, end_year + 1)


def _first_year(baseline: range, earliest_year: int | None = None) -> int:
    """One fetch range for both tools so they share the cache for a location.

    Starts two years before the baseline so 24-month accumulations are complete.
    """
    year = baseline.start - 2
    if earliest_year is not None:
        year = min(year, earliest_year)
    return max(year, FIRST_ERA5_YEAR)


def _provenance(meta: dict, lat: float, lon: float, how: str, baseline: range, method: str) -> dict:
    return {
        "dataset": "ERA5 reanalysis (ECMWF, Copernicus Climate Change Service), daily aggregates",
        "access": "Open-Meteo Historical Weather API",
        "model": ERA5_MODEL,
        "requested_location": {"latitude": round(lat, 5), "longitude": round(lon, 5), "from": how},
        **meta,
        "baseline_years": f"{baseline.start}-{baseline.stop - 1}",
        "method": method,
        "data_latency": "ERA5 is published with about 5-7 days of delay",
        "licence": "ERA5: Copernicus licence; Open-Meteo: CC BY 4.0 (attribution required)",
        "server": f"{SERVER_NAME} MCP server v{SERVER_VERSION}",
    }


def _error(message: str) -> str:
    logger.warning("Tool error: %s", message)
    return json.dumps({"error": message})


# ─── Tools ───────────────────────────────────────────────────────────────────


@mcp.tool()
async def climate_anomaly(
    start_date: str,
    end_date: str,
    latitude: float | None = None,
    longitude: float | None = None,
    geometry: dict | str | None = None,
    baseline_start_year: int = DEFAULT_BASELINE[0],
    baseline_end_year: int = DEFAULT_BASELINE[1],
) -> str:
    """
    Compare temperature and precipitation of a period with the climate normal.

    Use this to answer questions such as "was last week unusually hot in
    Madrid?" or "how much rain fell this season compared with normal?". The
    period is compared with the same calendar window in every baseline year
    (default 1991-2020, the WMO standard normal), using ERA5 reanalysis data.

    Args:
        start_date:  First day of the period, YYYY-MM-DD.
        end_date:    Last day of the period, YYYY-MM-DD (at most 366 days after
                     start_date; ERA5 lags real time by about a week).
        latitude:    Latitude in decimal degrees (WGS84). Use with longitude.
        longitude:   Longitude in decimal degrees (WGS84).
        geometry:    Alternatively, a GeoJSON Point/Polygon/Feature (object or string);
                     its centroid is used.
        baseline_start_year: First year of the reference climate (default 1991).
        baseline_end_year:   Last year of the reference climate (default 2020).

    Returns:
        JSON with, for mean temperature (°C), mean daily maximum temperature (°C)
        and total precipitation (mm): the period value, the baseline mean,
        the anomaly, a z-score, the percentile within the baseline and, for
        precipitation, the percent of normal. Includes a provenance block.
    """
    try:
        lat, lon, how = climate.resolve_location(latitude, longitude, geometry)
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if end < start:
            raise ValueError("end_date is before start_date")
        if (end - start).days > 366:
            raise ValueError("period longer than 366 days")
        baseline = _baseline(baseline_start_year, baseline_end_year)

        series, meta = await _load(lat, lon, _first_year(baseline, start.year))
        latest = climate.last_complete_day(series)
        if end > latest:
            raise ValueError(f"ERA5 data is available until {latest}; choose an earlier end_date")

        result = {
            "location": {"latitude": lat, "longitude": lon},
            "period": {"start": start.isoformat(), "end": end.isoformat(), "days": (end - start).days + 1},
            "temperature_mean_c": climate.period_anomaly(
                series, "temperature_2m_mean", start, end, baseline, "mean"
            ),
            "temperature_max_mean_c": climate.period_anomaly(
                series, "temperature_2m_max", start, end, baseline, "mean"
            ),
            "precipitation_total_mm": climate.period_anomaly(
                series, "precipitation_sum", start, end, baseline, "sum"
            ),
            "provenance": _provenance(
                meta, lat, lon, how, baseline,
                "Period mean (temperature) or sum (precipitation) compared with the same "
                "calendar window in each baseline year; percentile is the empirical rank.",
            ),
        }
        return json.dumps(result)
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the agent as JSON
        return _error(str(exc))


@mcp.tool()
async def drought_index(
    latitude: float | None = None,
    longitude: float | None = None,
    geometry: dict | str | None = None,
    end_month: str | None = None,
    scales: list[int] | None = None,
    baseline_start_year: int = DEFAULT_BASELINE[0],
    baseline_end_year: int = DEFAULT_BASELINE[1],
) -> str:
    """
    Drought indices (SPI and SPEI) and recent heat stress for a location.

    Use this to assess agricultural or hydrological drought risk for a farm,
    a plot or a city, for example before sourcing crops from it. SPI uses
    precipitation only; SPEI uses precipitation minus reference
    evapotranspiration (FAO-56), so it also captures heat-driven drying.
    Negative values are drier than normal: -1 to -1.5 moderately dry,
    -1.5 to -2 severely dry, below -2 extremely dry.

    Args:
        latitude:  Latitude in decimal degrees (WGS84). Use with longitude.
        longitude: Longitude in decimal degrees (WGS84).
        geometry:  Alternatively, a GeoJSON Point/Polygon/Feature (object or string);
                   its centroid is used.
        end_month: Last month of the accumulation window, YYYY-MM. Defaults to
                   the latest complete month available in ERA5.
        scales:    Accumulation windows in months, each 1-24 (default [3, 6]).
                   3 months reflects soil moisture / crops, 6-12 months
                   reservoirs and groundwater.
        baseline_start_year: First year of the reference climate (default 1991).
        baseline_end_year:   Last year of the reference climate (default 2020).

    Returns:
        JSON with SPI and SPEI per scale (value, index, McKee class, risk
        level), hot days in the last 90 days compared with what is normal
        there, an overall risk_level ("low", "medium" or "high") and a
        provenance block.
    """
    try:
        lat, lon, how = climate.resolve_location(latitude, longitude, geometry)
        scales = sorted(set(scales or [3, 6]))
        if any(s < 1 or s > 24 for s in scales):
            raise ValueError("each scale must be between 1 and 24 months")
        baseline = _baseline(baseline_start_year, baseline_end_year)

        series, meta = await _load(lat, lon, _first_year(baseline))

        precip = climate.monthly_totals(series, "precipitation_sum")
        balance = climate.water_balance(series)
        latest_year, latest_month = climate.last_complete_month(precip)
        if end_month:
            year, month = (int(part) for part in end_month.split("-"))
            if (year, month) > (latest_year, latest_month):
                raise ValueError(
                    f"latest complete ERA5 month is {latest_year}-{latest_month:02d}; choose an earlier end_month"
                )
        else:
            year, month = latest_year, latest_month

        spi = [climate.standardized_for_month(precip, year, month, s, baseline) for s in scales]
        spei = [climate.standardized_for_month(balance, year, month, s, baseline) for s in scales]
        heat = climate.hot_days(series, climate.last_complete_day(series), 90, baseline)

        overall = climate.worst_risk([e["risk_level"] for e in spi + spei] + [heat["risk_level"]])
        driest = min(spi + spei, key=lambda e: e["index"])
        result = {
            "location": {"latitude": lat, "longitude": lon},
            "end_month": f"{year}-{month:02d}",
            "spi": spi,
            "spei": spei,
            "hot_days_last_90": heat,
            "risk_level": overall,
            "summary": (
                f"Overall {overall} risk. Driest signal: "
                f"{'SPEI' if driest in spei else 'SPI'}-{driest['scale_months']} = {driest['index']} "
                f"({driest['class']}); {heat['hot_days']} hot days in the last 90 "
                f"(normal ≈ {heat['expected_hot_days']})."
            ),
            "provenance": _provenance(
                meta, lat, lon, how, baseline,
                "Empirical (non-parametric) SPI/SPEI: the accumulated value is ranked against "
                "the same months in each baseline year (Gringorten plotting position) and "
                "mapped to a standard normal. Classes after McKee et al. (1993).",
            ),
        }
        return json.dumps(result)
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the agent as JSON
        return _error(str(exc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Climate Risk MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="http",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")

