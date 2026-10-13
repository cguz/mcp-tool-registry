"""
Climate statistics for the Climate Risk MCP server.

Pure functions only (no network, no MCP), so they can be unit-tested offline.
A daily series is a ``dict[date, dict[str, float | None]]`` keyed by the
variable names in ``VARIABLES``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, timedelta
from statistics import NormalDist, mean, pstdev
from typing import Any

VARIABLES = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
)

Series = dict[date, dict[str, float | None]]

# McKee et al. (1993) classes, shared by SPI and SPEI: (lower bound, label, risk).
_CLASSES = (
    (2.0, "extremely wet", "low"),
    (1.5, "very wet", "low"),
    (1.0, "moderately wet", "low"),
    (-1.0, "near normal", "low"),
    (-1.5, "moderately dry", "medium"),
    (-2.0, "severely dry", "high"),
)
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


# ─── Location ────────────────────────────────────────────────────────────────


def _positions(obj: Any) -> Iterable[tuple[float, float]]:
    """Yield every (lon, lat) position found in a GeoJSON object."""
    if isinstance(obj, dict):
        kind = obj.get("type")
        if kind == "FeatureCollection":
            for feature in obj.get("features", []):
                yield from _positions(feature)
        elif kind == "Feature":
            yield from _positions(obj.get("geometry"))
        elif kind == "GeometryCollection":
            for geom in obj.get("geometries", []):
                yield from _positions(geom)
        elif "coordinates" in obj:
            yield from _positions(obj["coordinates"])
    elif isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
            yield float(obj[0]), float(obj[1])
        else:
            for item in obj:
                yield from _positions(item)


def geometry_centroid(geometry: str | dict) -> tuple[float, float]:
    """Return (lat, lon) as the mean of all vertices of a GeoJSON geometry.

    Good enough for farm-sized plots: ERA5 cells are ~25 km wide, so the exact
    centroid formula does not change which cell is selected.
    """
    obj = json.loads(geometry) if isinstance(geometry, str) else geometry
    points = list(_positions(obj))
    if not points:
        raise ValueError("geometry has no coordinates")
    lon = mean(p[0] for p in points)
    lat = mean(p[1] for p in points)
    return lat, lon


def resolve_location(
    latitude: float | None,
    longitude: float | None,
    geometry: str | dict | None,
) -> tuple[float, float, str]:
    """Return (lat, lon, how) from either a lat/lon pair or a GeoJSON geometry."""
    if geometry:
        lat, lon = geometry_centroid(geometry)
        how = "geometry centroid"
    elif latitude is not None and longitude is not None:
        lat, lon, how = float(latitude), float(longitude), "point"
    else:
        raise ValueError("provide latitude and longitude, or a GeoJSON geometry")
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"coordinates out of range: lat={lat}, lon={lon}")
    return lat, lon, how


# ─── Dates ───────────────────────────────────────────────────────────────────


def shift_years(d: date, years: int) -> date:
    """Move a date by whole years; 29 February becomes 28 February."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def last_complete_day(series: Series, var: str = "precipitation_sum") -> date:
    """Latest day with a value (ERA5 arrives with a few days of delay)."""
    days = [d for d, v in series.items() if v.get(var) is not None]
    if not days:
        raise ValueError("no data in series")
    return max(days)


def window_values(series: Series, var: str, start: date, end: date) -> list[float] | None:
    """Daily values from start to end inclusive, or None if any day is missing."""
    out: list[float] = []
    day = start
    while day <= end:
        value = series.get(day, {}).get(var)
        if value is None:
            return None
        out.append(value)
        day += timedelta(days=1)
    return out


# ─── Statistics ──────────────────────────────────────────────────────────────


def empirical_percentile(value: float, baseline: list[float]) -> float:
    """Percentile rank (0–100) of value within the baseline; ties count half."""
    below = sum(1 for b in baseline if b < value)
    equal = sum(1 for b in baseline if b == value)
    return 100.0 * (below + 0.5 * equal) / len(baseline)


def standardized_index(value: float, baseline: list[float]) -> float:
    """Non-parametric standardized index (empirical SPI/SPEI).

    Ranks the value within the baseline with the Gringorten plotting position
    and maps that probability through the inverse standard normal. With a
    30-year baseline the index is bounded to about ±2.3, which still spans
    every McKee class.
    """
    if len(baseline) < 10:
        raise ValueError("baseline needs at least 10 values")
    n = len(baseline) + 1
    rank = 1 + sum(1 for b in baseline if b < value) + 0.5 * sum(1 for b in baseline if b == value)
    p = (rank - 0.44) / (n + 0.12)
    return NormalDist().inv_cdf(p)


def classify_index(index: float) -> tuple[str, str]:
    """Return (McKee class, risk level) for an SPI/SPEI value."""
    for lower, label, risk in _CLASSES:
        if index >= lower:
            return label, risk
    return "extremely dry", "high"


def worst_risk(levels: Iterable[str]) -> str:
    return max(levels, key=_RISK_ORDER.__getitem__, default="low")


# ─── Monthly accumulations (SPI / SPEI) ──────────────────────────────────────


def monthly_totals(series: Series, var: str) -> dict[tuple[int, int], float]:
    """Monthly sums, only for months where every day has a value."""
    sums: dict[tuple[int, int], float] = {}
    counts: dict[tuple[int, int], int] = {}
    bad: set[tuple[int, int]] = set()
    for day, values in series.items():
        key = (day.year, day.month)
        value = values.get(var)
        if value is None:
            bad.add(key)
            continue
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
    complete = {}
    for (year, month), total in sums.items():
        days_in_month = ((date(year + month // 12, month % 12 + 1, 1)) - date(year, month, 1)).days
        if (year, month) not in bad and counts[(year, month)] == days_in_month:
            complete[(year, month)] = total
    return complete


def water_balance(series: Series) -> dict[tuple[int, int], float]:
    """Monthly precipitation minus FAO-56 reference evapotranspiration (mm)."""
    precip = monthly_totals(series, "precipitation_sum")
    et0 = monthly_totals(series, "et0_fao_evapotranspiration")
    return {k: precip[k] - et0[k] for k in precip.keys() & et0.keys()}


def _month_back(year: int, month: int, back: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) - back
    return index // 12, index % 12 + 1


def accumulate(monthly: dict[tuple[int, int], float], year: int, month: int, scale: int) -> float | None:
    """Sum of the `scale` months ending at (year, month), or None if incomplete."""
    total = 0.0
    for back in range(scale):
        key = _month_back(year, month, back)
        if key not in monthly:
            return None
        total += monthly[key]
    return total


def last_complete_month(monthly: dict[tuple[int, int], float]) -> tuple[int, int]:
    if not monthly:
        raise ValueError("no complete month in series")
    return max(monthly)


def standardized_for_month(
    monthly: dict[tuple[int, int], float],
    year: int,
    month: int,
    scale: int,
    baseline_years: range,
) -> dict[str, Any]:
    """SPI/SPEI-style index for the `scale`-month window ending at (year, month)."""
    current = accumulate(monthly, year, month, scale)
    if current is None:
        raise ValueError(f"no complete {scale}-month data ending {year}-{month:02d}")
    baseline = [
        v for y in baseline_years if (v := accumulate(monthly, y, month, scale)) is not None
    ]
    index = standardized_index(current, baseline)
    label, risk = classify_index(index)
    return {
        "scale_months": scale,
        "period": f"{_month_back(year, month, scale - 1)[0]}-{_month_back(year, month, scale - 1)[1]:02d}"
        f" to {year}-{month:02d}",
        "value_mm": round(current, 1),
        "baseline_median_mm": round(sorted(baseline)[len(baseline) // 2], 1),
        "index": round(index, 2),
        "class": label,
        "risk_level": risk,
        "baseline_samples": len(baseline),
    }


# ─── Period anomalies ────────────────────────────────────────────────────────


def period_anomaly(
    series: Series,
    var: str,
    start: date,
    end: date,
    baseline_years: range,
    aggregate: str,
) -> dict[str, Any]:
    """Compare one period with the same calendar window in each baseline year.

    aggregate is "mean" (temperatures) or "sum" (precipitation).
    """
    agg = mean if aggregate == "mean" else sum
    values = window_values(series, var, start, end)
    if values is None:
        raise ValueError(f"no complete {var} data between {start} and {end}")
    current = agg(values)

    baseline = []
    for year in baseline_years:
        shift = year - start.year
        window = window_values(series, var, shift_years(start, shift), shift_years(end, shift))
        if window is not None:
            baseline.append(agg(window))
    if len(baseline) < 10:
        raise ValueError("not enough baseline years with complete data")

    normal = mean(baseline)
    spread = pstdev(baseline)
    result = {
        "value": round(current, 2),
        "baseline_mean": round(normal, 2),
        "anomaly": round(current - normal, 2),
        "z_score": round((current - normal) / spread, 2) if spread else None,
        "percentile": round(empirical_percentile(current, baseline), 1),
        "baseline_min": round(min(baseline), 2),
        "baseline_max": round(max(baseline), 2),
        "baseline_samples": len(baseline),
    }
    if aggregate == "sum":
        result["percent_of_normal"] = round(100.0 * current / normal, 1) if normal else None
    return result


def hot_days(
    series: Series,
    end: date,
    days: int,
    baseline_years: range,
    percentile: float = 90.0,
) -> dict[str, Any]:
    """Count days whose maximum temperature exceeds the local baseline percentile.

    The threshold is the given percentile of all daily maxima in the same
    calendar window across the baseline years, so "hot" is relative to what is
    normal at this place and season.
    """
    start = end - timedelta(days=days - 1)
    current = window_values(series, "temperature_2m_max", start, end)
    if current is None:
        raise ValueError(f"no complete temperature data between {start} and {end}")
    pool: list[float] = []
    for year in baseline_years:
        shift = year - end.year
        window = window_values(
            series, "temperature_2m_max", shift_years(start, shift), shift_years(end, shift)
        )
        if window is not None:
            pool.extend(window)
    if not pool:
        raise ValueError("no baseline temperature data")
    pool.sort()
    threshold = pool[min(len(pool) - 1, int(len(pool) * percentile / 100.0))]
    count = sum(1 for v in current if v > threshold)
    expected = days * (100.0 - percentile) / 100.0
    ratio = count / expected if expected else 0.0
    risk = "high" if ratio >= 3 else "medium" if ratio >= 2 else "low"
    return {
        "period": f"{start} to {end}",
        "threshold_c": round(threshold, 1),
        "threshold_definition": f"p{percentile:g} of daily max temperature, same window, baseline years",
        "hot_days": count,
        "expected_hot_days": round(expected, 1),
        "ratio_to_expected": round(ratio, 2),
        "risk_level": risk,
    }
