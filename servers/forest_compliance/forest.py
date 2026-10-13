"""
Geometry and EUDR assessment logic for the Forest Compliance MCP server.

Pure functions only (no network, no raster I/O), so they can be unit-tested
offline. Rasters reach this module as numpy arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from typing import Any

import numpy as np

EARTH_RADIUS_M = 6_378_137.0  # WGS84 semi-major axis, as used by GeoJSON area tools
# EUDR Art. 2(13): products must come from land not deforested after 31 December 2020.
CUTOFF_YEAR = 2020
# EUDR Art. 9(1)(d): plots of more than 4 ha need a polygon; up to 4 ha a single
# point is enough, so a point is assessed as a circle of 4 ha around it.
POINT_PLOT_HA = 4.0
MAX_PLOT_HA = 10_000.0
TILE_DEG = 10

# A tolerance for 30 m misregistration along field edges: below it, post-cutoff
# loss is reported as "medium" (review) rather than "high".
HIGH_RISK_MIN_HA = 0.1
HIGH_RISK_MIN_FRACTION = 0.01


# ─── Geometry ────────────────────────────────────────────────────────────────


def _load(obj: dict | str) -> dict:
    return json.loads(obj) if isinstance(obj, str) else obj


def circle(lat: float, lon: float, area_ha: float = POINT_PLOT_HA, vertices: int = 64) -> dict:
    """GeoJSON Polygon approximating a circle of the given area around a point."""
    radius = math.sqrt(area_ha * 10_000 / math.pi)
    dlat = radius / 111_320.0
    dlon = radius / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    ring = [
        [lon + dlon * math.cos(2 * math.pi * i / vertices), lat + dlat * math.sin(2 * math.pi * i / vertices)]
        for i in range(vertices)
    ]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def to_plot_geometry(
    geometry: dict | str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> tuple[dict, str]:
    """Normalize the input to a Polygon/MultiPolygon and say how it was built.

    Accepts a GeoJSON geometry, Feature, or single-feature FeatureCollection, or a
    lat/lon pair. Points become a 4 ha circle (the EUDR point threshold).
    """
    if geometry is None:
        if latitude is None or longitude is None:
            raise ValueError("provide a GeoJSON geometry, or latitude and longitude")
        geom = {"type": "Point", "coordinates": [float(longitude), float(latitude)]}
    else:
        geom = _load(geometry)

    if geom.get("type") == "FeatureCollection":
        features = geom.get("features", [])
        if len(features) != 1:
            raise ValueError("a FeatureCollection must hold exactly one plot here; use eudr_check_plots for several")
        geom = features[0]
    if geom.get("type") == "Feature":
        geom = geom.get("geometry") or {}

    kind = geom.get("type")
    if kind == "Point":
        lon, lat = geom["coordinates"][:2]
        _check_lat_lon(lat, lon)
        return circle(lat, lon), f"point buffered to a {POINT_PLOT_HA:g} ha circle"
    if kind in ("Polygon", "MultiPolygon"):
        for lon, lat in _vertices(geom):
            _check_lat_lon(lat, lon)
        return geom, "polygon"
    raise ValueError(f"unsupported geometry type: {kind!r} (use Point, Polygon or MultiPolygon)")


def _check_lat_lon(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"coordinates out of range: lat={lat}, lon={lon}")


def _polygons(geom: dict) -> list[list[list[list[float]]]]:
    return [geom["coordinates"]] if geom["type"] == "Polygon" else list(geom["coordinates"])


def _vertices(geom: dict):
    for polygon in _polygons(geom):
        for ring in polygon:
            for position in ring:
                yield position[0], position[1]


def _ring_area_m2(ring: list[list[float]]) -> float:
    """Spherical ring area (Chamberlain & Duquette 2007), signed."""
    total = 0.0
    n = len(ring)
    for i in range(n):
        lon1, lat1 = ring[i][0], ring[i][1]
        lon2, lat2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        total += math.radians(lon2 - lon1) * (2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2)))
    return total * EARTH_RADIUS_M**2 / 2.0


def area_ha(geom: dict) -> float:
    """Area of a Polygon/MultiPolygon in hectares (outer ring minus holes)."""
    total = 0.0
    for polygon in _polygons(geom):
        outer, *holes = polygon
        total += abs(_ring_area_m2(outer)) - sum(abs(_ring_area_m2(h)) for h in holes)
    return total / 10_000.0


def bbox(geom: dict) -> tuple[float, float, float, float]:
    lons, lats = zip(*_vertices(geom), strict=True)
    return min(lons), min(lats), max(lons), max(lats)


def plot_id(geom: dict) -> str:
    """Stable fingerprint of the geometry, for the audit trail."""
    canonical = json.dumps(geom, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─── Hansen tiles ────────────────────────────────────────────────────────────


def hansen_tile_name(lat_top: int, lon_left: int) -> str:
    """Hansen tiles are named by their top-left corner, e.g. 10N_010W."""
    ns = f"{abs(lat_top):02d}{'N' if lat_top >= 0 else 'S'}"
    ew = f"{abs(lon_left):03d}{'E' if lon_left >= 0 else 'W'}"
    return f"{ns}_{ew}"


def hansen_tiles(box: tuple[float, float, float, float]) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Hansen tiles intersecting a bbox, each with the part of the bbox it covers."""
    west, south, east, north = box
    tiles = []
    # A tile covers latitudes (top - 10, top]; its first row starts at the top edge.
    lat_top = math.ceil(north / TILE_DEG) * TILE_DEG
    while lat_top > south:
        lon_left = math.floor(west / TILE_DEG) * TILE_DEG
        while lon_left < east:
            part = (
                max(west, lon_left),
                max(south, lat_top - TILE_DEG),
                min(east, lon_left + TILE_DEG),
                min(north, lat_top),
            )
            if part[0] < part[2] and part[1] < part[3]:
                tiles.append((hansen_tile_name(lat_top, lon_left), part))
            lon_left += TILE_DEG
        lat_top -= TILE_DEG
    return tiles


# ─── Assessment ──────────────────────────────────────────────────────────────


def tally(forest_2020: np.ndarray, loss_year: np.ndarray, inside: np.ndarray, pixel_ha: float) -> dict[str, Any]:
    """Area statistics for one aligned window.

    forest_2020: 1 where JRC GFC2020 maps forest on 31 Dec 2020, else 0.
    loss_year:   Hansen lossyear code (0 = no loss, 1..N = year 2000+N), same grid.
    inside:      True for pixels whose centre is inside the plot.
    """
    forest = (forest_2020 == 1) & inside
    loss = (loss_year > 0) & inside
    after = loss & (loss_year > CUTOFF_YEAR - 2000)
    deforested = after & forest

    by_year: dict[int, float] = {}
    codes, counts = np.unique(loss_year[deforested], return_counts=True)
    for code, count in zip(codes.tolist(), counts.tolist(), strict=True):
        by_year[2000 + int(code)] = count * pixel_ha
    return {
        "pixels": int(inside.sum()),
        "plot_ha": float(inside.sum()) * pixel_ha,
        "forest_2020_ha": float(forest.sum()) * pixel_ha,
        "loss_before_cutoff_ha": float((loss & ~after).sum()) * pixel_ha,
        "loss_after_cutoff_ha": float(after.sum()) * pixel_ha,
        "deforestation_after_cutoff_ha": float(deforested.sum()) * pixel_ha,
        "deforestation_by_year_ha": by_year,
    }


def merge_tallies(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("plot does not overlap any data")
    merged: dict[str, Any] = {k: 0.0 for k in parts[0] if k != "deforestation_by_year_ha"}
    by_year: dict[int, float] = {}
    for part in parts:
        for key in merged:
            merged[key] += part[key]
        for year, ha in part["deforestation_by_year_ha"].items():
            by_year[year] = by_year.get(year, 0.0) + ha
    merged["pixels"] = int(merged["pixels"])
    merged["deforestation_by_year_ha"] = dict(sorted(by_year.items()))
    return merged


def assess(stats: dict[str, Any], plot_area_ha: float) -> dict[str, Any]:
    """Turn area statistics into a risk level, a verdict and review notes."""
    deforested = stats["deforestation_after_cutoff_ha"]
    forest = stats["forest_2020_ha"]
    other_loss = stats["loss_after_cutoff_ha"] - deforested
    threshold = max(HIGH_RISK_MIN_HA, HIGH_RISK_MIN_FRACTION * plot_area_ha)
    notes: list[str] = []

    if deforested >= threshold:
        risk = "high"
        years = ", ".join(str(y) for y in stats["deforestation_by_year_ha"])
        verdict = (
            f"Possible deforestation after the EUDR cut-off: {deforested:.2f} ha of land mapped as forest "
            f"on 31 Dec 2020 lost tree cover in {years}. Not deforestation-free unless proven otherwise."
        )
    elif deforested > 0:
        risk = "medium"
        verdict = (
            f"Small post-cut-off loss inside 2020 forest ({deforested:.2f} ha, below the "
            f"{threshold:.2f} ha review threshold); may be edge misregistration. Needs review."
        )
    elif forest > 0:
        risk = "low"
        verdict = "No loss detected in land that was forest on 31 Dec 2020."
        notes.append(
            f"{forest:.2f} ha of the plot was forest at the cut-off date; converting it later would breach the EUDR."
        )
    else:
        risk = "low"
        verdict = "No forest on 31 Dec 2020 and no deforestation detected after the cut-off."

    if other_loss > 0.01:
        notes.append(
            f"{other_loss:.2f} ha of tree cover loss after 2020 on land not mapped as forest in 2020 "
            "(e.g. plantation or agroforestry rotation); not deforestation under the EUDR definition, but worth checking."
        )
    if stats["loss_before_cutoff_ha"] > 0.01:
        notes.append(
            f"{stats['loss_before_cutoff_ha']:.2f} ha of tree cover loss before the cut-off (2001-2020); "
            "allowed under the EUDR."
        )
    return {"risk_level": risk, "verdict": verdict, "notes": notes, "review_threshold_ha": round(threshold, 3)}


def plot_area_check(geom: dict) -> float:
    plot_ha = area_ha(geom)
    if plot_ha <= 0:
        raise ValueError("plot has zero area")
    if plot_ha > MAX_PLOT_HA:
        raise ValueError(f"plot is {plot_ha:,.0f} ha; at most {MAX_PLOT_HA:,.0f} ha per plot")
    return plot_ha


def pixel_area_ha(lat: float, deg: float) -> float:
    """Area of a square lat/lon pixel of `deg` degrees at latitude `lat`."""
    side_m = deg * math.pi / 180.0 * EARTH_RADIUS_M
    return side_m * side_m * math.cos(math.radians(lat)) / 10_000.0


# ─── Map rendering (dependency-free PNG) ─────────────────────────────────────

MAP_COLOURS = {
    "non_forest_2020": (236, 232, 222),
    "forest_2020": (46, 125, 50),
    "loss_before_cutoff": (251, 192, 45),
    "loss_after_cutoff_non_forest": (230, 124, 115),
    "deforestation_after_cutoff": (198, 40, 40),
    "plot_outline": (20, 20, 20),
}


def classify_pixels(forest_2020: np.ndarray, loss_year: np.ndarray) -> np.ndarray:
    """Per-pixel class index into the MAP_COLOURS order (0..4)."""
    classes = np.zeros(forest_2020.shape, dtype=np.uint8)
    classes[forest_2020 == 1] = 1
    loss = loss_year > 0
    after = loss & (loss_year > CUTOFF_YEAR - 2000)
    classes[loss & ~after] = 2
    classes[after & (forest_2020 != 1)] = 3
    classes[after & (forest_2020 == 1)] = 4
    return classes


def render_map(forest_2020: np.ndarray, loss_year: np.ndarray, inside: np.ndarray, min_size: int = 480) -> bytes:
    """RGB PNG of the plot surroundings: classes outside the plot are faded and
    the plot boundary is drawn in black."""
    palette = np.array(list(MAP_COLOURS.values()), dtype=np.uint8)
    rgb = palette[classify_pixels(forest_2020, loss_year)].astype(np.float32)
    faded = rgb * 0.45 + 255 * 0.55
    rgb = np.where(inside[..., None], rgb, faded)

    edge = inside & ~(
        np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
    )
    scale = max(1, math.ceil(min_size / max(inside.shape)))
    rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
    edge = np.repeat(np.repeat(edge, scale, 0), scale, 1)
    rgb[edge] = MAP_COLOURS["plot_outline"]
    return encode_png(rgb.astype(np.uint8))


def encode_png(rgb: np.ndarray) -> bytes:
    """Minimal PNG encoder for an (h, w, 3) uint8 array."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
