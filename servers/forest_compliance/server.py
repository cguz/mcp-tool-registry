"""
Forest Compliance — MCP Server
==============================
EU Deforestation Regulation (EUDR) screening for farm plots: was the land
forest on 31 December 2020 (the EUDR cut-off date), and has that forest been
lost since? Uses two public datasets read in place, no API key required:

    * JRC Global Forest Cover 2020 V4 (10 m) — the EU Observatory reference map
    * Hansen Global Forest Change v1.12 (30 m) — year of tree cover loss to 2024

Tools:
    eudr_check_plot  — screen one plot (polygon, point or lat/lon)
    eudr_check_plots — screen a supplier list (GeoJSON FeatureCollection)
    eudr_plot_map    — PNG map of a plot: 2020 forest and loss before/after the cut-off

Results are a risk screening for due diligence, not a legal determination.
Every response carries a ``provenance`` block for the audit trail.

Usage:
    python server.py                               # HTTP transport (default)
    python server.py --transport stdio             # stdio transport
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

import forest
import sources
from mcp.server.fastmcp import FastMCP, Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("forest-compliance-mcp")

SERVER_NAME = "Forest Compliance"
SERVER_VERSION = "0.1.0"
MAX_PLOTS_PER_CALL = 25
MAX_MAP_PLOT_HA = 2_500.0

mcp = FastMCP(SERVER_NAME, host="0.0.0.0", port=8000, stateless_http=True)

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _provenance(tiles: list[str], how: str, geom_id: str) -> dict[str, Any]:
    return {
        "plot_id": geom_id,
        "plot_geometry_from": how,
        "datasets": sources.DATASETS,
        "hansen_tiles": tiles,
        "cutoff_date": f"{forest.CUTOFF_YEAR}-12-31",
        "method": (
            "Pixels whose centre lies inside the plot are counted on the 10 m GFC2020 grid; Hansen "
            "loss years are resampled onto it (nearest). Deforestation after the cut-off = forest in "
            "GFC2020 AND Hansen loss year 2021 or later."
        ),
        "data_gap": (
            f"Hansen loss is available up to {sources.HANSEN_LAST_YEAR}; clearing in "
            f"{sources.HANSEN_LAST_YEAR + 1} or later is not yet covered. Check recent Sentinel-2 imagery."
        ),
        "disclaimer": "Risk screening for due diligence, not a legal determination under Regulation (EU) 2023/1115.",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "server": f"{SERVER_NAME} MCP server v{SERVER_VERSION}",
    }


def _round(stats: dict[str, Any]) -> dict[str, Any]:
    out = {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}
    out["deforestation_by_year_ha"] = {str(y): round(v, 3) for y, v in stats["deforestation_by_year_ha"].items()}
    return out


def _check_one(geom: dict, how: str) -> dict[str, Any]:
    """Blocking: geometry checks, raster reads and assessment for one plot."""
    started = time.monotonic()
    plot_ha = forest.plot_area_check(geom)
    stats, tiles = sources.measure(geom)
    result = {
        "plot_area_ha": round(plot_ha, 3),
        **forest.assess(stats, plot_ha),
        "areas_ha": _round(stats),
        "provenance": _provenance(tiles, how, forest.plot_id(geom)),
    }
    logger.info("Checked %.1f ha plot in %.1fs -> %s", plot_ha, time.monotonic() - started, result["risk_level"])
    return result


def _error(message: str) -> str:
    logger.warning("Tool error: %s", message)
    return json.dumps({"error": message})


@mcp.tool()
async def eudr_check_plot(
    geometry: dict | str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> str:
    """
    Screen one farm plot for deforestation after the EUDR cut-off date (31 Dec 2020).

    Use this to check whether a commodity (cocoa, coffee, soy, palm oil, rubber,
    cattle, wood) from a plot can be deforestation-free under the EU
    Deforestation Regulation. It answers: was the plot forest on 31 Dec 2020
    (JRC GFC2020, the EU reference map) and has that forest lost tree cover
    since (Hansen Global Forest Change, up to 2024)?

    Args:
        geometry:  The plot as GeoJSON (Polygon, MultiPolygon, Point, Feature or a
                   one-feature FeatureCollection), as an object or a string,
                   WGS84 lon/lat. Points are treated as a 4 ha circle, the EUDR
                   threshold below which a point is accepted instead of a polygon.
        latitude:  Alternatively, the plot location as latitude (decimal degrees)...
        longitude: ...and longitude; treated as a 4 ha circle.

    Returns:
        JSON with risk_level ("low", "medium" or "high"), a verdict sentence,
        review notes, areas in hectares (plot, forest in 2020, loss before and
        after the cut-off, deforestation after the cut-off by year) and a
        provenance block (datasets, versions, URLs, plot fingerprint, data gap).
        This is a risk screening, not a legal determination.
    """
    try:
        geom, how = forest.to_plot_geometry(geometry, latitude, longitude)
        return json.dumps(await asyncio.to_thread(_check_one, geom, how))
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the agent as JSON
        return _error(str(exc))


@mcp.tool()
async def eudr_check_plots(plots: dict | str) -> str:
    """
    Screen a list of supplier plots for EUDR deforestation risk in one call.

    Use this for a due-diligence batch, e.g. every farm of a cocoa cooperative.
    Each plot is assessed exactly like eudr_check_plot.

    Args:
        plots: GeoJSON FeatureCollection (object or string) with up to 25
               features. Each feature is a Polygon, MultiPolygon or Point (a
               point becomes a 4 ha circle). An "id" or "name" in the feature
               properties is echoed back to identify the plot.

    Returns:
        JSON with one entry per plot (id, risk_level, verdict, key areas or an
        error), a summary (count per risk level, total area, total deforestation
        after the cut-off, share of plots at high risk) and a provenance block.
    """
    try:
        collection = json.loads(plots) if isinstance(plots, str) else plots
        if collection.get("type") != "FeatureCollection":
            raise ValueError("plots must be a GeoJSON FeatureCollection")
        features = collection.get("features", [])
        if not features:
            raise ValueError("the FeatureCollection has no features")
        if len(features) > MAX_PLOTS_PER_CALL:
            raise ValueError(f"at most {MAX_PLOTS_PER_CALL} plots per call; split the list")

        results = []
        for index, feature in enumerate(features):
            props = feature.get("properties") or {}
            plot_ref = props.get("id") or props.get("name") or f"plot-{index + 1}"
            try:
                geom, how = forest.to_plot_geometry(feature)
                full = await asyncio.to_thread(_check_one, geom, how)
                areas = full["areas_ha"]
                results.append({
                    "id": plot_ref,
                    "risk_level": full["risk_level"],
                    "verdict": full["verdict"],
                    "notes": full["notes"],
                    "plot_area_ha": full["plot_area_ha"],
                    "forest_2020_ha": areas["forest_2020_ha"],
                    "deforestation_after_cutoff_ha": areas["deforestation_after_cutoff_ha"],
                    "deforestation_by_year_ha": areas["deforestation_by_year_ha"],
                    "plot_id": full["provenance"]["plot_id"],
                    "hansen_tiles": full["provenance"]["hansen_tiles"],
                })
            except Exception as exc:  # noqa: BLE001 - one bad plot must not sink the batch
                results.append({"id": plot_ref, "error": str(exc)})

        assessed = [r for r in results if "error" not in r]
        counts = {level: sum(1 for r in assessed if r["risk_level"] == level) for level in RISK_ORDER}
        summary = {
            "plots": len(results),
            "assessed": len(assessed),
            "failed": len(results) - len(assessed),
            "by_risk_level": counts,
            "high_risk_share": round(counts["high"] / len(assessed), 3) if assessed else None,
            "total_area_ha": round(sum(r["plot_area_ha"] for r in assessed), 2),
            "total_deforestation_after_cutoff_ha": round(
                sum(r["deforestation_after_cutoff_ha"] for r in assessed), 3
            ),
            "worst_risk_level": max((r["risk_level"] for r in assessed), key=RISK_ORDER.get, default=None),
        }
        provenance = _provenance(
            sorted({t for r in assessed for t in r["hansen_tiles"]}), "per plot, see results", "per plot, see results"
        )
        return json.dumps({"summary": summary, "results": results, "provenance": provenance})
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc))


# Image + text content; no structured (JSON-schema) output for this tool.
@mcp.tool(structured_output=False)
async def eudr_plot_map(
    geometry: dict | str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> list:
    """
    Draw a map of a plot for an EUDR due-diligence report.

    Use this after eudr_check_plot when the user wants to see the evidence.
    The PNG shows the plot (black outline) and its surroundings (faded):
    dark green = forest on 31 Dec 2020 (JRC GFC2020); dark red = that forest
    lost after the cut-off (deforestation risk); light red = tree cover lost
    after the cut-off on land that was not forest in 2020; yellow = loss
    before the cut-off (2001-2020, allowed); beige = not forest in 2020.

    Args:
        geometry:  The plot as GeoJSON (same formats as eudr_check_plot), at
                   most 2,500 ha and inside one 10° tile.
        latitude:  Alternatively, latitude of a point (drawn as a 4 ha circle)...
        longitude: ...and longitude.

    Returns:
        The PNG image followed by a JSON legend with the plot fingerprint and
        the data sources.
    """
    try:
        geom, how = forest.to_plot_geometry(geometry, latitude, longitude)
        plot_ha = forest.plot_area_check(geom)
        if plot_ha > MAX_MAP_PLOT_HA:
            raise ValueError(f"maps are limited to {MAX_MAP_PLOT_HA:,.0f} ha plots")
        forest_2020, loss_year, inside = await asyncio.to_thread(sources.plot_rasters, geom)
        png = forest.render_map(forest_2020, loss_year, inside)
        legend = {
            "legend": {
                "dark green": "forest on 31 Dec 2020 (JRC GFC2020)",
                "dark red": "2020 forest lost after the cut-off (deforestation risk)",
                "light red": "tree cover lost after the cut-off on non-forest land",
                "yellow": "tree cover lost 2001-2020 (before the cut-off)",
                "beige": "not forest in 2020",
                "black line": "plot boundary; surroundings are faded",
            },
            "plot_area_ha": round(plot_ha, 3),
            "provenance": _provenance([t for t, _ in forest.hansen_tiles(forest.bbox(geom))], how, forest.plot_id(geom)),
        }
        return [Image(data=png, format="png"), json.dumps(legend)]
    except Exception as exc:  # noqa: BLE001
        return [_error(str(exc))]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forest Compliance MCP Server")
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
