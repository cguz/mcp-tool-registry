"""
The cascade figure of the Wildfire Flood MCP server (matplotlib, headless).

Left panel: burn severity over a hillshade, with built-up land and the AOI
outline. Right panel: rainfall per 15 minutes (or per hour) against the
post-fire debris-flow threshold. Together they tell the cascade at a glance:
where the scar is, and whether the coming (or past) rain crosses the line.
"""

from __future__ import annotations

import io
from datetime import datetime

import burn
import matplotlib
import numpy as np
from PIL import Image as PILImage  # Pillow ships with matplotlib

matplotlib.use("Agg")  # no display in a server
import matplotlib.dates as mdates  # noqa: E402 - the backend must be chosen first
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Colours for severity classes 1..4 (low, moderate-low, moderate-high, high).
SEVERITY_COLOURS = ["#fee08b", "#fdae61", "#f46d43", "#a50026"]
BUILT_UP_COLOUR = "#3949ab"
MAX_MAP_PX = 450  # longest side of the map raster, to keep the PNG small
SHADE_LEVELS = 4


def _downsample(array: np.ndarray, step: int) -> np.ndarray:
    return array[::step, ::step] if step > 1 else array


def cascade_png(
    scene,
    outline_xy: list[list[tuple[float, float]]],
    rain: dict,
    hazard_result: dict,
    title: str,
    footer: str,
) -> bytes:
    """Render the two-panel figure and return it as a compact PNG.

    The map shows severity inside the AOI only, while built-up land is drawn
    over the whole grid (the AOI plus the exposure margin).

    Args:
        scene: sources.BurnScene with grid, dnbr, shade, land_cover.
        outline_xy: AOI rings in the grid CRS (metres).
        rain: series from sources.fetch_rain.
        hazard_result: output of hazard.assess.
        title: figure title.
        footer: source line printed at the bottom.
    """
    grid = scene.grid
    step = max(1, int(np.ceil(max(grid.shape) / MAX_MAP_PX)))
    left, top = grid.transform.c, grid.transform.f
    right, bottom = left + grid.width * grid.pixel_m, top - grid.height * grid.pixel_m
    extent = (0.0, (right - left) / 1000, 0.0, (top - bottom) / 1000)  # km from the lower-left corner

    fig, (ax_map, ax_rain) = plt.subplots(
        1, 2, figsize=(11, 4.8), dpi=90, gridspec_kw={"width_ratios": [1.05, 1]}
    )

    # Map: hillshade, severity classes 1-4, built-up land, AOI outline.
    # Posterized hillshade: a few grey levels keep the terrain readable and the PNG small.
    shade = np.round(_downsample(scene.shade, step) * SHADE_LEVELS) / SHADE_LEVELS
    ax_map.imshow(shade, cmap="gray", vmin=-0.3, vmax=1, extent=extent, interpolation="nearest")
    # Severity only inside the AOI: the margin is read for exposure, and on shaded
    # slopes far from the fire it shows illumination speckle, not burn.
    footprint = burn.burn_footprint(scene.dnbr) & scene.inside
    classes = burn.severity(_downsample(np.where(footprint, scene.dnbr, np.nan), step)).astype("float32")
    classes[classes < 1] = np.nan
    ax_map.imshow(classes, cmap=ListedColormap(SEVERITY_COLOURS), vmin=0.5, vmax=4.5, extent=extent,
                  alpha=0.8, interpolation="nearest")
    built = _downsample(scene.land_cover == burn.BUILT_UP, step).astype("float32")
    built[built == 0] = np.nan
    ax_map.imshow(built, cmap=ListedColormap([BUILT_UP_COLOUR]), extent=extent, alpha=0.45,
                  interpolation="nearest")
    for ring in outline_xy:
        xs = [(x - left) / 1000 for x, _ in ring]
        ys = [(y - bottom) / 1000 for _, y in ring]
        ax_map.plot(xs, ys, color="black", linewidth=1.2)
    ax_map.set_xlabel("km", fontsize=8)
    ax_map.set_ylabel("km", fontsize=8)
    ax_map.tick_params(labelsize=7)
    stats = scene.stats
    ax_map.set_title(
        f"Burn severity (Sentinel-2 dNBR): {stats.burned_ha:,.0f} ha burned\n"
        f"{stats.steep_mod_high_share:.0%} of it moderate/high severity on slopes ≥ 23°",
        fontsize=8.5, loc="left",
    )
    handles = [Patch(color=c, label=n) for c, n in zip(SEVERITY_COLOURS, ["low", "moderate-low",
               "moderate-high", "high"], strict=True)]
    handles.append(Patch(color=BUILT_UP_COLOUR, alpha=0.6, label="built-up (WorldCover)"))
    ax_map.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)

    # Rain: bars per step, 50 % threshold line, peak annotation.
    minutes = hazard_result["burst_duration_min"]
    times = rain["times_15"] if minutes == 15 else rain["times_60"]
    values = rain["mm_15"] if minutes == 15 else rain["mm_60"]
    stamps = [datetime.fromisoformat(t) for t in times]
    heights = [v or 0.0 for v in values]
    threshold = hazard_result["by_duration"][f"{minutes}_min"]["threshold_50pct_mm"]
    colours = ["#c62828" if threshold is not None and h >= threshold else "#1e88e5" for h in heights]
    ax_rain.bar(stamps, heights, width=minutes / 1440, color=colours, align="edge")
    if threshold is not None:
        ax_rain.axhline(threshold, color="#c62828", linestyle="--", linewidth=1.2,
                        label=f"50 % debris-flow threshold: {threshold:g} mm / {minutes} min")
        ax_rain.legend(fontsize=7.5, loc="upper left")
    ax_rain.set_ylim(0, max([*heights, threshold or 0, 1.0]) * 1.25)
    ax_rain.set_ylabel(f"rain per {minutes} min (mm)", fontsize=8)
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax_rain.tick_params(labelsize=7)
    ax_rain.grid(axis="y", color="#dddddd", linewidth=0.6)
    level = hazard_result["risk_level"]
    ax_rain.set_title(
        f"Rain vs post-fire debris-flow threshold\nlikelihood {hazard_result['likelihood']:.0%} ({level} risk)",
        fontsize=8.5, loc="left", color={"high": "#c62828", "medium": "#ef6c00"}.get(level, "#2e7d32"),
    )

    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.01, footer, fontsize=6.5, color="#525252")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    rendered = io.BytesIO()
    fig.savefig(rendered, format="png")
    plt.close(fig)  # free the figure; the server renders many over its lifetime
    return compact_png(rendered.getvalue())


def compact_png(png: bytes, colours: int = 48) -> bytes:
    """Re-encode as an optimized palette PNG so it stays small enough to return
    inline (EVE's MCP file convention: a few tens of KB)."""
    image = PILImage.open(io.BytesIO(png)).convert("RGB").quantize(colors=colours)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()
