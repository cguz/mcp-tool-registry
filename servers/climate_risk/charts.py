"""
Chart rendering for the Climate Risk MCP server (matplotlib, headless).

Kept apart from the statistics so the numbers can be tested without a
plotting backend, and the charts without network access.
"""

from __future__ import annotations

import io

import matplotlib
from PIL import Image as PILImage  # Pillow ships with matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - the backend must be chosen first
from matplotlib.patches import Patch  # noqa: E402

# Dry → wet, after the usual drought-monitor palette.
CLASS_COLOURS = {
    "extremely dry": "#7f2704",
    "severely dry": "#d94801",
    "moderately dry": "#fd8d3c",
    "near normal": "#bdbdbd",
    "moderately wet": "#74c476",
    "very wet": "#31a354",
    "extremely wet": "#006d2c",
}


def drought_chart_png(
    spei: list[dict],
    spi: list[dict],
    scale: int,
    title: str,
    footer: str,
) -> bytes:
    """Bar chart of monthly SPEI coloured by McKee class, SPI as a line."""
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=100)
    months = [e["month"] for e in spei]
    x = range(len(months))
    ax.bar(x, [e["index"] for e in spei], color=[CLASS_COLOURS[e["class"]] for e in spei], width=0.85, zorder=2)
    spi_by_month = {e["month"]: e["index"] for e in spi}
    ax.plot(x, [spi_by_month.get(m) for m in months], color="#08306b", linewidth=1.4, label=f"SPI-{scale}", zorder=3)

    for level, style in ((-1.0, ":"), (-1.5, "--"), (-2.0, "-")):
        ax.axhline(level, color="#7f2704", linestyle=style, linewidth=0.8, zorder=1)
    ax.axhline(0, color="#525252", linewidth=0.8, zorder=1)
    ax.set_ylim(-2.6, 2.6)
    ax.set_ylabel(f"SPEI-{scale}")
    step = max(1, len(months) // 12)
    ax.set_xticks(list(x)[::step], months[::step], rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linewidth=0.3, zorder=0)
    ax.set_title(title, fontsize=10, loc="left")

    used = [c for c in CLASS_COLOURS if any(e["class"] == c for e in spei)]
    handles = [Patch(color=CLASS_COLOURS[c], label=c) for c in used]
    handles.append(ax.get_lines()[0])
    ax.legend(handles=handles, fontsize=7, ncol=min(4, len(handles)), loc="upper left", frameon=False)
    fig.text(0.01, 0.01, footer, fontsize=6.5, color="#525252")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    rendered = io.BytesIO()
    fig.savefig(rendered, format="png")
    plt.close(fig)
    return _compact_png(rendered.getvalue())


def _compact_png(png: bytes, colours: int = 64) -> bytes:
    """Re-encode as an optimized palette PNG: flat chart colours survive a
    64-colour palette and the file shrinks several-fold, which keeps it small
    enough to return inline (EVE's MCP file convention: a few tens of KB)."""
    image = PILImage.open(io.BytesIO(png)).convert("RGB").quantize(colors=colours)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()
