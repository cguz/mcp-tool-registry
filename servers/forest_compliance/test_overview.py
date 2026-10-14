"""Offline tests for the overview map (no network, no LLM).

Run from this directory: python -m pytest test_overview.py -q
"""

import struct

import forest
import numpy as np
import overview
import pytest

SQUARE = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0]]]}


def test_centroid_ignores_closing_vertex():
    assert forest.centroid(SQUARE) == pytest.approx((0.005, 0.005))


def test_map_bounds_pads_and_enforces_minimum_span():
    west, south, east, north = overview.map_bounds([SQUARE])
    assert east - west == pytest.approx(overview.MIN_SPAN_DEG * 1.24)
    assert (west + east) / 2 == pytest.approx(0.005)


def test_map_bounds_covers_all_plots():
    far = {"type": "Point", "coordinates": [1.0, 1.0]}
    circle, _ = forest.to_plot_geometry(far)
    west, south, east, north = overview.map_bounds([SQUARE, circle])
    assert west < 0 and south < 0 and east > 1 and north > 1


def test_overview_png_is_small_valid_png():
    plots = [
        ("farm-01", SQUARE, "high"),
        ("farm-02", forest.to_plot_geometry(latitude=0.02, longitude=0.02)[0], "low"),
        ("farm-03", forest.to_plot_geometry(latitude=0.03, longitude=0.0)[0], "medium"),
    ]
    bounds = overview.map_bounds([g for _, g, _ in plots])
    fraction = np.linspace(0, 1, 100 * 100, dtype=np.float32).reshape(100, 100)
    png = overview.overview_png(plots, fraction, bounds, "Test cooperative", "footer")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    assert width == 800 and 380 <= height <= 900
    assert len(png) < 60_000


def test_figure_size_follows_ground_aspect():
    wide = overview.figure_size((-8.0, 5.7, -6.4, 6.3))
    square = overview.figure_size((0.0, 0.0, 0.1, 0.1))
    assert wide[0] == square[0] == 8.0
    assert wide[1] < square[1]
    assert 3.8 <= wide[1] <= 9.0
