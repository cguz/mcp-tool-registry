"""
Remote raster access for the Forest Compliance MCP server.

Both datasets are public Cloud-Optimized / tiled GeoTIFFs read in place with
HTTP range requests, so only the pixels around a plot are downloaded and no API
key is needed.
"""

from __future__ import annotations

import forest
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

GFC2020_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFC2020/LATEST/single-cog/JRC_GFC2020_V4_COG.tif"
)
HANSEN_VERSION = "GFC-2024-v1.12"
HANSEN_URL = (
    "https://storage.googleapis.com/earthenginepartners-hansen/"
    f"{HANSEN_VERSION}/Hansen_{HANSEN_VERSION}_lossyear_{{tile}}.tif"
)
HANSEN_LAST_YEAR = 2024
GFC2020_LAT_RANGE = (-60.0, 80.0)
HANSEN_LAT_RANGE = (-50.0, 80.0)

DATASETS = [
    {
        "name": "JRC Global Forest Cover 2020 (GFC2020) V4",
        "role": "forest / non-forest on 31 December 2020 (EUDR reference map of the EU Observatory)",
        "provider": "European Commission, Joint Research Centre",
        "resolution_m": 10,
        "url": GFC2020_URL,
        "info": "https://forobs.jrc.ec.europa.eu/GFC",
        "licence": "© European Union; reuse authorised under Commission Decision 2011/833/EU",
    },
    {
        "name": f"Hansen/UMD/Google/USGS/NASA Global Forest Change {HANSEN_VERSION} (lossyear)",
        "role": f"year of tree cover loss, 2001-{HANSEN_LAST_YEAR}",
        "provider": "University of Maryland (Hansen et al. 2013), also used by Global Forest Watch",
        "resolution_m": 30,
        "url": HANSEN_URL.format(tile="<tile>"),
        "info": "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html",
        "licence": "CC BY 4.0",
    },
]

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


def _check_coverage(box: tuple[float, float, float, float]) -> None:
    _, south, _, north = box
    low = max(GFC2020_LAT_RANGE[0], HANSEN_LAT_RANGE[0])
    high = min(GFC2020_LAT_RANGE[1], HANSEN_LAT_RANGE[1])
    if south < low or north > high:
        raise ValueError(f"plot is outside the data coverage (latitudes {low}..{high})")


def measure(geom: dict) -> tuple[dict, list[str]]:
    """Area statistics for a plot, read on the 10 m GFC2020 grid.

    For each Hansen tile the plot overlaps, the GFC2020 window is read at native
    resolution and the Hansen loss-year raster is resampled onto it (nearest
    neighbour; each 30 m Hansen pixel is exactly 3x3 GFC pixels). Pixels count
    as inside when their centre falls inside the plot. Blocking: call it from a
    worker thread.
    """
    box = forest.bbox(geom)
    _check_coverage(box)
    parts = []
    tiles_used = []
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + GFC2020_URL) as gfc:
        for tile, part in forest.hansen_tiles(box):
            window = from_bounds(*part, transform=gfc.transform)
            window = window.round_offsets(op="floor").round_lengths(op="ceil")
            forest_2020 = gfc.read(1, window=window)
            if forest_2020.size == 0:
                continue
            transform = gfc.window_transform(window)
            west, south, east, north = gfc.window_bounds(window)

            with rasterio.open("/vsicurl/" + HANSEN_URL.format(tile=tile)) as hansen:
                hansen_window = from_bounds(west, south, east, north, transform=hansen.transform)
                loss_year = hansen.read(
                    1,
                    window=hansen_window,
                    out_shape=forest_2020.shape,
                    resampling=Resampling.nearest,
                    boundless=True,
                    fill_value=0,
                )

            inside = geometry_mask([geom], out_shape=forest_2020.shape, transform=transform, invert=True)
            centre_lat = (south + north) / 2
            pixel_ha = forest.pixel_area_ha(centre_lat, abs(transform.e))
            parts.append(forest.tally(forest_2020, loss_year, inside, pixel_ha))
            tiles_used.append(tile)
    return forest.merge_tallies(parts), tiles_used


def plot_rasters(geom: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GFC2020, Hansen loss year and plot mask for a single-tile plot (for maps)."""
    box = forest.bbox(geom)
    _check_coverage(box)
    tiles = forest.hansen_tiles(box)
    if len(tiles) != 1:
        raise ValueError("maps are only drawn for plots inside a single 10° tile")
    tile, part = tiles[0]
    pad = max(part[2] - part[0], part[3] - part[1]) * 0.15
    padded = (part[0] - pad, part[1] - pad, part[2] + pad, part[3] + pad)
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + GFC2020_URL) as gfc:
        window = from_bounds(*padded, transform=gfc.transform).round_offsets(op="floor").round_lengths(op="ceil")
        forest_2020 = gfc.read(1, window=window)
        transform = gfc.window_transform(window)
        bounds = gfc.window_bounds(window)
        with rasterio.open("/vsicurl/" + HANSEN_URL.format(tile=tile)) as hansen:
            loss_year = hansen.read(
                1,
                window=from_bounds(*bounds, transform=hansen.transform),
                out_shape=forest_2020.shape,
                resampling=Resampling.nearest,
                boundless=True,
                fill_value=0,
            )
    inside = geometry_mask([geom], out_shape=forest_2020.shape, transform=transform, invert=True)
    return forest_2020, loss_year, inside



def forest_background(box: tuple[float, float, float, float], max_px: int = 900) -> tuple[np.ndarray, tuple]:
    """Fraction of GFC2020 forest per output pixel over a bbox, read from the COG
    overviews so large areas stay cheap. Returns (fraction, (west, south, east, north))."""
    _check_coverage(box)
    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + GFC2020_URL) as gfc:
        window = from_bounds(*box, transform=gfc.transform).round_offsets(op="floor").round_lengths(op="ceil")
        scale = max(window.width, window.height) / max_px
        shape = (max(1, round(window.height / max(scale, 1))), max(1, round(window.width / max(scale, 1))))
        fraction = gfc.read(
            1, window=window, out_shape=shape, out_dtype="float32", resampling=Resampling.average
        )
        bounds = gfc.window_bounds(window)
    return fraction, bounds
