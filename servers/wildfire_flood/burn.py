"""
Burn-scar and terrain maths for the Wildfire Flood MCP server (NumPy only, no I/O).

All arrays are on one common grid: the Sentinel-2 20 m grid of the scene's UTM
zone (or a coarser multiple of it for large areas), so pixels are square and
their size is known in metres.

    * Burn severity: differenced Normalized Burn Ratio, dNBR = NBR_pre - NBR_post,
      with NBR = (NIR - SWIR2) / (NIR + SWIR2) (Sentinel-2 B08 and B12),
      classified with the USGS FIREMON thresholds (Key & Benson 2006).
    * Terrain: slope in degrees from the Copernicus DEM.
    * Land cover: ESA WorldCover 2021 classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sentinel-2 Scene Classification (SCL) values kept as clear land. 2 "dark area"
# is kept on purpose: fresh burn scars are dark and often land in that class.
# Dropped: 0 no data, 1 saturated, 3 cloud shadow, 6 water, 8-10 cloud/cirrus, 11 snow.
CLEAR_SCL = (2, 4, 5, 7)

# dNBR severity classes (x1000 scale), USGS FIREMON / Key & Benson (2006).
SEVERITY_CLASSES = [
    ("unburned", -10_000, 100),
    ("low", 100, 270),
    ("moderate-low", 270, 440),
    ("moderate-high", 440, 660),
    ("high", 660, 10_000),
]
BURNED_MIN_DNBR = 100  # lower bound of "low" severity: the burn footprint
MOD_HIGH_MIN_DNBR = 270  # moderate or high severity, the input to the debris-flow model
STEEP_SLOPE_DEG = 23.0  # Staley et al. (2017): slopes >= 23° drive post-fire debris flows

# ESA WorldCover 2021 class codes and names.
WORLDCOVER_CLASSES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare / sparse vegetation",
    70: "snow and ice",
    80: "permanent water",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}
BUILT_UP = 50


def reflectance(dn: np.ndarray, scale: float, offset: float, nodata: int = 0) -> np.ndarray:
    """Surface reflectance from Sentinel-2 L2A digital numbers; nodata -> NaN.

    Products from processing baseline 04.00 (2022) on carry an offset of -0.1,
    so the scale and offset must come from each asset's STAC metadata.
    """
    values = dn.astype("float32") * scale + offset
    values[dn == nodata] = np.nan
    return values


MIN_NBR_SIGNAL = 0.02  # NIR + SWIR2 reflectance below this is noise (deep shadow, water)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio, NaN where there is too little signal.

    With the -0.1 offset of recent products, reflectance can be ~0 or slightly
    negative in deep shadow; the ratio would blow up there, so such pixels are
    dropped and the result is clipped to its physical range [-1, 1].
    """
    total = nir + swir2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (nir - swir2) / total
    ratio[~np.isfinite(ratio) | ~(total > MIN_NBR_SIGNAL)] = np.nan
    return np.clip(ratio, -1.0, 1.0)


def clear_mask(scl: np.ndarray) -> np.ndarray:
    """True where the Scene Classification marks clear land."""
    return np.isin(scl, CLEAR_SCL)


def dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray, clear: np.ndarray) -> np.ndarray:
    """Differenced NBR on the x1000 scale used by the severity thresholds;
    NaN where `clear` is False (cloud, shadow or no data on either date)."""
    out = (nbr_pre - nbr_post) * 1000.0
    out[~clear] = np.nan
    return out


def severity(dnbr_x1000: np.ndarray) -> np.ndarray:
    """Severity class index per pixel (0 = unburned ... 4 = high, -1 = no data)."""
    classes = np.full(dnbr_x1000.shape, -1, dtype="int8")
    for index, (_, low, high) in enumerate(SEVERITY_CLASSES):
        classes[(dnbr_x1000 >= low) & (dnbr_x1000 < high)] = index
    return classes


def slope_degrees(dem: np.ndarray, pixel_m: float) -> np.ndarray:
    """Slope in degrees from elevation (m) on a square grid of pixel_m metres."""
    dz_dy, dz_dx = np.gradient(dem.astype("float64"), pixel_m)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")


def hillshade(dem: np.ndarray, pixel_m: float, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Hillshade (0-1) for map backgrounds, lit from the north-west."""
    dz_dy, dz_dx = np.gradient(dem.astype("float64"), pixel_m)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1).astype("float32")


def box_sum(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Count of True pixels in the (2r+1) x (2r+1) window around each pixel,
    via cumulative sums (O(n), no SciPy dependency)."""

    def along(values: np.ndarray, axis: int) -> np.ndarray:
        size = values.shape[axis]
        pad_before = np.zeros_like(values.take([0], axis=axis)).repeat(radius_px + 1, axis=axis)
        pad_after = np.zeros_like(values.take([0], axis=axis)).repeat(radius_px, axis=axis)
        sums = np.cumsum(np.concatenate([pad_before, values, pad_after], axis=axis), axis=axis)
        upper = sums.take(range(2 * radius_px + 1, 2 * radius_px + 1 + size), axis=axis)
        return upper - sums.take(range(size), axis=axis)

    return along(along(mask.astype("int32"), 0), 1)


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation with a square window of +-radius_px."""
    return mask.copy() if radius_px <= 0 else box_sum(mask, radius_px) > 0


def burn_footprint(dnbr_x1000: np.ndarray, radius_px: int = 2, min_share: float = 0.5) -> np.ndarray:
    """Burned pixels (dNBR >= 100) that sit in a mostly burned neighbourhood.

    Pre- and post-fire scenes weeks apart see different sun angles, so shaded
    slopes far from the fire show scattered "low severity" pixels. A real burn
    scar is contiguous; requiring half of the 5 x 5 window to be burned removes
    that speckle while keeping the scar and its edges.
    """
    burned = np.nan_to_num(dnbr_x1000, nan=-1e9) >= BURNED_MIN_DNBR
    window = (2 * radius_px + 1) ** 2
    return burned & (box_sum(burned, radius_px) >= min_share * window)


@dataclass
class BurnStats:
    """Area statistics of a burn scar inside the AOI."""

    aoi_ha: float
    clear_share: float  # share of the AOI with a valid dNBR on both dates
    burned_ha: float
    severity_ha: dict[str, float]
    mean_dnbr_burned: float | None
    mod_high_ha: float
    steep_mod_high_ha: float
    steep_mod_high_share: float  # of the burned area: the "T" term of Staley M1
    mean_slope_burned_deg: float | None
    land_cover_burned_ha: dict[str, float]
    built_up_burned_ha: float
    built_up_near_ha: float
    near_km: float


def burn_stats(
    dnbr_x1000: np.ndarray,
    slope_deg: np.ndarray,
    land_cover: np.ndarray,
    inside: np.ndarray,
    pixel_m: float,
    near_km: float = 2.0,
) -> BurnStats:
    """Burn-scar statistics over the AOI.

    The burn footprint (dNBR >= 100, despeckled by burn_footprint) is treated as one drainage area, a
    screening simplification of the per-basin USGS assessment. Built-up land
    "near" the scar is WorldCover built-up within near_km of moderate/high burn
    on steep slopes, where runoff and debris would come from; no flow routing.
    It is counted over the whole grid, which the caller reads with a margin
    around the AOI so that communities just outside a fire perimeter count.
    """
    pixel_ha = pixel_m * pixel_m / 10_000
    valid = inside & np.isfinite(dnbr_x1000)
    footprint = burn_footprint(dnbr_x1000)
    classes = severity(np.where(valid & footprint, dnbr_x1000, np.nan))
    burned = valid & footprint
    mod_high = burned & (dnbr_x1000 >= MOD_HIGH_MIN_DNBR)
    steep = slope_deg >= STEEP_SLOPE_DEG
    steep_mod_high = mod_high & steep
    burned_px = int(burned.sum())

    source = dilate(steep_mod_high, round(near_km * 1000 / pixel_m))
    built_up = land_cover == BUILT_UP

    land_cover_burned = {}
    for code, name in WORLDCOVER_CLASSES.items():
        hectares = float((burned & (land_cover == code)).sum()) * pixel_ha
        if hectares > 0:
            land_cover_burned[name] = round(hectares, 1)

    return BurnStats(
        aoi_ha=round(float(inside.sum()) * pixel_ha, 1),
        clear_share=round(float(valid.sum()) / max(int(inside.sum()), 1), 3),
        burned_ha=round(burned_px * pixel_ha, 1),
        severity_ha={
            name: round(float((classes == index).sum()) * pixel_ha, 1)
            for index, (name, _, _) in enumerate(SEVERITY_CLASSES)
            if index > 0
        },
        mean_dnbr_burned=round(float(np.nanmean(dnbr_x1000[burned])), 1) if burned_px else None,
        mod_high_ha=round(float(mod_high.sum()) * pixel_ha, 1),
        steep_mod_high_ha=round(float(steep_mod_high.sum()) * pixel_ha, 1),
        steep_mod_high_share=round(float(steep_mod_high.sum()) / burned_px, 3) if burned_px else 0.0,
        mean_slope_burned_deg=round(float(slope_deg[burned].mean()), 1) if burned_px else None,
        land_cover_burned_ha=land_cover_burned,
        built_up_burned_ha=round(float((burned & built_up).sum()) * pixel_ha, 1),
        built_up_near_ha=round(float((source & built_up & ~burned).sum()) * pixel_ha, 1),
        near_km=near_km,
    )
