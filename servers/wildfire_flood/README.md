# Wildfire Flood — MCP server

**Compound hazard: wildfire scar → debris flow / flash flood.** After a fire,
burned slopes lose the vegetation and litter that held water and soil, and the
soil surface often becomes water-repellent. Short bursts of rain that were
harmless before the fire now run off and carry debris downslope, often months
after the flames are out (e.g. Montecito 2018, Los Angeles February 2025).

This server answers, for a burned area: how big and how severe is the scar,
will the rain in the coming days (or a past storm) cross the rainfall threshold
for debris flows, and how much built-up land lies close to the steep burned
slopes. All data are free and need no API key.

| Step | Data | How |
|---|---|---|
| Burn scar | Copernicus Sentinel-2 L2A (B08, B12, SCL), via the Earth Search STAC API, COGs on AWS | dNBR = NBR(pre) − NBR(post), USGS FIREMON severity classes |
| Terrain | Copernicus DEM GLO-30 (AWS Open Data) | slope; share of moderate/high burn on slopes ≥ 23° |
| Land cover | ESA WorldCover 2021 v200 (AWS Open Data) | what burned; built-up land near the scar |
| Rain | Open-Meteo forecast and historical-forecast APIs | 15-minute and hourly precipitation |
| Hazard | USGS M1 model, Staley et al. (2017) | debris-flow likelihood and the 50 % rainfall threshold |

## Tools

| Tool | Input | Output |
|---|---|---|
| `burn_severity` | `fire_start` (YYYY-MM-DD), optional `fire_end` (containment, in the past); the area as `geometry` (GeoJSON polygon, e.g. the fire perimeter), `bbox` [W, S, E, N] or `latitude` + `longitude` (+ `radius_km`, default 5) | burned ha, ha per severity class, mean dNBR, share of moderate/high burn on steep slopes, burned land cover, built-up land burned and within 2 km, scenes used, provenance |
| `postfire_flood_risk` | as `burn_severity`, plus `rain_start` / `rain_end` (default: today and the next 7 days, forecast) and optional `soil_kf` | `verdict` sentence, `risk_level` (low / medium / high), `likelihood`, threshold and peak rain for 15, 30 and 60 min, rain bursts above the threshold, burn summary, provenance |
| `cascade_figure` | as `postfire_flood_risk`, plus optional `title` | PNG (~45-70 KB): severity map over terrain with built-up land, and the rain bars against the threshold line; then a JSON block with verdict and provenance |

The area is limited to 2,500 km² (larger areas are read on a coarser grid
above ~1,000 km²). The rain window is at most 31 days and reaches at most
15 days into the future.

Images are returned inline as MCP `ImageContent`, in line with EVE's MCP file
convention (`backend/docs/mcp-file-return-convention.md`): EVE stores them as
chat artifacts and keeps the bytes out of the LLM context.

## Method

1. **Scenes.** Sentinel-2 L2A scenes up to 120 days before `fire_start` and
   after `fire_end` (scene cloud cover < 60 %) are searched on Earth Search.
   Pre and post scenes must come from the same MGRS tile so their pixels line
   up; the tile covering the area best is used. For each date the closest scene
   that is *usable* over at least 90 % of the area is taken (clear in the
   Scene Classification and with enough NIR + SWIR2 signal), else the most
   usable of five. The signal check matters in winter: steep slopes facing
   away from a low sun are in deep shadow and give no reliable NBR.
2. **Grid.** Everything is warped onto one UTM grid at 20 m (coarser for very
   large areas) covering the area plus a 2 km margin, with GDAL WarpedVRTs
   over HTTP range requests, so only the needed blocks are downloaded.
   Reflectance uses each asset's STAC `scale` / `offset` (products since 2022
   carry a −0.1 offset).
3. **Burn severity.** NBR = (B08 − B12) / (B08 + B12); dNBR × 1000 classified
   as low (100–269), moderate-low (270–439), moderate-high (440–659) and high
   (≥ 660) (Key & Benson 2006). The burn footprint keeps burned pixels whose
   5 × 5 neighbourhood is at least half burned, which removes isolated
   illumination speckle and keeps the contiguous scar.
4. **Hazard (USGS M1, Staley et al. 2017).**
   `likelihood = 1 / (1 + exp(−(b0 + b1·T·R + b2·F·R + b3·S·R)))` with
   T = share of the burned area at moderate/high severity on slopes ≥ 23°,
   F = mean dNBR / 1000, S = soil erodibility KF (default 0.25) and R = peak
   rain accumulation over 15, 30 or 60 minutes. The 50 % threshold is the R that
   gives a likelihood of 0.5; the controlling duration is the one with the
   highest likelihood. Risk level: high ≥ 50 %, medium ≥ 20 %, else low.
5. **Exposure.** WorldCover built-up land within 2 km of moderate/high burn on
   steep slopes (outside the scar itself) counts as exposed.

Coefficients (b0, b1, b2, b3): 15 min −3.63, 0.41, 0.67, 0.70; 30 min −3.61,
0.26, 0.39, 0.50; 60 min −3.21, 0.17, 0.20, 0.22.

## Validation: Eaton fire, Los Angeles (7–31 January 2025)

`test.py` replays the storm of 13–14 February 2025, which triggered debris
flows in the Eaton and Palisades burn areas:

| Check | Result |
|---|---|
| Burned area in the bbox [-118.17, 34.17, -118.02, 34.27] | ~6,000 ha (official perimeter: ~5,700 ha) |
| 50 % threshold | ~3.8 mm in 15 min (~15 mm/h) |
| Peak rain, 14 Feb 2025 01:15–01:30 UTC | 12.4 mm in 15 min → likelihood ~100 %, **high** |
| Dry week, 1–7 July 2025 | likelihood ~4 %, **low** |

## Files

| File | Contents |
|---|---|
| `server.py` | FastMCP, the three tools, the burn-analysis cache and provenance |
| `aoi.py` | Area input (GeoJSON, bbox, point + radius), geodesic area, fingerprint, tile coverage |
| `burn.py` | NumPy maths: reflectance, NBR, dNBR, severity, slope, hillshade, despeckling, statistics |
| `hazard.py` | USGS M1 model, thresholds, rain peaks and bursts |
| `sources.py` | All network I/O: STAC search, scene choice, warped COG reads, Open-Meteo |
| `figure.py` | Two-panel cascade figure (matplotlib, headless), palette PNG |
| `test_aoi.py`, `test_burn.py`, `test_hazard.py` | Offline tests (pytest), no network, no LLM |
| `test.py` | Live test over MCP/HTTP on the Eaton fire (no LLM); writes `cascade.png` |

## Run and test

From `mcp-tool-registry/`:

```bash
pip install -r servers/wildfire_flood/requirements.txt
cd servers/wildfire_flood
python -m pytest -q test_aoi.py test_burn.py test_hazard.py   # offline, < 1 s
python server.py --port 8103 &                                 # HTTP on :8103/mcp
python test.py http://localhost:8103/mcp                       # live, ~1 min the first time
```

The first call for a fire downloads the scenes (~20-60 s); later calls on the
same fire and area reuse an in-memory cache and take well under a second.

Register in a local EVE: `POST /mcp-servers` with
`{"name": "wildfire_flood", "enabled": true, "config": {"url": "http://localhost:8103/mcp", "transport": "streamable_http"}}`.

## Limitations

- A screening for situational awareness, **not an official warning**; follow
  local authorities (in the US, the USGS post-fire debris-flow assessments).
- M1 was calibrated in the western United States; elsewhere the likelihood
  is indicative.
- The burn footprint is treated as one drainage basin; the USGS assessment
  works per basin with flow routing.
- Soil erodibility (KF) is not mapped globally; pass `soil_kf` if known.
- Exposure is a distance buffer, not a flow path: it does not say which
  houses lie downstream.
- 15-minute rain is native only in North America and central Europe; elsewhere
  Open-Meteo interpolates it from hourly data, which smooths short bursts (the
  hourly model is then often the controlling one).
- Scenes weeks apart can differ in illumination; shaded slopes may show
  scattered low-severity pixels. Passing the fire perimeter as `geometry`
  gives the cleanest statistics.
- Open-Meteo's free API is for non-commercial use.
