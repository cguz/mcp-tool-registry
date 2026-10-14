"""
Supply-chain resilience matrix: EUDR risk combined with drought risk per plot.

The matrix is built here, deterministically, so the agent only has to copy it
into its answer instead of interpreting two sets of tool results. Drought risk
comes from outside this server (typically the climate_risk MCP server's
drought_index tool) and is passed in by the agent.
"""

from __future__ import annotations

import json
from typing import Any

RISK_LEVELS = ("low", "medium", "high")
UNKNOWN = "unknown"
RISK_ICONS = {"low": "🟢", "medium": "🟠", "high": "🔴", UNKNOWN: "⚪"}

# Supply status: can the plot supply the EU market now, and how reliably?
STATUS_BLOCKED = "blocked"
STATUS_AT_RISK = "at risk"
STATUS_REVIEW = "review"
STATUS_SECURE = "secure"
STATUS_ICONS = {STATUS_BLOCKED: "🔴", STATUS_REVIEW: "🟠", STATUS_AT_RISK: "🟠", STATUS_SECURE: "🟢"}

EUDR_ACTIONS = {
    "low": "EUDR: eligible; keep the geolocation on file.",
    "medium": "EUDR: verify the small post-2020 loss (imagery or field visit) before sourcing.",
    "high": "EUDR: hold sourcing and field-verify the post-2020 forest loss.",
    UNKNOWN: "EUDR: not assessed; fix the plot geometry and re-run.",
}
DROUGHT_ACTIONS = {
    "low": "Drought: no action.",
    "medium": "Drought: monitor rainfall; plan shade trees and mulching.",
    "high": "Drought: expect lower yields; prioritise for drought support or insurance and line up alternative supply.",
    UNKNOWN: "Drought: no drought result provided.",
}


def drought_level(value: Any) -> str:
    """Drought risk level from what the agent passed for one plot.

    Accepts a risk level string ("low", "medium", "high"), a climate_risk
    drought_index result (object or JSON string, its top-level risk_level is
    used), or anything else, which yields "unknown".
    """
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in RISK_LEVELS:
            return text.lower()
        try:
            value = json.loads(text)
        except ValueError:
            return UNKNOWN
    if isinstance(value, dict):
        level = str(value.get("risk_level", "")).lower()
        return level if level in RISK_LEVELS else UNKNOWN
    return UNKNOWN


def parse_drought(drought: dict | str | None) -> dict[str, str]:
    """Map of plot id -> drought risk level from the tool argument."""
    if drought is None:
        return {}
    mapping = json.loads(drought) if isinstance(drought, str) else drought
    if not isinstance(mapping, dict):
        raise ValueError("drought must be an object mapping plot id to a risk level or a drought_index result")
    return {str(plot_ref): drought_level(value) for plot_ref, value in mapping.items()}


def supply_status(eudr: str, drought: str) -> str:
    """Overall supply status: EUDR decides market access, drought decides reliability."""
    if eudr == "high":
        return STATUS_BLOCKED
    if eudr in ("medium", UNKNOWN):
        return STATUS_REVIEW
    if drought in ("medium", "high"):
        return STATUS_AT_RISK
    return STATUS_SECURE


def build_rows(eudr_results: list[dict], drought_by_id: dict[str, str]) -> list[dict[str, Any]]:
    """One matrix row per plot, in the order of the EUDR results."""
    rows = []
    for result in eudr_results:
        plot_ref = str(result["id"])
        eudr = result.get("risk_level", UNKNOWN)
        drought = drought_by_id.get(plot_ref, UNKNOWN)
        rows.append({
            "id": plot_ref,
            "eudr_risk": eudr,
            "deforestation_after_cutoff_ha": result.get("deforestation_after_cutoff_ha"),
            "drought_risk": drought,
            "supply_status": supply_status(eudr, drought),
            "recommendation": f"{EUDR_ACTIONS[eudr]} {DROUGHT_ACTIONS[drought]}",
        })
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """The resilience matrix as a markdown table, ready to paste into an answer."""
    lines = [
        "| Farm | EUDR risk | Drought risk | Supply status | Recommendation |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        eudr = f"{RISK_ICONS[row['eudr_risk']]} {row['eudr_risk']}"
        if row["deforestation_after_cutoff_ha"]:
            eudr += f" ({row['deforestation_after_cutoff_ha']:g} ha lost after 2020)"
        drought = f"{RISK_ICONS[row['drought_risk']]} {row['drought_risk']}"
        status = f"{STATUS_ICONS[row['supply_status']]} {row['supply_status']}"
        lines.append(f"| {row['id']} | {eudr} | {drought} | {status} | {row['recommendation']} |")
    return "\n".join(lines)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per supply status plus the plots that still lack a drought result."""
    statuses = (STATUS_SECURE, STATUS_AT_RISK, STATUS_REVIEW, STATUS_BLOCKED)
    return {
        "plots": len(rows),
        "by_supply_status": {status: sum(1 for r in rows if r["supply_status"] == status) for status in statuses},
        "missing_drought": [r["id"] for r in rows if r["drought_risk"] == UNKNOWN],
    }
