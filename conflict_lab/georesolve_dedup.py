"""Agent 7 & Agent 9: Malian georesolution, spatiotemporal deduplication (±2d, 15km) & batched Critic audit."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import httpx

from conflict_lab import AGENT_CAST
from conflict_lab.gazetteer import (
    haversine_km,
    normalize_actor_name,
    normalize_tactic_name,
    resolve_mali_location,
)
from conflict_lab.providers import extract_json_object, invoke_vol2_agent
from conflict_lab.schema import RawOsintRecord, VerifiedIncident


def _parse_date_days(date_str: str) -> int:
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.toordinal()
    except Exception:
        return datetime(2024, 1, 1).toordinal()


def _bucket_compatible(
    existing: Dict[str, Any],
    *,
    date_ord: int,
    lat: float,
    lon: float,
    cercle_id: str,
    actor: str,
    tactic: str,
) -> bool:
    if abs(existing["date_ord"] - date_ord) > 2:
        return False
    if existing["cercle_id"] != cercle_id and haversine_km(existing["lat"], existing["lon"], lat, lon) > 15.0:
        return False
    if haversine_km(existing["lat"], existing["lon"], lat, lon) > 20.0:
        return False
    # Different actors & tactics within the same day/cercle still merge only when one is Unattributed / other
    if (
        actor != "Unattributed Armed Group"
        and existing["actor_primary"] != "Unattributed Armed Group"
        and actor != existing["actor_primary"]
        and tactic != existing["tactic"]
    ):
        return False
    return True


def deduplicate_and_georesolve(records: Sequence[RawOsintRecord]) -> List[VerifiedIncident]:
    """Merge multi-source records within (±2d, <=15km) into VerifiedIncident records."""
    ordered = sorted(records, key=lambda r: (r.event_date, -r.confidence, r.record_id))
    clusters: List[Dict[str, Any]] = []

    for rec in ordered:
        cercle, precision = resolve_mali_location(
            raw_location=rec.raw_location,
            admin1_hint=rec.admin1_hint,
            admin2_hint=rec.admin2_hint,
            lat=rec.lat,
            lon=rec.lon,
        )
        lat = float(rec.lat) if (rec.lat is not None and precision == "exact_locality") else cercle.lat
        lon = float(rec.lon) if (rec.lon is not None and precision == "exact_locality") else cercle.lon
        d_ord = _parse_date_days(rec.event_date)
        actor = normalize_actor_name(rec.actor_raw)
        tactic = rec.tactic_hint if rec.tactic_hint != "other" else normalize_tactic_name(f"{rec.title} {rec.summary}")

        matched: Optional[Dict[str, Any]] = None
        for b in reversed(clusters):
            if _bucket_compatible(
                b,
                date_ord=d_ord,
                lat=lat,
                lon=lon,
                cercle_id=cercle.cercle_id,
                actor=actor,
                tactic=tactic,
            ):
                matched = b
                break

        if matched is None:
            clusters.append(
                {
                    "date_start": rec.event_date[:10],
                    "date_end": rec.event_date[:10],
                    "date_ord": d_ord,
                    "precision_level": precision,
                    "admin1": cercle.admin1,
                    "admin2": cercle.name,
                    "cercle_id": cercle.cercle_id,
                    "locality": rec.raw_location or cercle.name,
                    "lat": lat,
                    "lon": lon,
                    "actor_primary": actor,
                    "tactic": tactic,
                    "fatalities_best": max(0, int(rec.fatalities_reported)),
                    "sources": [rec.source_url or rec.source_name],
                    "source_names": {rec.source_name},
                    "perspectives": [rec.source_perspective],
                    "headline": rec.title,
                    "summary": rec.summary,
                    "adjudicator_note": "",
                }
            )
        else:
            if rec.event_date[:10] < matched["date_start"]:
                matched["date_start"] = rec.event_date[:10]
            if rec.event_date[:10] > matched["date_end"]:
                matched["date_end"] = rec.event_date[:10]
            if precision == "exact_locality" and matched["precision_level"] != "exact_locality":
                matched["lat"] = lat
                matched["lon"] = lon
                matched["precision_level"] = precision
                matched["locality"] = rec.raw_location or matched["locality"]
            if matched["actor_primary"] == "Unattributed Armed Group" and actor != "Unattributed Armed Group":
                matched["actor_primary"] = actor
            if matched["tactic"] == "other" and tactic != "other":
                matched["tactic"] = tactic
            matched["fatalities_best"] = max(matched["fatalities_best"], max(0, int(rec.fatalities_reported)))
            src_ref = rec.source_url or rec.source_name
            if src_ref and src_ref not in matched["sources"]:
                matched["sources"].append(src_ref)
            matched["source_names"].add(rec.source_name)
            if rec.source_perspective not in matched["perspectives"]:
                matched["perspectives"].append(rec.source_perspective)

    out: List[VerifiedIncident] = []
    for b in clusters:
        uniq_src_count = max(len(b["source_names"]), len(b["sources"]))
        if uniq_src_count >= 3 or len(b["perspectives"]) >= 2:
            tier = "high"
        elif uniq_src_count == 2 or "ucdp_ged_hdx" in b["source_names"]:
            tier = "medium"
        else:
            tier = "single_source"

        inc_hash = hashlib.sha256(
            f"{b['cercle_id']}|{b['date_start']}|{b['actor_primary']}|{b['tactic']}|{b['headline'][:60]}".encode("utf-8")
        ).hexdigest()[:14]
        out.append(
            VerifiedIncident(
                incident_id=f"inc_{inc_hash}",
                date_start=b["date_start"],
                date_end=b["date_end"],
                precision_level=b["precision_level"],
                admin1=b["admin1"],
                admin2=b["admin2"],
                cercle_id=b["cercle_id"],
                locality=b["locality"][:120],
                lat=round(float(b["lat"]), 4),
                lon=round(float(b["lon"]), 4),
                actor_primary=b["actor_primary"],
                tactic=b["tactic"],  # type: ignore[arg-type]
                fatalities_best=int(b["fatalities_best"]),
                source_count=uniq_src_count,
                sources=b["sources"][:10],
                perspectives=b["perspectives"],
                corroboration_tier=tier,  # type: ignore[arg-type]
                headline=b["headline"][:200],
                summary=b["summary"][:320],
                adjudicator_note=b["adjudicator_note"],
            )
        )
    out.sort(key=lambda x: (x.date_start, x.incident_id), reverse=True)
    return out


async def adjudicate_recent_incidents_batched(
    client: httpx.AsyncClient,
    incidents: List[VerifiedIncident],
    *,
    batch_size: int = 20,
    max_to_audit: int = 40,
) -> List[VerifiedIncident]:
    """Agent 9 (Gemini 3.1 Flash-Lite): Audit recent OSINT-extracted incidents in batches of 15-25."""
    spec = {sp.agent_id: sp for sp in AGENT_CAST}["a9-adjudicator"]
    candidates = [inc for inc in incidents if "independent_event_db" not in inc.perspectives][:max_to_audit]
    if not candidates:
        return incidents

    drop_ids = set()
    notes: Dict[str, str] = {}
    for start in range(0, len(candidates), max(1, min(25, batch_size))):
        chunk = candidates[start : start + batch_size]
        lines = [
            f"[{i}] Date={inc.date_start} | Cercle={inc.admin2} ({inc.admin1}) | Actor={inc.actor_primary} | "
            f"Tactic={inc.tactic} | Perspectives={','.join(inc.perspectives)} | Headline={inc.headline[:120]}"
            for i, inc in enumerate(chunk)
        ]
        prompt = (
            "You are Agent 9 (Gemini 3.1 Flash-Lite OSINT Critic). Review this batch of extracted Mali incidents.\n"
            "Mark verdict='REJECT' ONLY if clearly non-Mali, duplicate commentary, or generic non-event press noise.\n"
            "Keep official FAMa DIRPA & corroborated media reports (`ACCEPT`), noting any claim vs external corroboration nuance.\n\n"
            + "\n".join(lines)
            + '\n\nRespond strictly as JSON:\n{"audits":[{"idx":0,"verdict":"ACCEPT|REJECT","note":"short rationale"}]}'
        )
        raw, _, err = await invoke_vol2_agent(
            client,
            spec,
            [{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1,
        )
        parsed = extract_json_object(raw) if not err else None
        audits = (parsed or {}).get("audits") if isinstance(parsed, dict) else None
        if isinstance(audits, list):
            for a in audits:
                if not isinstance(a, dict) or a.get("idx") is None:
                    continue
                try:
                    idx = int(a["idx"])
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(chunk):
                    inc = chunk[idx]
                    if str(a.get("verdict") or "").upper() == "REJECT":
                        drop_ids.add(inc.incident_id)
                    elif a.get("note"):
                        notes[inc.incident_id] = str(a["note"])[:220]

    kept: List[VerifiedIncident] = []
    for inc in incidents:
        if inc.incident_id in drop_ids:
            continue
        if inc.incident_id in notes:
            inc.adjudicator_note = notes[inc.incident_id]
        kept.append(inc)
    return kept
