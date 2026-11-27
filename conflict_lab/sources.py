"""Parallel public OSINT retrievers for Volume II (HDX UCDP, ACLED summary, ReliefWeb, GDELT, fama.mil.ml DIRPA, public X)."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote_plus

import httpx

from conflict_lab import AGENT_CAST, OsintAgentSpec
from conflict_lab.gazetteer import normalize_actor_name, normalize_tactic_name
from conflict_lab.providers import extract_json_object, invoke_vol2_agent
from conflict_lab.schema import RawOsintRecord

_AGENT_MAP: Dict[str, OsintAgentSpec] = {sp.agent_id: sp for sp in AGENT_CAST}
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(raw: str) -> str:
    return _WS_RE.sub(" ", _HTML_TAG_RE.sub(" ", raw or "")).strip()


def _make_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"raw_{h}"


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def fetch_ucdp_mali(
    client: httpx.AsyncClient,
    *,
    max_rows: int = 1200,
) -> List[RawOsintRecord]:
    """Agent 1: Fetch disaggregated village/day UCDP GED Mali CSV from HDX CKAN."""
    pkg_url = "https://data.humdata.org/api/3/action/package_show?id=ucdp-data-for-mali"
    try:
        resp = await client.get(pkg_url, timeout=30.0)
        if resp.status_code != 200:
            return []
        resources = (resp.json().get("result") or {}).get("resources") or []
        csv_url = ""
        for res in resources:
            fmt = str(res.get("format") or "").lower()
            dl = str(res.get("download_url") or res.get("url") or "")
            if fmt == "csv" or dl.lower().endswith(".csv"):
                csv_url = dl
                break
        if not csv_url:
            return []

        csv_resp = await client.get(csv_url, timeout=60.0, follow_redirects=True)
        if csv_resp.status_code != 200 or not csv_resp.text.strip():
            return []

        text = csv_resp.text
        lines = text.splitlines()
        # HDX UCDP files sometimes include a second HXL `#...` header row right below column names
        if len(lines) >= 2 and lines[1].startswith("#"):
            lines = [lines[0]] + lines[2:]
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        parsed: List[RawOsintRecord] = []
        for row in reader:
            d_start = str(row.get("date_start") or "")[:10]
            if not d_start or len(d_start) < 10:
                continue
            lat_v: Optional[float] = None
            lon_v: Optional[float] = None
            try:
                lat_v = float(row.get("latitude") or "")
                lon_v = float(row.get("longitude") or "")
            except (TypeError, ValueError):
                lat_v, lon_v = None, None
            try:
                fat = int(float(row.get("best") or row.get("best_est") or 0))
            except (TypeError, ValueError):
                fat = 0
            side_a = str(row.get("side_a") or "").strip()
            side_b = str(row.get("side_b") or "").strip()
            actor_raw = f"{side_b or side_a}"
            if side_a and side_b:
                actor_raw = f"{side_b} vs {side_a}"
            adm1 = str(row.get("adm_1") or "").replace(" region", "").strip()
            adm2 = str(row.get("adm_2") or "").replace(" cercle", "").strip()
            loc_str = str(row.get("where_coordinates") or row.get("where_description") or adm2 or adm1).strip()
            headline = str(row.get("source_headline") or f"UCDP Event ({adm2 or adm1})").strip()
            ev_id = str(row.get("id") or row.get("relid") or _make_id("ucdp", d_start, loc_str, actor_raw))
            t_vio = str(row.get("type_of_violence") or "")
            tactic = normalize_tactic_name(f"{headline} {row.get('source_article', '')}")
            if tactic == "other" and t_vio == "3":
                tactic = "raid_assault"
            parsed.append(
                RawOsintRecord(
                    record_id=f"ucdp_{ev_id}",
                    source_name="ucdp_ged_hdx",
                    source_perspective="independent_event_db",
                    source_url=csv_url,
                    published_date=d_start,
                    event_date=d_start,
                    title=headline[:180],
                    summary=f"UCDP GED {row.get('source_article', '')} | Actors: {actor_raw}"[:320],
                    raw_location=loc_str,
                    admin1_hint=adm1,
                    admin2_hint=adm2,
                    lat=lat_v,
                    lon=lon_v,
                    actor_raw=actor_raw,
                    tactic_hint=tactic,
                    fatalities_reported=max(0, fat),
                    confidence=0.92,
                )
            )
        parsed.sort(key=lambda r: r.event_date, reverse=True)
        return parsed[:max_rows]
    except Exception:
        return []


async def fetch_hdx_acled_metadata(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Agent 2: Inspect HDX ACLED aggregated Mali conflict package metadata cleanly."""
    url = "https://data.humdata.org/api/3/action/package_show?id=mali-acled-conflict-data"
    try:
        resp = await client.get(url, timeout=20.0)
        if resp.status_code != 200:
            return {"ok": False, "resources": 0}
        res = (resp.json().get("result") or {})
        return {
            "ok": True,
            "metadata_modified": str(res.get("metadata_modified") or "")[:19],
            "resources": len(res.get("resources") or []),
            "dataset_url": "https://data.humdata.org/dataset/mali-acled-conflict-data",
        }
    except Exception:
        return {"ok": False, "resources": 0}


async def _llm_extract_events(
    client: httpx.AsyncClient,
    spec: OsintAgentSpec,
    *,
    source_name: str,
    perspective: str,
    items: Sequence[Dict[str, str]],
) -> List[RawOsintRecord]:
    """Use an assigned NVIDIA / OpenRouter :free agent to structure raw articles into candidate incidents."""
    if not items:
        return []
    snippets = []
    for i, it in enumerate(items[:14]):
        snippets.append(
            f"[{i}] Date={it.get('date', '')[:10]} | URL={it.get('url', '')[:120]} | "
            f"Title={it.get('title', '')[:140]} | Text={it.get('text', '')[:260]}"
        )
    prompt = (
        f"You are {spec.name} ({spec.role}). Inspect these Mali security/OSINT items from {source_name}.\n"
        "Extract concrete conflict, military, blockade, or security incidents occurring IN MALI. "
        "If an item omits an explicit event date, use the item's published Date.\n"
        "For each distinct Mali incident, return idx, event_date (YYYY-MM-DD), locality, admin2_cercle, admin1_region, "
        "actor_raw (e.g., JNIM, ISGS, FAMa, CSP-PSD, Dozo), tactic (ied_mine|ambush|raid_assault|indirect_fire|kidnapping_intimidation|airstrike_clash|blockade_siege|other), "
        "fatalities (int >= 0), and 1-sentence summary.\n\n"
        + "\n".join(snippets)
        + '\n\nRespond strictly as JSON:\n{"incidents":[{"idx":0,"event_date":"YYYY-MM-DD","locality":"...","admin2":"","admin1":"","actor_raw":"...","tactic":"ambush","fatalities":0,"summary":"..."}]}'
    )
    raw, _, err = await invoke_vol2_agent(
        client,
        spec,
        [{"role": "user", "content": prompt}],
        max_tokens=2400,
        temperature=0.1,
    )
    parsed = extract_json_object(raw) if not err else None
    out: List[RawOsintRecord] = []
    rows = (parsed or {}).get("incidents") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("idx", -1))
        except (TypeError, ValueError):
            idx = -1
        src_item = items[idx] if 0 <= idx < len(items) else {}
        ev_date = str(r.get("event_date") or src_item.get("date") or _today_iso())[:10]
        if len(ev_date) < 10:
            ev_date = _today_iso()
        loc = str(r.get("locality") or src_item.get("title") or "Mali").strip()
        actor = str(r.get("actor_raw") or "").strip()
        summary = str(r.get("summary") or src_item.get("title") or "").strip()
        try:
            fat = max(0, int(r.get("fatalities") or 0))
        except (TypeError, ValueError):
            fat = 0
        url = str(src_item.get("url") or "")
        out.append(
            RawOsintRecord(
                record_id=_make_id(source_name, ev_date, loc, actor, url),
                source_name=source_name,
                source_perspective=perspective,  # type: ignore[arg-type]
                source_url=url,
                published_date=str(src_item.get("date") or ev_date)[:10],
                event_date=ev_date,
                title=str(src_item.get("title") or summary)[:180],
                summary=summary[:320],
                raw_location=loc,
                admin1_hint=str(r.get("admin1") or ""),
                admin2_hint=str(r.get("admin2") or ""),
                actor_raw=normalize_actor_name(actor) if actor else "Unattributed Armed Group",
                tactic_hint=normalize_tactic_name(f"{r.get('tactic', '')} {summary}"),
                fatalities_reported=fat,
                confidence=0.72,
            )
        )
    return out


def _parse_rss_date(pub_date: str) -> str:
    from email.utils import parsedate_to_datetime

    s = (pub_date or "").strip()
    if not s:
        return _today_iso()
    try:
        return parsedate_to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return s[:10] if len(s) >= 10 and s[4] == "-" else _today_iso()


async def _fetch_rss_items(client: httpx.AsyncClient, url: str, *, keyword_filter: bool = False, limit: int = 16) -> List[Dict[str, str]]:
    import xml.etree.ElementTree as ET

    try:
        resp = await client.get(url, timeout=20.0, follow_redirects=True)
        if resp.status_code != 200 or "<item>" not in resp.text:
            return []
        root = ET.fromstring(resp.text)
        out: List[Dict[str, str]] = []
        for it in root.findall(".//item"):
            title = _strip_html(it.findtext("title", ""))
            link = (it.findtext("link", "") or "").strip()
            desc = _strip_html(it.findtext("description", ""))
            dt = _parse_rss_date(it.findtext("pubDate", ""))
            if keyword_filter and "mali" not in (title + " " + desc).lower():
                continue
            if title:
                out.append({"date": dt, "url": link, "title": title, "text": (desc or title)[:450]})
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


async def fetch_reliefweb_mali(client: httpx.AsyncClient, *, use_llm: bool = True) -> List[RawOsintRecord]:
    """Agent 3: Query ReliefWeb & UN humanitarian RSS feeds for Mali security & displacement reports."""
    raw_items: List[Dict[str, str]] = []
    for rss_url in (
        "https://reliefweb.int/updates/rss.xml?search=primary_country.iso3%3AMLI",
        "https://news.google.com/rss/search?q=Mali+OCHA+OR+UNHCR+OR+MINUSMA+OR+attaque+humanitaire&hl=fr&gl=FR&ceid=FR:fr",
    ):
        items = await _fetch_rss_items(client, rss_url, keyword_filter=True, limit=12)
        for it in items:
            if any(
                k in (it["title"] + " " + it["text"]).lower()
                for k in ("attack", "attaque", "security", "sécurit", "armed", "armé", "violence", "ied", "displacement", "blocus", "jnim", "kidal", "menaka", "mopti", "gao", "tombouctou")
            ):
                raw_items.append(it)
        if len(raw_items) >= 12:
            break

    if not raw_items:
        return []
    if use_llm:
        extracted = await _llm_extract_events(
            client,
            _AGENT_MAP["a3-reliefweb-envoy"],
            source_name="reliefweb_api",
            perspective="multilateral_un",
            items=raw_items[:14],
        )
        if extracted:
            return extracted
    return [
        RawOsintRecord(
            record_id=_make_id("reliefweb", it["date"], it["title"]),
            source_name="reliefweb_api",
            source_perspective="multilateral_un",
            source_url=it["url"],
            published_date=it["date"],
            event_date=it["date"],
            title=it["title"][:180],
            summary=it["text"][:280],
            raw_location=it["title"],
            tactic_hint=normalize_tactic_name(it["title"] + " " + it["text"]),
            confidence=0.68,
        )
        for it in raw_items[:10]
    ]


async def fetch_gdelt_mali(client: httpx.AsyncClient, *, use_llm: bool = True) -> List[RawOsintRecord]:
    """Agent 4: Query Google News FR + RFI Afrique + GDELT Doc 2.0 for live Mali security press."""
    items: List[Dict[str, str]] = []
    items.extend(
        await _fetch_rss_items(
            client,
            "https://news.google.com/rss/search?q=Mali+(attaque+OR+embuscade+OR+JNIM+OR+EIGS+OR+Kidal+OR+Menaka+OR+Mopti+OR+Segou)&hl=fr&gl=FR&ceid=FR:fr",
            limit=14,
        )
    )
    items.extend(await _fetch_rss_items(client, "https://www.rfi.fr/fr/afrique/rss", keyword_filter=True, limit=8))

    if len(items) < 6:
        q = quote_plus(
            "(Mali AND (attaque OR attack OR embuscade OR ambush OR IED OR JNIM OR FAMa OR Kidal OR Menaka OR Mopti))"
        )
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=ArtList&maxrecords=20&format=json&sort=DateDesc"
        try:
            resp = await client.get(url, timeout=20.0)
            if resp.status_code == 200 and "articles" in resp.text:
                for a in (resp.json().get("articles") or [])[:12]:
                    seendate = str(a.get("seendate") or "")
                    dt = f"{seendate[:4]}-{seendate[4:6]}-{seendate[6:8]}" if len(seendate) >= 8 else _today_iso()
                    items.append({
                        "date": dt,
                        "url": str(a.get("url") or ""),
                        "title": str(a.get("title") or ""),
                        "text": f"Domain={a.get('domain', '')} | Title={a.get('title', '')}",
                    })
        except Exception:
            pass

    if not items:
        return []
    if use_llm:
        extracted = await _llm_extract_events(
            client,
            _AGENT_MAP["a4-gdelt-wire"],
            source_name="gdelt_doc2",
            perspective="sahel_press",
            items=items[:14],
        )
        if extracted:
            return extracted
    return [
        RawOsintRecord(
            record_id=_make_id("gdelt", it["date"], it["url"], it["title"]),
            source_name="gdelt_doc2",
            source_perspective="sahel_press",
            source_url=it["url"],
            published_date=it["date"],
            event_date=it["date"],
            title=it["title"][:180],
            summary=it["text"][:260],
            raw_location=it["title"],
            actor_raw=normalize_actor_name(it["title"]),
            tactic_hint=normalize_tactic_name(it["title"]),
            confidence=0.62,
        )
        for it in items[:12]
    ]


async def fetch_fama_dirpa_official(client: httpx.AsyncClient, *, use_llm: bool = True) -> List[RawOsintRecord]:
    """Agent 5: Pull official Malian Armed Forces (FAMa DIRPA) communiqués from fama.mil.ml & indexed RSS."""
    items: List[Dict[str, str]] = []
    for candidate_url in (
        "https://fama.mil.ml/wp-json/wp/v2/posts?per_page=12",
        "https://fama.mil.ml/feed/",
        "https://fama.mil.ml/",
    ):
        try:
            resp = await client.get(candidate_url, timeout=14.0, follow_redirects=True)
            if resp.status_code != 200:
                continue
            if "wp-json" in candidate_url:
                posts = resp.json()
                if isinstance(posts, list):
                    for p in posts[:12]:
                        dt = str(p.get("date") or _today_iso())[:10]
                        title = _strip_html(str((p.get("title") or {}).get("rendered") or ""))
                        excerpt = _strip_html(str((p.get("excerpt") or {}).get("rendered") or ""))
                        content = _strip_html(str((p.get("content") or {}).get("rendered") or ""))
                        items.append({
                            "date": dt,
                            "url": str(p.get("link") or candidate_url),
                            "title": title,
                            "text": (excerpt or content)[:500],
                        })
                    if items:
                        break
            elif candidate_url.endswith("/feed/"):
                rss_items = await _fetch_rss_items(client, candidate_url, limit=12)
                if rss_items:
                    items.extend(rss_items)
                    break
            else:
                links = re.findall(r'<a[^>]+href="(https?://fama\.mil\.ml/[^"]+)"[^>]*>([^<]{15,180})</a>', resp.text, re.I)
                for href, label in links[:12]:
                    items.append({"date": _today_iso(), "url": href, "title": _strip_html(label), "text": _strip_html(label)})
                if items:
                    break
        except Exception:
            continue

    if not items:
        items.extend(
            await _fetch_rss_items(
                client,
                "https://news.google.com/rss/search?q=FAMa+DIRPA+Mali+OR+site:fama.mil.ml&hl=fr&gl=FR&ceid=FR:fr",
                keyword_filter=True,
                limit=12,
            )
        )

    if not items:
        return []
    if use_llm:
        extracted = await _llm_extract_events(
            client,
            _AGENT_MAP["a5-fama-dirpa"],
            source_name="fama_dirpa_official",
            perspective="official_fama_dirpa",
            items=items[:14],
        )
        if extracted:
            return extracted
    return [
        RawOsintRecord(
            record_id=_make_id("fama", it["date"], it["url"], it["title"]),
            source_name="fama_dirpa_official",
            source_perspective="official_fama_dirpa",
            source_url=it["url"],
            published_date=it["date"],
            event_date=it["date"],
            title=it["title"][:180],
            summary=it["text"][:280],
            raw_location=it["title"],
            actor_raw="FAMa / Allied Security Forces",
            tactic_hint=normalize_tactic_name(it["title"] + " " + it["text"]),
            confidence=0.78,
        )
        for it in items[:10]
    ]


async def fetch_public_x_osint(client: httpx.AsyncClient, *, use_llm: bool = True) -> List[RawOsintRecord]:
    """Agent 6: Pull public X posts via sanctioned X API v2 (when X_BEARER_TOKEN exists) or public RSS index."""
    items: List[Dict[str, str]] = []
    x_tok = (os.environ.get("X_BEARER_TOKEN") or "").strip()
    if x_tok:
        try:
            q = quote_plus('(from:FAMa_DIRPA OR (Mali (JNIM OR Kidal OR Menaka OR Mopti OR attaque OR embuscade))) -is:retweet')
            url = f"https://api.twitter.com/2/tweets/search/recent?query={q}&max_results=20&tweet.fields=created_at,text"
            resp = await client.get(url, headers={"Authorization": f"Bearer {x_tok}"}, timeout=20.0)
            if resp.status_code == 200:
                for tw in (resp.json().get("data") or []):
                    tid = str(tw.get("id") or "")
                    dt = str(tw.get("created_at") or _today_iso())[:10]
                    txt = str(tw.get("text") or "")
                    items.append({
                        "date": dt,
                        "url": f"https://x.com/i/web/status/{tid}" if tid else "https://x.com",
                        "title": txt[:120],
                        "text": txt[:400],
                    })
        except Exception:
            pass

    if not items:
        items.extend(
            await _fetch_rss_items(
                client,
                "https://news.google.com/rss/search?q=Mali+JNIM+OR+FAMa+OR+Kidal+OR+Tonka+OR+x.com&hl=fr&gl=FR&ceid=FR:fr",
                keyword_filter=True,
                limit=12,
            )
        )

    if not items:
        return []
    if use_llm:
        extracted = await _llm_extract_events(
            client,
            _AGENT_MAP["a6-public-x-osint"],
            source_name="public_x_osint",
            perspective="public_x_post",
            items=items,
        )
        if extracted:
            return extracted
    return [
        RawOsintRecord(
            record_id=_make_id("x_osint", it["date"], it["url"], it["title"]),
            source_name="public_x_osint",
            source_perspective="public_x_post",
            source_url=it["url"],
            published_date=it["date"],
            event_date=it["date"],
            title=it["title"][:180],
            summary=it["text"][:280],
            raw_location=it["title"],
            actor_raw=normalize_actor_name(it["title"]),
            tactic_hint=normalize_tactic_name(it["title"] + " " + it["text"]),
            confidence=0.58,
        )
        for it in items[:8]
    ]


async def gather_all_sources(
    client: httpx.AsyncClient,
    *,
    max_ucdp_rows: int = 1200,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Run all 6 retrieval agents concurrently."""
    ucdp, acled_meta, reliefweb, gdelt, fama, x_osint = await asyncio.gather(
        fetch_ucdp_mali(client, max_rows=max_ucdp_rows),
        fetch_hdx_acled_metadata(client),
        fetch_reliefweb_mali(client, use_llm=use_llm),
        fetch_gdelt_mali(client, use_llm=use_llm),
        fetch_fama_dirpa_official(client, use_llm=use_llm),
        fetch_public_x_osint(client, use_llm=use_llm),
    )
    records = list(ucdp) + list(reliefweb) + list(gdelt) + list(fama) + list(x_osint)
    return {
        "records": records,
        "source_counts": {
            "ucdp_ged_hdx": len(ucdp),
            "reliefweb_api": len(reliefweb),
            "gdelt_doc2": len(gdelt),
            "fama_dirpa_official": len(fama),
            "public_x_osint": len(x_osint),
        },
        "acled_hdx_meta": acled_meta,
    }
