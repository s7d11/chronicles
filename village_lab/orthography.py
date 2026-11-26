"""Gate 1 & Gate 2: DNAFLA/AMALAN Standardization & Lemma-vs-Rule Boundary."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from beni.utils.normalize import normalize_homoglyphs

from bench.metrics import _EN_FR_NOISE
from swarm_lab import normalize_lang_code
from swarm_lab.resources import _norm_key, load_baseline_resources
from village_lab.schema import OrthographyCheck

# DNAFLA (1982) / AMALAN standardized Bambara letter inventory + hyphen/apostrophe
BAMBARA_ALLOWED_CHARS: Set[str] = set("abcdeɛfghijklmnoɔprstuwyzɲŋn-'’\u0300\u0301\u0302\u030c")
_EXTRA_NON_MALI_NOISE: Set[str] = {"bonjour", "merci", "voila", "alors", "donc", "exemple", "dictionary"}

# Common Bambara regular inflection/derivation patterns already governed by .gram rules
_BAM_INFLECTION_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("w", "regular plural noun/adj suffix (-w -> inflection_PFVINTR_PROG_PL)"),
    ("baga", "agentive suffix (-baga -> n_v_derivation)"),
    ("baa", "agentive suffix (-baa -> n_v_derivation)"),
    ("len", "resultative participle (-len -> participles_PTCPRES)"),
    ("nen", "resultative participle (-nen -> participles_PTCPRES)"),
    ("tɔ", "progressive participle (-tɔ -> participles)"),
    ("ta", "potential/participle suffix (-ta -> participles)"),
    ("li", "verbal noun suffix (-li -> n_v_derivation)"),
    ("ni", "verbal noun suffix (-ni -> n_v_derivation)"),
    ("ya", "abstract/inchoative derivation (-ya -> n_v_derivation)"),
    ("la", "locative/instrument suffix (-la -> inflection_MNT12_OPT2_DIR)"),
    ("ra", "perfective intransitive (-ra -> inflection_PFVINTR_PROG_PL)"),
    ("na", "nasalized aspect/locative (-na -> inflection_PFVINTR_PROG_PL)"),
    ("bali", "privative derivation (-bali -> participles)"),
    ("ntan", "privative derivation (-ntan -> n_derivation)"),
)

_TOKEN_CLEAN_RE = re.compile(r"^[^\w\u00C0-\u024F\u0250-\u02AF\u1E00-\u1EFF]+|[^\w\u00C0-\u024F\u0250-\u02AF\u1E00-\u1EFF]+$")


def strip_combining_marks(text: str) -> str:
    """Return lowercase base string without tone diacritics for stem comparison."""
    nfd = unicodedata.normalize("NFD", text or "")
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped).lower()


@lru_cache(maxsize=32)
def _baseline_char_inventory(lang: str) -> Set[str]:
    """Derive observed canonical alphabet directly from read-only baseline.dict."""
    canon = normalize_lang_code(lang)
    if canon == "bam":
        return set(BAMBARA_ALLOWED_CHARS)
    res = load_baseline_resources(canon)
    chars: Set[str] = set("abcdefghijklmnopqrstuvwxyz-'’")
    for key in list(res.get("dict_index", {}).keys())[:4000]:
        for ch in unicodedata.normalize("NFC", key.lower()):
            if ch.isalpha() or ch in "-'’":
                chars.add(ch)
    return chars


def normalize_bambara_dnafla(word: str) -> Tuple[str, List[str]]:
    """Apply DNAFLA Bambara normalization checks and return (standardized, notes)."""
    notes: List[str] = []
    txt = unicodedata.normalize("NFC", normalize_homoglyphs(word or "")).strip()
    txt = _TOKEN_CLEAN_RE.sub("", txt).lower()
    if not txt:
        return "", ["empty_token"]

    orig = txt
    # Legacy French digraphs / vowels often typed instead of official DNAFLA letters
    if "ny" in txt:
        txt = txt.replace("ny", "ɲ")
        notes.append("normalized French digraph 'ny' -> DNAFLA 'ɲ'")
    if txt.startswith("ng") or "ng" in txt:
        # Word-initial or post-vocalic 'ng' commonly represents velar nasal ŋ in non-standard spelling
        if txt.startswith("ng"):
            txt = "ŋ" + txt[2:]
            notes.append("normalized initial 'ng' -> DNAFLA 'ŋ'")
    if "è" in txt:
        txt = txt.replace("è", "ɛ")
        notes.append("normalized accent 'è' -> DNAFLA 'ɛ'")
    if "ò" in txt:
        txt = txt.replace("ò", "ɔ")
        notes.append("normalized accent 'ò' -> DNAFLA 'ɔ'")
    if "é" in txt:
        txt = txt.replace("é", "é")
        notes.append("normalized 'é' -> DNAFLA e+acute tone mark")

    txt = unicodedata.normalize("NFC", txt)
    allowed = BAMBARA_ALLOWED_CHARS
    bad_chars = sorted({ch for ch in unicodedata.normalize("NFD", txt) if ch not in allowed})
    if bad_chars:
        notes.append(f"non_DNAFLA_characters:{''.join(bad_chars)}")
    if txt != orig and not notes:
        notes.append("normalized_homoglyph_or_nfc")
    return txt, notes


def check_standardization(
    word: str,
    lang: str,
    *,
    working_dict_keys: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> OrthographyCheck:
    """Gate 1: Verify orthography, anti-loanword hygiene, and baseline deduplication."""
    canon = normalize_lang_code(lang)
    base_res = load_baseline_resources(canon)
    dict_index: Dict[str, List[Dict[str, str]]] = working_dict_keys or base_res["dict_index"]

    raw_clean = _TOKEN_CLEAN_RE.sub("", unicodedata.normalize("NFC", word or "").strip())
    if not raw_clean or len(raw_clean) < 2 or len(raw_clean) > 48:
        return OrthographyCheck(
            raw_form=word,
            standardized_form=raw_clean,
            is_standard=False,
            route_hint="reject",
            issues=["invalid_length"],
            rationale="Token is empty, too short (<2), or longer than 48 characters.",
        )

    if any(ch.isdigit() for ch in raw_clean):
        return OrthographyCheck(
            raw_form=word,
            standardized_form=raw_clean,
            is_standard=False,
            route_hint="reject",
            issues=["contains_digits"],
            rationale="Candidate contains numeric digits.",
        )

    if strip_combining_marks(raw_clean) in (_EN_FR_NOISE | _EXTRA_NON_MALI_NOISE):
        return OrthographyCheck(
            raw_form=word,
            standardized_form=raw_clean.lower(),
            is_standard=False,
            route_hint="reject",
            issues=["en_fr_noise"],
            rationale="Rejected by MALI13 hygiene guard as French/English function word or noise.",
        )

    issues: List[str] = []
    if canon == "bam":
        std_form, notes = normalize_bambara_dnafla(raw_clean)
        issues.extend(notes)
        has_bad = any(n.startswith("non_DNAFLA_characters:") or n == "empty_token" for n in notes)
        is_std = not has_bad
    else:
        std_form = _norm_key(raw_clean)
        inv = _baseline_char_inventory(canon)
        bad = sorted({ch for ch in unicodedata.normalize("NFC", std_form) if ch not in inv})
        if bad:
            issues.append(f"non_standard_chars:{''.join(bad)}")
        is_std = bool(std_form) and not bad

    if not is_std:
        return OrthographyCheck(
            raw_form=word,
            standardized_form=std_form or raw_clean,
            is_standard=False,
            normalized_changed=(std_form != raw_clean.lower()),
            route_hint="reject",
            issues=issues or ["orthography_rejected"],
            rationale=f"Fails {canon} standardized character inventory: {', '.join(issues)}",
        )

    nk_std = _norm_key(std_form)
    if nk_std in dict_index:
        existing_lx = dict_index[nk_std][0].get("lx", nk_std)
        return OrthographyCheck(
            raw_form=word,
            standardized_form=std_form,
            is_standard=True,
            normalized_changed=(std_form != raw_clean.lower()),
            already_in_dict=True,
            existing_headword=existing_lx,
            route_hint="reject",
            issues=issues + ["already_attested_in_dict"],
            rationale=f"Already attested under standardized headword '\\lx {existing_lx}'.",
        )

    # Check toneless stem match -> route as standardized variant (\va) if close match exists
    toneless = strip_combining_marks(std_form)
    for k, recs in dict_index.items():
        if strip_combining_marks(k) == toneless and recs:
            return OrthographyCheck(
                raw_form=word,
                standardized_form=std_form,
                is_standard=True,
                normalized_changed=(std_form != raw_clean.lower()),
                already_in_dict=False,
                existing_headword=recs[0].get("lx", k),
                route_hint="mdf_va",
                issues=issues + ["tonal_or_dialect_variant_of_existing_lx"],
                rationale=(
                    f"Matches tonal/orthographic stem of '\\lx {recs[0].get('lx', k)}'; "
                    "integrate as \\va variant rather than duplicate \\lx."
                ),
            )

    return OrthographyCheck(
        raw_form=word,
        standardized_form=std_form,
        is_standard=True,
        normalized_changed=(std_form != raw_clean.lower()),
        already_in_dict=False,
        route_hint="mdf_lx",
        issues=issues,
        rationale=(
            f"Conforms to standardized {canon} orthography"
            + (f" (normalized: {'; '.join(issues)})" if issues else "")
            + "."
        ),
    )


def classify_lemma_vs_rule(
    word: str,
    pos: str,
    lang: str,
    *,
    working_dict_keys: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> Tuple[str, str]:
    """Gate 2: Determine if candidate belongs in MDF (\\lx/\\va) vs. .gram morphotactics."""
    canon = normalize_lang_code(lang)
    base_res = load_baseline_resources(canon)
    dict_index = working_dict_keys or base_res["dict_index"]
    nk = _norm_key(word)
    toneless = strip_combining_marks(nk)

    if "-" in nk:
        parts = [p for p in nk.split("-") if len(p) >= 2]
        if parts and all((_norm_key(p) in dict_index) for p in parts):
            return (
                "gram_rule",
                f"Compound '{nk}' decomposes cleanly into attested stems ({' + '.join(parts)}); model via .gram composition rather than bloating .dict.",
            )

    if canon == "bam" and len(toneless) >= 4:
        for suf, desc in _BAM_INFLECTION_SUFFIXES:
            if toneless.endswith(suf) and len(toneless) - len(suf) >= 2:
                stem = toneless[: -len(suf)]
                # Check direct or toneless match for root stem
                for k, recs in dict_index.items():
                    if strip_combining_marks(k) == stem and recs:
                        return (
                            "gram_rule",
                            f"Appears to be derived/inflected from attested root '\\lx {recs[0].get('lx', k)}' via {desc}.",
                        )

    return (
        "mdf_lx",
        f"Unattested lexical stem for pos={pos or 'n'}; eligible for Council & DabaX Gate as standardized MDF entry.",
    )


def parse_working_dict_index(dict_path: Path) -> Tuple[Dict[str, List[Dict[str, str]]], int, int]:
    """Count \\lx and \\va entries and index headwords from an evolving MDF .dict file."""
    index: Dict[str, List[Dict[str, str]]] = {}
    lx_count = 0
    va_count = 0
    cur: Dict[str, str] = {}
    variants: List[str] = []

    def _flush() -> None:
        nonlocal lx_count, cur, variants
        if not cur.get("lx"):
            cur = {}
            variants = []
            return
        lx_count += 1
        rec = {
            "lx": cur.get("lx", ""),
            "ps": cur.get("ps", ""),
            "ge": cur.get("ge", "") or cur.get("gve", ""),
            "gf": cur.get("gf", "") or cur.get("gvf", ""),
        }
        for key in [_norm_key(cur["lx"])] + [_norm_key(v) for v in variants]:
            if key:
                bucket = index.setdefault(key, [])
                if len(bucket) < 4:
                    bucket.append(rec)
        cur = {}
        variants = []

    for raw in dict_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            _flush()
            continue
        if line.startswith("\\lx "):
            _flush()
            cur["lx"] = line[4:].strip()
        elif line.startswith("\\va "):
            va_count += 1
            variants.append(line[4:].strip())
        elif line.startswith("\\ps ") and "ps" not in cur:
            cur["ps"] = line[4:].strip()
        elif line.startswith("\\ge ") and "ge" not in cur:
            cur["ge"] = line[4:].strip()
        elif line.startswith("\\gf ") and "gf" not in cur:
            cur["gf"] = line[4:].strip()
        elif line.startswith("\\gve ") and "gve" not in cur:
            cur["gve"] = line[5:].strip()
        elif line.startswith("\\gvf ") and "gvf" not in cur:
            cur["gvf"] = line[5:].strip()
    _flush()
    return index, lx_count, va_count
