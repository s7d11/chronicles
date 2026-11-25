"""Read-only Sebeni resource loaders, baseline diagnostics, and isolated Shadow DabaX verification."""

from __future__ import annotations

import csv
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from beni.core.compute.helpers import is_empr_token
from beni.core.morphotactic.dabax import DabaX
from beni.utils.normalize import normalize_homoglyphs

from bench import LANG_NAMES
from bench.dataset import DEFAULT_OUT_DIR, load_split
from bench.metrics import _EN_FR_NOISE, _WORD_RE
from swarm_lab import ALL_LANGS, normalize_lang_code
from swarm_lab.schema import DictDelta, NativeGenResult, ShadowPatchResult

SEBENI_DATA_DIR = Path("/home/seb/sebenx/sebeni/beni/data")
TEST_JSON_PATH = SEBENI_DATA_DIR / "test.json"
BASELINES_DIR = SEBENI_DATA_DIR / "baselines"
PROMPT_NATIVE_PATH = Path(__file__).resolve().parent.parent / "prompt_native.csv"

_BASELINE_CACHE: Dict[str, Dict[str, Any]] = {}
_DEVMINI_MORPH_CACHE: Dict[str, List[str]] = {}


def _norm_key(word: str) -> str:
    txt = normalize_homoglyphs(word or "")
    txt = unicodedata.normalize("NFC", txt).strip().lower()
    txt = re.sub(r"^[^\w\u00C0-\u024F\u0250-\u02AF\u1E00-\u1EFF]+|[^\w\u00C0-\u024F\u0250-\u02AF\u1E00-\u1EFF]+$", "", txt)
    return txt


def load_test_texts() -> Dict[str, str]:
    """Load test.json from Sebeni read-only and canonicalize ISO keys (mlq -> mku)."""
    raw = json.loads(TEST_JSON_PATH.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for k, v in raw.items():
        canon = normalize_lang_code(k)
        if canon in ALL_LANGS and isinstance(v, str) and v.strip():
            out[canon] = v.strip()
    return out


def load_prompt_native() -> Dict[str, Dict[str, str]]:
    """Load prompt_native.csv and normalize keys (seq -> spp)."""
    out: Dict[str, Dict[str, str]] = {}
    with PROMPT_NATIVE_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None}
            canon = normalize_lang_code(cleaned.get("Language", ""))
            if not canon:
                continue
            out[canon] = {
                "Text": cleaned.get("Text", "Text"),
                "Title": cleaned.get("Title", "Title"),
                "Content": cleaned.get("Content", "Content"),
                "Context": cleaned.get("Context", "Context"),
                "Grammar": cleaned.get("Grammar", "Grammar"),
            }
    return out


def _parse_mdf_dict(dict_path: Path) -> Tuple[Dict[str, List[Dict[str, str]]], int]:
    """Index Toolbox/MDF entries by normalized headword/variant without mutating files."""
    index: Dict[str, List[Dict[str, str]]] = {}
    entry_count = 0
    cur: Dict[str, str] = {}
    variants: List[str] = []

    def _flush() -> None:
        nonlocal entry_count, cur, variants
        if not cur.get("lx"):
            cur = {}
            variants = []
            return
        entry_count += 1
        rec = {
            "lx": cur.get("lx", ""),
            "ps": cur.get("ps", ""),
            "ge": cur.get("ge", "") or cur.get("gve", ""),
            "gf": cur.get("gf", "") or cur.get("gvf", ""),
        }
        keys = [_norm_key(cur["lx"])] + [_norm_key(v) for v in variants]
        for k in keys:
            if k:
                bucket = index.setdefault(k, [])
                if len(bucket) < 4:
                    bucket.append(rec)
        cur = {}
        variants = []

    for raw_line in dict_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            _flush()
            continue
        if line.startswith("\\lx "):
            _flush()
            cur["lx"] = line[4:].strip()
        elif line.startswith("\\va "):
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
    return index, entry_count


def load_baseline_resources(lang: str) -> Dict[str, Any]:
    """Read baseline.gram and indexed baseline.dict read-only for one language."""
    canon = normalize_lang_code(lang)
    if canon in _BASELINE_CACHE:
        return _BASELINE_CACHE[canon]
    bdir = BASELINES_DIR / canon
    gram_path = bdir / "baseline.gram"
    dict_path = bdir / "baseline.dict"
    gram_text = gram_path.read_text(encoding="utf-8", errors="replace") if gram_path.is_file() else ""
    dict_index, n_entries = _parse_mdf_dict(dict_path) if dict_path.is_file() else ({}, 0)

    gram_lines = gram_text.splitlines()
    excerpt_lines = [ln for ln in gram_lines if ln.strip() and not ln.strip().startswith("# version")][:85]
    gram_excerpt = "\n".join(excerpt_lines)[:3400]

    res = {
        "lang": canon,
        "gram_path": gram_path,
        "dict_path": dict_path,
        "gram_excerpt": gram_excerpt,
        "gram_chars": len(gram_text),
        "dict_index": dict_index,
        "dict_entries": n_entries,
    }
    _BASELINE_CACHE[canon] = res
    return res


def lookup_dict_candidates(lang: str, surface: str, limit: int = 3) -> List[Dict[str, str]]:
    """Find exact and longest-prefix/stem dictionary matches for a surface token."""
    res = load_baseline_resources(lang)
    idx: Dict[str, List[Dict[str, str]]] = res["dict_index"]
    nk = _norm_key(surface)
    if not nk:
        return []
    hits: List[Dict[str, str]] = []
    seen = set()

    def _add(rec: Dict[str, str], match_kind: str) -> None:
        key = (rec["lx"], rec["ps"], rec["ge"] or rec["gf"])
        if key in seen:
            return
        seen.add(key)
        hits.append({**rec, "match": match_kind})

    for rec in idx.get(nk, []):
        _add(rec, "exact")

    # Strip common hyphenated segments & prefixes/suffixes for stems
    parts = [p for p in re.split(r"[-’']", nk) if len(p) >= 2]
    for part in parts:
        for rec in idx.get(part, []):
            _add(rec, "segment")

    # Shortest peel (prefix up to 3 chars, suffix up to 4 chars)
    if len(hits) < limit and len(nk) >= 4:
        for trim_l in range(0, min(3, len(nk) - 2)):
            for trim_r in range(0, min(4, len(nk) - trim_l - 2)):
                sub = nk[trim_l : len(nk) - trim_r if trim_r else None]
                if len(sub) < 3 or sub == nk:
                    continue
                for rec in idx.get(sub, []):
                    _add(rec, "stem")
                    if len(hits) >= limit:
                        return hits[:limit]

    return hits[:limit]


def score_text_with_dabax(dx: DabaX, text: str) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Run DabaX on text and return (true_parse_pct, phantom_pct, token_rows)."""
    morph_map = dx.extract_morphology(text or "")
    rows: List[Dict[str, Any]] = []
    total = 0
    parsed = 0
    phantom = 0
    for sent_tokens in morph_map.values():
        for tok in sent_tokens:
            surf = str(tok.get("token", "")).strip()
            if not surf:
                continue
            total += 1
            stage = str(tok.get("stage", "-1"))
            analyses = tok.get("analyses") or []
            if is_empr_token(tok):
                parsed += 1
            else:
                try:
                    ok_stage = stage not in ("-1", "tokenizer") and int(float(stage)) >= 0
                except (TypeError, ValueError):
                    ok_stage = False
                has_gloss = any(
                    bool((a or {}).get("gloss")) or bool((a or {}).get("morphemes"))
                    for a in analyses
                )
                if ok_stage and has_gloss:
                    parsed += 1
                elif ok_stage and not has_gloss:
                    phantom += 1
            rows.append({"token": surf, "stage": stage, "analyses": analyses})
    if total == 0:
        return 0.0, 0.0, rows
    return round(100.0 * parsed / total, 2), round(100.0 * phantom / total, 2), rows


def diagnose_test_text(
    lang: str,
    text: str,
    *,
    runtime_dir: Optional[Path] = None,
    max_targets: int = 8,
) -> Dict[str, Any]:
    """Analyze test.json[lang] using DabaX(strict_parse=True) and pick target words."""
    canon = normalize_lang_code(lang)
    rt = runtime_dir or (Path(__file__).resolve().parent.parent / "runs" / "edition2" / "_rt_base" / canon)
    dx = DabaX(canon, runtime_dir=rt, strict_parse=True, normalize_homoglyphs=True)
    parse_pct, phantom_pct, token_rows = score_text_with_dabax(dx, text)
    base_res = load_baseline_resources(canon)

    unparsed: List[Dict[str, Any]] = []
    parsed_complex: List[Dict[str, Any]] = []
    seen = set()

    for row in token_rows:
        surf = row["token"]
        nk = _norm_key(surf)
        if not nk or len(nk) < 2 or nk.isdigit() or nk in seen:
            continue
        seen.add(nk)
        stage = row["stage"]
        analyses = row["analyses"]
        first_a = analyses[0] if analyses else {}
        morphemes = [str(m.get("form", "")) for m in (first_a.get("morphemes") or []) if m.get("form")]
        item = {
            "surface": surf,
            "stage": stage,
            "baseline_ps": first_a.get("ps", []),
            "baseline_gloss": first_a.get("gloss", ""),
            "baseline_morphemes": morphemes or ([first_a.get("form", "")] if first_a.get("form") else []),
            "dict_hits": lookup_dict_candidates(canon, surf, limit=2),
        }
        if stage in ("-1", "tokenizer"):
            unparsed.append(item)
        elif len(morphemes) >= 2 or "-" in surf or len(nk) >= 7:
            parsed_complex.append(item)

    targets = (unparsed[:12] + parsed_complex[:max_targets])[:max_targets]
    return {
        "lang": canon,
        "lang_name": LANG_NAMES.get(canon, canon),
        "n_tokens": len(token_rows),
        "baseline_parse_pct": parse_pct,
        "baseline_phantom_pct": phantom_pct,
        "gram_excerpt": base_res["gram_excerpt"],
        "dict_entries_total": base_res["dict_entries"],
        "target_words": targets,
        "unparsed_count": len(unparsed),
    }


def _devmini_texts_for_lang(lang: str) -> List[str]:
    canon = normalize_lang_code(lang)
    if canon in _DEVMINI_MORPH_CACHE:
        return _DEVMINI_MORPH_CACHE[canon]
    samples = load_split(DEFAULT_OUT_DIR, "morph")
    texts = [s.text for s in samples if s.lang == canon]
    _DEVMINI_MORPH_CACHE[canon] = texts
    return texts


def _sanitize_mdf_entry(delta: DictDelta) -> Optional[str]:
    lx = re.sub(r"[\r\n\\]+", "", delta.lx).strip()
    if not lx or len(lx) > 64 or " " in lx:
        return None
    ps = re.sub(r"[^A-Za-z0-9_./-]", "", delta.ps or "n")[:16] or "n"
    ge = re.sub(r"[\r\n\\]+", " ", delta.ge or "gloss").strip()[:120] or "gloss"
    gf = re.sub(r"[\r\n\\]+", " ", delta.gf or ge).strip()[:120] or ge
    return "\n".join(
        [
            f"\\lx {lx}",
            f"\\ps {ps}",
            f"\\ge {ge}",
            f"\\gf {gf}",
            "\\rem SWARM_CANDIDATE",
            "",
        ]
    )


def _mean_parse_across(dx: DabaX, texts: Sequence[str]) -> Tuple[float, float]:
    if not texts:
        return 0.0, 0.0
    p_sum, ph_sum = 0.0, 0.0
    for txt in texts:
        p, ph, _ = score_text_with_dabax(dx, txt)
        p_sum += p
        ph_sum += ph
    return round(p_sum / len(texts), 2), round(ph_sum / len(texts), 2)


def run_shadow_dabax_test(
    lang: str,
    test_text: str,
    dict_deltas: Sequence[DictDelta],
    gram_rules: Sequence[str],
    shadow_dir: Path,
) -> ShadowPatchResult:
    """Copy baseline.{dict,gram} into isolated shadow_dir, apply consensus deltas, and score."""
    canon = normalize_lang_code(lang)
    base_res = load_baseline_resources(canon)
    shadow_dir.mkdir(parents=True, exist_ok=True)
    devmini_texts = _devmini_texts_for_lang(canon)

    base_rt = shadow_dir / "rt_base"
    base_dx = DabaX(canon, runtime_dir=base_rt, strict_parse=True, normalize_homoglyphs=True)
    base_test_pct, _, _ = score_text_with_dabax(base_dx, test_text)
    base_dev_pct, _ = _mean_parse_across(base_dx, devmini_texts)

    existing_keys = set(base_res["dict_index"].keys())
    new_blocks: List[str] = []
    seen_new = set()
    for d in dict_deltas:
        nk = _norm_key(d.lx)
        if not nk or nk in existing_keys or nk in seen_new:
            continue
        block = _sanitize_mdf_entry(d)
        if block:
            seen_new.add(nk)
            new_blocks.append(block)

    sh_dict = shadow_dir / "baseline.dict"
    sh_gram = shadow_dir / "baseline.gram"
    shutil.copyfile(base_res["dict_path"], sh_dict)
    shutil.copyfile(base_res["gram_path"], sh_gram)

    if new_blocks:
        with sh_dict.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(new_blocks))

    dict_compile_ok = False
    gram_compile_ok = True
    shadow_dx: Optional[DabaX] = None
    err_msg: Optional[str] = None

    try:
        shadow_dx = DabaX(
            canon,
            gram=sh_gram,
            ldict=sh_dict,
            runtime_dir=shadow_dir / "rt_dict",
            strict_parse=True,
            normalize_homoglyphs=True,
        )
        dict_compile_ok = True
    except Exception as exc:
        err_msg = f"dict compile failed: {type(exc).__name__}: {str(exc)[:140]}"

    clean_rules = [
        r.strip()
        for r in gram_rules
        if r and r.strip() and not r.strip().startswith("#") and ("->" in r or "pattern" in r)
    ]
    if dict_compile_ok and clean_rules:
        sh_gram_patched = shadow_dir / "baseline.patched.gram"
        shutil.copyfile(sh_gram, sh_gram_patched)
        with sh_gram_patched.open("a", encoding="utf-8") as f:
            f.write("\n# SWARM_CANDIDATE_RULES\n" + "\n".join(clean_rules) + "\n")
        try:
            gram_dx = DabaX(
                canon,
                gram=sh_gram_patched,
                ldict=sh_dict,
                runtime_dir=shadow_dir / "rt_both",
                strict_parse=True,
                normalize_homoglyphs=True,
            )
            shadow_dx = gram_dx
            gram_compile_ok = True
        except Exception as exc:
            gram_compile_ok = False
            err_msg = f"gram rule fallback to dict-only ({type(exc).__name__}: {str(exc)[:120]})"

    if shadow_dx is None:
        return ShadowPatchResult(
            dict_entries_added=len(new_blocks),
            gram_rules_proposed=len(clean_rules),
            dict_compile_ok=False,
            gram_compile_ok=False,
            baseline_test_parse_pct=base_test_pct,
            shadow_test_parse_pct=base_test_pct,
            test_delta_pct=0.0,
            baseline_devmini_parse_pct=base_dev_pct,
            shadow_devmini_parse_pct=base_dev_pct,
            devmini_delta_pct=0.0,
            phantom_rate_pct=0.0,
            error=err_msg,
        )

    sh_test_pct, sh_phantom, _ = score_text_with_dabax(shadow_dx, test_text)
    sh_dev_pct, _ = _mean_parse_across(shadow_dx, devmini_texts)

    return ShadowPatchResult(
        dict_entries_added=len(new_blocks),
        gram_rules_proposed=len(clean_rules),
        dict_compile_ok=dict_compile_ok,
        gram_compile_ok=gram_compile_ok,
        baseline_test_parse_pct=base_test_pct,
        shadow_test_parse_pct=sh_test_pct,
        test_delta_pct=round(sh_test_pct - base_test_pct, 2),
        baseline_devmini_parse_pct=base_dev_pct,
        shadow_devmini_parse_pct=sh_dev_pct,
        devmini_delta_pct=round(sh_dev_pct - base_dev_pct, 2),
        phantom_rate_pct=sh_phantom,
        error=err_msg,
    )


def evaluate_native_generation(
    lang: str,
    command_template: str,
    native_labels: Dict[str, str],
    output_text: str,
    shadow_dir: Optional[Path] = None,
    verified_by_peer: bool = False,
) -> NativeGenResult:
    """Evaluate native-command generation against baseline/shadow DabaX & field headers."""
    canon = normalize_lang_code(lang)
    base_dx = DabaX(
        canon,
        runtime_dir=(shadow_dir / "rt_base") if shadow_dir else None,
        strict_parse=True,
        normalize_homoglyphs=True,
    )
    base_pct, _, _ = score_text_with_dabax(base_dx, output_text)

    sh_pct = base_pct
    if shadow_dir and (shadow_dir / "baseline.dict").is_file():
        rt_use = shadow_dir / ("rt_both" if (shadow_dir / "rt_both").exists() else "rt_dict")
        gram_use = shadow_dir / ("baseline.patched.gram" if (shadow_dir / "baseline.patched.gram").is_file() else "baseline.gram")
        try:
            sh_dx = DabaX(
                canon,
                gram=gram_use,
                ldict=shadow_dir / "baseline.dict",
                runtime_dir=rt_use,
                strict_parse=True,
                normalize_homoglyphs=True,
            )
            sh_pct, _, _ = score_text_with_dabax(sh_dx, output_text)
        except Exception:
            sh_pct = base_pct

    words = _WORD_RE.findall(output_text or "")
    noise = sum(1 for w in words if w.lower() in _EN_FR_NOISE)
    leak_pct = round(100.0 * noise / max(1, len(words)), 2)

    check_keys = ("Title", "Content", "Context", "Grammar")
    lower_out = (output_text or "").lower()
    matched = sum(1 for k in check_keys if native_labels.get(k, "").lower() in lower_out)
    compliance = round(100.0 * matched / len(check_keys), 1)

    return NativeGenResult(
        command_template=command_template,
        native_fields=native_labels,
        output_text=output_text,
        baseline_parse_pct=base_pct,
        shadow_parse_pct=sh_pct,
        leakage_pct=leak_pct,
        header_compliance=compliance,
        verified_by_peer=verified_by_peer,
    )
