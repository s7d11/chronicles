"""Gate 3: Isolated Shadow DabaX(strict_parse=True) compilation & zero-regression check."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from beni.core.morphotactic.dabax import DabaX

from swarm_lab import normalize_lang_code
from swarm_lab.resources import _devmini_texts_for_lang, _mean_parse_across, score_text_with_dabax
from village_lab.schema import DabaXCheck

_DEVMINI_BASE_SCORE_CACHE: Dict[str, float] = {}


def _baseline_devmini_score(canon: str, dict_path: Path, gram_path: Path) -> float:
    cache_key = f"{canon}:{dict_path.resolve()}:{gram_path.resolve()}"
    if cache_key in _DEVMINI_BASE_SCORE_CACHE:
        return _DEVMINI_BASE_SCORE_CACHE[cache_key]
    dev_texts = _devmini_texts_for_lang(canon)[:20]
    if not dev_texts:
        _DEVMINI_BASE_SCORE_CACHE[cache_key] = 0.0
        return 0.0
    with tempfile.TemporaryDirectory() as tmp:
        dx = DabaX(
            canon,
            gram=gram_path,
            ldict=dict_path,
            runtime_dir=Path(tmp) / "rt",
            strict_parse=True,
            normalize_homoglyphs=True,
        )
        pct, _ = _mean_parse_across(dx, dev_texts)
    _DEVMINI_BASE_SCORE_CACHE[cache_key] = pct
    return pct


def _sanitize_gram_rule(rule_text: str, section_name: str = "section ed4_council_rules") -> Optional[str]:
    lines = [ln.strip() for ln in (rule_text or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    pat_lines = [ln for ln in lines if ln.startswith("pattern ")]
    if not pat_lines:
        return None
    has_section = any(ln.startswith("section ") for ln in lines)
    if has_section:
        return "\n".join(lines) + "\n"
    return f"{section_name}\n" + "\n".join(pat_lines) + "\n"


def verify_shadow_candidate(
    *,
    lang: str,
    working_dict_path: Path,
    working_gram_path: Path,
    candidate_term: str,
    mdf_block: str = "",
    va_addition: Optional[Dict[str, str]] = None,
    gram_rule: str = "",
) -> DabaXCheck:
    """Compile working dict + candidate in an isolated temp dir with strict DabaX."""
    canon = normalize_lang_code(lang)
    dev_texts = _devmini_texts_for_lang(canon)[:20]
    base_dev = _baseline_devmini_score(canon, working_dict_path, working_gram_path)

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        sh_dict = tdir / "shadow.dict"
        sh_gram = tdir / "shadow.gram"
        shutil.copyfile(working_dict_path, sh_dict)
        shutil.copyfile(working_gram_path, sh_gram)

        if mdf_block.strip():
            with sh_dict.open("a", encoding="utf-8") as fh:
                fh.write("\n" + mdf_block.strip() + "\n")
        elif va_addition and va_addition.get("va") and va_addition.get("target_lx"):
            _append_va_to_dict_file(
                sh_dict,
                target_lx=va_addition["target_lx"],
                va_form=va_addition["va"],
            )

        clean_gram = _sanitize_gram_rule(gram_rule) if gram_rule.strip() else None
        if gram_rule.strip() and not clean_gram:
            return DabaXCheck(
                dict_compile_ok=False,
                gram_compile_ok=False,
                candidate_parsed=False,
                baseline_devmini_pct=base_dev,
                shadow_devmini_pct=base_dev,
                regression_pct=0.0,
                error="Invalid .gram syntax (must include valid 'pattern ...' declaration)",
            )
        if clean_gram:
            with sh_gram.open("a", encoding="utf-8") as fh:
                fh.write("\n" + clean_gram + "\n")

        try:
            dx = DabaX(
                canon,
                gram=sh_gram,
                ldict=sh_dict,
                runtime_dir=tdir / "rt_shadow",
                strict_parse=True,
                normalize_homoglyphs=True,
            )
        except Exception as exc:
            return DabaXCheck(
                dict_compile_ok=not bool(mdf_block or va_addition),
                gram_compile_ok=not bool(clean_gram),
                candidate_parsed=False,
                baseline_devmini_pct=base_dev,
                shadow_devmini_pct=base_dev,
                regression_pct=0.0,
                error=f"DabaX compile error: {type(exc).__name__}: {str(exc)[:180]}",
            )

        cand_pct, _, _ = score_text_with_dabax(dx, candidate_term)
        sh_dev, _ = _mean_parse_across(dx, dev_texts) if dev_texts else (0.0, 0.0)
        regression = round(max(0.0, base_dev - sh_dev), 2)
        return DabaXCheck(
            dict_compile_ok=True,
            gram_compile_ok=True,
            candidate_parsed=(cand_pct >= 99.0),
            baseline_devmini_pct=base_dev,
            shadow_devmini_pct=sh_dev,
            regression_pct=regression,
            error="" if regression <= 0.5 else f"Regression {regression:.2f}% on dev-mini",
        )


def verify_shadow_batch(
    *,
    lang: str,
    working_dict_path: Path,
    working_gram_path: Path,
    items: Sequence[Dict[str, Any]],
) -> List[DabaXCheck]:
    """Optimistically verify multiple candidates in one DabaX compile; isolate per-item on failure."""
    if not items:
        return []
    if len(items) == 1:
        it = items[0]
        return [
            verify_shadow_candidate(
                lang=lang,
                working_dict_path=working_dict_path,
                working_gram_path=working_gram_path,
                candidate_term=str(it.get("candidate_term") or ""),
                mdf_block=str(it.get("mdf_block") or ""),
                va_addition=it.get("va_addition"),
                gram_rule=str(it.get("gram_rule") or ""),
            )
        ]

    canon = normalize_lang_code(lang)
    dev_texts = _devmini_texts_for_lang(canon)[:20]
    base_dev = _baseline_devmini_score(canon, working_dict_path, working_gram_path)

    cleaned_grams: List[Optional[str]] = []
    for it in items:
        gr = str(it.get("gram_rule") or "").strip()
        if gr:
            cg = _sanitize_gram_rule(gr)
            if not cg:
                # Bad .gram rule in batch -> isolate individually
                return [
                    verify_shadow_candidate(
                        lang=lang,
                        working_dict_path=working_dict_path,
                        working_gram_path=working_gram_path,
                        candidate_term=str(x.get("candidate_term") or ""),
                        mdf_block=str(x.get("mdf_block") or ""),
                        va_addition=x.get("va_addition"),
                        gram_rule=str(x.get("gram_rule") or ""),
                    )
                    for x in items
                ]
            cleaned_grams.append(cg)
        else:
            cleaned_grams.append(None)

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        sh_dict = tdir / "shadow.dict"
        sh_gram = tdir / "shadow.gram"
        shutil.copyfile(working_dict_path, sh_dict)
        shutil.copyfile(working_gram_path, sh_gram)

        for it, cg in zip(items, cleaned_grams):
            mdf = str(it.get("mdf_block") or "").strip()
            va_add = it.get("va_addition")
            if mdf:
                with sh_dict.open("a", encoding="utf-8") as fh:
                    fh.write("\n" + mdf + "\n")
            elif isinstance(va_add, dict) and va_add.get("va") and va_add.get("target_lx"):
                _append_va_to_dict_file(
                    sh_dict,
                    target_lx=va_add["target_lx"],
                    va_form=va_add["va"],
                )
            if cg:
                with sh_gram.open("a", encoding="utf-8") as fh:
                    fh.write("\n" + cg + "\n")

        try:
            dx = DabaX(
                canon,
                gram=sh_gram,
                ldict=sh_dict,
                runtime_dir=tdir / "rt_shadow",
                strict_parse=True,
                normalize_homoglyphs=True,
            )
            sh_dev, _ = _mean_parse_across(dx, dev_texts) if dev_texts else (0.0, 0.0)
            regression = round(max(0.0, base_dev - sh_dev), 2)
            if regression <= 0.5:
                out: List[DabaXCheck] = []
                for it in items:
                    term = str(it.get("candidate_term") or "")
                    cand_pct, _, _ = score_text_with_dabax(dx, term)
                    out.append(
                        DabaXCheck(
                            dict_compile_ok=True,
                            gram_compile_ok=True,
                            candidate_parsed=(cand_pct >= 99.0),
                            baseline_devmini_pct=base_dev,
                            shadow_devmini_pct=sh_dev,
                            regression_pct=regression,
                            error="",
                        )
                    )
                return out
        except Exception:
            pass

    return [
        verify_shadow_candidate(
            lang=lang,
            working_dict_path=working_dict_path,
            working_gram_path=working_gram_path,
            candidate_term=str(x.get("candidate_term") or ""),
            mdf_block=str(x.get("mdf_block") or ""),
            va_addition=x.get("va_addition"),
            gram_rule=str(x.get("gram_rule") or ""),
        )
        for x in items
    ]


def _append_va_to_dict_file(dict_path: Path, *, target_lx: str, va_form: str) -> bool:
    """Insert a standardized \\va line directly under matching \\lx in MDF file."""
    lines = dict_path.read_text(encoding="utf-8", errors="replace").splitlines()
    target_lower = target_lx.strip().lower()
    out: list[str] = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if not inserted and ln.startswith("\\lx ") and ln[4:].strip().lower() == target_lower:
            out.append(f"\\va {va_form.strip()}")
            inserted = True
    if not inserted:
        return False
    dict_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True
