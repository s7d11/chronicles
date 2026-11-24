"""Deterministic MALI13 dev-mini split builder with strict anti-leakage checks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import beni
from beni.core.compute.metrics import MorphologyScorer
from beni.core.morphotactic.dabax import DabaX
from beni.utils.normalize import clean_corpus_lines, normalize_homoglyphs

from bench import LANGS
from bench.schema import (
    ClozeSample,
    EfficiencySample,
    GenSample,
    LidSample,
    MorphSample,
    RefToken,
    SplitManifest,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "dev-mini"
DIACRITIC_CHARS = set("ɛɔɲŋɓɗƴăəɣḍḥṭẓʼƐƆƝŊƁƊƳĂƏƔḌḤṬẒàáâèéêìíîòóôùúû")
CONFUSABLE_PEERS: Dict[str, Tuple[str, ...]] = {
    "bam": ("mku", "kao"),
    "mku": ("bam", "kao"),
    "kao": ("mku", "bam"),
    "myk": ("spp", "bam"),
    "spp": ("myk", "bmq"),
    "boz": ("snk", "bam"),
    "snk": ("boz", "mku"),
    "mey": ("taq", "ful"),
    "taq": ("mey", "ses"),
    "dtm": ("bmq", "spp"),
    "bmq": ("dtm", "spp"),
    "ful": ("ses", "snk"),
    "ses": ("ful", "snk"),
}
_WORD_RE = re.compile(r"[^\s.,;:!?\"'()\[\]{}<>]+")


def _beni_data_dir() -> Path:
    return Path(beni.__file__).resolve().parent / "data"


def _norm_key(text: str) -> str:
    return " ".join(normalize_homoglyphs(text or "").strip().lower().split())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_excluded_hashes(data_dir: Optional[Path] = None) -> Set[str]:
    """Hash all lines from training/distillation sets for zero-overlap enforcement."""
    base = data_dir or _beni_data_dir()
    hashes: Set[str] = set()
    for fname in ("dataset_300_samples.jsonl", "dataset_10_samples.jsonl"):
        p = base / "raw" / fname
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text") or obj.get("sentence") or ""
            except json.JSONDecodeError:
                txt = line
            nk = _norm_key(str(txt))
            if nk:
                hashes.add(_sha256_text(nk))
    return hashes


def _load_clean_candidates(
    lang: str,
    excluded_hashes: Set[str],
    rng: random.Random,
    data_dir: Optional[Path] = None,
) -> List[str]:
    base = data_dir or _beni_data_dir()
    raw_file = base / "raw" / f"{lang}.txt"
    if not raw_file.is_file():
        raise FileNotFoundError(f"Missing raw corpus file: {raw_file}")

    raw_lines = raw_file.read_text(encoding="utf-8", errors="replace").splitlines()
    cleaned = clean_corpus_lines(raw_lines, normalize=True)

    # Also include test.json sentences for diversity if not in excluded_hashes
    test_json_path = base / "test.json"
    if test_json_path.is_file():
        try:
            tj = json.loads(test_json_path.read_text(encoding="utf-8"))
            keys = [lang] + (["mlq"] if lang == "mku" else [])
            for k in keys:
                val = tj.get(k)
                if isinstance(val, list):
                    cleaned.extend(clean_corpus_lines([str(x) for x in val], normalize=True))
                elif isinstance(val, str):
                    cleaned.extend(clean_corpus_lines(val.splitlines(), normalize=True))
        except Exception:
            pass

    seen: Set[str] = set()
    unique: List[str] = []
    for line in cleaned:
        words = line.split()
        if len(words) < 4 or len(words) > 42:
            continue
        if len(line) < 16 or len(line) > 320:
            continue
        # Skip lines dominated by digits or URLs
        if "http" in line.lower() or sum(ch.isdigit() for ch in line) > len(line) * 0.25:
            continue
        nk = _norm_key(line)
        h = _sha256_text(nk)
        if h in excluded_hashes or h in seen:
            continue
        seen.add(h)
        unique.append(line.strip())

    rng.shuffle(unique)
    return unique


def _batch_dabax_parse(dx: DabaX, lines: Sequence[str], chunk_size: int = 40) -> List[Any]:
    """Parse lines through DabaX, preserving 1-to-1 alignment with input lines."""
    results: List[Any] = []
    for i in range(0, len(lines), chunk_size):
        chunk = list(lines[i : i + chunk_size])
        try:
            parsed = dx.loader("\n".join(chunk))
            if len(parsed) == len(chunk):
                results.extend(parsed)
                continue
        except Exception:
            pass
        for single in chunk:
            try:
                p = dx.loader(single)
                results.append(p[0] if p else None)
            except Exception:
                results.append(None)
    return results


def _count_inflected_and_morphemes(sent: Any) -> Tuple[int, List[RefToken], List[str], List[str]]:
    ref_tokens: List[RefToken] = []
    flat_morphemes: List[str] = []
    stages: List[str] = []
    n_inflected = 0
    for tok in getattr(sent, "tokens", []) or []:
        st = str(getattr(tok, "stage", "-1"))
        morphs = [str(m) for m in tok.best_morphemes() if str(m).strip()]
        if not morphs:
            morphs = [str(tok.surface)]
        if st in {"0", "1"} and len(morphs) >= 2:
            n_inflected += 1
        ref_tokens.append(RefToken(surface=str(tok.surface), stage=st, morphemes=morphs))
        flat_morphemes.extend(morphs)
        stages.append(st)
    return n_inflected, ref_tokens, flat_morphemes, stages


def _word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _has_diacritic(word: str) -> bool:
    return any(ch in DIACRITIC_CHARS for ch in word)


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = row.model_dump() if hasattr(row, "model_dump") else dict(row)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_dev_mini(
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 2026,
    runtime_dir: Optional[Path] = None,
) -> SplitManifest:
    """Deterministically build the 728-row dev-mini MALI13 splits and manifest."""
    rng = random.Random(seed)
    excluded_hashes = load_excluded_hashes()
    rt_dir = runtime_dir or Path("/tmp/chronicles_dabax_runtime")
    rt_dir.mkdir(parents=True, exist_ok=True)
    scorer = MorphologyScorer()

    morph_rows: List[MorphSample] = []
    gen_rows: List[GenSample] = []
    lid_rows: List[LidSample] = []
    cloze_rows: List[ClozeSample] = []
    eff_rows: List[EfficiencySample] = []

    # Collect per-language pools first so cloze distractors can draw from confusable peers
    lang_pools: Dict[str, List[str]] = {}
    lang_parsed: Dict[str, List[Tuple[str, Any, float, int]]] = {}
    vocab_by_lang: Dict[str, List[str]] = {}

    for lang in LANGS:
        candidates = _load_clean_candidates(lang, excluded_hashes, rng)
        pool = candidates[:260]
        lang_pools[lang] = candidates
        dx = DabaX(lang, runtime_dir=rt_dir, strict_parse=True, normalize_homoglyphs=True)
        sents = _batch_dabax_parse(dx, pool)
        annotated: List[Tuple[str, Any, float, int]] = []
        inflected_vocab: Set[str] = set()
        for line, sent in zip(pool, sents):
            if sent is None or not getattr(sent, "tokens", None):
                continue
            rate = float(scorer.strict_parsed_rate(sent))
            n_inf, _, _, _ = _count_inflected_and_morphemes(sent)
            for tok in sent.tokens:
                if str(tok.stage) in {"0", "1"} and len(tok.best_morphemes()) >= 2:
                    inflected_vocab.add(str(tok.surface))
            annotated.append((line, sent, rate, n_inf))
        lang_parsed[lang] = annotated
        vocab_by_lang[lang] = sorted(inflected_vocab) or ["..."]

    for lang in LANGS:
        used_hashes: Set[str] = set()
        annotated = lang_parsed[lang]

        def _mark_used(line: str) -> None:
            used_hashes.add(_sha256_text(_norm_key(line)))

        def _is_unused(line: str) -> bool:
            return _sha256_text(_norm_key(line)) not in used_hashes

        # 1. morph.jsonl (10 per lang, length-stratified)
        qualifying = [
            item for item in annotated
            if item[2] >= 0.85 and item[3] >= 1 and 4 <= len(item[0].split()) <= 24
        ]
        if len(qualifying) < 10:
            qualifying = [
                item for item in annotated
                if item[2] >= 0.70 and len(item[0].split()) <= 26
            ]
        if len(qualifying) < 10:
            qualifying = sorted(annotated, key=lambda x: (-x[2], -x[3]))

        short = [x for x in qualifying if len(x[0].split()) <= 7]
        med = [x for x in qualifying if 8 <= len(x[0].split()) <= 14]
        long_ = [x for x in qualifying if len(x[0].split()) >= 15]
        picked_morph: List[Tuple[str, Any, float, int]] = []
        for bucket, target_n in ((short, 3), (med, 4), (long_, 3)):
            for item in bucket:
                if len(picked_morph) >= 10:
                    break
                if _is_unused(item[0]):
                    picked_morph.append(item)
                    _mark_used(item[0])
                if len([p for p in picked_morph if p in bucket]) >= target_n:
                    break
        for item in qualifying:
            if len(picked_morph) >= 10:
                break
            if _is_unused(item[0]):
                picked_morph.append(item)
                _mark_used(item[0])

        for idx, (line, sent, rate, n_inf) in enumerate(picked_morph[:10], start=1):
            _, ref_toks, flat_m, ref_st = _count_inflected_and_morphemes(sent)
            morph_rows.append(
                MorphSample(
                    id=f"{lang}-morph-{idx:04d}",
                    lang=lang,
                    text=line,
                    ref_tokens=ref_toks,
                    ref_morphemes=flat_m,
                    ref_stages=ref_st,
                    ref_strict_rate=round(rate, 4),
                    n_inflected=n_inf,
                )
            )

        # 2. gen.jsonl (8 per lang, >=12 words split at 5..7 words)
        gen_pool = [
            item[0] for item in annotated
            if _is_unused(item[0]) and len(item[0].split()) >= 12
        ]
        if len(gen_pool) < 8:
            gen_pool.extend(
                line for line in lang_pools[lang]
                if _is_unused(line) and len(line.split()) >= 10
            )
        if len(gen_pool) < 8:
            gen_pool.extend(
                line for line in lang_pools[lang]
                if _is_unused(line) and len(line.split()) >= 8
            )
        for idx, line in enumerate(gen_pool[:8], start=1):
            _mark_used(line)
            words = line.split()
            cut = min(7, max(5, len(words) // 2))
            gen_rows.append(
                GenSample(
                    id=f"{lang}-gen-{idx:04d}",
                    lang=lang,
                    prompt_prefix=" ".join(words[:cut]),
                    ref_continuation=" ".join(words[cut:]),
                )
            )

        # 3. lid.jsonl (10 per lang, 8..30 words)
        lid_pool = [
            line for line in lang_pools[lang]
            if _is_unused(line) and 8 <= len(line.split()) <= 30
        ]
        if len(lid_pool) < 10:
            lid_pool.extend(line for line in lang_pools[lang] if _is_unused(line))
        for idx, line in enumerate(lid_pool[:10], start=1):
            _mark_used(line)
            lid_rows.append(LidSample(id=f"{lang}-lid-{idx:04d}", lang=lang, text=line))

        # 4. cloze.jsonl (8 per lang, mask one inflected token)
        cloze_picked = 0
        for line, sent, _rate, _ninf in annotated:
            if cloze_picked >= 8:
                break
            if not _is_unused(line):
                continue
            target_tok = None
            for tok in sent.tokens:
                surf = str(tok.surface)
                if (
                    str(tok.stage) in {"0", "1"}
                    and len(tok.best_morphemes()) >= 2
                    and len(surf) >= 3
                    and re.search(rf"(?<!\w){re.escape(surf)}(?!\w)", line)
                ):
                    target_tok = tok
                    break
            if target_tok is None:
                continue
            target = str(target_tok.surface)
            masked = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", "[MASK]", line, count=1)
            if "[MASK]" not in masked:
                continue

            morphs = [str(m) for m in target_tok.best_morphemes() if str(m).strip()]
            bare = morphs[0] if morphs and morphs[0].lower() != target.lower() else target[: max(2, len(target) - 2)]
            same_lang_pool = [w for w in vocab_by_lang[lang] if w.lower() not in {target.lower(), bare.lower()}]
            same_d = rng.choice(same_lang_pool) if same_lang_pool else target + "w"
            peer_lang = rng.choice(CONFUSABLE_PEERS[lang])
            peer_pool = [w for w in vocab_by_lang.get(peer_lang, []) if w.lower() not in {target.lower(), bare.lower(), same_d.lower()}]
            peer_d = rng.choice(peer_pool) if peer_pool else bare + "lu"

            raw_opts = [
                ("gold", target),
                ("bare_stem", bare),
                ("same_lang_inflected", same_d),
                ("confusable_lang", peer_d),
            ]
            # Guarantee 4 distinct surface choices
            seen_opt = {target.lower()}
            final_opts: List[Tuple[str, str]] = [("gold", target)]
            for kind, val in raw_opts[1:]:
                cand = val
                while cand.lower() in seen_opt:
                    cand = cand + rng.choice(["a", "n", "w", "e"])
                seen_opt.add(cand.lower())
                final_opts.append((kind, cand))

            rng.shuffle(final_opts)
            letters = ["A", "B", "C", "D"]
            choices = {letters[i]: final_opts[i][1] for i in range(4)}
            answer = next(letters[i] for i in range(4) if final_opts[i][0] == "gold")
            distractor_types = [final_opts[i][0] for i in range(4) if final_opts[i][0] != "gold"]

            _mark_used(line)
            cloze_picked += 1
            cloze_rows.append(
                ClozeSample(
                    id=f"{lang}-cloze-{cloze_picked:04d}",
                    lang=lang,
                    masked_text=masked,
                    target=target,
                    choices=choices,
                    answer=answer,
                    distractor_types=distractor_types,
                )
            )

        # Fallback if a language yielded <8 inflected cloze rows
        if cloze_picked < 8:
            for line, sent, _r, _n in annotated:
                if cloze_picked >= 8:
                    break
                if not _is_unused(line):
                    continue
                words = [w for w in _word_tokens(line) if len(w) >= 4]
                if not words:
                    continue
                target = words[len(words) // 2]
                masked = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", "[MASK]", line, count=1)
                if "[MASK]" not in masked:
                    continue
                bare = target[: max(2, len(target) - 1)]
                peer_lang = rng.choice(CONFUSABLE_PEERS[lang])
                peer_d = rng.choice(vocab_by_lang.get(peer_lang, ["kɛ"]))
                same_d = rng.choice(vocab_by_lang.get(lang, [target + "w"]))
                opts = [("gold", target), ("bare_stem", bare), ("same_lang_inflected", same_d), ("confusable_lang", peer_d)]
                seen_opt = {target.lower()}
                fixed = [("gold", target)]
                for k, v in opts[1:]:
                    c = v
                    while c.lower() in seen_opt:
                        c = c + "a"
                    seen_opt.add(c.lower())
                    fixed.append((k, c))
                rng.shuffle(fixed)
                letters = ["A", "B", "C", "D"]
                choices = {letters[i]: fixed[i][1] for i in range(4)}
                answer = next(letters[i] for i in range(4) if fixed[i][0] == "gold")
                _mark_used(line)
                cloze_picked += 1
                cloze_rows.append(
                    ClozeSample(
                        id=f"{lang}-cloze-{cloze_picked:04d}",
                        lang=lang,
                        masked_text=masked,
                        target=target,
                        choices=choices,
                        answer=answer,
                        distractor_types=[fixed[i][0] for i in range(4) if fixed[i][0] != "gold"],
                    )
                )

        # 5. lm_efficiency.jsonl (20 per lang)
        eff_pool = [item for item in annotated if _is_unused(item[0])]
        if len(eff_pool) < 20:
            eff_pool.extend(item for item in annotated if item not in eff_pool)
        for idx, (line, sent, _r, _n) in enumerate(eff_pool[:20], start=1):
            _mark_used(line)
            _, _, flat_m, _ = _count_inflected_and_morphemes(sent)
            words = _word_tokens(line)
            ascii_toks = [w for w in words if not _has_diacritic(w)]
            diac_toks = [w for w in words if _has_diacritic(w)]
            eff_rows.append(
                EfficiencySample(
                    id=f"{lang}-eff-{idx:04d}",
                    lang=lang,
                    text=line,
                    utf8_bytes=len(line.encode("utf-8")),
                    n_words=max(1, len(words)),
                    n_dabax_morphemes=max(1, len(flat_m)),
                    ascii_tokens=ascii_toks,
                    diacritic_tokens=diac_toks,
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        "morph.jsonl": morph_rows,
        "gen.jsonl": gen_rows,
        "lid.jsonl": lid_rows,
        "cloze.jsonl": cloze_rows,
        "lm_efficiency.jsonl": eff_rows,
    }
    files_meta: Dict[str, Dict[str, Any]] = {}
    for fname, rows in split_map.items():
        p = out_dir / fname
        _write_jsonl(p, rows)
        files_meta[fname] = {"rows": len(rows), "sha256": _sha256_file(p)}

    # Anti-leakage verification over generated files
    overlap = audit_zero_overlap(out_dir, excluded_hashes)
    if overlap != 0:
        raise AssertionError(f"Anti-leakage check failed: {overlap} overlapping rows found")

    manifest = SplitManifest(
        tier="dev-mini",
        seed=seed,
        anti_leakage_overlap=overlap,
        excluded_hash_count=len(excluded_hashes),
        total_rows=sum(len(r) for r in split_map.values()),
        langs=list(LANGS),
        files=files_meta,
    )
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def audit_zero_overlap(data_dir: Path, excluded_hashes: Optional[Set[str]] = None) -> int:
    """Count any overlap between split text rows and excluded training/distillation hashes."""
    excl = excluded_hashes if excluded_hashes is not None else load_excluded_hashes()
    overlap = 0
    for fname in ("morph.jsonl", "gen.jsonl", "lid.jsonl", "cloze.jsonl", "lm_efficiency.jsonl"):
        p = data_dir / fname
        if not p.is_file():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            candidates = [
                obj.get("text"),
                obj.get("masked_text"),
                f"{obj.get('prompt_prefix', '')} {obj.get('ref_continuation', '')}".strip(),
            ]
            for c in candidates:
                if c and _sha256_text(_norm_key(str(c))) in excl:
                    overlap += 1
    return overlap


def verify_manifest(data_dir: Path = DEFAULT_OUT_DIR) -> SplitManifest:
    """Verify every frozen JSONL file matches manifest.json SHA-256 and row counts."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = SplitManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for fname, meta in manifest.files.items():
        fpath = data_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing split file: {fpath}")
        actual_sha = _sha256_file(fpath)
        if actual_sha != meta["sha256"]:
            raise ValueError(f"Checksum mismatch for {fname}: {actual_sha} != {meta['sha256']}")
        actual_rows = sum(1 for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip())
        if actual_rows != meta["rows"]:
            raise ValueError(f"Row count mismatch for {fname}: {actual_rows} != {meta['rows']}")
    return manifest


def load_split(data_dir: Path, task: str) -> List[Any]:
    """Load typed samples for morph|gen|lid|cloze|efficiency."""
    fname_map = {
        "morph": ("morph.jsonl", MorphSample),
        "gen": ("gen.jsonl", GenSample),
        "lid": ("lid.jsonl", LidSample),
        "cloze": ("cloze.jsonl", ClozeSample),
        "efficiency": ("lm_efficiency.jsonl", EfficiencySample),
    }
    fname, cls = fname_map[task]
    p = data_dir / fname
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(cls.model_validate_json(line))
    return out
