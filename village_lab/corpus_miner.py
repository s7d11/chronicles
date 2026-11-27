"""Corpus OOV miner reading beni/data/raw/<lang>.txt read-only for Edition No. 4."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from beni.core.compute.helpers import is_empr_token
from beni.core.morphotactic.dabax import DabaX

from bench.metrics import _EN_FR_NOISE
from swarm_lab import normalize_lang_code
from swarm_lab.resources import SEBENI_DATA_DIR, _norm_key
from village_lab.orthography import strip_combining_marks

_TOK_RE = re.compile(
    r"[A-Za-z\u00C0-\u024F\u0250-\u02AF\u1E00-\u1EFF]+"
    r"(?:-[A-Za-z\u00C0-\u024F\u0250-\u02AF\u1E00-\u1EFF]+)?"
)
_EXTRA_SKIP = {"bonjour", "merci", "voila", "alors", "donc", "exemple", "article", "section", "page", "figure", "edition"}
_BAM_DIAC = set("ɛɔɲŋƐƆƝŊ")
_BAM_MARKERS = frozenset({
    "bɛ", "tɛ", "kɔnɔ", "bɛɛ", "mɔgɔ", "fɛn", "don", "ye", "ka", "ni", "wa", "min", "ani", "nka", "fana",
})


def _is_plausible_bam_line(line: str) -> bool:
    """Reject non-Bambara lines (French/Malay/Nzema/Senufo noise) lacking Bambara markers."""
    if any(ch in _BAM_DIAC for ch in line):
        return True
    return any(tok.lower() in _BAM_MARKERS for tok in _TOK_RE.findall(line))


@dataclass(frozen=True)
class CorpusCandidate:
    term: str
    freq: int
    example_sentence: str
    source_file: str


def _resolve_raw_corpus_path(lang: str) -> Optional[Path]:
    canon = normalize_lang_code(lang)
    raw_dir = Path(SEBENI_DATA_DIR) / "raw"
    cand = raw_dir / f"{canon}.txt"
    if cand.is_file():
        return cand
    # Fallbacks for ISO aliases (mku<->mlq, spp<->seq)
    for alias in {"mku": ["mlq"], "spp": ["seq"]}.get(canon, []):
        alt = raw_dir / f"{alias}.txt"
        if alt.is_file():
            return alt
    return None


class CorpusOOVMiner:
    """Streams real unparsed, out-of-dictionary words & sentences from <lang>.txt."""

    def __init__(
        self,
        lang: str,
        *,
        working_dict_index: Dict[str, list],
        working_dict_path: Path,
        working_gram_path: Path,
        max_scan_lines: int = 60_000,
    ):
        self.lang = normalize_lang_code(lang)
        self.working_dict_path = Path(working_dict_path)
        self.working_gram_path = Path(working_gram_path)
        self.raw_path = _resolve_raw_corpus_path(self.lang)
        self._toneless_dict: Set[str] = {
            strip_combining_marks(k) for k in working_dict_index.keys() if k
        }
        self._ranked: List[CorpusCandidate] = []
        self._cursor = 0
        self._dx: Optional[DabaX] = None
        if self.raw_path is not None:
            self._build_index(max_scan_lines=max_scan_lines)

    def _build_index(self, *, max_scan_lines: int) -> None:
        counts: Counter[str] = Counter()
        lower_counts: Counter[str] = Counter()
        mid_cap_counts: Counter[str] = Counter()
        best_sent: Dict[str, str] = {}
        noise = _EN_FR_NOISE | _EXTRA_SKIP

        assert self.raw_path is not None
        with self.raw_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for idx, raw_line in enumerate(fh):
                if idx >= max_scan_lines:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                if self.lang == "bam" and not _is_plausible_bam_line(line):
                    continue
                for tok_idx, tok in enumerate(_TOK_RE.findall(line)):
                    nk = _norm_key(tok)
                    if not nk or len(nk) < 4 or len(nk) > 30:
                        continue
                    if nk.startswith("-") or nk.endswith("-") or any(c.isdigit() for c in nk):
                        continue
                    toneless = strip_combining_marks(nk)
                    if toneless in noise or toneless in self._toneless_dict:
                        continue
                    counts[nk] += 1
                    if tok.islower():
                        lower_counts[nk] += 1
                    elif tok_idx > 0 and tok[0].isupper():
                        mid_cap_counts[nk] += 1
                    prev = best_sent.get(nk)
                    if prev is None:
                        best_sent[nk] = line[:240]
                    else:
                        # Prefer informative, clean mid-length sentences (40..150 chars)
                        p_good = 40 <= len(prev) <= 150
                        c_good = 40 <= len(line) <= 150
                        if c_good and not p_good:
                            best_sent[nk] = line[:240]

        min_freq = 2 if len(counts) > 200 else 1
        src_name = self.raw_path.name
        self._ranked = [
            CorpusCandidate(
                term=term,
                freq=freq,
                example_sentence=best_sent.get(term, term),
                source_file=src_name,
            )
            for term, freq in counts.most_common(6000)
            if freq >= min_freq and not (mid_cap_counts[term] >= 2 and lower_counts[term] == 0)
        ]

    def _ensure_dabax(self) -> Optional[DabaX]:
        if self._dx is not None:
            return self._dx
        try:
            rt_dir = self.working_dict_path.parent / "_rt_miner"
            self._dx = DabaX(
                self.lang,
                gram=self.working_gram_path,
                ldict=self.working_dict_path,
                runtime_dir=rt_dir,
                strict_parse=True,
                normalize_homoglyphs=True,
            )
        except Exception:
            self._dx = None
        return self._dx

    def _is_already_parsed(self, dx: DabaX, terms: Sequence[str]) -> Set[str]:
        if not terms:
            return set()
        parsed_set: Set[str] = set()
        try:
            morph_map = dx.extract_morphology(" ".join(terms))
            for sent_tokens in morph_map.values():
                for tok in sent_tokens:
                    surf = _norm_key(str(tok.get("token", "")))
                    if not surf:
                        continue
                    if is_empr_token(tok):
                        parsed_set.add(surf)
                        continue
                    stage = str(tok.get("stage", "-1"))
                    analyses = tok.get("analyses") or []
                    try:
                        ok_stage = stage not in ("-1", "tokenizer") and int(float(stage)) >= 0
                    except (TypeError, ValueError):
                        ok_stage = False
                    has_gloss = any(
                        bool((a or {}).get("gloss")) or bool((a or {}).get("morphemes"))
                        for a in analyses
                    )
                    if ok_stage and has_gloss:
                        parsed_set.add(surf)
        except Exception:
            pass
        return parsed_set

    def next_batch(
        self,
        n: int = 5,
        *,
        exclude: Optional[Set[str]] = None,
        working_dict_index: Optional[Dict[str, list]] = None,
    ) -> List[CorpusCandidate]:
        """Return up to `n` unparsed, out-of-dictionary real corpus words & sentences."""
        if not self._ranked:
            return []
        if working_dict_index is not None:
            self._toneless_dict = {strip_combining_marks(k) for k in working_dict_index.keys() if k}

        excl = {strip_combining_marks(x) for x in (exclude or set())}
        dx = self._ensure_dabax()
        selected: List[CorpusCandidate] = []
        pool_size = max(25, n * 6)

        while len(selected) < n and self._cursor < len(self._ranked):
            chunk = self._ranked[self._cursor : self._cursor + pool_size]
            self._cursor += len(chunk)
            filtered = [
                c
                for c in chunk
                if strip_combining_marks(c.term) not in excl
                and strip_combining_marks(c.term) not in self._toneless_dict
            ]
            if not filtered:
                continue
            parsed = self._is_already_parsed(dx, [c.term for c in filtered]) if dx is not None else set()
            for cand in filtered:
                if _norm_key(cand.term) in parsed:
                    continue
                selected.append(cand)
                excl.add(strip_combining_marks(cand.term))
                if len(selected) >= n:
                    break
        return selected
