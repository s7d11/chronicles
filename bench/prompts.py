"""Prompt builders for MALI13 tasks (reusing Sebeni SRLGrpoPrompt for morphology)."""

from __future__ import annotations

from typing import Dict, List

from beni.core.srl.config import SRLGrpoPrompt
from bench import LANGS, LANG_NAMES


def build_morph_messages(lang: str, text: str) -> List[Dict[str, str]]:
    """Use Sebeni's exact SRLGrpoPrompt system message so trained & API models match."""
    sys_prompt = SRLGrpoPrompt(languages=[lang]).prompt().strip()
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text},
    ]


def build_gen_messages(lang: str, prompt_prefix: str) -> List[Dict[str, str]]:
    name = LANG_NAMES.get(lang, lang)
    system = (
        f"You are a fluent native writer of {name} ({lang}) in Mali. "
        f"Continue the user's {name} sentence naturally with 1-2 sentences strictly in {name}. "
        "Do NOT translate into French or English. Do NOT repeat the prompt prefix. "
        "Do NOT add commentary, headings, or markdown. Output ONLY the continuation text."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_prefix},
    ]


def build_lid_messages(text: str) -> List[Dict[str, str]]:
    code_list = ", ".join(f"{c} ({LANG_NAMES[c]})" for c in LANGS)
    system = (
        "You are an expert African linguist performing 13-way Language Identification "
        "across Mali's 13 national languages (MULTI13).\n"
        f"Valid ISO codes: {code_list}.\n"
        'Identify the language of the input text and respond ONLY with a JSON object: {"lang": "<code>"}.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def build_cloze_messages(lang: str, masked_text: str, choices: Dict[str, str]) -> List[Dict[str, str]]:
    name = LANG_NAMES.get(lang, lang)
    opts = "\n".join(f"{k}: {v}" for k, v in sorted(choices.items()))
    system = (
        f"You are an expert linguist in {name} ({lang}). "
        "Choose the single best option (A, B, C, or D) to fill [MASK] in the sentence. "
        'Respond ONLY with a JSON object: {"choice": "A"} (or B, C, or D).'
    )
    user = f"Sentence: {masked_text}\nOptions:\n{opts}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
