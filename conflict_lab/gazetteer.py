"""Canonical Malian Admin-1/Admin-2 (cercle) gazetteer, spelling normalizer & corridor graph."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CercleEntry:
    cercle_id: str
    name: str
    admin1: str
    lat: float
    lon: float
    aliases: Tuple[str, ...]
    corridor_neighbors: Tuple[str, ...]


def _norm_geo(text: str) -> str:
    s = unicodedata.normalize("NFKD", (text or "").strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


CERCLE_GAZETTEER: Tuple[CercleEntry, ...] = (
    # Bamako & Koulikoro
    CercleEntry("bamako", "Bamako", "Bamako", 12.6392, -8.0029, ("bamako", "kati-bamako"), ("kati", "koulikoro", "kangaba")),
    CercleEntry("kati", "Kati", "Koulikoro", 12.7441, -8.0726, ("kati", "kati camp"), ("bamako", "nara", "kolokani", "banamba")),
    CercleEntry("nara", "Nara", "Koulikoro", 15.1686, -7.2878, ("nara", "mourdiah", "guire", "dilly"), ("kati", "nioro", "sokolo", "banamba")),
    CercleEntry("banamba", "Banamba", "Koulikoro", 13.5477, -7.4481, ("banamba", "touba"), ("kati", "nara", "segou", "kolokani")),
    CercleEntry("kolokani", "Kolokani", "Koulikoro", 13.5728, -8.0339, ("kolokani", "didieni"), ("kati", "banamba", "kita")),
    CercleEntry("koulikoro", "Koulikoro", "Koulikoro", 12.8627, -7.5599, ("koulikoro", "nyamina"), ("bamako", "segou", "dioila")),
    # Kayes & Nioro
    CercleEntry("kayes", "Kayes", "Kayes", 14.4469, -11.4445, ("kayes",), ("yelimane", "bafulabe")),
    CercleEntry("nioro", "Nioro du Sahel", "Kayes", 15.2286, -9.5897, ("nioro", "nioro du sahel", "diema", "sandare"), ("nara", "yelimane", "kita")),
    CercleEntry("kita", "Kita", "Kayes", 13.0349, -9.4895, ("kita",), ("kolokani", "nioro", "kayes")),
    CercleEntry("yelimane", "Yélimané", "Kayes", 15.1342, -10.5731, ("yelimane", "gogui"), ("kayes", "nioro")),
    CercleEntry("bafulabe", "Bafoulabé", "Kayes", 13.8065, -10.8321, ("bafoulabe", "bafulabe"), ("kayes", "kita")),
    # Sikasso & Bougouni / Koutiala
    CercleEntry("sikasso", "Sikasso", "Sikasso", 11.3176, -5.6665, ("sikasso", "kadiolo", "loulouni", "zégoua", "zegoua"), ("bougouni", "koutiala")),
    CercleEntry("koutiala", "Koutiala", "Sikasso", 12.3917, -5.4642, ("koutiala", "yorosso", "koury", "kouri"), ("sikasso", "segou", "san")),
    CercleEntry("bougouni", "Bougouni", "Sikasso", 11.4177, -7.4832, ("bougouni", "yanfolila", "kolondieba"), ("sikasso", "bamako", "dioila")),
    CercleEntry("dioila", "Dioïla", "Koulikoro", 12.4875, -6.7989, ("dioila", "fana"), ("koulikoro", "bougouni", "segou")),
    # Segou & San
    CercleEntry("segou", "Ségou", "Ségou", 13.4317, -6.2157, ("segou", "markala", "cinzana", "baraoueli", "bla"), ("banamba", "koulikoro", "niono", "macina", "san")),
    CercleEntry("niono", "Niono", "Ségou", 14.2526, -5.9930, ("niono", "diabaly", "sokolo", "dogofry", "molodo"), ("segou", "macina", "tenenkou", "nara")),
    CercleEntry("macina", "Macina", "Ségou", 13.9631, -5.3572, ("macina", "ke macina", "monimpebougou"), ("niono", "segou", "djenne", "tenenkou")),
    CercleEntry("san", "San", "Ségou", 13.3034, -4.8956, ("san", "tominian", "kimparana", "fangasso"), ("segou", "koutiala", "djenne", "bankass")),
    # Mopti / Bandiagara / Douentza
    CercleEntry("mopti", "Mopti", "Mopti", 14.4843, -4.1830, ("mopti", "sevare", "fatoma", "konna"), ("djenne", "bandiagara", "tenenkou", "douentza")),
    CercleEntry("djenne", "Djenné", "Mopti", 13.9061, -4.5533, ("djenne", "jenné", "sofola", "mourrah"), ("mopti", "macina", "san", "bandiagara")),
    CercleEntry("tenenkou", "Ténenkou", "Mopti", 14.4572, -4.9169, ("tenenkou", "diafarabe", "youwarou", "diondiori"), ("niono", "macina", "mopti", "niafunke")),
    CercleEntry("bandiagara", "Bandiagara", "Bandiagara", 14.3501, -3.6104, ("bandiagara", "parou", "songho", "kendie"), ("mopti", "bankass", "koro", "douentza")),
    CercleEntry("bankass", "Bankass", "Bandiagara", 14.0725, -3.5161, ("bankass", "socoura", "ouenkoro"), ("bandiagara", "koro", "san")),
    CercleEntry("koro", "Koro", "Bandiagara", 14.0642, -3.0744, ("koro", "dinangourou", "koporo"), ("bankass", "bandiagara", "douentza")),
    CercleEntry("douentza", "Douentza", "Douentza", 15.0014, -2.9498, ("douentza", "hombori", "bamba-douentza", "boni", "mondoro", "petaka"), ("mopti", "bandiagara", "koro", "gossi", "dire")),
    # Tombouctou & Taoudenit
    CercleEntry("tombouctou", "Tombouctou", "Tombouctou", 16.7735, -3.0074, ("tombouctou", "timbuktu", "ber", "taoudenit", "arawan"), ("dire", "goundam", "gourma-rharous")),
    CercleEntry("niafunke", "Niafunké", "Tombouctou", 15.9322, -3.9906, ("niafunke", "léré", "lere", "soboundou"), ("tenenkou", "dire", "goundam")),
    CercleEntry("dire", "Diré", "Tombouctou", 16.2647, -3.3958, ("dire", "tonka"), ("niafunke", "goundam", "tombouctou")),
    CercleEntry("goundam", "Goundam", "Tombouctou", 16.4145, -3.6708, ("goundam", "tonka-goundam", "léré-goundam", "raz el ma"), ("niafunke", "dire", "tombouctou")),
    CercleEntry("gourma-rharous", "Gourma-Rharous", "Tombouctou", 16.8794, -1.9228, ("gourma rharous", "rharous", "gossi", "banikane", "bambara maoude"), ("tombouctou", "douentza", "gao", "bourem")),
    # Gao & Menaka
    CercleEntry("gao", "Gao", "Gao", 16.2717, -0.0447, ("gao", "ntillit", "ntahaka", "soni ali ber"), ("gourma-rharous", "bourem", "ansongo", "menaka")),
    CercleEntry("ansongo", "Ansongo", "Gao", 15.6597, 0.5022, ("ansongo", "tessit", "labbezanga", "intillit", "talataye", "indelimane", "watagouna"), ("gao", "menaka", "anderamboukane")),
    CercleEntry("bourem", "Bourem", "Gao", 16.9489, -0.3547, ("bourem", "bamba", "temera", "tabankort"), ("gourma-rharous", "gao", "aneffif", "kidal")),
    CercleEntry("menaka", "Ménaka", "Ménaka", 15.9182, 2.4022, ("menaka", "inekar", "anderamboukane", "tidermene", "tamalat"), ("gao", "ansongo", "kidal")),
    # Kidal
    CercleEntry("kidal", "Kidal", "Kidal", 18.4411, 1.4078, ("kidal", "aneffif", "anefis", "essouk", "aguelhoc"), ("bourem", "menaka", "tessalit", "tin-essako")),
    CercleEntry("tessalit", "Tessalit", "Kidal", 20.1972, 1.0128, ("tessalit", "aguelhok", "aguelhoc", "in-khalil"), ("kidal", "tombouctou")),
    CercleEntry("abeibara", "Abeïbara", "Kidal", 19.1200, 1.7500, ("abeibara", "tin-essako", "tinzawatene", "tinzaouatene", "boghasa"), ("kidal", "tessalit")),
)

BY_CERCLE_ID: Dict[str, CercleEntry] = {c.cercle_id: c for c in CERCLE_GAZETTEER}
_ALIAS_TO_CERCLE: Dict[str, CercleEntry] = {}
for _c in CERCLE_GAZETTEER:
    _ALIAS_TO_CERCLE[_norm_geo(_c.name)] = _c
    _ALIAS_TO_CERCLE[_norm_geo(_c.cercle_id)] = _c
    for _al in _c.aliases:
        _ALIAS_TO_CERCLE[_norm_geo(_al)] = _c


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def resolve_mali_location(
    *,
    raw_location: str = "",
    admin1_hint: str = "",
    admin2_hint: str = "",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Tuple[CercleEntry, str]:
    """Resolve raw text or coordinates onto canonical Malian Admin-2 cercle & precision."""
    for candidate_txt, prec in (
        (raw_location, "exact_locality"),
        (admin2_hint, "cercle_centroid"),
        (admin1_hint, "region_centroid"),
    ):
        nk = _norm_geo(candidate_txt)
        if not nk:
            continue
        if nk in _ALIAS_TO_CERCLE:
            chosen = _ALIAS_TO_CERCLE[nk]
            if lat is not None and lon is not None and _in_mali_bbox(lat, lon):
                return chosen, "exact_locality"
            return chosen, prec
        for alias_k, entry in _ALIAS_TO_CERCLE.items():
            if len(alias_k) >= 4 and (alias_k in nk or nk in alias_k):
                return entry, "cercle_centroid"

    if lat is not None and lon is not None and _in_mali_bbox(lat, lon):
        nearest = min(CERCLE_GAZETTEER, key=lambda c: haversine_km(lat, lon, c.lat, c.lon))
        return nearest, "exact_locality"

    # Conservative fallback: Mopti central Niger Bend hub
    return BY_CERCLE_ID["mopti"], "region_centroid"


def _in_mali_bbox(lat: float, lon: float) -> bool:
    return 10.0 <= lat <= 25.5 and -12.5 <= lon <= 4.5


ACTOR_CANON_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:jnim|nusrat|macina|katiba|iyad|aqim|gsim)\b|ansar\s*dine", re.I), "JNIM / Katiba Coalition"),
    (re.compile(r"islamic state|\b(?:isgs|eigs|isis|daesh|is)\b|sahel province", re.I), "ISGS / Islamic State Sahel"),
    (re.compile(r"\b(?:fama|dirpa|fmds)\b|wagner|africa corps|government of russia|armed forces of mali|government of mali|forces arm", re.I), "FAMa / Allied Security Forces"),
    (re.compile(r"\b(?:csp|cma|mnla|hcua|maa|fla|gatia|msa|tuareg|azawad)\b|csp-psd|cadre strat", re.I), "CSP-PSD / Azawad Groups"),
    (re.compile(r"\b(?:donzo|dozos?|ganda)\b|dan na ambassagou|self[- ]defense|communal militia", re.I), "Local Militias / Self-Defense"),
)

TACTIC_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bied\b|mine|improvised explosive|engin explosif|mine antipersonnel", re.I), "ied_mine"),
    (re.compile(r"ambush|embuscade|convoy|convoi", re.I), "ambush"),
    (re.compile(r"blockade|siege|blocus|supply|fuel|cistern|cut off", re.I), "blockade_siege"),
    (re.compile(r"kidnap|abduct|enl[eè]v|hostage|intimidat|extort|tax", re.I), "kidnapping_intimidation"),
    (re.compile(r"mortar|shell|rocket|obus|roquette|indirect fire|drone strike", re.I), "indirect_fire"),
    (re.compile(r"airstrike|frappe a[ée]rienne|tb2|bayraktar|helicopter|air raid", re.I), "airstrike_clash"),
    (re.compile(r"attack|assault|raid|clash|combat|overran|post|camp", re.I), "raid_assault"),
)


def normalize_actor_name(raw_actor: str) -> str:
    txt = (raw_actor or "").strip()
    if not txt:
        return "Unattributed Armed Group"
    for pat, label in ACTOR_CANON_RULES:
        if pat.search(txt):
            return label
    return txt[:48]


def normalize_tactic_name(raw_text: str) -> str:
    txt = (raw_text or "").strip()
    for pat, label in TACTIC_RULES:
        if pat.search(txt):
            return label  # type: ignore[return-value]
    return "other"
