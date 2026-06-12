"""STT transcript correction for FAB domain terms.

The post-processor stays deterministic and conservative: common high-confidence
misrecognitions are fixed first, known vocab terms are normalized next, and fuzzy
matching is allowed only for vocab candidates when STT confidence is high enough.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RULES_PATH = Path(__file__).with_name("stt_correction_rules.json")
FUZZY_CONFIDENCE_THRESHOLD = 0.55
FUZZY_SCORE_CUTOFF = 0.86
MAX_ALIAS_COMBINATIONS = 1024
MAX_ALIASES_PER_PART = 8
REDUCED_ALIASES_PER_PART = 2

DEFAULT_KOREAN_FIXES: tuple[tuple[str, str], ...] = (
    (r"가장\s*먼저\s*봐야\s*할\s*트렌드(?:가|는|은)?", "가장 먼저 봐야 할 툴그룹"),
    (r"(?:세리\s*디\s*벨벳|세리디벨벳|쓰리\s*디\s*(?:맵|맵스|멥)|3\s*d\s*(?:맵|map))", "3D 맵"),
    (r"위치\s*(?:찍어\s*줄\s*수\s*있어|찍어\s*줄래|찍어줘|찍어|표시해|보여줘|잡아줘)", "위치 찍어줘"),
    (r"툴\s*(?:구름|그륨|그럼|크룹|그룸|그릅|구룹|그룹)", "툴그룹"),
    (r"(?:티\s*지|티지)(?=\s|$|[은는이가을를도의로])", "TG"),
    (r"구역\s*(?:발|벌|빨|별)(?=\s|$|[을를이가은는도의로])", "구역별"),
    (r"(?:해드멧|데프\s*(?:멧|맷|메트)|대프\s*(?:멧|맷))", "Def_Met"),
    (r"해던", "해당"),
    (r"했던(?=\s*(?:구역|공정|TG|티지|툴그룹|툴\s*그룹|설비|장비|라인|존))", "해당"),
    (r"리소\s*(?:멧|맷)", "Litho_Met"),
    (r"(?:리소|리쏘)(?=\s|$|[은는이가을를에의도])", "Litho"),
    (r"(?:다이|디)\s*일렉트릭", "Dielectric"),
    (r"디퓨[전젼션]", "Diffusion"),
    (r"드라이\s*에[치칭]", "Dry_Etch"),
    (r"(?:웻|웨트|웨)\s*에[치칭]", "Wet_Etch"),
    (r"임플란[트테]", "Implant"),
    (r"플라[나너]|플레이너", "Planar"),
    (r"큐\s*타임|규타임", "Q-time"),
    (r"(?:^|\s)(?:윕|위프)(?=\s|$|[은는이가을를도])", " WIP"),
    (r"오이이|오\s*이\s*이", "OEE"),
    (r"트랜드", "트렌드"),
    (r"가동\s*율", "가동률"),
    (r"가용\s*율", "가용률"),
)

DEFAULT_PART_ALIASES: dict[str, tuple[str, ...]] = {
    "BE": ("BE", "비이", "비 이"),
    "FE": ("FE", "에프이", "에프 이"),
    "TG": ("TG", "티지", "티 지"),
    "DEF": ("DEF", "데프", "대프", "디프"),
    "MET": ("MET", "멧", "맷", "메트"),
    "LITHO": ("LITHO", "리소", "리쏘"),
    "DRY": ("DRY", "드라이"),
    "ETCH": ("ETCH", "에치", "에칭"),
    "WET": ("WET", "웻", "웨트", "웨"),
    "DIELECTRIC": ("DIELECTRIC", "다이일렉트릭", "다이 일렉트릭", "디일렉트릭", "디 일렉트릭"),
    "DIFFUSION": ("DIFFUSION", "디퓨전", "디퓨젼", "디퓨션"),
    "IMPLANT": ("IMPLANT", "임플란트", "임플란테"),
    "PLANAR": ("PLANAR", "플라나", "플라너", "플레이너"),
}

DEFAULT_NUMERIC_WORDS: dict[int, str] = {
    0: "영",
    1: "일",
    2: "이",
    3: "삼",
    4: "사",
    5: "오",
    6: "육",
    7: "칠",
    8: "팔",
    9: "구",
    10: "십",
}


@dataclass(frozen=True)
class CorrectionRules:
    korean_fixes: tuple[tuple[str, str], ...]
    part_aliases: dict[str, tuple[str, ...]]
    numeric_words: dict[int, str]


@dataclass
class CorrectionResult:
    text: str
    raw: str
    applied_rules: list[str] = field(default_factory=list)
    changed_terms: list[dict[str, str]] = field(default_factory=list)
    fuzzy_applied: bool = False


def _default_rules() -> CorrectionRules:
    return CorrectionRules(DEFAULT_KOREAN_FIXES, DEFAULT_PART_ALIASES, DEFAULT_NUMERIC_WORDS)


def _load_rules_payload(path: Path = RULES_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_correction_rules() -> CorrectionRules:
    """Load externalized correction rules; fall back to safe built-ins on failure."""
    try:
        payload = _load_rules_payload()
        korean_fixes = tuple((str(pattern), str(replacement)) for pattern, replacement in payload["korean_fixes"])
        part_aliases = {
            str(key).upper(): tuple(str(item) for item in value)
            for key, value in dict(payload["part_aliases"]).items()
        }
        numeric_words = {int(key): str(value) for key, value in dict(payload["numeric_words"]).items()}
        return CorrectionRules(korean_fixes, part_aliases, numeric_words)
    except Exception:  # noqa: BLE001 - fallback preserves voice input availability.
        logger.warning("STT correction rules load failed; using built-in defaults", exc_info=True)
        return _default_rules()


def apply_korean_correction(text: str) -> str:
    """Fix common Korean STT misrecognitions in domain phrases."""
    return _apply_korean_correction(text, CorrectionResult(text=text or "", raw=text or "")).text


def _apply_korean_correction(text: str, result: CorrectionResult) -> CorrectionResult:
    corrected = text or ""
    for index, (pattern, replacement) in enumerate(load_correction_rules().korean_fixes):
        before = corrected
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        if corrected != before:
            result.applied_rules.append(f"korean:{index}")
            result.changed_terms.append({"from": before, "to": corrected})
    result.text = corrected.strip()
    return result


def _number_to_korean(value: int) -> str:
    words = load_correction_rules().numeric_words
    if value < 0 or value > 999:
        return str(value)
    if value <= 10:
        return words[value]

    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    tens, ones = divmod(remainder, 10)

    if hundreds:
        parts.append("" if hundreds == 1 else words[hundreds])
        parts.append("백")
    if tens:
        parts.append("" if tens == 1 else words[tens])
        parts.append("십")
    if ones:
        parts.append(words[ones])
    return "".join(parts)


def _numeric_variants(part: str) -> tuple[str, ...]:
    if not part.isdigit():
        return ()
    words = load_correction_rules().numeric_words
    value = int(part)
    korean = _number_to_korean(value)
    digit_names = "".join(words[int(ch)] for ch in part)
    variants = {part, korean, " ".join(korean), digit_names, " ".join(digit_names)}
    if len(part) == 2 and part.startswith("0"):
        variants.add(_number_to_korean(int(part[1])))
    return tuple(sorted(variants, key=len, reverse=True))


def _part_variants(part: str) -> tuple[str, ...]:
    normalized = part.upper()
    variants = {part, part.replace("-", "_"), part.replace("_", " "), part.replace("_", "")}
    variants.update(load_correction_rules().part_aliases.get(normalized, ()))
    variants.update(_numeric_variants(part))

    compact = re.sub(r"[^A-Za-z0-9]", "", part).upper()
    if compact == "DEFMET":
        variants.update(("DefMet", "Def_Met", "Def Met", "데프멧", "데프 멧", "대프멧", "대프 멧"))
    if compact == "LITHOMET":
        variants.update(("LithoMet", "Litho_Met", "Litho Met", "리소멧", "리소 멧", "리소맷", "리소 맷"))

    return tuple(sorted({v for v in variants if v}, key=len, reverse=True))


@lru_cache(maxsize=1024)
def _vocab_pattern(term: str) -> re.Pattern[str] | None:
    parts = [p for p in re.split(r"[_\s-]+", term) if p]
    if not parts:
        return None

    part_patterns = []
    for part in parts:
        aliases = [re.escape(alias) for alias in _part_variants(part)]
        part_patterns.append(r"(?:" + "|".join(aliases) + r")")

    body = r"[\s_-]*".join(part_patterns)
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", flags=re.IGNORECASE)


def _apply_vocab_exact(text: str, vocab: list[str], result: CorrectionResult) -> str:
    corrected = text
    for term in sorted(vocab, key=len, reverse=True):
        pattern = _vocab_pattern(term)
        if pattern is None:
            continue

        def replace(match: re.Match[str]) -> str:
            found = match.group(0)
            if found != term:
                result.applied_rules.append("vocab:exact")
                result.changed_terms.append({"from": found, "to": term})
            return term

        corrected = pattern.sub(replace, corrected)
    return corrected


def _normalize_for_fuzzy(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value).casefold()


def _term_alias_phrases(term: str) -> tuple[str, ...]:
    parts = [p for p in re.split(r"[_\s-]+", term) if p]
    if not parts:
        return ()
    phrases = {term, term.replace("_", " "), term.replace("_", ""), term.replace("_", "-")}
    phrase_parts = [_part_variants(part) for part in parts]
    if all(phrase_parts):
        phrases.add(" ".join(variants[0] for variants in phrase_parts))
        phrases.add("".join(variants[0] for variants in phrase_parts))
        total_combinations = 1
        for variants in phrase_parts:
            total_combinations *= min(len(variants), MAX_ALIASES_PER_PART)
        limit = REDUCED_ALIASES_PER_PART if total_combinations > MAX_ALIAS_COMBINATIONS else MAX_ALIASES_PER_PART
        for combo in product(*(variants[:limit] for variants in phrase_parts)):
            phrases.add(" ".join(combo))
            phrases.add("".join(combo))
    return tuple(sorted(phrases, key=len, reverse=True))


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _apply_vocab_fuzzy(text: str, vocab: list[str], result: CorrectionResult) -> str:
    spans = _token_spans(text)
    if not spans:
        return text

    best: tuple[float, int, int, str, str] | None = None
    for term in sorted(vocab, key=len, reverse=True):
        term_parts = [p for p in re.split(r"[_\s-]+", term) if p]
        min_window = max(1, len(term_parts) - 1)
        max_window = min(len(spans), len(term_parts) + 1)
        aliases = tuple(_normalize_for_fuzzy(alias) for alias in _term_alias_phrases(term))
        if not aliases:
            continue
        for window in range(min_window, max_window + 1):
            for start in range(0, len(spans) - window + 1):
                end = start + window - 1
                phrase = text[spans[start][1]:spans[end][2]]
                if phrase == term:
                    continue
                normalized = _normalize_for_fuzzy(phrase)
                score = max(SequenceMatcher(None, normalized, alias).ratio() for alias in aliases)
                if score >= FUZZY_SCORE_CUTOFF and (best is None or score > best[0]):
                    best = (score, spans[start][1], spans[end][2], phrase, term)

    if best is None:
        return text

    _, start, end, phrase, term = best
    result.applied_rules.append("vocab:fuzzy")
    result.changed_terms.append({"from": phrase, "to": term})
    result.fuzzy_applied = True
    return text[:start] + term + text[end:]


def apply_vocab_correction(text: str, vocab: list[str], confidence: float | None = None) -> str:
    """Normalize transcript variants to known area/TG names."""
    return correct_transcript_detailed(text, vocab, confidence=confidence).text


def correct_transcript_detailed(
    text: str,
    vocab: list[str] | None = None,
    confidence: float | None = None,
) -> CorrectionResult:
    """Apply deterministic STT correction pipeline and return correction metadata."""
    raw = text or ""
    result = CorrectionResult(text=raw, raw=raw)
    _apply_korean_correction(raw, result)
    vocab_list = [term for term in (vocab or []) if term]
    if vocab_list:
        result.text = _apply_vocab_exact(result.text, vocab_list, result)
        if confidence is not None and confidence >= FUZZY_CONFIDENCE_THRESHOLD:
            result.text = _apply_vocab_fuzzy(result.text, vocab_list, result)
    result.text = result.text.strip()
    return result


def correct_transcript(
    text: str,
    vocab: list[str] | None = None,
    confidence: float | None = None,
) -> str:
    """Apply deterministic STT correction pipeline."""
    return correct_transcript_detailed(text, vocab or [], confidence=confidence).text
