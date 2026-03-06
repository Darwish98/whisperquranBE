"""
tajweed_rules.py — Text-Based Tajweed Rule Identification
==========================================================
Phase 3: Identifies WHERE tajweed rules exist in the Quran text.
No audio analysis. Pure text analysis on diacritized Uthmani script.

Usage:
    from tajweed_rules import annotate_surah
    annotations = annotate_surah(1)  # Al-Fatiha
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger("whisperquran.tajweed")

# ── Arabic character constants ────────────────────────────────────────────────

FATHA    = '\u064E'
DAMMA    = '\u064F'
KASRA    = '\u0650'
SHADDA   = '\u0651'
SUKUN    = '\u0652'
FATHATAN = '\u064B'
DAMMATAN = '\u064C'
KASRATAN = '\u064D'
SUPERSCRIPT_ALEF = '\u0670'

NOON   = '\u0646'
MEEM   = '\u0645'
BA     = '\u0628'
LAAM   = '\u0644'
ALEF   = '\u0627'
ALEF_WASLA = '\u0671'
WAW    = '\u0648'
YA     = '\u064A'
ALEF_MAQSURA = '\u0649'
HAMZA  = '\u0621'
ALEF_HAMZA_ABOVE = '\u0623'
ALEF_HAMZA_BELOW = '\u0625'
WAW_HAMZA = '\u0624'
YA_HAMZA = '\u0626'
TATWEEL = '\u0640'
ALEF_MADDA = '\u0622'

QALQALAH_LETTERS = {'\u0642', '\u0637', '\u0628', '\u062C', '\u062F'}

IKHFA_LETTERS = {
    '\u062A', '\u062B', '\u062C', '\u062F', '\u0630',
    '\u0632', '\u0633', '\u0634', '\u0635', '\u0636',
    '\u0637', '\u0638', '\u0641', '\u0642', '\u0643',
}

IDGHAM_GHUNNA_LETTERS = {YA, NOON, MEEM, WAW}
IDGHAM_NO_GHUNNA_LETTERS = {LAAM, '\u0631'}

SHAMS_LETTERS = {
    '\u062A', '\u062B', '\u062F', '\u0630', '\u0631',
    '\u0632', '\u0633', '\u0634', '\u0635', '\u0636',
    '\u0637', '\u0638', LAAM, NOON,
}

HAMZA_CHARS = {HAMZA, '\u0654', '\u0655', ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW, WAW_HAMZA, YA_HAMZA}
HARAKAT = {FATHA, DAMMA, KASRA, SHADDA, SUKUN, FATHATAN, DAMMATAN, KASRATAN, SUPERSCRIPT_ALEF}
MADD_LETTERS = {ALEF, WAW, YA, ALEF_MAQSURA}

# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class TajweedAnnotation:
    rule: str
    rule_category: str
    char_start: int
    char_end: int
    description: str
    arabic_name: str
    harakat_count: Optional[int] = None

RULE_INFO = {
    "ghunnah":           {"cat": "ghunna",    "ar": "غُنَّة",               "desc": "Nasalization — hold for 2 counts"},
    "ikhfa":             {"cat": "ikhfa",     "ar": "إِخْفَاء",            "desc": "Hidden noon — light nasalization"},
    "ikhfa_shafawi":     {"cat": "ikhfa",     "ar": "إِخْفَاء شَفَوِي",     "desc": "Hidden meem before ba"},
    "idgham_ghunnah":    {"cat": "idgham",    "ar": "إِدْغَام بِغُنَّة",     "desc": "Merging with nasalization"},
    "idgham_no_ghunnah": {"cat": "idgham",    "ar": "إِدْغَام بِلَا غُنَّة", "desc": "Merging without nasalization"},
    "idgham_shafawi":    {"cat": "idgham",    "ar": "إِدْغَام شَفَوِي",     "desc": "Meem merged into meem"},
    "iqlab":             {"cat": "iqlab",     "ar": "إِقْلَاب",            "desc": "Noon becomes meem before ba"},
    "qalqalah":          {"cat": "qalqalah",  "ar": "قَلْقَلَة",           "desc": "Echoing bounce on stopped letter"},
    "madd_2":            {"cat": "madd",      "ar": "مَدّ طَبِيعِي",        "desc": "Natural elongation — 2 counts"},
    "madd_246":          {"cat": "madd",      "ar": "مَدّ عَارِض",          "desc": "Presented elongation — 2/4/6 counts"},
    "madd_muttasil":     {"cat": "madd",      "ar": "مَدّ مُتَّصِل",        "desc": "Connected elongation — 4-5 counts"},
    "madd_munfasil":     {"cat": "madd",      "ar": "مَدّ مُنْفَصِل",       "desc": "Separated elongation — 4-5 counts"},
    "madd_6":            {"cat": "madd",      "ar": "مَدّ لَازِم",          "desc": "Obligatory elongation — 6 counts"},
    "lam_shamsiyyah":    {"cat": "lam_shams", "ar": "لَام شَمْسِيَّة",      "desc": "Assimilated lam (sun letter)"},
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_base_letters(word: str) -> List[tuple]:
    letters = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in HARAKAT or ch == TATWEEL:
            i += 1
            continue
        base_idx = i
        harakat_list = []
        i += 1
        while i < len(chars) and chars[i] in HARAKAT:
            harakat_list.append(chars[i])
            i += 1
        letters.append((ch, base_idx, harakat_list))
    return letters

def _has_sukun(h): return SUKUN in h
def _has_shadda(h): return SHADDA in h
def _has_vowel(h): return bool(set(h) & {FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN})

def _is_noon_sakin(ch, harakat):
    return ch == NOON and (_has_sukun(harakat) or (not _has_vowel(harakat) and not _has_shadda(harakat)))

def _is_meem_sakin(ch, harakat):
    return ch == MEEM and (_has_sukun(harakat) or (not _has_vowel(harakat) and not _has_shadda(harakat)))

def _first_base_letter(word):
    if not word:
        return None
    for ch in word:
        if ch not in HARAKAT and ch != TATWEEL and ch != ALEF_WASLA and ch != ALEF:
            return ch
    letters = _get_base_letters(word)
    return letters[1][0] if len(letters) >= 2 else (letters[0][0] if letters else None)

def _make_annotation(rule, char_start, char_end):
    info = RULE_INFO.get(rule, {"cat": "unknown", "ar": rule, "desc": rule})
    hc = None
    if rule == "madd_2": hc = 2
    elif rule == "madd_246": hc = 6
    elif rule == "madd_muttasil": hc = 5
    elif rule == "madd_munfasil": hc = 5
    elif rule == "madd_6": hc = 6
    elif rule == "ghunnah": hc = 2
    return TajweedAnnotation(
        rule=rule, rule_category=info["cat"],
        char_start=char_start, char_end=char_end,
        description=info["desc"], arabic_name=info["ar"],
        harakat_count=hc,
    )

# ── Main detection ─────────────────────────────────────────────────────────────

def get_word_tajweed_rules(word, next_word=None, is_last_word_in_ayah=False):
    """Identify all tajweed rules that apply to a word."""
    rules = []
    letters = _get_base_letters(word)
    if not letters:
        return rules

    next_first_base = _first_base_letter(next_word) if next_word else None

    for idx, (ch, char_idx, harakat) in enumerate(letters):
        next_letter = letters[idx + 1] if idx + 1 < len(letters) else None
        end_idx = char_idx + 1 + len(harakat)

        # Ghunna: noon/meem with shadda
        if ch in (NOON, MEEM) and _has_shadda(harakat):
            rules.append(_make_annotation("ghunnah", char_idx, end_idx))

        # Noon sakin rules
        if _is_noon_sakin(ch, harakat):
            check = next_letter[0] if next_letter else next_first_base
            if check:
                if check == BA:
                    rules.append(_make_annotation("iqlab", char_idx, end_idx))
                elif check in IKHFA_LETTERS:
                    rules.append(_make_annotation("ikhfa", char_idx, end_idx))
                elif check in IDGHAM_GHUNNA_LETTERS and not next_letter:
                    rules.append(_make_annotation("idgham_ghunnah", char_idx, end_idx))
                elif check in IDGHAM_NO_GHUNNA_LETTERS and not next_letter:
                    rules.append(_make_annotation("idgham_no_ghunnah", char_idx, end_idx))

        # Meem sakin rules
        if _is_meem_sakin(ch, harakat):
            check = next_letter[0] if next_letter else next_first_base
            if check:
                if check == BA:
                    rules.append(_make_annotation("ikhfa_shafawi", char_idx, end_idx))
                elif check == MEEM and not next_letter:
                    rules.append(_make_annotation("idgham_shafawi", char_idx, end_idx))

        # Qalqalah
        if ch in QALQALAH_LETTERS:
            is_sakin = _has_sukun(harakat) or (
                not _has_vowel(harakat) and not _has_shadda(harakat) and
                (next_letter is None or is_last_word_in_ayah)
            )
            if is_sakin:
                rules.append(_make_annotation("qalqalah", char_idx, end_idx))

        # Madd
        if ch in MADD_LETTERS or ch == SUPERSCRIPT_ALEF or ch == ALEF_MAQSURA:
            _detect_madd(ch, char_idx, harakat, idx, letters, next_word, is_last_word_in_ayah, rules)

        # Alef Madda
        if ch == ALEF_MADDA:
            rules.append(_make_annotation("madd_2", char_idx, end_idx))

    # Lam Shamsiyyah
    _detect_lam_shamsiyyah(word, letters, rules)

    return rules


def _detect_madd(ch, char_idx, harakat, idx, letters, next_word, is_last, rules):
    next_letter = letters[idx + 1] if idx + 1 < len(letters) else None
    is_madd_carrier = False
    if ch in (ALEF, ALEF_MAQSURA, SUPERSCRIPT_ALEF):
        is_madd_carrier = True
    elif ch == WAW and _has_sukun(harakat):
        is_madd_carrier = True
    elif ch == YA and _has_sukun(harakat):
        is_madd_carrier = True
    if not is_madd_carrier:
        return

    end_idx = char_idx + 1 + len(harakat)

    if next_letter:
        next_ch, _, next_harakat = next_letter
        if _has_shadda(next_harakat) or _has_sukun(next_harakat):
            rules.append(_make_annotation("madd_6", char_idx, end_idx))
            return
        if next_ch in HAMZA_CHARS:
            rules.append(_make_annotation("madd_muttasil", char_idx, end_idx))
            return

    if next_word:
        nf = _first_base_letter(next_word)
        if nf in HAMZA_CHARS:
            rules.append(_make_annotation("madd_munfasil", char_idx, end_idx))
            return

    if next_letter is None and is_last:
        rules.append(_make_annotation("madd_246", char_idx, end_idx))
        return

    if next_letter is not None:
        rules.append(_make_annotation("madd_2", char_idx, end_idx))


def _detect_lam_shamsiyyah(word, letters, rules):
    for i in range(len(letters) - 1):
        ch, char_idx, harakat = letters[i]
        if ch in (ALEF, ALEF_WASLA) and i + 1 < len(letters):
            next_ch, _, _ = letters[i + 1]
            if next_ch == LAAM and i + 2 < len(letters):
                after_lam, _, _ = letters[i + 2]
                if after_lam in SHAMS_LETTERS:
                    lam_idx = letters[i + 1][1]
                    rules.append(_make_annotation("lam_shamsiyyah", lam_idx, lam_idx + 1))
            break


# ── WordTajweedInfo + annotate_surah ───────────────────────────────────────────

@dataclass
class WordTajweedInfo:
    global_index: int
    word_text: str
    ayah: int
    word_in_ayah: int
    rules: List[TajweedAnnotation]

    @property
    def has_rules(self):
        return len(self.rules) > 0

    @property
    def rule_names(self):
        return [r.rule for r in self.rules]

    @property
    def primary_rule(self):
        priority = ["madd_6", "madd_muttasil", "madd_munfasil", "madd_246", "madd_2",
                     "ghunnah", "qalqalah", "ikhfa", "ikhfa_shafawi",
                     "idgham_ghunnah", "idgham_no_ghunnah", "idgham_shafawi",
                     "iqlab", "lam_shamsiyyah"]
        for p in priority:
            if p in self.rule_names:
                return p
        return self.rules[0].rule if self.rules else None

    def to_dict(self):
        return {
            "index": self.global_index,
            "word": self.word_text,
            "ayah": self.ayah,
            "wordInAyah": self.word_in_ayah,
            "rules": [
                {
                    "rule": r.rule,
                    "category": r.rule_category,
                    "charStart": r.char_start,
                    "charEnd": r.char_end,
                    "description": r.description,
                    "arabicName": r.arabic_name,
                    "harakatCount": r.harakat_count,
                }
                for r in self.rules
            ],
            "primaryRule": self.primary_rule,
        }


def annotate_surah(surah: int, db=None) -> List[WordTajweedInfo]:
    """Annotate all words in a surah with tajweed rules."""
    from quran_db import get_quran_db
    if db is None:
        db = get_quran_db()

    verses = db.get_surah_verses(surah)
    if not verses:
        return []

    results = []
    global_idx = 0

    for verse in verses:
        words = verse.words
        for wi, word in enumerate(words):
            next_word = None
            is_last = False

            if wi + 1 < len(words):
                next_word = words[wi + 1]
            else:
                nv = db.get_verse(surah, verse.ayah + 1)
                if nv and nv.words:
                    next_word = nv.words[0]
                is_last = True

            rules = get_word_tajweed_rules(word, next_word, is_last)

            results.append(WordTajweedInfo(
                global_index=global_idx,
                word_text=word,
                ayah=verse.ayah,
                word_in_ayah=wi,
                rules=rules,
            ))
            global_idx += 1

    return results


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from quran_db import get_quran_db
    db = get_quran_db()

    print("═══ Al-Fatiha Tajweed Rules ═══\n")
    annotations = annotate_surah(1, db)
    for info in annotations:
        if info.has_rules:
            rules_str = ", ".join(f"{r.rule} ({r.arabic_name})" for r in info.rules)
            print(f"  [{info.global_index}] {info.ayah}:{info.word_in_ayah} '{info.word_text}' → {rules_str}")

    total_rules = sum(len(i.rules) for i in annotations)
    words_with = sum(1 for i in annotations if i.has_rules)
    print(f"\n  Total: {total_rules} rules across {words_with}/{len(annotations)} words")

    print("\n═══ Stats ═══")
    for s in [1, 2, 36, 112]:
        ann = annotate_surah(s, db)
        t = sum(len(a.rules) for a in ann)
        w = sum(1 for a in ann if a.has_rules)
        print(f"  Surah {s}: {len(ann)} words, {w} with rules, {t} annotations")