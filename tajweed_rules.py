"""
tajweed_rules.py — Text-Based Tajweed Rule Identification
==========================================================
Phase 3: Identifies WHERE tajweed rules exist in the Quran text.
No audio analysis. Pure text analysis on diacritized Uthmani script.

Fixes vs previous version:
  - YA/WAW without explicit harakat (implicit sukun) now detected as madd carriers
  - Tatweel + superscript alef (مَـٰ) pattern now detected correctly
  - Hamzat al-wasl (ALEF_WASLA ٱ) excluded from madd detection
  - Each madd subtype now has its own category for distinct frontend coloring
  - madd_246 harakat_count corrected to None (variable 2/4/6, not fixed)

Usage:
    from tajweed_rules import annotate_surah
    annotations = annotate_surah(1)  # Al-Fatiha
"""

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
ALEF_WASLA = '\u0671'       # hamzat al-wasl — silent joining alef, NOT a madd carrier
ALEF_HAMZA_ABOVE = '\u0623'
ALEF_HAMZA_BELOW = '\u0625'
ALEF_MADDA = '\u0622'
ALEF_MAQSURA = '\u0649'
WAW    = '\u0648'
WAW_HAMZA = '\u0624'
YA     = '\u064A'
YA_HAMZA = '\u0626'
HAMZA  = '\u0621'
TATWEEL = '\u0640'

QALQALAH_LETTERS = {'\u0642', '\u0637', '\u0628', '\u062C', '\u062F'}

IKHFA_LETTERS = {
    '\u062A', '\u062B', '\u062C', '\u062F', '\u0630',
    '\u0632', '\u0633', '\u0634', '\u0635', '\u0636',
    '\u0637', '\u0638', '\u0641', '\u0642', '\u0643',
}

IDGHAM_GHUNNA_LETTERS  = {YA, NOON, MEEM, WAW}
IDGHAM_NO_GHUNNA_LETTERS = {LAAM, '\u0631'}

SHAMS_LETTERS = {
    '\u062A', '\u062B', '\u062F', '\u0630', '\u0631',
    '\u0632', '\u0633', '\u0634', '\u0635', '\u0636',
    '\u0637', '\u0638', LAAM, NOON,
}

HAMZA_CHARS = {
    HAMZA, '\u0654', '\u0655',
    ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW,
    WAW_HAMZA, YA_HAMZA,
}

HARAKAT = {
    FATHA, DAMMA, KASRA, SHADDA, SUKUN,
    FATHATAN, DAMMATAN, KASRATAN, SUPERSCRIPT_ALEF,
}

MADD_LETTERS = {ALEF, WAW, YA, ALEF_MAQSURA}

# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class TajweedAnnotation:
    rule: str
    rule_category: str
    char_start: int
    char_end: int
    description: str
    arabic_name: str
    harakat_count: Optional[int] = None

# Each madd subtype has its own category so the frontend can color them differently
RULE_INFO = {
    "ghunnah":            {"cat": "ghunna",        "ar": "غُنَّة",              "desc": "Nasalization — hold for 2 counts"},
    "ikhfa":              {"cat": "ikhfa",          "ar": "إِخْفَاء",           "desc": "Hidden noon — light nasalization"},
    "ikhfa_shafawi":      {"cat": "ikhfa",          "ar": "إِخْفَاء شَفَوِي",    "desc": "Hidden meem before ba"},
    "idgham_ghunnah":     {"cat": "idgham",         "ar": "إِدْغَام بِغُنَّة",    "desc": "Merging with nasalization"},
    "idgham_no_ghunnah":  {"cat": "idgham",         "ar": "إِدْغَام بِلَا غُنَّة","desc": "Merging without nasalization"},
    "idgham_shafawi":     {"cat": "idgham",         "ar": "إِدْغَام شَفَوِي",    "desc": "Meem merged into meem"},
    "iqlab":              {"cat": "iqlab",          "ar": "إِقْلَاب",           "desc": "Noon becomes meem before ba"},
    "qalqalah":           {"cat": "qalqalah",       "ar": "قَلْقَلَة",          "desc": "Echoing bounce on stopped letter"},
    "lam_shamsiyyah":     {"cat": "lam_shams",      "ar": "لَام شَمْسِيَّة",     "desc": "Assimilated lam (sun letter)"},
    # ── Madd subtypes — distinct categories for distinct colors ──────────────
    "madd_2":             {"cat": "madd_2",         "ar": "مَدّ طَبِيعِي",       "desc": "Natural madd — 2 counts"},
    "madd_246":           {"cat": "madd_246",       "ar": "مَدّ عَارِض",         "desc": "Presented madd — 2, 4, or 6 counts (at waqf)"},
    "madd_muttasil":      {"cat": "madd_muttasil",  "ar": "مَدّ مُتَّصِل",       "desc": "Connected madd — 4-5 counts (obligatory)"},
    "madd_munfasil":      {"cat": "madd_munfasil",  "ar": "مَدّ مُنْفَصِل",      "desc": "Separated madd — 4-5 counts"},
    "madd_6":             {"cat": "madd_6",         "ar": "مَدّ لَازِم",         "desc": "Obligatory madd — 6 counts"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_base_letters(word: str) -> List[tuple]:
    """
    Returns list of (base_char, raw_string_index, [harakat]) for each
    base letter in the word. Skips TATWEEL and HARAKAT.
    """
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

def _has_sukun(h):  return SUKUN in h
def _has_shadda(h): return SHADDA in h
def _has_vowel(h):  return bool(set(h) & {FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN})

def _is_noon_sakin(ch, harakat):
    return ch == NOON and (_has_sukun(harakat) or
                           (not _has_vowel(harakat) and not _has_shadda(harakat)))

def _is_meem_sakin(ch, harakat):
    return ch == MEEM and (_has_sukun(harakat) or
                           (not _has_vowel(harakat) and not _has_shadda(harakat)))

def _first_base_letter(word):
    """Return the first base consonant in a word (skips wasla/article alef)."""
    if not word:
        return None
    for ch in word:
        if ch not in HARAKAT and ch != TATWEEL and ch != ALEF_WASLA and ch != ALEF:
            return ch
    letters = _get_base_letters(word)
    return letters[1][0] if len(letters) >= 2 else (letters[0][0] if letters else None)

def _is_wasla_alef(word_chars: list, char_idx: int) -> bool:
    """
    Returns True if the alef at char_idx is hamzat al-wasl (a joining/silent alef).
    Hamzat al-wasl is NEVER a madd carrier.
    - ALEF_WASLA (U+0671) is always wasla
    - Plain ALEF at position 0 before LAM = definite article (also not a madd)
    """
    ch = word_chars[char_idx]
    if ch == ALEF_WASLA:
        return True
    if ch == ALEF and char_idx == 0:
        j = char_idx + 1
        while j < len(word_chars) and word_chars[j] in HARAKAT:
            j += 1
        if j < len(word_chars) and word_chars[j] == LAAM:
            return True
    return False

def _make_annotation(rule, char_start, char_end):
    info = RULE_INFO.get(rule, {"cat": "unknown", "ar": rule, "desc": rule})
    # harakat_count = expected beat count for audio verification (Phase 5)
    hc = None
    if rule == "madd_2":        hc = 2
    elif rule == "madd_246":    hc = 2     # minimum expectation at waqf (can be 2/4/6 — we verify the floor)
    elif rule == "madd_muttasil": hc = 4
    elif rule == "madd_munfasil": hc = 4
    elif rule == "madd_6":      hc = 6
    elif rule == "ghunnah":     hc = 2
    return TajweedAnnotation(
        rule=rule,
        rule_category=info["cat"],
        char_start=char_start,
        char_end=char_end,
        description=info["desc"],
        arabic_name=info["ar"],
        harakat_count=hc,
    )

# ── Madd detection ────────────────────────────────────────────────────────────

def _detect_madd(ch, char_idx, harakat, idx, letters, next_word, is_last, rules,
                 word_chars=None, word=None):
    """
    Detect madd rule for a single letter position.

    Madd carriers:
      - ALEF (plain, not wasla/article) — always
      - ALEF_MAQSURA, SUPERSCRIPT_ALEF  — always
      - WAW with sukun OR no vowel      — madd extension of damma
      - YA  with sukun OR no vowel      — madd extension of kasra

    Wasla alef (ALEF_WASLA or article alef at pos 0 before lam) is excluded
    by the caller checking _is_wasla_alef before calling this function.

    Madd subtypes (in priority order):
      madd_6        — next letter has sukun or shadda (same word)
      madd_muttasil — next letter is hamza (same word)
      madd_munfasil — first letter of next word is hamza
      madd_246      — end of word at waqf position (last letter in ayah)
      madd_2        — all other cases (natural madd)
    """
    # Determine if this is a genuine madd carrier
    is_madd_carrier = False

    if ch in (ALEF, ALEF_MAQSURA, SUPERSCRIPT_ALEF, ALEF_MADDA):
        is_madd_carrier = True
    elif ch == WAW and (_has_sukun(harakat) or not _has_vowel(harakat)):
        # WAW with no vowel = madd extension (not conjunction waw which has damma)
        is_madd_carrier = True
    elif ch == YA and (_has_sukun(harakat) or not _has_vowel(harakat)):
        is_madd_carrier = True

    if not is_madd_carrier:
        return

    end_idx = char_idx + 1 + len(harakat)
    next_letter = letters[idx + 1] if idx + 1 < len(letters) else None

    # madd_6: immediately followed by sukun or shadda in same word
    if next_letter:
        next_ch, _, next_harakat = next_letter
        if _has_shadda(next_harakat) or _has_sukun(next_harakat):
            rules.append(_make_annotation("madd_6", char_idx, end_idx))
            return
        # madd_muttasil: followed by hamza in same word
        if next_ch in HAMZA_CHARS:
            rules.append(_make_annotation("madd_muttasil", char_idx, end_idx))
            return

    # madd_munfasil: next word starts with hamza
    if next_word:
        nf = _first_base_letter(next_word)
        if nf in HAMZA_CHARS:
            rules.append(_make_annotation("madd_munfasil", char_idx, end_idx))
            return

    # madd_246: waqf position
    #   Case A: madd letter is the very last letter of the word/ayah
    if next_letter is None and is_last:
        rules.append(_make_annotation("madd_246", char_idx, end_idx))
        return
    #   Case B: madd letter is second-to-last AND this is the last word of the ayah
    #   e.g. ي in يـنَ at end of surah — at waqf you stop on ي, the final نَ is silent
    MADD_CARRIERS = {ALEF, ALEF_MAQSURA, SUPERSCRIPT_ALEF, ALEF_MADDA, WAW, YA}
    if is_last and next_letter is not None:
        next_ch, _, _ = next_letter
        is_penultimate = (idx + 2 >= len(letters))  # only one more letter after this
        if is_penultimate and next_ch not in MADD_CARRIERS:
            rules.append(_make_annotation("madd_246", char_idx, end_idx))
            return

    # madd_2: natural madd (all other positions)
    rules.append(_make_annotation("madd_2", char_idx, end_idx))


def _detect_tatweel_madd(word: str, rules: list):
    """
    Detect the tatweel + superscript alef madd pattern.

    In many Uthmani words a long-a vowel is written as:
        consonant + fatha + tatweel (U+0640) + superscript alef (U+0670)
    Example: مَـٰلِكِ  ٱلرَّحْمَـٰنِ

    _get_base_letters skips TATWEEL entirely, so this pattern is never
    seen by the main letter loop. We scan raw chars here.

    The colored range starts at the consonant before the tatweel.
    """
    chars = list(word)
    n = len(chars)
    for i in range(n):
        if chars[i] != TATWEEL:
            continue
        # Find the base consonant before the tatweel (skip back over harakat)
        start = i - 1
        while start >= 0 and chars[start] in HARAKAT:
            start -= 1
        if start < 0:
            continue
        # End = after superscript alef if present, else after tatweel
        end = i + 1
        if end < n and chars[end] == SUPERSCRIPT_ALEF:
            end += 1
        rules.append(_make_annotation("madd_2", start, end))


def _detect_lam_shamsiyyah(word: str, letters: list, rules: list):
    """
    Detect lam shamsiyyah: the lam in ال is assimilated into the following
    sun letter (it's silent / doubled into the next letter).
    Triggered when: alef/wasla_alef → lam → sun letter
    """
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

# ── Main word detection ───────────────────────────────────────────────────────

def get_word_tajweed_rules(word, next_word=None, is_last_word_in_ayah=False):
    """
    Identify all tajweed rules that apply to a word.
    Returns list of TajweedAnnotation with char_start/char_end for coloring.
    """
    rules = []
    letters = _get_base_letters(word)
    if not letters:
        return rules

    word_chars = list(word)
    next_first_base = _first_base_letter(next_word) if next_word else None

    # Track char indices already claimed by tatweel-madd (avoid double annotation)
    tatweel_ranges = set()
    for i, ch_raw in enumerate(word_chars):
        if ch_raw == TATWEEL:
            start = i - 1
            while start >= 0 and word_chars[start] in HARAKAT:
                start -= 1
            if start >= 0:
                end = i + 2 if (i + 1 < len(word_chars) and
                                  word_chars[i + 1] == SUPERSCRIPT_ALEF) else i + 1
                for j in range(start, end):
                    tatweel_ranges.add(j)

    for idx, (ch, char_idx, harakat) in enumerate(letters):
        next_letter = letters[idx + 1] if idx + 1 < len(letters) else None
        end_idx = char_idx + 1 + len(harakat)

        # ── Ghunna: noon or meem with shadda ─────────────────────────────
        if ch in (NOON, MEEM) and _has_shadda(harakat):
            rules.append(_make_annotation("ghunnah", char_idx, end_idx))

        # ── Noon sakin rules ──────────────────────────────────────────────
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

        # ── Meem sakin rules ──────────────────────────────────────────────
        if _is_meem_sakin(ch, harakat):
            check = next_letter[0] if next_letter else next_first_base
            if check:
                if check == BA:
                    rules.append(_make_annotation("ikhfa_shafawi", char_idx, end_idx))
                elif check == MEEM and not next_letter:
                    rules.append(_make_annotation("idgham_shafawi", char_idx, end_idx))

        # ── Qalqalah ──────────────────────────────────────────────────────
        if ch in QALQALAH_LETTERS:
            is_sakin = _has_sukun(harakat) or (
                not _has_vowel(harakat) and not _has_shadda(harakat) and
                (next_letter is None or is_last_word_in_ayah)
            )
            if is_sakin:
                rules.append(_make_annotation("qalqalah", char_idx, end_idx))

        # ── Madd ─────────────────────────────────────────────────────────
        # Skip if already covered by tatweel pattern, or if wasla alef
        if char_idx in tatweel_ranges:
            continue
        if ch in MADD_LETTERS or ch == SUPERSCRIPT_ALEF or ch == ALEF_MAQSURA:
            if not _is_wasla_alef(word_chars, char_idx):
                _detect_madd(ch, char_idx, harakat, idx, letters,
                             next_word, is_last_word_in_ayah, rules,
                             word_chars=word_chars, word=word)

        # ── Alef Madda (آ = alef + madda sign) ───────────────────────────
        # Route through _detect_madd so it can become madd_6, muttasil, etc.
        if ch == ALEF_MADDA:
            _detect_madd(ch, char_idx, harakat, idx, letters,
                         next_word, is_last_word_in_ayah, rules,
                         word_chars=word_chars, word=word)

    # ── Tatweel + superscript alef madd ──────────────────────────────────
    _detect_tatweel_madd(word, rules)

    # ── Lam Shamsiyyah ────────────────────────────────────────────────────
    _detect_lam_shamsiyyah(word, letters, rules)

    return rules

# ── WordTajweedInfo + annotate_surah ─────────────────────────────────────────

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
        priority = [
            "madd_6", "madd_muttasil", "madd_munfasil", "madd_246", "madd_2",
            "ghunnah", "qalqalah", "ikhfa", "ikhfa_shafawi",
            "idgham_ghunnah", "idgham_no_ghunnah", "idgham_shafawi",
            "iqlab", "lam_shamsiyyah",
        ]
        for p in priority:
            if p in self.rule_names:
                return p
        return self.rules[0].rule if self.rules else None

    def to_dict(self):
        return {
            "index":       self.global_index,
            "word":        self.word_text,
            "ayah":        self.ayah,
            "wordInAyah":  self.word_in_ayah,
            "rules": [
                {
                    "rule":        r.rule,
                    "category":    r.rule_category,
                    "charStart":   r.char_start,
                    "charEnd":     r.char_end,
                    "description": r.description,
                    "arabicName":  r.arabic_name,
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

            word_rules = get_word_tajweed_rules(word, next_word, is_last)

            results.append(WordTajweedInfo(
                global_index=global_idx,
                word_text=word,
                ayah=verse.ayah,
                word_in_ayah=wi,
                rules=word_rules,
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
        word_chars = list(info.word_text)
        print(f"  [{info.global_index}] {info.ayah}:{info.word_in_ayah} '{info.word_text}'")
        if info.has_rules:
            for r in info.rules:
                colored = ''.join(word_chars[r.char_start:r.char_end])
                print(f"    → {r.rule} ({r.rule_category}) chars[{r.char_start}:{r.char_end}] = '{colored}'")
        else:
            print(f"    → (no rules)")

    total_rules = sum(len(i.rules) for i in annotations)
    words_with  = sum(1 for i in annotations if i.has_rules)
    print(f"\n  Total: {total_rules} rules across {words_with}/{len(annotations)} words")

    print("\n═══ Stats ═══")
    for s in [1, 2, 36, 112]:
        ann = annotate_surah(s, db)
        t = sum(len(a.rules) for a in ann)
        w = sum(1 for a in ann if a.has_rules)
        print(f"  Surah {s}: {len(ann)} words, {w} with rules, {t} annotations")