"""
ctc_matcher.py — Text Matching for WhisperQuran Phase 2
========================================================

Matches ASR transcription output against known Quran text.
Tracks word-level position within a surah.
Returns per-word match results with confidence scores.

TWO-TIER MATCHING (updated for Transducer decoder with tashkeel):
  1. If both spoken and expected have diacritics → compare WITH tashkeel
     (stricter, higher confidence, rewards correct harakat)
  2. If spoken lacks diacritics (CTC mode) → compare normalized (stripped)
     (lenient, still works, but capped at 0.90 max)

This way:
  - RNNT decoder outputs "بِسْمِ" → compared diacritized against "بِسْمِ" → 1.0
  - CTC decoder outputs "بسم" → compared normalized against "بسم" → 0.90 max
  - Wrong harakat "بُسْمِ" → diacritized comparison catches it → lower score
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from quran_db import QuranDB, get_quran_db, normalize_arabic, levenshtein, strip_diacritics

log = logging.getLogger("whisperquran.matcher")

# Regex to detect if a word has Arabic diacritics (harakat)
HARAKAT_RE = re.compile(r'[\u064B-\u0652\u0670]')


def has_tashkeel(text: str) -> bool:
    """Check if Arabic text contains diacritical marks."""
    return bool(HARAKAT_RE.search(text))


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class WordMatch:
    """Result of matching one spoken word against expected Quran word."""
    global_index: int           # Index in the flat surah word list
    expected: str               # Diacritized expected word
    expected_norm: str          # Normalized expected word
    spoken: str                 # What the ASR heard
    spoken_norm: str            # Normalized spoken word
    similarity: float           # 0.0 - 1.0
    matched: bool               # True if similarity >= threshold
    ayah: int                   # Which ayah this word belongs to
    word_in_ayah: int           # Word index within the ayah


@dataclass
class MatchResult:
    """Result of matching an ASR chunk against the current position."""
    words: List[WordMatch]      # Per-word match results
    words_matched: int          # How many words were successfully matched
    new_position: int           # Updated global word index after matching
    ayah: int                   # Current ayah number
    complete: bool              # True if we've reached end of surah


@dataclass
class SurahWord:
    """A single word in a surah's flat word list."""
    text: str                   # Diacritized text
    norm: str                   # Normalized text
    ayah: int                   # Ayah number
    word_in_ayah: int           # Word index within ayah
    global_index: int           # Index in flat surah list


# ── Session tracker ────────────────────────────────────────────────────────────

class RecitationSession:
    """
    Tracks a user's position in a surah across multiple audio chunks.
    
    One session per WebSocket connection. Maintains:
      - Current word position (global index within surah)
      - Expected word sequence for the surah
      - Match history for retry detection
    """
    
    def __init__(self, surah: int, db: Optional[QuranDB] = None):
        self.surah = surah
        self.db = db or get_quran_db()
        self.position = 0           # Current global word index
        self.total_words = 0
        self.words: List[SurahWord] = []
        self._retries: dict = {}    # global_index → retry count
        
        self._build_word_list()
    
    def _build_word_list(self):
        """Build flat word list for the surah from QuranDB."""
        verses = self.db.get_surah_verses(self.surah)
        if not verses:
            log.warning(f"No verses found for surah {self.surah}")
            return
        
        idx = 0
        for verse in verses:
            for i, (word, word_norm) in enumerate(zip(verse.words, verse.words_norm)):
                self.words.append(SurahWord(
                    text=word,
                    norm=word_norm,
                    ayah=verse.ayah,
                    word_in_ayah=i,
                    global_index=idx,
                ))
                idx += 1
        
        self.total_words = len(self.words)
        log.info(f"Session for surah {self.surah}: {self.total_words} words, "
                 f"{len(verses)} ayahs")
    
    def match_transcript(
        self,
        transcript: str,
        threshold: float = 0.65,
        max_lookahead: int = 3,
    ) -> MatchResult:
        """
        Match an ASR transcript against expected words at current position.
        
        Uses two-tier comparison:
          - With tashkeel: if RNNT decoder provided diacritics, compare
            diacritized text for higher accuracy (up to 1.0)
          - Without tashkeel: if CTC decoder or stripped output, compare
            normalized text (capped at 0.90)
        """
        if not transcript or not transcript.strip():
            return MatchResult(
                words=[], words_matched=0,
                new_position=self.position,
                ayah=self._current_ayah(),
                complete=self.position >= self.total_words,
            )
        
        spoken_words = self._split_arabic(transcript)
        if not spoken_words:
            return MatchResult(
                words=[], words_matched=0,
                new_position=self.position,
                ayah=self._current_ayah(),
                complete=self.position >= self.total_words,
            )
        
        matches = []
        pos = self.position
        words_matched = 0
        
        for spoken in spoken_words:
            if pos >= self.total_words:
                break
            
            spoken_norm = normalize_arabic(spoken)
            if not spoken_norm:
                continue
            
            spoken_has_tashkeel = has_tashkeel(spoken)
            
            # Try matching at current position + lookahead
            best_match = None
            best_sim = 0.0
            best_offset = 0
            
            for offset in range(min(max_lookahead, self.total_words - pos)):
                expected = self.words[pos + offset]
                sim = self._word_similarity(
                    spoken, spoken_norm, spoken_has_tashkeel,
                    expected.text, expected.norm,
                )
                
                if sim > best_sim:
                    best_sim = sim
                    best_match = expected
                    best_offset = offset
            
            if best_match and best_sim >= threshold:
                # Successful match
                wm = WordMatch(
                    global_index=best_match.global_index,
                    expected=best_match.text,
                    expected_norm=best_match.norm,
                    spoken=spoken,
                    spoken_norm=spoken_norm,
                    similarity=round(best_sim, 3),
                    matched=True,
                    ayah=best_match.ayah,
                    word_in_ayah=best_match.word_in_ayah,
                )
                matches.append(wm)
                words_matched += 1
                
                # Mark any skipped words as missed
                for skip in range(best_offset):
                    skipped = self.words[pos + skip]
                    matches.append(WordMatch(
                        global_index=skipped.global_index,
                        expected=skipped.text,
                        expected_norm=skipped.norm,
                        spoken="",
                        spoken_norm="",
                        similarity=0.0,
                        matched=False,
                        ayah=skipped.ayah,
                        word_in_ayah=skipped.word_in_ayah,
                    ))
                
                pos = best_match.global_index + 1
                
            else:
                # No match — record as incorrect attempt at current position
                expected = self.words[pos]
                wm = WordMatch(
                    global_index=expected.global_index,
                    expected=expected.text,
                    expected_norm=expected.norm,
                    spoken=spoken,
                    spoken_norm=spoken_norm,
                    similarity=round(best_sim, 3),
                    matched=False,
                    ayah=expected.ayah,
                    word_in_ayah=expected.word_in_ayah,
                )
                matches.append(wm)
                
                # Track retries
                self._retries[expected.global_index] = \
                    self._retries.get(expected.global_index, 0) + 1
                
                # Don't advance position on failed match
        
        # Update session position
        self.position = pos
        
        return MatchResult(
            words=matches,
            words_matched=words_matched,
            new_position=pos,
            ayah=self._current_ayah(),
            complete=pos >= self.total_words,
        )
    
    def reset(self):
        """Reset position to beginning of surah."""
        self.position = 0
        self._retries.clear()
    
    def set_position(self, global_index: int):
        """Jump to a specific word position."""
        self.position = max(0, min(global_index, self.total_words))
    
    def get_retries(self, global_index: int) -> int:
        """Get retry count for a specific word."""
        return self._retries.get(global_index, 0)
    
    # ── Helpers ────────────────────────────────────────────────────────────
    
    def _current_ayah(self) -> int:
        """Get the current ayah number based on position."""
        if self.position >= self.total_words:
            return self.words[-1].ayah if self.words else 1
        return self.words[self.position].ayah
    
    @staticmethod
    def _split_arabic(text: str) -> List[str]:
        """Split Arabic text into words, handling various whitespace."""
        text = re.sub(r'[،؟.!,?]', ' ', text)
        return [w for w in text.strip().split() if w]
    
    @staticmethod
    def _simplify_uthmani(t: str) -> str:
        """
        Simplify Uthmani-specific encoding differences that don't affect
        pronunciation but cause string mismatches between ASR and QuranDB.
        Keeps all harakat intact.
        """
        t = t.replace('\u0670', '')          # superscript alef (ٰ)
        t = t.replace('\u0671', '\u0627')    # alef wasla (ٱ) → regular alef (ا)
        # Remove Quran-specific pause/sajda marks
        t = re.sub(r'[\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', t)
        t = t.replace('\u0640', '')           # tatweel
        return t
    
    @staticmethod
    def _word_similarity(
        spoken_raw: str,
        spoken_norm: str,
        spoken_has_tashkeel: bool,
        expected_raw: str,
        expected_norm: str,
    ) -> float:
        """
        Two-tier word similarity for Arabic Quran matching.
        
        Tier 1 (diacritized): When ASR has tashkeel (RNNT decoder),
        compare full diacritized forms. Catches wrong harakat.
        Score range: 0.0 - 1.0
        
        Tier 2 (normalized): When ASR lacks tashkeel (CTC decoder),
        compare stripped consonant forms. Capped at 0.90.
        Score range: 0.0 - 0.90
        """
        # Exact diacritized match → perfect score
        if spoken_raw == expected_raw:
            return 1.0
        
        # Exact normalized match
        if spoken_norm == expected_norm:
            if spoken_has_tashkeel:
                return 0.85  # Consonants match but harakat differ
            return 0.90  # No tashkeel — can't tell, give benefit
        
        if not spoken_norm or not expected_norm:
            return 0.0
        
        # ── Tier 1: Diacritized comparison ─────────────────────────────────
        diacritized_sim = 0.0
        if spoken_has_tashkeel:
            spoken_simp = RecitationSession._simplify_uthmani(spoken_raw)
            expected_simp = RecitationSession._simplify_uthmani(expected_raw)
            
            if spoken_simp == expected_simp:
                return 0.98  # Near-perfect, just Uthmani encoding diffs
            
            dist = levenshtein(spoken_simp, expected_simp)
            max_len = max(len(spoken_simp), len(expected_simp))
            diacritized_sim = 1.0 - (dist / max_len) if max_len > 0 else 0.0
            
            if spoken_simp and expected_simp and spoken_simp[0] == expected_simp[0]:
                diacritized_sim += 0.02
        
        # ── Tier 2: Normalized comparison (consonants only) ────────────────
        dist_norm = levenshtein(spoken_norm, expected_norm)
        max_len_norm = max(len(spoken_norm), len(expected_norm))
        normalized_sim = 1.0 - (dist_norm / max_len_norm) if max_len_norm > 0 else 0.0
        
        if spoken_norm and expected_norm and spoken_norm[0] == expected_norm[0]:
            normalized_sim += 0.03
        
        # Article stripping bonus
        def strip_article(t: str) -> str:
            return re.sub(r'^(ال|ٱل|لل|لا)', '', t)
        
        spoken_stripped = strip_article(spoken_norm)
        expected_stripped = strip_article(expected_norm)
        
        if spoken_stripped and expected_stripped:
            if spoken_stripped == expected_stripped:
                normalized_sim = max(normalized_sim, 0.88)
            else:
                dist2 = levenshtein(spoken_stripped, expected_stripped)
                max_len2 = max(len(spoken_stripped), len(expected_stripped))
                alt_sim = 1.0 - (dist2 / max_len2) + 0.03 if max_len2 > 0 else 0.0
                normalized_sim = max(normalized_sim, alt_sim)
        
        # Cap normalized at 0.90 when spoken has no tashkeel
        if not spoken_has_tashkeel:
            normalized_sim = min(0.90, normalized_sim)
        
        final = max(diacritized_sim, normalized_sim)
        return min(1.0, final)
    
    def to_wire(self, result: MatchResult) -> dict:
        """Convert MatchResult to JSON for WebSocket."""
        return {
            "type": "match",
            "words": [
                {
                    "index": wm.global_index,
                    "expected": wm.expected,
                    "spoken": wm.spoken,
                    "similarity": wm.similarity,
                    "matched": wm.matched,
                    "ayah": wm.ayah,
                    "wordInAyah": wm.word_in_ayah,
                    "retries": self.get_retries(wm.global_index),
                }
                for wm in result.words
            ],
            "position": result.new_position,
            "ayah": result.ayah,
            "wordsMatched": result.words_matched,
            "totalWords": self.total_words,
            "complete": result.complete,
        }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    db = get_quran_db()
    
    # ── Test 1: RNNT output WITH tashkeel ──────────────────────────────────
    print("\n═══ Test 1: RNNT decoder (with tashkeel) ═══")
    session = RecitationSession(surah=1, db=db)
    print(f"Surah 1: {session.total_words} words")
    
    chunks_rnnt = [
        "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
        "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَالَمِينَ",
        "ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
        "مَالِكِ يَوْمِ ٱلدِّينِ",
    ]
    
    for chunk in chunks_rnnt:
        result = session.match_transcript(chunk)
        print(f"\n  Chunk: '{chunk[:50]}...'")
        print(f"  Matched: {result.words_matched}, position → {result.new_position}")
        for wm in result.words:
            status = "✓" if wm.matched else "✗"
            print(f"    {status} [{wm.global_index}] sim={wm.similarity:.2f}  "
                  f"'{wm.spoken}' → '{wm.expected}'")
    
    # ── Test 2: CTC output WITHOUT tashkeel ────────────────────────────────
    print("\n═══ Test 2: CTC decoder (no tashkeel) ═══")
    session2 = RecitationSession(surah=1, db=db)
    
    chunks_ctc = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
    ]
    
    for chunk in chunks_ctc:
        result = session2.match_transcript(chunk)
        print(f"\n  Chunk: '{chunk[:50]}...'")
        print(f"  Matched: {result.words_matched}, position → {result.new_position}")
        for wm in result.words:
            status = "✓" if wm.matched else "✗"
            print(f"    {status} [{wm.global_index}] sim={wm.similarity:.2f}  "
                  f"'{wm.spoken}' → '{wm.expected}'")
    
    # ── Test 3: Wrong harakat ──────────────────────────────────────────────
    print("\n═══ Test 3: Wrong harakat detection ═══")
    session3 = RecitationSession(surah=1, db=db)
    
    # رَبُّ instead of رَبِّ (wrong kasra→damma)
    result = session3.match_transcript("بِسْمُ ٱللَّهِ")  # wrong damma on meem
    print(f"\n  Wrong harakat: بِسْمُ (damma) vs بِسْمِ (kasra)")
    for wm in result.words:
        status = "✓" if wm.matched else "✗"
        print(f"    {status} sim={wm.similarity:.2f}  '{wm.spoken}' → '{wm.expected}'")
    
    print(f"\nRNNT position: {session.position}/{session.total_words}")
    print(f"CTC position:  {session2.position}/{session2.total_words}")