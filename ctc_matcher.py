"""
ctc_matcher.py — CTC Text Matching for WhisperQuran Phase 2
============================================================

Matches CTC transcription output against known Quran text.
Tracks word-level position within a surah.
Returns per-word match results with confidence scores.

This replaces the frontend's fuzzy string matching (arabicUtils.ts)
with server-side matching that has access to:
  - The full CTC transcript (with diacritics from FastConformer PCD)
  - The complete Quran text database (QuranDB)
  - Position tracking across multiple audio chunks

Architecture:
  Frontend sends: refText (surah words) + audio chunks
  Backend runs:   CTC inference → ctc_matcher → word-level results
  Backend sends:  {"type":"match","words":[...]} per chunk

The key insight: we know what the reciter SHOULD be saying (the Quran
is a closed corpus), so we score the transcript against known text
instead of treating it as free-form ASR.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from quran_db import QuranDB, get_quran_db, normalize_arabic, levenshtein, strip_diacritics

log = logging.getLogger("whisperquran.matcher")


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class WordMatch:
    """Result of matching one spoken word against expected Quran word."""
    global_index: int           # Index in the flat surah word list
    expected: str               # Diacritized expected word
    expected_norm: str          # Normalized expected word
    spoken: str                 # What the CTC heard
    spoken_norm: str            # Normalized spoken word
    similarity: float           # 0.0 - 1.0
    matched: bool               # True if similarity >= threshold
    ayah: int                   # Which ayah this word belongs to
    word_in_ayah: int           # Word index within the ayah


@dataclass
class MatchResult:
    """Result of matching a CTC chunk against the current position."""
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
        Match a CTC transcript against expected words at current position.
        
        Args:
            transcript: Diacritized Arabic text from FastConformer CTC
            threshold: Minimum similarity to accept a word match
            max_lookahead: How many positions ahead to search for a match
                          (handles word skips / insertions by ASR)
        
        Returns:
            MatchResult with per-word details and updated position
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
            
            # Try matching at current position + lookahead
            best_match = None
            best_sim = 0.0
            best_offset = 0
            
            for offset in range(min(max_lookahead, self.total_words - pos)):
                expected = self.words[pos + offset]
                sim = self._word_similarity(spoken_norm, expected.norm)
                
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
        # Remove common punctuation that FastConformer may include
        text = re.sub(r'[،؟.!,?]', ' ', text)
        return [w for w in text.strip().split() if w]
    
    @staticmethod
    def _word_similarity(spoken_norm: str, expected_norm: str) -> float:
        """
        Compute similarity between two normalized Arabic words.
        Returns 0.0 - 1.0.
        
        Uses Levenshtein distance on normalized text.
        Includes bonuses for matching first character and article handling.
        """
        if spoken_norm == expected_norm:
            return 1.0
        
        if not spoken_norm or not expected_norm:
            return 0.0
        
        # Base Levenshtein similarity
        dist = levenshtein(spoken_norm, expected_norm)
        max_len = max(len(spoken_norm), len(expected_norm))
        sim = 1.0 - (dist / max_len)
        
        # Bonus for matching first consonant
        if spoken_norm[0] == expected_norm[0]:
            sim += 0.03
        
        # Try with article stripped (ال / ٱل)
        def strip_article(t):
            return re.sub(r'^(ال|ٱل|لل|لا)', '', t)
        
        spoken_stripped = strip_article(spoken_norm)
        expected_stripped = strip_article(expected_norm)
        
        if spoken_stripped and expected_stripped:
            if spoken_stripped == expected_stripped:
                return max(sim, 0.90)
            
            dist2 = levenshtein(spoken_stripped, expected_stripped)
            max_len2 = max(len(spoken_stripped), len(expected_stripped))
            sim2 = 1.0 - (dist2 / max_len2) + 0.03
            sim = max(sim, sim2)
        
        return min(1.0, sim)
    
    def to_wire(self, result: MatchResult) -> dict:
        """
        Convert MatchResult to JSON-serializable dict for WebSocket.
        
        Wire format matches what the frontend expects:
        {
            "type": "match",
            "words": [
                {
                    "index": 0,           // global word index in surah
                    "expected": "ٱلْحَمْدُ",
                    "spoken": "الحمد",
                    "similarity": 0.92,
                    "matched": true,
                    "ayah": 2,
                    "wordInAyah": 0,
                    "retries": 0
                },
                ...
            ],
            "position": 4,
            "ayah": 2,
            "wordsMatched": 4,
            "totalWords": 29,
            "complete": false
        }
        """
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
    session = RecitationSession(surah=1, db=db)
    
    print(f"\nSurah 1: {session.total_words} words")
    print(f"Words: {[w.text for w in session.words[:10]]}")
    
    # Simulate recitation chunks
    chunks = [
        "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
        "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَالَمِينَ",
        "ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
        "مَالِكِ يَوْمِ ٱلدِّينِ",
    ]
    
    for chunk in chunks:
        result = session.match_transcript(chunk)
        wire = session.to_wire(result)
        
        print(f"\nChunk: '{chunk[:50]}...'")
        print(f"  Matched: {result.words_matched} words, position → {result.new_position}")
        for wm in result.words:
            status = "✓" if wm.matched else "✗"
            print(f"    {status} [{wm.global_index}] '{wm.spoken}' → '{wm.expected}' "
                  f"(sim={wm.similarity:.2f}, ayah={wm.ayah})")
    
    print(f"\nFinal position: {session.position}/{session.total_words}")
    print(f"Complete: {session.position >= session.total_words}")
