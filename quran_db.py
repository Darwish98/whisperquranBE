"""
quran_db.py — Quran Verse Database for WhisperQuran Phase 2
============================================================

All 6,236 verses with Uthmani diacritized text.
Indexed by (surah, ayah) for O(1) lookup.
Normalized (stripped diacritics) versions for fuzzy matching.

Data source: alquran.cloud API (Quran Uthmani edition)
Cached locally as JSON after first download.

Usage:
    db = QuranDB()
    verse = db.get_verse(2, 255)  # Al-Baqarah, Ayat al-Kursi
    words = db.get_words(2, 255)  # List of diacritized words
    
    # Get all words for a surah (for position tracking)
    surah_words = db.get_surah_words(2)
    
    # Fuzzy search
    results = db.search("الحمد لله رب العالمين", top_k=5)
"""

import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("whisperquran.qurandb")

# ── Diacritics removal ────────────────────────────────────────────────────────

DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
    r'\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED'
    r'\u0653-\u0655\uFE70-\uFE7F]'
)

def strip_diacritics(text: str) -> str:
    """Remove all Arabic diacritics/tashkeel."""
    return DIACRITICS_RE.sub('', text)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fuzzy comparison."""
    t = strip_diacritics(text.strip())
    t = re.sub(r'[\u0622\u0623\u0625\u0671\u0672\u0673\u0627]', '\u0627', t)  # unify alef
    t = t.replace('\u0629', '\u0647')    # ta marbuta → ha
    t = t.replace('\u0649', '\u064A')    # alef maqsura → ya
    t = t.replace('\u0640', '')           # tatweel
    t = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2060-\u2069\uFEFF]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance on strings."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(b)]


# ── Verse data structure ──────────────────────────────────────────────────────

class Verse:
    """A single Quran verse with both diacritized and normalized text."""
    __slots__ = ('surah', 'ayah', 'text', 'text_norm', 'words', 'words_norm', 'page')
    
    def __init__(self, surah: int, ayah: int, text: str, page: int = 0):
        self.surah = surah
        self.ayah = ayah
        self.text = text                                    # Original Uthmani with tashkeel
        self.text_norm = normalize_arabic(text)             # Stripped for fuzzy matching
        self.words = text.split()                           # Diacritized word list
        self.words_norm = self.text_norm.split()            # Normalized word list
        self.page = page                                    # Madinah Mushaf page (1-604)
    
    def __repr__(self):
        return f"Verse({self.surah}:{self.ayah} [{len(self.words)} words])"


# ── QuranDB ───────────────────────────────────────────────────────────────────

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quran_uthmani.json")
API_URL = "https://api.alquran.cloud/v1/quran/quran-uthmani"


class QuranDB:
    """
    In-memory database of all 6,236 Quran verses.
    
    Indexed by (surah, ayah) for O(1) lookup.
    Downloads and caches Uthmani text from alquran.cloud on first use.
    """
    
    def __init__(self, cache_path: str = CACHE_FILE):
        self._verses: Dict[Tuple[int, int], Verse] = {}
        self._surah_verses: Dict[int, List[Verse]] = {}
        self._cache_path = cache_path
        self._load()
    
    def _load(self):
        """Load from cache or download from API."""
        raw_data = None
        
        # Try cache first
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                log.info(f"Loaded Quran data from cache: {self._cache_path}")
            except Exception as e:
                log.warning(f"Cache load failed: {e}")
        
        # Download if no cache
        if raw_data is None:
            raw_data = self._download()
            # Save cache
            try:
                with open(self._cache_path, 'w', encoding='utf-8') as f:
                    json.dump(raw_data, f, ensure_ascii=False, indent=2)
                log.info(f"Cached Quran data to: {self._cache_path}")
            except Exception as e:
                log.warning(f"Cache save failed: {e}")
        
        # Parse into Verse objects
        self._parse(raw_data)
    
    def _download(self) -> dict:
        """Download full Quran from alquran.cloud API."""
        log.info(f"Downloading Quran data from {API_URL}...")
        req = urllib.request.Request(API_URL, headers={'User-Agent': 'WhisperQuran/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        if data.get('code') != 200:
            raise RuntimeError(f"API error: {data}")
        
        log.info("Quran data downloaded successfully")
        return data
    
    def _parse(self, data: dict):
        """Parse API response into Verse objects."""
        surahs = data.get('data', {}).get('surahs', [])
        total = 0
        
        for surah_data in surahs:
            surah_num = surah_data['number']
            surah_verses = []
            
            for ayah_data in surah_data['ayahs']:
                ayah_num = ayah_data['numberInSurah']
                text = ayah_data['text']
                # Strip BOM and zero-width characters that alquran.cloud includes
                text = text.replace('\uFEFF', '').replace('\u200B', '').strip()
                page = ayah_data.get('page', 0)
                
                verse = Verse(surah_num, ayah_num, text, page)
                self._verses[(surah_num, ayah_num)] = verse
                surah_verses.append(verse)
                total += 1
            
            self._surah_verses[surah_num] = surah_verses
        
        log.info(f"Loaded {total} verses across {len(self._surah_verses)} surahs")
    
    # ── Lookups ────────────────────────────────────────────────────────────
    
    def get_verse(self, surah: int, ayah: int) -> Optional[Verse]:
        """Get a single verse by surah and ayah number."""
        return self._verses.get((surah, ayah))
    
    def get_surah_verses(self, surah: int) -> List[Verse]:
        """Get all verses in a surah, ordered by ayah."""
        return self._surah_verses.get(surah, [])
    
    def get_surah_words(self, surah: int) -> List[Tuple[str, int, int]]:
        """
        Get all words in a surah as a flat list.
        Returns: [(word_text, ayah_number, word_index_in_ayah), ...]
        """
        result = []
        for verse in self.get_surah_verses(surah):
            for i, word in enumerate(verse.words):
                result.append((word, verse.ayah, i))
        return result
    
    def get_words(self, surah: int, ayah: int) -> List[str]:
        """Get diacritized word list for a specific verse."""
        verse = self.get_verse(surah, ayah)
        return verse.words if verse else []
    
    def get_words_norm(self, surah: int, ayah: int) -> List[str]:
        """Get normalized word list for a specific verse."""
        verse = self.get_verse(surah, ayah)
        return verse.words_norm if verse else []
    
    @property
    def total_verses(self) -> int:
        return len(self._verses)
    
    @property
    def total_surahs(self) -> int:
        return len(self._surah_verses)
    
    # ── Fuzzy search ───────────────────────────────────────────────────────
    
    def search(self, text: str, top_k: int = 5) -> List[Tuple[Verse, float]]:
        """
        Fuzzy search for a text snippet across all verses.
        Returns top_k matches with similarity scores [0, 1].
        """
        query_norm = normalize_arabic(text)
        if not query_norm:
            return []
        
        scored = []
        for verse in self._verses.values():
            # Quick length filter — skip verses much shorter than query
            if len(verse.text_norm) < len(query_norm) * 0.3:
                continue
            
            dist = levenshtein(query_norm, verse.text_norm)
            max_len = max(len(query_norm), len(verse.text_norm))
            similarity = 1.0 - (dist / max_len) if max_len > 0 else 0.0
            
            if similarity > 0.3:  # Don't bother with very low matches
                scored.append((verse, similarity))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    # ── Word-level matching ────────────────────────────────────────────────
    
    def match_words(
        self,
        spoken_words: List[str],
        surah: int,
        start_ayah: int,
        start_word: int,
        threshold: float = 0.70,
    ) -> List[dict]:
        """
        Match a sequence of spoken words against expected Quran text.
        
        Starting from (start_ayah, start_word), compares each spoken word
        against the expected word using normalized Levenshtein similarity.
        
        Returns a list of match results:
        [
            {
                "spoken": "الحمد",
                "expected": "ٱلْحَمْدُ",
                "expected_norm": "الحمد",
                "ayah": 1,
                "word_idx": 0,
                "similarity": 0.95,
                "matched": True,
            },
            ...
        ]
        """
        results = []
        verses = self.get_surah_verses(surah)
        if not verses:
            return results
        
        # Build flat word list starting from position
        expected_words = []
        for verse in verses:
            if verse.ayah < start_ayah:
                continue
            for i, (word, word_norm) in enumerate(zip(verse.words, verse.words_norm)):
                if verse.ayah == start_ayah and i < start_word:
                    continue
                expected_words.append({
                    'text': word,
                    'norm': word_norm,
                    'ayah': verse.ayah,
                    'word_idx': i,
                })
        
        if not expected_words:
            return results
        
        exp_idx = 0
        for spoken in spoken_words:
            if exp_idx >= len(expected_words):
                break
            
            exp = expected_words[exp_idx]
            spoken_norm = normalize_arabic(spoken)
            
            # Calculate similarity
            dist = levenshtein(spoken_norm, exp['norm'])
            max_len = max(len(spoken_norm), len(exp['norm']))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 0.0
            
            matched = sim >= threshold
            
            results.append({
                'spoken': spoken,
                'expected': exp['text'],
                'expected_norm': exp['norm'],
                'ayah': exp['ayah'],
                'word_idx': exp['word_idx'],
                'similarity': round(sim, 3),
                'matched': matched,
            })
            
            if matched:
                exp_idx += 1
            else:
                # Try skipping one expected word (ASR may merge words)
                if exp_idx + 1 < len(expected_words):
                    next_exp = expected_words[exp_idx + 1]
                    dist2 = levenshtein(spoken_norm, next_exp['norm'])
                    max_len2 = max(len(spoken_norm), len(next_exp['norm']))
                    sim2 = 1.0 - (dist2 / max_len2) if max_len2 > 0 else 0.0
                    
                    if sim2 >= threshold:
                        # Skipped word — mark current as missed, match next
                        results[-1]['skip_match'] = True
                        results[-1]['expected'] = next_exp['text']
                        results[-1]['expected_norm'] = next_exp['norm']
                        results[-1]['ayah'] = next_exp['ayah']
                        results[-1]['word_idx'] = next_exp['word_idx']
                        results[-1]['similarity'] = round(sim2, 3)
                        results[-1]['matched'] = True
                        exp_idx += 2
                        continue
                
                # No match — stay at same position
                exp_idx += 1  # Move forward anyway to avoid getting stuck
        
        return results


# ── Singleton ──────────────────────────────────────────────────────────────────

_db: Optional[QuranDB] = None

def get_quran_db() -> QuranDB:
    """Get or create the singleton QuranDB instance."""
    global _db
    if _db is None:
        _db = QuranDB()
    return _db


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = get_quran_db()
    
    print(f"\nTotal verses: {db.total_verses}")
    print(f"Total surahs: {db.total_surahs}")
    
    # Test lookups
    v = db.get_verse(1, 1)
    if v:
        print(f"\n1:1 = {v.text}")
        print(f"  words: {v.words}")
        print(f"  norm:  {v.words_norm}")
        print(f"  page:  {v.page}")
    
    v = db.get_verse(2, 255)
    if v:
        print(f"\n2:255 = {v.text[:80]}...")
    
    # Test fuzzy search
    print("\nSearching for 'الحمد لله رب العالمين'...")
    results = db.search("الحمد لله رب العالمين", top_k=3)
    for verse, score in results:
        print(f"  {verse.surah}:{verse.ayah} ({score:.2f}) {verse.text[:60]}")
    
    # Test word matching
    print("\nMatching spoken words against Al-Fatiha...")
    matches = db.match_words(
        spoken_words=["الحمد", "لله", "رب", "العالمين"],
        surah=1,
        start_ayah=2,
        start_word=0,
    )
    for m in matches:
        status = "✓" if m['matched'] else "✗"
        print(f"  {status} '{m['spoken']}' → '{m['expected']}' (sim={m['similarity']:.2f})")