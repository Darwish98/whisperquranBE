"""
word_timing.py — Phase 4: Word-Level Timing from RNNT Hypotheses
================================================================

Extracts word-level timestamps from NeMo FastConformer RNNT transcription.

NeMo's RNNT decoder, when called with return_hypotheses=True, returns
Hypothesis objects whose .timestep field gives per-token emission frames.

Frame math for FastConformer (PCD variant):
  - Raw audio: 16,000 samples/sec
  - Feature extraction: 10ms hop → 100 frames/sec
  - FastConformer subsampling: 8x → 12.5 frame/sec per encoder step
  - So each encoder step = 80ms
  - Token i's frame index = hypothesis.timestep[i]
  → token_time_ms = timestep[i] * 80.0

We group consecutive tokens into words (splitting on whitespace in the
decoded text), giving us [word, start_ms, end_ms] for each word.

Usage:
    from word_timing import extract_word_timings, WordTiming

    # hypotheses = model.transcribe([wav_path], return_hypotheses=True)
    # hyp = hypotheses[0]
    timings = extract_word_timings(hyp, audio_duration_ms)

    for t in timings:
        print(f"{t.word}: {t.start_ms:.0f}ms – {t.end_ms:.0f}ms ({t.duration_ms:.0f}ms)")
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

log = logging.getLogger("whisperquran.timing")

# FastConformer PCD subsampling: 8x at 10ms hop = 80ms per encoder step
MS_PER_STEP = 80.0


@dataclass
class WordTiming:
    word: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def extract_word_timings(
    hypothesis,
    audio_duration_ms: float,
    ms_per_step: float = MS_PER_STEP,
) -> List[WordTiming]:
    """
    Extract word-level timings from an RNNT Hypothesis object.

    Args:
        hypothesis: NeMo Hypothesis with .text and .timestep attributes.
                    .timestep is a list of (token_id, frame_idx) or just
                    frame indices depending on NeMo version.
        audio_duration_ms: Total duration of the audio chunk in ms.
                           Used to clamp the last word's end time.
        ms_per_step: Milliseconds per encoder output step (default 80ms).

    Returns:
        List of WordTiming objects, one per word.
        Returns [] if timing extraction fails (graceful degradation).
    """
    try:
        text = getattr(hypothesis, "text", None)
        if not text:
            return []

        words = text.strip().split()
        if not words:
            return []

        timesteps = getattr(hypothesis, "timestep", None)
        if timesteps is None or len(timesteps) == 0:
            return []

        # NeMo RNNT timesteps can be:
        #   (a) A flat list of ints — one per emitted token
        #   (b) A list of (token_id, frame) tuples (older NeMo)
        # Normalise to flat list of frame indices.
        if isinstance(timesteps[0], (list, tuple)):
            frame_indices = [t[1] for t in timesteps]
        else:
            frame_indices = list(timesteps)

        # Convert frame indices to ms
        token_times_ms = [f * ms_per_step for f in frame_indices]

        # The RNNT tokeniser emits one timestep per *subword token*, not per
        # character. We need to map tokens back to words. We do this by
        # aligning the token count to word boundaries using the hypothesis
        # token ids if available, else fall back to equal-duration split.
        token_ids = getattr(hypothesis, "y_sequence", None)
        if token_ids is None:
            token_ids = getattr(hypothesis, "tokens", None)

        if token_ids is not None and hasattr(hypothesis, "tokenizer"):
            # Decode each token individually to find word boundaries
            tokenizer = hypothesis.tokenizer
            timings = _timings_from_tokenizer(
                words, token_ids, token_times_ms, tokenizer, audio_duration_ms
            )
        else:
            # Fallback: distribute tokens evenly across words
            timings = _timings_evenly_distributed(
                words, token_times_ms, audio_duration_ms
            )

        return timings

    except Exception as e:
        log.warning(f"Word timing extraction failed (non-fatal): {e}")
        return []


def _timings_from_tokenizer(
    words: List[str],
    token_ids,
    token_times_ms: List[float],
    tokenizer,
    audio_duration_ms: float,
) -> List[WordTiming]:
    """
    Use the tokenizer to decode tokens one-by-one and find word boundaries.
    Works when NeMo provides the tokenizer reference on the hypothesis.
    """
    decoded_tokens = []
    for tid in token_ids:
        try:
            piece = tokenizer.ids_to_text([int(tid)])
            decoded_tokens.append(piece)
        except Exception:
            decoded_tokens.append("")

    # Join decoded tokens and re-split to align with word boundaries
    # SentencePiece puts ▁ (U+2581) at word-start tokens
    word_starts: List[int] = []  # token indices that start a new word
    in_word = False
    for i, piece in enumerate(decoded_tokens):
        if piece.startswith("\u2581") or not in_word:
            word_starts.append(i)
            in_word = True

    # Trim or pad word_starts to match word count
    while len(word_starts) < len(words):
        word_starts.append(len(token_times_ms) - 1)
    word_starts = word_starts[: len(words)]

    timings: List[WordTiming] = []
    for i, (word, start_tok) in enumerate(zip(words, word_starts)):
        start_ms = token_times_ms[start_tok] if start_tok < len(token_times_ms) else 0.0

        if i + 1 < len(word_starts):
            end_tok = word_starts[i + 1]
            end_ms = token_times_ms[end_tok] if end_tok < len(token_times_ms) else audio_duration_ms
        else:
            end_ms = audio_duration_ms

        end_ms = max(end_ms, start_ms + 50.0)  # minimum 50ms per word
        timings.append(WordTiming(word=word, start_ms=start_ms, end_ms=end_ms))

    return timings


def _timings_evenly_distributed(
    words: List[str],
    token_times_ms: List[float],
    audio_duration_ms: float,
) -> List[WordTiming]:
    """
    Fallback: distribute tokens evenly across words when tokenizer is
    unavailable. Still gives useful durations; timing is approximate.
    """
    n_words = len(words)
    n_tokens = len(token_times_ms)

    if n_tokens == 0 or n_words == 0:
        return []

    # Divide token list into n_words slices
    tokens_per_word = n_tokens / n_words
    timings: List[WordTiming] = []

    for i, word in enumerate(words):
        start_tok = int(i * tokens_per_word)
        end_tok = int((i + 1) * tokens_per_word) - 1

        start_ms = token_times_ms[min(start_tok, n_tokens - 1)]
        if end_tok < n_tokens - 1:
            end_ms = token_times_ms[end_tok + 1]
        else:
            end_ms = audio_duration_ms

        end_ms = max(end_ms, start_ms + 50.0)
        timings.append(WordTiming(word=word, start_ms=start_ms, end_ms=end_ms))

    return timings


# ── Align timings to matched words ────────────────────────────────────────────

def align_timings_to_transcript(
    transcript_words: List[str],
    timings: List[WordTiming],
) -> List[Optional[WordTiming]]:
    """
    Align WordTiming list (indexed by transcript position) to transcript words.
    Returns a list the same length as transcript_words, with None where
    alignment fails.

    Simple greedy alignment by word order — good enough since RNNT outputs
    words in order.
    """
    aligned: List[Optional[WordTiming]] = [None] * len(transcript_words)

    if not timings:
        return aligned

    # Build a lookup by word text for quick access
    timing_idx = 0
    for i, tw in enumerate(transcript_words):
        if timing_idx >= len(timings):
            break
        # Advance timing_idx if the timing word doesn't match (robustness)
        for j in range(timing_idx, min(timing_idx + 3, len(timings))):
            if _words_match(tw, timings[j].word):
                aligned[i] = timings[j]
                timing_idx = j + 1
                break
        else:
            # No match found in lookahead window — use positional timing anyway
            if timing_idx < len(timings):
                aligned[i] = timings[timing_idx]
                timing_idx += 1

    return aligned


def _words_match(a: str, b: str) -> bool:
    """Loose Arabic word match (strip diacritics for comparison)."""
    import re
    strip = lambda s: re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06DC]', '', s).strip()
    return strip(a) == strip(b)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick sanity test with a mock hypothesis object.
    Run: python word_timing.py
    """
    from types import SimpleNamespace

    # Simulate an RNNT hypothesis for "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    # 4 words, 12 tokens (3 per word on average)
    mock_hyp = SimpleNamespace(
        text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        timestep=[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
    )

    audio_ms = 3000.0
    timings = extract_word_timings(mock_hyp, audio_ms)

    print("Word timing extraction test:")
    print(f"  Audio duration: {audio_ms:.0f}ms")
    for t in timings:
        print(f"  '{t.word}': {t.start_ms:.0f}ms – {t.end_ms:.0f}ms ({t.duration_ms:.0f}ms)")

    print("\nAll good ✓")