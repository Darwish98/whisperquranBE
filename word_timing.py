"""
word_timing.py — Phase 4: Word-Level Timing from RNNT Hypotheses
================================================================

NeMo FastConformer with compute_timestamps=True populates:
    hypothesis.timestamp = {
        'timestep': [...],   # per-token frame indices
        'char': [...],       # per-char timestamps
        'word': [            # per-word timestamps  ← we use this
            {'word': 'بِسْمِ', 'start_offset': 2, 'end_offset': 8},
            ...
        ],
        'segment': [...]
    }

Frame math:
  FastConformer PCD: 8x subsampling × 10ms hop = 80ms per encoder step
  start_ms = start_offset * 80.0
  end_ms   = end_offset   * 80.0
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

log = logging.getLogger("whisperquran.timing")

MS_PER_STEP = 80.0  # FastConformer PCD: 8x subsampling at 10ms hop


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
    Extract word-level timings from a NeMo RNNT Hypothesis object.

    Uses hypothesis.timestamp['word'] which NeMo populates when
    compute_timestamps=True is set in the decoding config.

    Falls back to hypothesis.timestep (old attribute) if timestamp dict
    is not present, then to evenly-distributed fallback.

    Returns [] on any failure (graceful degradation).
    """
    try:
        # ── Primary path: hypothesis.timestamp['word'] ────────────────────
        timestamp = getattr(hypothesis, "timestamp", None)
        if timestamp and isinstance(timestamp, dict):
            word_ts = timestamp.get("word", [])
            if word_ts:
                timings = []
                for i, entry in enumerate(word_ts):
                    start_ms = entry.get("start_offset", 0) * ms_per_step
                    # Use next word's start as this word's end for realistic duration.
                    # NeMo's end_offset is the last token frame (too tight).
                    if i + 1 < len(word_ts):
                        end_ms = word_ts[i + 1].get("start_offset",
                                     entry.get("end_offset", 0)) * ms_per_step
                    else:
                        # Last word: use NeMo's end_offset if available,
                        # otherwise cap at start + 1200ms to avoid chunk
                        # boundary artifact inflating the duration.
                        nemo_end = entry.get("end_offset", 0) * ms_per_step
                        if nemo_end > start_ms + 50.0:
                            end_ms = nemo_end
                        else:
                            # Fallback cap: reasonable max for one Arabic word
                            end_ms = min(audio_duration_ms, start_ms + 1200.0)
                    end_ms = max(end_ms, start_ms + 50.0)
                    timings.append(WordTiming(
                        word=entry.get("word", ""),
                        start_ms=start_ms,
                        end_ms=end_ms,
                    ))
                log.debug(f"Word timings from timestamp['word']: {len(timings)} words")
                return timings

            # timestamp dict exists but word list empty — try timestep list
            timestep_list = timestamp.get("timestep", [])
            if timestep_list:
                return _timings_from_timestep_list(
                    hypothesis, timestep_list, audio_duration_ms, ms_per_step
                )

        # ── Fallback: legacy hypothesis.timestep attribute ────────────────
        timesteps = getattr(hypothesis, "timestep", None)
        if timesteps is not None and len(timesteps) > 0:
            return _timings_from_timestep_list(
                hypothesis, timesteps, audio_duration_ms, ms_per_step
            )

        log.debug("No timing data on hypothesis — returning []")
        return []

    except Exception as e:
        log.warning(f"Word timing extraction failed (non-fatal): {e}")
        return []


def _timings_from_timestep_list(
    hypothesis,
    timesteps,
    audio_duration_ms: float,
    ms_per_step: float,
) -> List[WordTiming]:
    """
    Build word timings from a flat list of per-token frame indices.
    Used as fallback when word-level timestamps aren't available.
    """
    try:
        text = getattr(hypothesis, "text", None)
        if not text:
            return []
        words = text.strip().split()
        if not words:
            return []

        # Normalise timesteps to flat int list
        if hasattr(timesteps[0], "__iter__"):
            frame_indices = [t[1] for t in timesteps]
        else:
            frame_indices = [int(t) for t in timesteps]

        token_times_ms = [f * ms_per_step for f in frame_indices]

        # Try tokenizer-based word boundary detection
        token_ids = getattr(hypothesis, "y_sequence", None)
        if token_ids is None:
            token_ids = getattr(hypothesis, "tokens", None)

        if token_ids is not None and hasattr(hypothesis, "tokenizer"):
            return _timings_from_tokenizer(
                words, token_ids, token_times_ms,
                hypothesis.tokenizer, audio_duration_ms
            )

        # Last resort: distribute tokens evenly
        return _timings_evenly_distributed(words, token_times_ms, audio_duration_ms)

    except Exception as e:
        log.warning(f"Timestep fallback failed: {e}")
        return []


def _timings_from_tokenizer(
    words, token_ids, token_times_ms, tokenizer, audio_duration_ms
):
    decoded = []
    for tid in token_ids:
        try:
            decoded.append(tokenizer.ids_to_text([int(tid)]))
        except Exception:
            decoded.append("")

    word_starts = []
    in_word = False
    for i, piece in enumerate(decoded):
        if piece.startswith("\u2581") or not in_word:
            word_starts.append(i)
            in_word = True

    while len(word_starts) < len(words):
        word_starts.append(len(token_times_ms) - 1)
    word_starts = word_starts[:len(words)]

    timings = []
    for i, (word, start_tok) in enumerate(zip(words, word_starts)):
        start_ms = token_times_ms[start_tok] if start_tok < len(token_times_ms) else 0.0
        if i + 1 < len(word_starts):
            end_tok = word_starts[i + 1]
            end_ms = token_times_ms[end_tok] if end_tok < len(token_times_ms) else audio_duration_ms
        else:
            end_ms = audio_duration_ms
        end_ms = max(end_ms, start_ms + 50.0)
        timings.append(WordTiming(word=word, start_ms=start_ms, end_ms=end_ms))
    return timings


def _timings_evenly_distributed(words, token_times_ms, audio_duration_ms):
    n_words  = len(words)
    n_tokens = len(token_times_ms)
    if not n_tokens or not n_words:
        return []
    tpw = n_tokens / n_words
    timings = []
    for i, word in enumerate(words):
        st = int(i * tpw)
        et = int((i + 1) * tpw) - 1
        start_ms = token_times_ms[min(st, n_tokens - 1)]
        end_ms   = token_times_ms[et + 1] if et + 1 < n_tokens else audio_duration_ms
        end_ms   = max(end_ms, start_ms + 50.0)
        timings.append(WordTiming(word=word, start_ms=start_ms, end_ms=end_ms))
    return timings


def align_timings_to_transcript(
    transcript_words: List[str],
    timings: List[WordTiming],
) -> List[Optional[WordTiming]]:
    """
    Align WordTiming list to transcript words by greedy word-order matching.
    Returns list same length as transcript_words, None where alignment fails.
    """
    aligned: List[Optional[WordTiming]] = [None] * len(transcript_words)
    if not timings:
        return aligned

    timing_idx = 0
    for i, tw in enumerate(transcript_words):
        if timing_idx >= len(timings):
            break
        for j in range(timing_idx, min(timing_idx + 3, len(timings))):
            if _words_match(tw, timings[j].word):
                aligned[i] = timings[j]
                timing_idx = j + 1
                break
        else:
            if timing_idx < len(timings):
                aligned[i] = timings[timing_idx]
                timing_idx += 1
    return aligned


def _words_match(a: str, b: str) -> bool:
    import re
    strip = lambda s: re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06DC]', '', s).strip()
    return strip(a) == strip(b)