"""
tajweed_duration.py — Phase 5: Duration-Based Tajweed Verification
===================================================================

Verifies Madd (elongation) and Ghunna (nasalization) rules by comparing
the measured word duration against the minimum expected duration derived
from the harakat_count in the text-based rule identification.

Design principles:
  - Only flag violations when VERY confident (duration far below threshold)
  - Never flag a violation if timing data is unavailable → falls back to
    Phase 3 "confirmation" (educational mode, no verdict)
  - Confidence score reflects how far below threshold the actual duration is
  - Rules not verifiable by duration (Qalqalah, Ikhfa, Idgham, Iqlab) are
    always returned as confirmations regardless of audio

Timing model (at ~80–100 BPM recitation speed, ~150ms per count):
  - 1 harakat count ≈ 150ms
  - Madd 2 counts → expected ≥ 300ms, violation threshold = 150ms (50%)
  - Madd 4-5 counts → expected ≥ 600ms, violation threshold = 300ms (50%)
  - Madd 6 counts → expected ≥ 900ms, violation threshold = 450ms (50%)
  - Ghunna 2 counts → expected ≥ 300ms, violation threshold = 150ms (50%)

The 50% threshold is deliberately conservative — a reciter going at double
speed is still getting SOME elongation in; we only flag when it's clearly
absent. This avoids false positives which would destroy user trust.

Usage:
    from tajweed_duration import verify_word_tajweed

    verdict = verify_word_tajweed(
        rule="madd_2",
        harakat_count=2,
        actual_duration_ms=120.0,   # from Phase 4 word timing
    )
    # → TajweedVerdict(correct=False, confidence=0.87, ...)
"""

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("whisperquran.tajweed_duration")

# ── Duration model ────────────────────────────────────────────────────────────

# ms per harakat count at moderate recitation speed (~120ms is more realistic
# than 150ms — measured against actual recitation samples)
MS_PER_COUNT = 120.0

# Violation threshold: flag if actual < this fraction of expected.
# 0.70 = must hold at least 70% of expected duration.
VIOLATION_THRESHOLD_RATIO = 0.70

# Minimum confidence to report a violation.
# 0.35 is strict enough to be meaningful without hair-trigger false positives.
MIN_VIOLATION_CONFIDENCE = 0.35

# Rules verifiable by duration alone.
# Ghunna REMOVED — nasalization requires frequency analysis, not just duration.
DURATION_VERIFIABLE = {
    "madd_2", "madd_246", "madd_muttasil", "madd_munfasil", "madd_6",
}

# Rules NOT verifiable by duration — always return as educational note only
DURATION_NOT_VERIFIABLE = {
    "ghunnah",       # needs nasal frequency analysis — out of scope for MVP
    "ikhfa", "ikhfa_shafawi",
    "idgham_ghunnah", "idgham_no_ghunnah", "idgham_shafawi",
    "iqlab",
    "qalqalah",      # needs burst-energy detection — high false-positive risk
    "lam_shamsiyyah",
}


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class TajweedVerdict:
    """Result of verifying one tajweed rule against measured audio duration."""
    rule: str                      # e.g. "madd_2"
    rule_category: str             # e.g. "madd"
    correct: bool                  # True = confirmed correct, False = violation
    confidence: float              # 0.0–1.0 how confident we are in the verdict
    expected_duration_ms: float    # minimum expected duration
    actual_duration_ms: Optional[float]  # measured duration (None if unavailable)
    verifiable: bool               # False = no audio data, falls back to confirmation
    details: str                   # human-readable explanation


# ── Core verification ─────────────────────────────────────────────────────────

def verify_word_tajweed(
    rule: str,
    harakat_count: Optional[int],
    actual_duration_ms: Optional[float],
) -> TajweedVerdict:
    """
    Verify a single tajweed rule against a measured word duration.

    Args:
        rule: The tajweed rule name (e.g. "madd_2", "ghunnah")
        harakat_count: Number of counts from text analysis (None if unknown)
        actual_duration_ms: Measured word duration in ms from Phase 4 timing.
                            None means timing was unavailable — falls back to
                            Phase 3 confirmation mode.

    Returns:
        TajweedVerdict with correct/violation + confidence score.
    """
    # Rules not verifiable by duration → always confirmation
    if rule in DURATION_NOT_VERIFIABLE:
        return TajweedVerdict(
            rule=rule,
            rule_category=_get_category(rule),
            correct=True,
            confidence=1.0,
            expected_duration_ms=0.0,
            actual_duration_ms=actual_duration_ms,
            verifiable=False,
            details="Rule cannot be verified by duration alone — educational note only.",
        )

    # No harakat count → can't compute expected duration
    if harakat_count is None or harakat_count == 0:
        return TajweedVerdict(
            rule=rule,
            rule_category=_get_category(rule),
            correct=True,
            confidence=1.0,
            expected_duration_ms=0.0,
            actual_duration_ms=actual_duration_ms,
            verifiable=False,
            details="Duration requirement unknown for this rule variant.",
        )

    expected_ms = harakat_count * MS_PER_COUNT
    threshold_ms = expected_ms * VIOLATION_THRESHOLD_RATIO

    # No timing data → fall back to Phase 3 confirmation
    if actual_duration_ms is None:
        return TajweedVerdict(
            rule=rule,
            rule_category=_get_category(rule),
            correct=True,
            confidence=1.0,
            expected_duration_ms=expected_ms,
            actual_duration_ms=None,
            verifiable=False,
            details=f"No timing data — expected ≥ {expected_ms:.0f}ms for {harakat_count} counts.",
        )

    # Compare actual vs threshold
    if actual_duration_ms >= threshold_ms:
        # ── State: VERIFIED GOOD ─────────────────────────────────────────────
        # Duration is sufficient. Confidence = how comfortably above threshold.
        margin = (actual_duration_ms - threshold_ms) / expected_ms
        confidence = min(1.0, 0.7 + margin * 0.3)
        return TajweedVerdict(
            rule=rule,
            rule_category=_get_category(rule),
            correct=True,
            confidence=round(confidence, 3),
            expected_duration_ms=expected_ms,
            actual_duration_ms=actual_duration_ms,
            verifiable=True,
            details=(
                f"Duration {actual_duration_ms:.0f}ms — "
                f"meets {harakat_count}-count requirement (≥{threshold_ms:.0f}ms)."
            ),
        )
    else:
        # Duration is below threshold
        shortfall_ratio = (threshold_ms - actual_duration_ms) / threshold_ms
        confidence = round(min(1.0, shortfall_ratio * 1.5), 3)

        if confidence < MIN_VIOLATION_CONFIDENCE:
            # ── State: BORDERLINE — not confident enough to flag ─────────────
            # Return verifiable=True, correct=True so UI shows neutral "~" badge
            # rather than GOOD (which implies verified) or IMPROVE (false alarm).
            return TajweedVerdict(
                rule=rule,
                rule_category=_get_category(rule),
                correct=True,
                confidence=round(confidence, 3),
                expected_duration_ms=expected_ms,
                actual_duration_ms=actual_duration_ms,
                verifiable=True,
                details=(
                    f"Duration {actual_duration_ms:.0f}ms — borderline "
                    f"(need ≥{threshold_ms:.0f}ms). Try holding a little longer."
                ),
            )

        # ── State: NEEDS PRACTICE ─────────────────────────────────────────────
        # Clearly too short. Flag it.
        return TajweedVerdict(
            rule=rule,
            rule_category=_get_category(rule),
            correct=False,
            confidence=confidence,
            expected_duration_ms=expected_ms,
            actual_duration_ms=actual_duration_ms,
            verifiable=True,
            details=(
                f"Duration {actual_duration_ms:.0f}ms is too short — "
                f"hold for at least {expected_ms:.0f}ms ({harakat_count} counts)."
            ),
        )


def _get_category(rule: str) -> str:
    """Map rule name to category."""
    if rule.startswith("madd"):
        return "madd"
    if rule == "ghunnah":
        return "ghunna"
    if rule.startswith("ikhfa"):
        return "ikhfa"
    if rule.startswith("idgham"):
        return "idgham"
    if rule == "iqlab":
        return "iqlab"
    if rule == "qalqalah":
        return "qalqalah"
    return "other"


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═══ tajweed_duration.py self-test ═══\n")

    cases = [
        # (rule, harakat_count, actual_ms, expected_outcome)
        ("madd_2",       2, 350.0,  "correct"),   # comfortably long enough
        ("madd_2",       2, 200.0,  "correct"),   # above 50% threshold (150ms)
        ("madd_2",       2,  80.0,  "violation"), # clearly too short
        ("madd_muttasil",5, 700.0,  "correct"),   # 5 counts, threshold=375ms
        ("madd_muttasil",5, 200.0,  "violation"), # too short for 5 counts
        ("madd_6",       6, 500.0,  "correct"),   # 6 counts, threshold=450ms
        ("madd_6",       6, 300.0,  "correct"),   # borderline — conservative, don't flag
        ("ghunnah",      2, 320.0,  "correct"),
        ("ghunnah",      2,  90.0,  "violation"),
        ("ikhfa",        2, 100.0,  "confirmation"), # not duration-verifiable
        ("qalqalah",  None,  80.0,  "confirmation"), # not duration-verifiable
        ("madd_2",       2,  None,  "no timing"),  # no data → confirmation
    ]

    all_pass = True
    for rule, hc, dur, expected in cases:
        v = verify_word_tajweed(rule, hc, dur)
        outcome = (
            "no timing" if not v.verifiable and dur is None
            else "confirmation" if not v.verifiable
            else "violation" if not v.correct and v.confidence >= 0.60
            else "correct"
        )
        status = "✓" if outcome == expected else "✗"
        if outcome != expected:
            all_pass = False
        print(
            f"  {status} {rule:<20} hc={str(hc):<4} dur={str(dur):<8} "
            f"→ {outcome:<14} conf={v.confidence:.2f}  {v.details[:60]}"
        )

    print(f"\n{'All tests passed ✓' if all_pass else 'SOME TESTS FAILED ✗'}")