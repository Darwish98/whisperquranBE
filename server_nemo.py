"""
WhisperQuran Backend — FastConformer + Quran Matching + Tajweed
================================================================
Phase 2: ASR transcription matched against known Quran text.
Phase 3: Text-based tajweed rule annotations (no audio analysis).
Phase 4: Word-level timing from RNNT hypotheses.
Phase 5: Duration-based tajweed verification.

Auth: Supabase uses ES256 for user tokens — verified via /auth/v1/user API.
      JWT_SECRET is NOT used for user tokens (only anon/service keys use HS256).

Requirements:
  pip install nemo_toolkit[asr] fastapi uvicorn websockets python-dotenv soundfile cuda-python>=12.3 PyJWT cryptography
"""

import asyncio
import json
import logging
import os
import re
import time
import tempfile
import urllib.request
import urllib.error
from collections import deque
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from quran_db import get_quran_db
from ctc_matcher import RecitationSession
from tajweed_rules import annotate_surah
from word_timing import extract_word_timings, align_timings_to_transcript
from tajweed_duration import verify_word_tajweed

# ── Load .env ─────────────────────────────────────────────────────────────────

load_dotenv(override=True)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID      = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
SAMPLE_RATE   = 16000
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "1.5"))
PORT          = int(os.getenv("PORT", "8000"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

SUPABASE_URL      = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = (
    os.getenv("SUPABASE_ANON_KEY") or
    os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY") or ""
).strip().strip("\"'")
SKIP_AUTH       = os.getenv("SKIP_AUTH", "false").lower() == "true"
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.65"))
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8080"
).split(",")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("whisperquran")

# Startup diagnostics
log.info(f"SUPABASE_URL set: {bool(SUPABASE_URL)}")
log.info(f"SUPABASE_ANON_KEY set: {bool(SUPABASE_ANON_KEY)}")
log.info(f"SKIP_AUTH: {SKIP_AUTH}")

# ── Load NeMo FastConformer ───────────────────────────────────────────────────

log.info(f"Loading FastConformer model: {MODEL_ID}")
log.info(f"Device: {DEVICE}")

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
model.eval()
if DEVICE == "cuda":
    model = model.cuda()

from nemo.collections.asr.models.rnnt_models import RNNTDecodingConfig
from omegaconf import OmegaConf

decoding_cfg = RNNTDecodingConfig()
decoding_cfg.strategy = "greedy_batch"
decoding_cfg.greedy.max_symbols = 10
decoding_cfg.greedy.loop_labels = True
decoding_cfg.greedy.use_cuda_graph_decoder = True
model.change_decoding_strategy(OmegaConf.structured(decoding_cfg), decoder_type="rnnt")
log.info("FastConformer loaded — RNNT decoder (tashkeel, GPU greedy_batch + CUDA graphs)")

# ── Load QuranDB ──────────────────────────────────────────────────────────────

log.info("Loading Quran database...")
quran_db = get_quran_db()
log.info(f"QuranDB: {quran_db.total_verses} verses, {quran_db.total_surahs} surahs")

# ── Warmup ────────────────────────────────────────────────────────────────────

def _warmup_model():
    try:
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = tf.name
        sf.write(tmp, silence, SAMPLE_RATE)
        with torch.no_grad():
            model.transcribe([tmp])
        os.remove(tmp)
        log.info("Model warmup complete — ready for real-time inference")
    except Exception as e:
        log.warning(f"Model warmup failed (non-fatal): {e}")

_warmup_model()

# ── Auth ──────────────────────────────────────────────────────────────────────
#
# Supabase user session tokens use ES256 (elliptic curve asymmetric signing).
# The SUPABASE_JWT_SECRET only signs anon/service keys (HS256) — it cannot
# verify user tokens. The correct approach is the /auth/v1/user API endpoint.
#
# This adds ~50ms latency on the WebSocket handshake only (not per audio chunk).

def validate_token(token: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Supabase JWT. Returns (valid, user_id).
    Uses Supabase /auth/v1/user API — works for all token types (ES256, RS256, HS256).
    """
    if SKIP_AUTH:
        return True, "dev-user"

    if not token:
        return False, None

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        log.error("SUPABASE_URL or SUPABASE_ANON_KEY not set — cannot validate token")
        return False, None

    try:
        req = urllib.request.Request(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY,
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                uid = data.get("id")
                if uid:
                    log.info(f"Auth OK: user={uid[:8]}...")
                    return True, uid
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        log.warning(f"Auth failed: HTTP {e.code} {body[:120]}")
    except Exception as e:
        log.warning(f"Auth error: {type(e).__name__}: {e}")

    return False, None

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_repetition_guard(max_repeats: int = 3, window: int = 6):
    recent: deque = deque(maxlen=window)

    def strip_diacritics(text: str) -> str:
        return re.sub(r'[\u064B-\u065F\u0670]', '', text).strip()

    def check(text: str) -> bool:
        bare = strip_diacritics(text)
        if not bare:
            return True
        count = sum(1 for t in recent if t == bare)
        recent.append(bare)
        return count >= max_repeats

    return check


def detect_surah_from_config(msg: dict) -> Optional[int]:
    surah = msg.get("surah")
    if isinstance(surah, int) and 1 <= surah <= 114:
        return surah
    ref_text = msg.get("refText", "")
    if ref_text:
        results = quran_db.search(ref_text[:100], top_k=1)
        if results:
            verse, score = results[0]
            if score > 0.5:
                log.info(f"Detected surah {verse.surah} from refText (score={score:.2f})")
                return verse.surah
    return None


def build_tajweed_cache(surah_num: int) -> dict:
    cache = {}
    for a in annotate_surah(surah_num, quran_db):
        if a.has_rules:
            cache[a.global_index] = [
                {
                    "rule": r.rule,
                    "category": r.rule_category,
                    "description": r.description,
                    "arabicName": r.arabic_name,
                    "harakatCount": r.harakat_count,
                }
                for r in a.rules
            ]
    return cache

# ── Transcription ─────────────────────────────────────────────────────────────

MIN_AUDIO_BYTES = int(SAMPLE_RATE * 2 * 0.3)
MIN_RMS         = 0.005


def transcribe(audio_pcm16: bytes) -> Tuple[Optional[str], Optional[object]]:
    if len(audio_pcm16) < MIN_AUDIO_BYTES:
        return None, None

    samples = np.frombuffer(audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < MIN_RMS:
        return None, None

    audio_duration_ms = (len(samples) / SAMPLE_RATE) * 1000.0

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = tf.name
    try:
        sf.write(tmp, samples, SAMPLE_RATE)
        with torch.no_grad():
            hypotheses = model.transcribe([tmp], return_hypotheses=True)

        hyp = hypotheses[0]
        if isinstance(hyp, str):
            text = hyp.strip()
            hypothesis = None
        else:
            text = getattr(hyp, "text", str(hyp)).strip()
            hyp._audio_duration_ms = audio_duration_ms
            hypothesis = hyp

        if not text or len(text) <= 2:
            return None, None

        return text, hypothesis

    except Exception as e:
        log.error(f"Transcription error: {e}")
        return None, None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="WhisperQuran Backend (FastConformer + QuranDB + Tajweed)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "decoder": "rnnt",
        "chunk_seconds": CHUNK_SECONDS,
        "quran_db": {
            "verses": quran_db.total_verses,
            "surahs": quran_db.total_surahs,
        },
        "features": ["transcription", "quran_matching", "tajweed_annotations", "word_timing"],
        "auth": {
            "skip": SKIP_AUTH,
            "supabase_url": bool(SUPABASE_URL),
            "anon_key": bool(SUPABASE_ANON_KEY),
        },
    }

# ── Tajweed REST endpoints ────────────────────────────────────────────────────

@app.get("/tajweed/surah/{surah_num}")
def get_tajweed_annotations(surah_num: int):
    if not 1 <= surah_num <= 114:
        return {"error": "Invalid surah number"}
    annotations = annotate_surah(surah_num, quran_db)
    return {
        "surah": surah_num,
        "totalWords": len(annotations),
        "wordsWithRules": sum(1 for a in annotations if a.has_rules),
        "words": [a.to_dict() for a in annotations if a.has_rules],
    }


from pydantic import BaseModel
from typing import List, Optional as _Optional
import time as _time


class WordTimingInput(BaseModel):
    word_index: int
    duration_ms: _Optional[float] = None


class AnalyzeTajweedRequest(BaseModel):
    audio_base64: str = ""
    ayah_words: List[str] = []
    word_timings: List[WordTimingInput] = []


@app.post("/analyze-tajweed")
def analyze_tajweed(req: AnalyzeTajweedRequest):
    t0 = _time.time()
    words = req.ayah_words

    if not words:
        return {
            "rules_found": 0, "rules_checked": 0,
            "violations": [], "confirmations": [],
            "score": 1.0, "processing_time_ms": 0,
            "alignment_method": "text_based",
        }

    from tajweed_rules import get_word_tajweed_rules

    timing_map = {
        wt.word_index: wt.duration_ms
        for wt in req.word_timings
        if wt.duration_ms is not None
    }
    log.info(f"analyze_tajweed: words={len(words)}, timings={list(timing_map.items())[:5]}")

    confirmations = []
    violations    = []
    rules_found   = 0
    rules_verified = 0

    for i, word in enumerate(words):
        next_word = words[i + 1] if i + 1 < len(words) else None
        is_last   = (i == len(words) - 1)
        rules     = get_word_tajweed_rules(word, next_word, is_last)
        rules_found += len(rules)
        actual_ms = timing_map.get(i)

        for r in rules:
            verdict = verify_word_tajweed(
                rule=r.rule,
                harakat_count=r.harakat_count,
                actual_duration_ms=actual_ms,
            )
            entry = {
                "rule":              r.rule_category,
                "sub_type":          r.rule,
                "word":              word,
                "word_index":        i,
                "correct":           verdict.correct,
                "confidence":        verdict.confidence,
                "verifiable":        verdict.verifiable,
                "expected_duration": verdict.expected_duration_ms / 1000.0 if verdict.expected_duration_ms else None,
                "actual_duration":   actual_ms / 1000.0 if actual_ms else None,
                "timestamp":         None,
                "details":           verdict.details,
            }
            if verdict.verifiable:
                rules_verified += 1
            if not verdict.correct:
                violations.append(entry)
            else:
                confirmations.append(entry)

    if rules_verified > 0:
        score = round(max(0.0, 1.0 - len(violations) / max(rules_verified, 1)), 3)
    else:
        score = 1.0

    return {
        "rules_found":       rules_found,
        "rules_checked":     rules_verified,
        "violations":        violations,
        "confirmations":     confirmations,
        "score":             score,
        "processing_time_ms": round((_time.time() - t0) * 1000, 1),
        "alignment_method":  "duration_based" if timing_map else "text_based",
    }

# ── WebSocket handler ─────────────────────────────────────────────────────────

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    client = ws.client.host if ws.client else "unknown"
    log.info(f"Client connected: {client}")

    user_id       = None
    session       = None
    tajweed_cache = {}
    audio_buffer  = bytearray()
    last_tx_time  = time.time()
    last_text     = ""
    is_repeated   = make_repetition_guard(max_repeats=3, window=6)

    async def send(msg: dict):
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    try:
        # ── Config handshake ──────────────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(ws.receive(), timeout=10.0)
        except asyncio.TimeoutError:
            await send({"type": "error", "message": "Config timeout", "code": "TIMEOUT"})
            await ws.close(4001)
            return

        try:
            msg = json.loads(raw.get("text") or raw.get("bytes", b"{}").decode())
        except Exception:
            await ws.close(4002)
            return

        if msg.get("type") != "config":
            await ws.close(4002)
            return

        valid, uid = validate_token(msg.get("token", ""))
        if not valid:
            await send({"type": "error", "message": "Authentication required", "code": "AUTH_REQUIRED"})
            await ws.close(4003)
            return

        user_id = uid

        surah_num = detect_surah_from_config(msg)
        if surah_num:
            session       = RecitationSession(surah=surah_num, db=quran_db)
            tajweed_cache = build_tajweed_cache(surah_num)
            log.info(f"[{user_id}] Session: surah={surah_num} words={session.total_words} tajweed={len(tajweed_cache)}")
        else:
            log.info(f"[{user_id}] No surah detected — raw transcript mode")

        await send({"type": "ready"})

        # ── Audio loop ────────────────────────────────────────────────────────
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                break

            if "text" in raw:
                try:
                    ctrl = json.loads(raw["text"])
                except Exception:
                    continue

                ctrl_type = ctrl.get("type", "")

                if ctrl_type == "updateRefText":
                    new_surah = ctrl.get("surah")
                    if isinstance(new_surah, int) and 1 <= new_surah <= 114:
                        session       = RecitationSession(surah=new_surah, db=quran_db)
                        tajweed_cache = build_tajweed_cache(new_surah)
                        log.info(f"[{user_id}] Switched to surah {new_surah}")
                    elif ctrl.get("refText"):
                        detected = detect_surah_from_config(ctrl)
                        if detected:
                            session       = RecitationSession(surah=detected, db=quran_db)
                            tajweed_cache = build_tajweed_cache(detected)
                            log.info(f"[{user_id}] Detected surah {detected} from refText")

                elif ctrl_type == "reset":
                    if session:
                        session.reset()
                        log.info(f"[{user_id}] Position reset")

                elif ctrl_type == "setPosition":
                    pos = ctrl.get("position", 0)
                    if session and isinstance(pos, int):
                        session.set_position(pos)
                        log.info(f"[{user_id}] Position set to {pos}")

                continue

            chunk = raw.get("bytes")
            if not chunk:
                continue

            audio_buffer.extend(chunk)

            now = time.time()
            if now - last_tx_time < CHUNK_SECONDS or len(audio_buffer) < MIN_AUDIO_BYTES:
                continue

            snapshot = bytes(audio_buffer)
            audio_buffer.clear()
            last_tx_time = now

            loop = asyncio.get_event_loop()
            text, hypothesis = await loop.run_in_executor(None, transcribe, snapshot)

            if text is None:
                last_text = ""
                continue

            if is_repeated(text):
                log.info(f"Repetition rejected: {text[:40]}")
                last_text = ""
                continue

            if text == last_text:
                continue

            last_text = text

            # ── Phase 4: Word timings ─────────────────────────────────────────
            word_timings = []
            if hypothesis is not None:
                audio_dur_ms = getattr(hypothesis, "_audio_duration_ms", 3000.0)
                word_timings = extract_word_timings(hypothesis, audio_dur_ms)

            # ── Phase 2+3: Match + tajweed + timings ──────────────────────────
            if session:
                result = session.match_transcript(text, threshold=MATCH_THRESHOLD)
                wire   = session.to_wire(result)
                wire["transcript"] = text

                transcript_words  = text.strip().split()
                aligned_timings   = align_timings_to_transcript(transcript_words, word_timings)
                timing_by_spoken  = {i: t for i, t in enumerate(aligned_timings) if t is not None}

                spoken_idx = 0
                for w in wire["words"]:
                    taj = tajweed_cache.get(w["index"])
                    if taj:
                        w["tajweed"] = taj

                    if w.get("spoken"):
                        timing = timing_by_spoken.get(spoken_idx)
                        spoken_idx += 1
                    else:
                        timing = None

                    if timing is not None:
                        w["startMs"]    = round(timing.start_ms)
                        w["endMs"]      = round(timing.end_ms)
                        w["durationMs"] = round(timing.duration_ms)

                await send(wire)
                log.info(
                    f"[{user_id[:8] if user_id else '?'}] "
                    f"matched={result.words_matched}/{len(result.words)} "
                    f"pos={result.new_position} '{text[:50]}'"
                )
            else:
                await send({"type": "final", "text": text, "words": []})
                log.info(f"[{user_id[:8] if user_id else '?'}] {text[:80]}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error [{client}]: {e}")
        try:
            await send({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        log.info(f"Client disconnected: {client} (user={user_id})")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting WhisperQuran backend on port {PORT}")
    log.info(f"Model:   {MODEL_ID}")
    log.info(f"Device:  {DEVICE}")
    log.info(f"Decoder: RNNT (with tashkeel)")
    log.info(f"Chunk:   {CHUNK_SECONDS}s")
    log.info(f"QuranDB: {quran_db.total_verses} verses")
    log.info(f"Auth:    SKIP={SKIP_AUTH} | Supabase API via /auth/v1/user")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")