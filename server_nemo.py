"""
WhisperQuran Backend — FastConformer + Quran Matching + Tajweed
================================================================
Phase 2: ASR transcription matched against known Quran text.
Phase 3: Text-based tajweed rule annotations (no audio analysis).

Features:
  - RNNT decoder with tashkeel (diacritized Arabic output)
  - QuranDB: all 6,236 verses with Uthmani text
  - RecitationSession: word-level position tracking per connection
  - Tajweed annotations: 14 rule types identified from text alone
  - REST endpoint: GET /tajweed/surah/{n} for frontend preloading
  - WebSocket match messages include per-word tajweed rules

Protocol:
  1. Client sends JSON config: {"type":"config","locale":"ar-SA","surah":2,"token":"JWT"}
  2. Server validates JWT, loads session + tajweed, responds: {"type":"ready"}
  3. Client streams binary PCM16 @ 16kHz mono
  4. Server accumulates chunks, runs RNNT inference every CHUNK_SECONDS
  5. Server responds: {"type":"match","words":[{..., "tajweed":[...]}],...}

Requirements:
  pip install nemo_toolkit[asr] fastapi uvicorn websockets python-dotenv soundfile cuda-python>=12.3
"""

import asyncio
import json
import logging
import os
import re
import time
import tempfile
from collections import deque
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Phase 2 imports
from quran_db import get_quran_db, QuranDB
from ctc_matcher import RecitationSession
from tajweed_rules import annotate_surah
from word_timing import extract_word_timings, align_timings_to_transcript


# ── Load .env ────────────────────────────────────────────────────────────────

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_ID       = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
SAMPLE_RATE    = 16000
CHUNK_SECONDS  = float(os.getenv("CHUNK_SECONDS", "1.5"))
PORT           = int(os.getenv("PORT", "8000"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Auth
SUPABASE_URL       = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY  = os.getenv("SUPABASE_ANON_KEY", "")
JWT_SECRET         = os.getenv("SUPABASE_JWT_SECRET", "")
SKIP_AUTH          = os.getenv("SKIP_AUTH", "false").lower() == "true"

# Matching
MATCH_THRESHOLD    = float(os.getenv("MATCH_THRESHOLD", "0.65"))

# CORS
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8080"
).split(",")

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("whisperquran")

# ── Load NeMo FastConformer Model ────────────────────────────────────────────

log.info(f"Loading FastConformer model: {MODEL_ID}")
log.info(f"Device: {DEVICE}")

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
model.eval()
if DEVICE == "cuda":
    model = model.cuda()

# Decoder: RNNT (Transducer) — outputs diacritical marks (tashkeel).
# The PCD model was trained with Transducer as its primary decoder.
# RNNT is NOT Whisper's autoregressive seq2seq — it doesn't hallucinate.
# Strategy: greedy_batch runs on GPU with CUDA graph acceleration.
from nemo.collections.asr.models.rnnt_models import RNNTDecodingConfig
from omegaconf import OmegaConf

decoding_cfg = RNNTDecodingConfig()
decoding_cfg.strategy = "greedy_batch"
decoding_cfg.greedy.max_symbols = 10
decoding_cfg.greedy.loop_labels = True
decoding_cfg.greedy.use_cuda_graph_decoder = True
model.change_decoding_strategy(OmegaConf.structured(decoding_cfg), decoder_type="rnnt")
log.info("FastConformer model loaded — RNNT decoder (tashkeel, GPU greedy_batch + CUDA graphs)")

# ── Load QuranDB (Phase 2) ───────────────────────────────────────────────────

log.info("Loading Quran database...")
quran_db = get_quran_db()
log.info(f"QuranDB loaded: {quran_db.total_verses} verses, {quran_db.total_surahs} surahs")

# ── Warmup: run one dummy inference so RNNT JIT/Lhotse init happens now ──────

def _warmup_model():
    """Run a short silent audio through the model to trigger JIT compilation
    and Lhotse initialization. This avoids a 5-10s delay on the first real request."""
    try:
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
        tmp = tempfile.mktemp(suffix=".wav")
        sf.write(tmp, silence, SAMPLE_RATE)
        with torch.no_grad():
            model.transcribe([tmp])
        os.remove(tmp)
        log.info("Model warmup complete — ready for real-time inference")
    except Exception as e:
        log.warning(f"Model warmup failed (non-fatal): {e}")

_warmup_model()

# ── Auth Helper ──────────────────────────────────────────────────────────────

def validate_token(token: str) -> Tuple[bool, Optional[str]]:
    """Validate Supabase JWT token. Returns (valid, user_id)."""
    if SKIP_AUTH:
        return True, "dev-user"

    if not token:
        return False, None

    try:
        import jwt as pyjwt
        payload = pyjwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return True, payload.get("sub")
    except Exception:
        if not JWT_SECRET:
            return True, "unknown"
        return False, None

# ── Repetition Guard ─────────────────────────────────────────────────────────

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

# ── Surah detection from refText ─────────────────────────────────────────────

def detect_surah_from_config(msg: dict) -> Optional[int]:
    """Extract surah number from the config message."""
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

# ── Phase 3: Build tajweed cache for a surah ─────────────────────────────────

def build_tajweed_cache(surah_num: int) -> dict:
    """
    Build a dict mapping global_index → tajweed rules for a surah.
    Called once per session/surah change, cached for the connection lifetime.
    """
    ann_list = annotate_surah(surah_num, quran_db)
    cache = {}
    for a in ann_list:
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

# ── Transcription ────────────────────────────────────────────────────────────

MIN_AUDIO_BYTES = int(SAMPLE_RATE * 2 * 0.3)
MIN_RMS         = 0.005

def transcribe(audio_pcm16: bytes) -> Tuple[Optional[str], Optional[object]]:
    """
    Run FastConformer RNNT inference on PCM16 audio.
    Returns (text, hypothesis) where hypothesis carries token-level timestamps.
    Returns (None, None) if audio is too short, silent, or transcription fails.

    Phase 4: return_hypotheses=True gives us .timestep per token for word timing.
    """
    if len(audio_pcm16) < MIN_AUDIO_BYTES:
        return None, None

    samples = np.frombuffer(audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < MIN_RMS:
        return None, None

    audio_duration_ms = (len(samples) / SAMPLE_RATE) * 1000.0
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, samples, SAMPLE_RATE)

        with torch.no_grad():
            # return_hypotheses=True → get Hypothesis objects with .timestep
            hypotheses = model.transcribe([tmp], return_hypotheses=True)

        hyp = hypotheses[0]

        # Hypothesis object has .text; plain-string fallback for safety
        if isinstance(hyp, str):
            text = hyp.strip()
            hypothesis = None
        else:
            text = getattr(hyp, "text", str(hyp)).strip()
            # Attach audio duration so word_timing.py can clamp the last word
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
# ── FastAPI ──────────────────────────────────────────────────────────────────

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
        "features": ["transcription", "quran_matching", "tajweed_annotations"],
    }

# ── Phase 3: Tajweed REST endpoints ──────────────────────────────────────────

@app.get("/tajweed/surah/{surah_num}")
def get_tajweed_annotations(surah_num: int):
    """Return tajweed rule annotations for every word in a surah."""
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
from typing import List
import time as _time

class AnalyzeTajweedRequest(BaseModel):
    audio_base64: str = ""
    ayah_words: List[str] = []

@app.post("/analyze-tajweed")
def analyze_tajweed(req: AnalyzeTajweedRequest):
    """
    Analyze tajweed for a list of ayah words.
    
    Phase 3: Returns text-based rule identification for each word.
             Audio is accepted but not analyzed yet (Phase 5).
             All identified rules are returned as "confirmations" with
             confidence=1.0 since we can only identify rules, not verify them.
    
    Phase 5 (future): Will use audio_base64 for duration-based verification
             of Ghunna and Madd rules, returning violations when detected.
    """
    t0 = _time.time()
    words = req.ayah_words
    
    if not words:
        return {
            "rules_found": 0,
            "rules_checked": 0,
            "violations": [],
            "confirmations": [],
            "score": 1.0,
            "processing_time_ms": 0,
            "alignment_method": "text_based",
        }
    
    from tajweed_rules import get_word_tajweed_rules
    
    confirmations = []
    violations = []
    rules_found = 0
    
    for i, word in enumerate(words):
        next_word = words[i + 1] if i + 1 < len(words) else None
        is_last = (i == len(words) - 1)
        
        rules = get_word_tajweed_rules(word, next_word, is_last)
        rules_found += len(rules)
        
        for r in rules:
            # Phase 3: All rules are "confirmed" (we can't detect violations
            # without audio analysis). Confidence = 1.0 for rule identification.
            # Phase 5 will add actual violation detection for Ghunna/Madd.
            confirmations.append({
                "rule": r.rule_category,
                "sub_type": r.rule,
                "word": word,
                "word_index": i,
                "correct": True,
                "confidence": 1.0,
                "expected_duration": (r.harakat_count * 0.2) if r.harakat_count else None,
                "actual_duration": None,  # Phase 5: measured from audio
                "timestamp": None,
                "details": r.description,
            })
    
    elapsed = (_time.time() - t0) * 1000
    rules_checked = rules_found  # We "checked" all rules we found
    score = 1.0 if rules_found > 0 else 1.0  # Phase 3: assume correct, Phase 5: calculate
    
    return {
        "rules_found": rules_found,
        "rules_checked": rules_checked,
        "violations": violations,
        "confirmations": confirmations,
        "score": score,
        "processing_time_ms": round(elapsed, 1),
        "alignment_method": "text_based",
    }

# ── WebSocket handler ────────────────────────────────────────────────────────

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    client = ws.client.host if ws.client else "unknown"
    log.info(f"Client connected: {client}")

    authenticated    = False
    user_id          = None
    session          = None             # RecitationSession — Phase 2
    tajweed_cache    = {}               # Phase 3: global_index → tajweed rules
    audio_buffer     = bytearray()
    last_tx_time     = time.time()
    last_text        = ""
    is_repeated      = make_repetition_guard(max_repeats=3, window=6)

    async def send(msg: dict):
        try:
            await ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    try:
        # ── Config handshake ─────────────────────────────────────────────
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

        authenticated = True
        user_id       = uid

        # ── Phase 2 + 3: Create recitation session + tajweed cache ────────
        surah_num = detect_surah_from_config(msg)
        if surah_num:
            session = RecitationSession(surah=surah_num, db=quran_db)
            tajweed_cache = build_tajweed_cache(surah_num)
            log.info(f"[{user_id}] Session: surah {surah_num}, "
                     f"{session.total_words} words, {len(tajweed_cache)} tajweed annotations")
        else:
            log.info(f"[{user_id}] No surah detected — raw transcript mode")

        await send({"type": "ready"})

        # ── Audio loop ───────────────────────────────────────────────────
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                break

            if "text" in raw:
                # Control messages
                try:
                    ctrl = json.loads(raw["text"])
                except Exception:
                    continue
                
                ctrl_type = ctrl.get("type", "")
                
                # Handle surah change mid-session
                if ctrl_type == "updateRefText":
                    new_surah = ctrl.get("surah")
                    if isinstance(new_surah, int) and 1 <= new_surah <= 114:
                        session = RecitationSession(surah=new_surah, db=quran_db)
                        tajweed_cache = build_tajweed_cache(new_surah)
                        log.info(f"[{user_id}] Switched to surah {new_surah}")
                    elif ctrl.get("refText"):
                        detected = detect_surah_from_config(ctrl)
                        if detected:
                            session = RecitationSession(surah=detected, db=quran_db)
                            tajweed_cache = build_tajweed_cache(detected)
                            log.info(f"[{user_id}] Detected surah {detected} from refText update")
                
                # Handle position reset
                elif ctrl_type == "reset":
                    if session:
                        session.reset()
                        log.info(f"[{user_id}] Position reset")
                
                # Handle position jump
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

            now     = time.time()
            elapsed = now - last_tx_time

            if elapsed >= CHUNK_SECONDS and len(audio_buffer) >= MIN_AUDIO_BYTES:
                snapshot = bytes(audio_buffer)
                audio_buffer.clear()
                last_tx_time = now

                # Run RNNT inference in thread pool so event loop stays alive
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

                # ── Phase 4: Extract word timings from RNNT hypothesis ────
                word_timings = []
                if hypothesis is not None:
                    audio_dur_ms = getattr(hypothesis, "_audio_duration_ms", 3000.0)
                    word_timings = extract_word_timings(hypothesis, audio_dur_ms)

                # ── Phase 2 + 3: Match + attach tajweed ──────────────────
                if session:
                    result = session.match_transcript(
                        text,
                        threshold=MATCH_THRESHOLD,
                    )
                    wire = session.to_wire(result)
                    wire["transcript"] = text

                    # Phase 3: Attach tajweed rules to each matched word
                    # Phase 4: Attach word timings
                    transcript_words = text.strip().split()
                    aligned_timings = align_timings_to_transcript(
                        transcript_words, word_timings
                    )
                    # Build a quick lookup: spoken_word_index → timing
                    timing_by_spoken: dict = {}
                    for i, t in enumerate(aligned_timings):
                        if t is not None:
                            timing_by_spoken[i] = t

                    # We need to map each wire word to its position in the
                    # transcript. wire["words"] are in the order they were
                    # emitted, which matches transcript order.
                    for spoken_idx, w in enumerate(wire["words"]):
                        taj = tajweed_cache.get(w["index"])
                        if taj:
                            w["tajweed"] = taj

                        # Attach timing if available
                        timing = timing_by_spoken.get(spoken_idx)
                        if timing:
                            w["startMs"] = round(timing.start_ms)
                            w["endMs"] = round(timing.end_ms)
                            w["durationMs"] = round(timing.duration_ms)

                    await send(wire)
                    
                    matched_str = f"{result.words_matched}/{len(result.words)}"
                    log.info(f"[{user_id[:8] if user_id else '?'}] "
                             f"matched={matched_str} pos={result.new_position} "
                             f"'{text[:50]}'")
                else:
                    # Fallback: raw transcript (no session/surah detected)
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

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting Quran backend on port {PORT}")
    log.info(f"Model:   {MODEL_ID}")
    log.info(f"Device:  {DEVICE}")
    log.info(f"Decoder: RNNT (with tashkeel)")
    log.info(f"Chunk:   {CHUNK_SECONDS}s")
    log.info(f"QuranDB: {quran_db.total_verses} verses loaded")
    log.info(f"Tajweed: 14 rule types (text-based, Phase 3)")
    log.info(f"Match threshold: {MATCH_THRESHOLD}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")