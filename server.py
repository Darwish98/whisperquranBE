"""
Quran Recitation Backend — Python / Tarteel Whisper
====================================================
"""

import asyncio
import json
import logging
import os
import time
import base64
import re
from typing import Optional
from collections import deque

import numpy as np
import requests
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperForConditionalGeneration, WhisperProcessor

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PORT            = int(os.getenv("PORT", 8000))
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
if not SUPABASE_URL:
    logging.warning("SUPABASE_URL not set — JWT signature check skipped")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
DEVICE          = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SECONDS   = float(os.getenv("CHUNK_SECONDS", 2.0))
SAMPLE_RATE     = 16000
MODEL_ID        = "tarteel-ai/whisper-base-ar-quran"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Load model ────────────────────────────────────────────────────────────────

log.info(f"Loading {MODEL_ID} on {DEVICE}...")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision="refs/pr/3")
model.to(DEVICE)
model.eval()
log.info(f"Model loaded. Device: {DEVICE} | dtype: {next(model.parameters()).dtype}")

# ── JWKS / JWT ────────────────────────────────────────────────────────────────

_jwks_cache: dict = {}
_jwks_cache_time: float = 0.0

def get_jwks() -> list:
    global _jwks_cache, _jwks_cache_time
    now = time.time()
    if _jwks_cache and now - _jwks_cache_time < 3600:
        return _jwks_cache.get("keys", [])
    if not SUPABASE_URL:
        return []
    try:
        r = requests.get(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", timeout=5)
        r.raise_for_status()
        _jwks_cache = r.json()
        _jwks_cache_time = now
        log.info(f"JWKS loaded — {len(_jwks_cache.get('keys', []))} key(s)")
        return _jwks_cache.get("keys", [])
    except Exception as e:
        log.error(f"Failed to fetch JWKS: {e}")
        return []

def validate_token(token: str) -> tuple[bool, Optional[str]]:
    if not token:
        return False, None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False, None

        def b64decode(s):
            s += "=" * (-len(s) % 4)
            return json.loads(base64.urlsafe_b64decode(s).decode())

        header  = b64decode(parts[0])
        payload = b64decode(parts[1])

        if payload.get("exp", 0) < time.time():
            return False, None
        if "supabase" not in payload.get("iss", ""):
            return False, None

        alg = header.get("alg", "HS256")
        if alg == "ES256":
            from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicNumbers, SECP256R1
            from cryptography.hazmat.primitives.hashes import SHA256
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
            from cryptography.exceptions import InvalidSignature

            kid  = header.get("kid")
            keys = get_jwks()
            jwk  = next((k for k in keys if k.get("kid") == kid), keys[0] if keys else None)

            if not jwk:
                log.warning("No JWKS key — skipping signature check")
                return True, payload.get("sub")

            x = int.from_bytes(base64.urlsafe_b64decode(jwk["x"] + "=="), "big")
            y = int.from_bytes(base64.urlsafe_b64decode(jwk["y"] + "=="), "big")
            pub_key = EllipticCurvePublicNumbers(x, y, SECP256R1()).public_key(default_backend())

            raw_sig = base64.urlsafe_b64decode(parts[2] + "==")
            half    = len(raw_sig) // 2
            der_sig = encode_dss_signature(
                int.from_bytes(raw_sig[:half], "big"),
                int.from_bytes(raw_sig[half:], "big"),
            )
            try:
                pub_key.verify(der_sig, f"{parts[0]}.{parts[1]}".encode(), ECDSA(SHA256()))
                return True, payload.get("sub")
            except InvalidSignature:
                return False, None
        else:
            log.warning(f"Algorithm {alg} — skipping signature check")
            return True, payload.get("sub")

    except Exception as e:
        log.error(f"Token validation error: {e}")
        return False, None

# ── Silence detection ─────────────────────────────────────────────────────────

# RMS threshold — audio below this is considered silence
SILENCE_RMS   = 0.008
# Minimum fraction of 10ms frames that must have signal
MIN_ACTIVE    = 0.10

def is_silent(samples: np.ndarray) -> bool:
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < SILENCE_RMS:
        return True
    frame_len = SAMPLE_RATE // 100  # 10ms
    frames    = [samples[i:i+frame_len] for i in range(0, len(samples) - frame_len, frame_len)]
    if not frames:
        return True
    active = sum(1 for f in frames if np.sqrt(np.mean(f ** 2)) > SILENCE_RMS)
    return (active / len(frames)) < MIN_ACTIVE

# ── Hallucination filter ──────────────────────────────────────────────────────

# Strip diacritics for comparison
def strip_diacritics(text: str) -> str:
    return re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', text)

# Common Whisper hallucinations on Quran model when given silence/noise
HALLUCINATION_BARE = {
    strip_diacritics(t) for t in {
        "والمؤمنين", "وَالْمُؤْمِنِينَ",
        "سبحانك", "سُبْحَانَكَ",
        "آمين", "أمين",
        "صدق الله العظيم",
        "الحمد لله",
        "لا إله إلا الله",
        "الله أكبر",
        "بسم الله",
        "وبحمده",
        "سبحان الله وبحمده",
        "اللهم صل على محمد",
        "جزاكم الله خيرا",
        "شكرا",
    }
}

def is_hallucination(text: str) -> bool:
    bare = strip_diacritics(text.strip())
    # Exact match
    if bare in HALLUCINATION_BARE:
        return True
    # Very short output (1-2 chars) is noise
    if len(bare.replace(" ", "")) < 3:
        return True
    return False

# ── Repetition guard ──────────────────────────────────────────────────────────

def make_repetition_guard(max_repeats: int = 2, window: int = 6):
    """
    Returns a function that returns True if text has been seen too many
    times in the recent window, indicating hallucination loop.
    """
    recent: deque = deque(maxlen=window)

    def check(text: str) -> bool:
        bare = strip_diacritics(text.strip())
        count = sum(1 for t in recent if t == bare)
        recent.append(bare)
        return count >= max_repeats

    return check

# ── Transcription ─────────────────────────────────────────────────────────────

MIN_AUDIO_BYTES = int(SAMPLE_RATE * 2 * 0.5)  # at least 0.5s of audio

def transcribe(audio_pcm16: bytes) -> Optional[str]:
    if len(audio_pcm16) < MIN_AUDIO_BYTES:
        return None

    samples = np.frombuffer(audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0

    if is_silent(samples):
        log.debug("Silent chunk — skipped")
        return None

    # Use most recent 30s (Whisper max)
    max_samples = 30 * SAMPLE_RATE
    if len(samples) > max_samples:
        samples = samples[-max_samples:]

    inputs = processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_new_tokens=128)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()

    if not text:
        return None

    if is_hallucination(text):
        log.info(f"Hallucination rejected: {text}")
        return None

    return text

# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Quran Recitation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    client = ws.client.host if ws.client else "unknown"
    log.info(f"Client connected: {client}")

    authenticated    = False
    user_id          = None
    audio_buffer     = bytearray()
    last_tx_time     = time.time()
    last_text        = ""
    is_repeated      = make_repetition_guard(max_repeats=2, window=6)

    async def send(msg: dict):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    try:
        # Config handshake
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
        log.info(f"User authenticated: {user_id}")
        await send({"type": "ready"})

        # Audio loop
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                break

            if "text" in raw:
                continue  # control messages ignored

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

                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, transcribe, snapshot)

                if text is None:
                    # Silence or hallucination — reset repeat guard context
                    last_text = ""
                    continue

                # Skip if exact same output as last chunk (repetition loop)
                bare = strip_diacritics(text)
                if bare == strip_diacritics(last_text):
                    log.info(f"Skipping duplicate: {text}")
                    continue

                # Skip if seen too many times recently (hallucination loop)
                if is_repeated(text):
                    log.info(f"Repetition loop rejected: {text}")
                    last_text = ""
                    continue

                last_text = text
                await send({"type": "final", "text": text, "words": []})
                log.info(f"Transcribed [{user_id[:8] if user_id else '?'}]: {text[:80]}")

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

if __name__ == "__main__":
    log.info(f"Starting Quran backend on port {PORT}")
    log.info(f"Model:  {MODEL_ID}")
    log.info(f"Device: {DEVICE}")
    log.info(f"Chunk:  {CHUNK_SECONDS}s")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
