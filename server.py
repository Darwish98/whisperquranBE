"""
Quran Recitation Backend — Python / Tarteel Whisper
====================================================
Replaces the Node.js Azure backend with a local GPU-accelerated server
using tarteel-ai/whisper-base-ar-quran — a Whisper model fine-tuned
specifically on Quran recitation audio.

Architecture:
  Browser mic → PCM16 @ 16kHz → WebSocket → buffer audio →
  Whisper inference (GPU) → interim/final JSON → frontend word matching

Requirements:
  pip install fastapi uvicorn websockets torch transformers
              numpy soundfile python-dotenv requests

Run:
  python server.py

.env:
  SUPABASE_URL=https://hkiskbdykjaxvxjoxoqy.supabase.co
  ALLOWED_ORIGINS=http://localhost:5173
  PORT=8000
  DEVICE=cuda          # or cpu
  CHUNK_SECONDS=2.5    # how often to transcribe (lower = more responsive)
"""

import asyncio
import json
import logging
import os
import time
import base64
from typing import Optional

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
    logging.warning("SUPABASE_URL not set in .env — JWT signature check will be skipped")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
DEVICE          = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SECONDS   = float(os.getenv("CHUNK_SECONDS", 1.5))
SAMPLE_RATE     = 16000
MODEL_ID        = "tarteel-ai/whisper-base-ar-quran"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Load model at startup ─────────────────────────────────────────────────────

log.info(f"Loading {MODEL_ID} on {DEVICE}...")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision="refs/pr/3")
model.to(DEVICE)
model.eval()

log.info(f"Model loaded. Device: {DEVICE} | dtype: {next(model.parameters()).dtype}")

# ── JWKS cache for ES256 JWT verification ─────────────────────────────────────

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
    """Returns (valid, user_id)"""
    if not token:
        return False, None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False, None

        # Decode header and payload
        def b64decode(s):
            s += "=" * (-len(s) % 4)
            return json.loads(base64.urlsafe_b64decode(s).decode())

        header  = b64decode(parts[0])
        payload = b64decode(parts[1])

        # Check expiry
        if payload.get("exp", 0) < time.time():
            log.warning("Token expired")
            return False, None

        # Check issuer
        iss = payload.get("iss", "")
        if "supabase" not in iss:
            log.warning(f"Invalid issuer: {iss}")
            return False, None

        alg = header.get("alg", "HS256")

        if alg == "ES256":
            # Verify using JWKS public key
            from cryptography.hazmat.primitives.asymmetric.ec import (
                ECDSA, EllipticCurvePublicKey
            )
            from cryptography.hazmat.primitives.hashes import SHA256
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.asymmetric.utils import (
                decode_dss_signature, encode_dss_signature
            )
            from cryptography.exceptions import InvalidSignature
            import struct

            kid  = header.get("kid")
            keys = get_jwks()
            jwk  = next((k for k in keys if k.get("kid") == kid), keys[0] if keys else None)

            if not jwk:
                log.warning("No JWKS key available — skipping signature check")
                return True, payload.get("sub")

            # Import JWK as public key using cryptography library
            from cryptography.hazmat.primitives.asymmetric.ec import (
                EllipticCurvePublicNumbers, SECP256R1
            )
            x = int.from_bytes(base64.urlsafe_b64decode(jwk["x"] + "=="), "big")
            y = int.from_bytes(base64.urlsafe_b64decode(jwk["y"] + "=="), "big")
            pub_numbers = EllipticCurvePublicNumbers(x, y, SECP256R1())
            pub_key = pub_numbers.public_key(default_backend())

            # The JWT signature is IEEE P1363 format (raw r||s, 64 bytes for P-256)
            # cryptography library needs DER format
            raw_sig = base64.urlsafe_b64decode(parts[2] + "==")
            half    = len(raw_sig) // 2
            r       = int.from_bytes(raw_sig[:half], "big")
            s       = int.from_bytes(raw_sig[half:], "big")
            der_sig = encode_dss_signature(r, s)

            signing_input = f"{parts[0]}.{parts[1]}".encode()
            try:
                pub_key.verify(der_sig, signing_input, ECDSA(SHA256()))
                return True, payload.get("sub")
            except InvalidSignature:
                log.warning("ES256 signature invalid")
                return False, None

        else:
            # HS256 or no secret — trust expiry+issuer
            log.warning(f"Algorithm {alg} — skipping signature check")
            return True, payload.get("sub")

    except Exception as e:
        log.error(f"Token validation error: {e}")
        return False, None

# ── Whisper transcription ─────────────────────────────────────────────────────

def transcribe(audio_pcm16: bytes) -> Optional[str]:
    """Transcribe raw PCM16 mono 16kHz audio bytes using Tarteel Whisper."""
    if len(audio_pcm16) < 3200:  # < 0.1s — skip
        return None

    # Convert PCM16 → float32 numpy array
    samples = np.frombuffer(audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0

    # Pad or truncate to Whisper's 30s window
    max_samples = 30 * SAMPLE_RATE
    if len(samples) > max_samples:
        samples = samples[-max_samples:]  # keep most recent

    # Whisper feature extraction
    inputs = processor(
        samples,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=128,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    # Also strip any remaining special token artifacts the tokenizer misses
    import re
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return text if text else None

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Quran Recitation Backend — Tarteel Whisper")

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
        "chunk_seconds": CHUNK_SECONDS,
    }

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    client = ws.client.host if ws.client else "unknown"
    log.info(f"Client connected: {client}")

    authenticated = False
    user_id       = None
    audio_buffer  = bytearray()   # accumulates PCM16 chunks
    last_transcribe_time = time.time()
    last_interim_text    = ""

    async def send(msg: dict):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    try:
        # ── Config handshake timeout ──────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(ws.receive(), timeout=10.0)
        except asyncio.TimeoutError:
            await send({"type": "error", "message": "Config timeout", "code": "TIMEOUT"})
            await ws.close(4001)
            return

        # Parse config message
        try:
            msg = json.loads(raw.get("text") or raw.get("bytes", b"{}").decode())
        except Exception:
            await send({"type": "error", "message": "Invalid config", "code": "BAD_CONFIG"})
            await ws.close(4002)
            return

        if msg.get("type") != "config":
            await send({"type": "error", "message": "Expected config message", "code": "BAD_CONFIG"})
            await ws.close(4002)
            return

        # Validate auth token
        token = msg.get("token", "")
        valid, uid = validate_token(token)
        if not valid:
            await send({"type": "error", "message": "Authentication required", "code": "AUTH_REQUIRED"})
            await ws.close(4003)
            return

        authenticated = True
        user_id       = uid
        log.info(f"User authenticated: {user_id}")

        # Tell frontend we're ready
        await send({"type": "ready"})

        # ── Main audio loop ───────────────────────────────────────────────────
        chunk_bytes = int(CHUNK_SECONDS * SAMPLE_RATE * 2)  # 2 bytes per PCM16 sample

        while True:
            try:
                raw = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                break  # Client gone

            # JSON control messages (updateRefText, etc.)
            if "text" in raw:
                try:
                    ctrl = json.loads(raw["text"])
                    # Future: handle updateRefText for pronunciation assessment
                    log.debug(f"Control message: {ctrl.get('type')}")
                except Exception:
                    pass
                continue

            # Binary audio data
            chunk = raw.get("bytes")
            if not chunk:
                continue

            audio_buffer.extend(chunk)

            now = time.time()
            elapsed = now - last_transcribe_time

            # Run Whisper every CHUNK_SECONDS (or when buffer is large enough)
            if elapsed >= CHUNK_SECONDS and len(audio_buffer) >= 3200:
                # Run inference in thread pool so we don't block the event loop
                audio_snapshot = bytes(audio_buffer)

                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, transcribe, audio_snapshot)

                last_transcribe_time = now

                if text and text != last_interim_text:
                    last_interim_text = text
                    # Send as interim while listening
                    await send({"type": "interim", "text": text})
                    log.info(f"Interim [{user_id[:8]}]: {text[:60]}")

            # When buffer exceeds 8s, emit a final result and reset
            if len(audio_buffer) >= int(8 * SAMPLE_RATE * 2):
                audio_snapshot = bytes(audio_buffer)
                audio_buffer.clear()
                last_interim_text = ""

                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, transcribe, audio_snapshot)

                if text:
                    await send({"type": "final", "text": text, "words": []})
                    log.info(f"Final [{user_id[:8]}]: {text[:60]}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error [{client}]: {e}")
        try:
            await send({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        # Flush remaining audio as final result
        if authenticated and len(audio_buffer) >= 3200:
            try:
                text = transcribe(bytes(audio_buffer))
                if text:
                    await send({"type": "final", "text": text, "words": []})
                    log.info(f"Flush final [{user_id[:8] if user_id else '?'}]: {text[:60]}")
            except Exception:
                pass

        log.info(f"Client disconnected: {client} (user={user_id})")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"Starting Quran backend on port {PORT}")
    log.info(f"Model:  {MODEL_ID}")
    log.info(f"Device: {DEVICE}")
    log.info(f"Chunk:  {CHUNK_SECONDS}s")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
