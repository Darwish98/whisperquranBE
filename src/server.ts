/**
 * Quran Recitation Backend
 * Real-time Arabic speech recognition via Azure Cognitive Services Speech SDK
 * WebSocket server: receives raw PCM audio chunks → streams to Azure → returns transcription
 *
 * ENV vars required (set in Azure App Service / .env):
 *   AZURE_SPEECH_KEY        – Azure Cognitive Services Speech resource key
 *   AZURE_SPEECH_REGION     – e.g. "eastus", "westeurope"
 *   PORT                    – (optional) defaults to 8000
 *   ALLOWED_ORIGINS         – comma-separated origins, e.g. "https://yourapp.com"
 */

import * as sdk from "microsoft-cognitiveservices-speech-sdk";
import { WebSocketServer, WebSocket } from "ws";
import http from "http";
import "dotenv/config";

const PORT = Number(process.env.PORT ?? 8000);
const SPEECH_KEY = process.env.AZURE_SPEECH_KEY ?? "";
const SPEECH_REGION = process.env.AZURE_SPEECH_REGION ?? "";
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS ?? "*")
  .split(",")
  .map((s) => s.trim());

if (!SPEECH_KEY || !SPEECH_REGION) {
  console.error("❌  AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set.");
  process.exit(1);
}

// ──────────────────────────────────────────────
// HTTP server (health check endpoint)
// ──────────────────────────────────────────────
const httpServer = http.createServer((req, res) => {
  if (req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", region: SPEECH_REGION }));
    return;
  }
  res.writeHead(404);
  res.end();
});

// ──────────────────────────────────────────────
// WebSocket server
// ──────────────────────────────────────────────
const wss = new WebSocketServer({
  server: httpServer,
  path: "/ws/transcribe",
  verifyClient: ({ origin }: { origin: string }) => {
    if (ALLOWED_ORIGINS.includes("*")) return true;
    return ALLOWED_ORIGINS.includes(origin);
  },
});

wss.on("connection", (ws: WebSocket) => {
  console.log("🔌  Client connected");

  // ── Azure Speech SDK setup ──────────────────
  const speechConfig = sdk.SpeechConfig.fromSubscription(
    SPEECH_KEY,
    SPEECH_REGION,
  );
  speechConfig.speechRecognitionLanguage = "ar-SA";

  // Enable detailed output (confidence + ITN)
  speechConfig.outputFormat = sdk.OutputFormat.Detailed;

  // Enable profanity filter OFF so Arabic words aren't masked
  speechConfig.setProfanity(sdk.ProfanityOption.Raw);

  // Enable word-level timestamps (needed for per-word feedback)
  speechConfig.requestWordLevelTimestamps();

  // Push-stream: we feed raw PCM16 from the browser
  const pushStream = sdk.AudioInputStream.createPushStream(
    sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1), // 16kHz, 16-bit, mono
  );
  const audioConfig = sdk.AudioConfig.fromStreamInput(pushStream);

  const recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

  // ── Recognizer events ───────────────────────

  // Interim hypothesis (very fast, <100 ms)
  recognizer.recognizing = (_s, e) => {
    if (e.result.reason === sdk.ResultReason.RecognizingSpeech) {
      const text = e.result.text?.trim();
      if (text && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "interim", text }));
      }
    }
  };

  // Final result
  recognizer.recognized = (_s, e) => {
    if (e.result.reason === sdk.ResultReason.RecognizedSpeech) {
      const text = e.result.text?.trim();
      if (!text) return;

      // Extract word-level details from JSON if available
      let words: Array<{ word: string; confidence: number }> = [];
      try {
        const detail = JSON.parse(e.result.json);
        const best = detail?.NBest?.[0];
        if (best?.Words) {
          words = best.Words.map((w: any) => ({
            word: w.Word,
            confidence: w.Confidence ?? 1,
          }));
        }
      } catch {
        // fallback – send plain text
      }

      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "final", text, words }));
      }
    } else if (e.result.reason === sdk.ResultReason.NoMatch) {
      console.log("⚠️  No speech recognized");
    }
  };

  recognizer.canceled = (_s, e) => {
    if (e.reason === sdk.CancellationReason.Error) {
      console.error("❌  Azure Speech error:", e.errorDetails);
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "error", message: e.errorDetails }));
      }
    }
  };

  recognizer.startContinuousRecognitionAsync(
    () => console.log("🎙️  Azure recognizer started"),
    (err) => console.error("❌  Failed to start recognizer:", err),
  );

  // ── Receive audio from browser ──────────────
  ws.on("message", (data: Buffer | ArrayBuffer | Buffer[]) => {
    try {
      let ab: ArrayBuffer;
      if (data instanceof ArrayBuffer) {
        ab = data;
      } else if (Buffer.isBuffer(data)) {
        ab = data.buffer.slice(
          data.byteOffset,
          data.byteOffset + data.byteLength,
        ) as ArrayBuffer;
      } else {
        // Buffer[] (rare)
        const merged = Buffer.concat(data as Buffer[]);
        ab = merged.buffer.slice(
          merged.byteOffset,
          merged.byteOffset + merged.byteLength,
        ) as ArrayBuffer;
      }
      pushStream.write(ab);
    } catch (err) {
      console.error("Audio write error:", err);
    }
  });

  ws.on("close", () => {
    console.log("🔌  Client disconnected");
    pushStream.close();
    recognizer.stopContinuousRecognitionAsync(
      () => recognizer.close(),
      (err) => {
        console.error("Stop error:", err);
        recognizer.close();
      },
    );
  });

  ws.on("error", (err) => {
    console.error("WS error:", err);
  });
});

httpServer.listen(PORT, () => {
  console.log(`✅  Quran recitation backend listening on port ${PORT}`);
  console.log(`   WebSocket: ws://0.0.0.0:${PORT}/ws/transcribe`);
  console.log(`   Health:    http://0.0.0.0:${PORT}/health`);
});
