/**
 * Quran Recitation Backend — Hardened
 *
 * Real-time Arabic speech recognition via Azure Cognitive Services Speech SDK.
 * WebSocket server: receives raw PCM audio chunks → streams to Azure → returns transcription.
 *
 * SECURITY:
 * - Per-IP rate limiting (max concurrent connections + message throttle)
 * - Supabase JWT auth validation on WebSocket config handshake
 * - Cooldown between sessions per IP
 * - All speech processing server-side only (no API keys in frontend)
 *
 * SPEECH MODEL:
 * - Azure Cognitive Services Speech SDK (NOT OpenAI Whisper)
 * - Language: ar-SA (Arabic - Saudi Arabia)
 * - Streaming real-time recognition with word-level timestamps
 * - Runs ONLY on this backend server, never in browser
 *
 * ENV vars required:
 *   AZURE_SPEECH_KEY        – Azure Cognitive Services Speech resource key
 *   AZURE_SPEECH_REGION     – e.g. "eastus", "westeurope"
 *   SUPABASE_JWT_SECRET     – Supabase JWT secret for token validation (optional, recommended)
 *   PORT                    – (optional) defaults to 8000
 *   ALLOWED_ORIGINS         – comma-separated origins, e.g. "https://yourapp.com"
 *   MAX_CONNECTIONS_PER_IP  – (optional) default 3
 *   RATE_LIMIT_MESSAGES     – (optional) max audio messages per window, default 100
 *   RATE_LIMIT_WINDOW_MS    – (optional) rate limit window in ms, default 10000
 *   SESSION_COOLDOWN_MS     – (optional) cooldown between new connections per IP, default 5000
 */

import * as sdk from "microsoft-cognitiveservices-speech-sdk";
import { WebSocketServer, WebSocket } from "ws";
import http from "http";
import "dotenv/config";

// ── Configuration ─────────────────────────────────────────────────────────────

const PORT                  = Number(process.env.PORT ?? 8000);
const SPEECH_KEY            = process.env.AZURE_SPEECH_KEY ?? "";
const SPEECH_REGION         = process.env.AZURE_SPEECH_REGION ?? "";
const SUPABASE_JWT_SECRET   = process.env.SUPABASE_JWT_SECRET ?? "";
const ALLOWED_ORIGINS       = (process.env.ALLOWED_ORIGINS ?? "*").split(",").map(s => s.trim());
const MAX_CONNECTIONS_PER_IP = Number(process.env.MAX_CONNECTIONS_PER_IP ?? 3);
const RATE_LIMIT_MESSAGES   = Number(process.env.RATE_LIMIT_MESSAGES ?? 100);
const RATE_LIMIT_WINDOW_MS  = Number(process.env.RATE_LIMIT_WINDOW_MS ?? 10000);
const SESSION_COOLDOWN_MS   = Number(process.env.SESSION_COOLDOWN_MS ?? 5000);

if (!SPEECH_KEY || !SPEECH_REGION) {
  console.error("❌  AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set.");
  process.exit(1);
}

// ── Rate Limiting State ───────────────────────────────────────────────────────

interface IPState {
  connections: number;
  lastDisconnect: number;
  messageTimestamps: number[];
}

const ipStates = new Map<string, IPState>();

function getIPState(ip: string): IPState {
  if (!ipStates.has(ip)) {
    ipStates.set(ip, { connections: 0, lastDisconnect: 0, messageTimestamps: [] });
  }
  return ipStates.get(ip)!;
}

function checkRateLimit(ip: string): { allowed: boolean; reason?: string } {
  const state = getIPState(ip);

  // Check concurrent connection limit
  if (state.connections >= MAX_CONNECTIONS_PER_IP) {
    return { allowed: false, reason: `Max ${MAX_CONNECTIONS_PER_IP} concurrent connections per IP` };
  }

  // Check session cooldown
  const timeSinceLastDisconnect = Date.now() - state.lastDisconnect;
  if (state.lastDisconnect > 0 && timeSinceLastDisconnect < SESSION_COOLDOWN_MS) {
    const waitSec = Math.ceil((SESSION_COOLDOWN_MS - timeSinceLastDisconnect) / 1000);
    return { allowed: false, reason: `Please wait ${waitSec}s before starting a new session` };
  }

  return { allowed: true };
}

function checkMessageRate(ip: string): boolean {
  const state = getIPState(ip);
  const now = Date.now();

  // Remove old timestamps outside the window
  state.messageTimestamps = state.messageTimestamps.filter(
    t => now - t < RATE_LIMIT_WINDOW_MS
  );

  if (state.messageTimestamps.length >= RATE_LIMIT_MESSAGES) {
    return false; // Rate limited
  }

  state.messageTimestamps.push(now);
  return true;
}

// ── JWT Validation (lightweight, without full library) ────────────────────────

function base64UrlDecode(str: string): string {
  const padded = str + "=".repeat((4 - (str.length % 4)) % 4);
  return Buffer.from(padded, "base64url").toString("utf8");
}

async function validateSupabaseToken(token: string): Promise<{ valid: boolean; userId?: string; error?: string }> {
  if (!token) {
    return { valid: false, error: "No auth token provided" };
  }

  try {
    const parts = token.split(".");
    if (parts.length !== 3) {
      return { valid: false, error: "Invalid token format" };
    }

    const payload = JSON.parse(base64UrlDecode(parts[1]));

    // Check expiration
    if (payload.exp && payload.exp < Date.now() / 1000) {
      return { valid: false, error: "Token expired" };
    }

    // Check issuer (should be supabase)
    if (payload.iss && !payload.iss.includes("supabase")) {
      return { valid: false, error: "Invalid token issuer" };
    }

    // If we have a JWT secret, verify the signature properly
    if (SUPABASE_JWT_SECRET) {
      const crypto = await import("crypto");
      const signatureInput = parts[0] + "." + parts[1];
      const expectedSig = crypto
        .createHmac("sha256", SUPABASE_JWT_SECRET)
        .update(signatureInput)
        .digest("base64url");

      if (expectedSig !== parts[2]) {
        return { valid: false, error: "Invalid token signature" };
      }
    }

    return { valid: true, userId: payload.sub };
  } catch (err) {
    return { valid: false, error: "Token validation failed" };
  }
}

// ── Logging ───────────────────────────────────────────────────────────────────

function log(level: "INFO" | "WARN" | "ERROR", msg: string, meta?: Record<string, unknown>) {
  const ts = new Date().toISOString();
  const metaStr = meta ? " " + JSON.stringify(meta) : "";
  console[level === "ERROR" ? "error" : level === "WARN" ? "warn" : "log"](
    `[${ts}] [${level}] ${msg}${metaStr}`
  );
}

// ── HTTP Server ───────────────────────────────────────────────────────────────

const httpServer = http.createServer((req, res) => {
  // CORS headers
  const origin = req.headers.origin ?? "";
  if (ALLOWED_ORIGINS.includes("*") || ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin || "*");
  }
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      status: "ok",
      region: SPEECH_REGION,
      model: "Azure Cognitive Services Speech SDK (ar-SA)",
      note: "Speech processing runs only on this backend server",
      activeConnections: [...ipStates.values()].reduce((sum, s) => sum + s.connections, 0),
    }));
    return;
  }

  if (req.url === "/architecture") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      speechModel: "Azure Cognitive Services Speech SDK",
      modelNote: "This is NOT OpenAI Whisper. Uses Azure's enterprise speech service.",
      language: "ar-SA (Arabic - Saudi Arabia)",
      processing: "Server-side only (this Node.js backend)",
      endpoint: "/ws/transcribe (WebSocket)",
      frontendRole: "Captures microphone audio, sends raw PCM16 via WebSocket",
      backendRole: "Receives audio, streams to Azure Speech, returns transcription",
      security: {
        auth: "Supabase JWT token required in config handshake",
        rateLimiting: `${MAX_CONNECTIONS_PER_IP} concurrent per IP, ${RATE_LIMIT_MESSAGES} msgs/${RATE_LIMIT_WINDOW_MS}ms`,
        cooldown: `${SESSION_COOLDOWN_MS}ms between sessions`,
      },
    }));
    return;
  }

  res.writeHead(404);
  res.end();
});

// ── WebSocket Server ──────────────────────────────────────────────────────────

const wss = new WebSocketServer({
  server: httpServer,
  path: "/ws/transcribe",
  verifyClient: ({ origin, req }: { origin: string; req: http.IncomingMessage }) => {
    // Check origin
    if (!ALLOWED_ORIGINS.includes("*") && !ALLOWED_ORIGINS.includes(origin)) {
      log("WARN", "Rejected connection: origin not allowed", { origin });
      return false;
    }

    // Check IP rate limit
    const ip = (req.headers["x-forwarded-for"] as string)?.split(",")[0]?.trim() ||
               req.socket.remoteAddress || "unknown";
    const rateCheck = checkRateLimit(ip);
    if (!rateCheck.allowed) {
      log("WARN", "Rejected connection: rate limited", { ip, reason: rateCheck.reason });
      return false;
    }

    return true;
  },
});

wss.on("connection", (ws: WebSocket, req: http.IncomingMessage) => {
  const ip = (req.headers["x-forwarded-for"] as string)?.split(",")[0]?.trim() ||
             req.socket.remoteAddress || "unknown";
  const state = getIPState(ip);
  state.connections++;

  log("INFO", "Client connected", { ip, connections: state.connections });

  let authenticated = false;
  let userId: string | undefined;
  let configReceived = false;
  let recognizer: sdk.SpeechRecognizer | null = null;
  let pushStream: sdk.PushAudioInputStream | null = null;

  // Auto-close if no config message within 10 seconds
  const configTimeout = setTimeout(() => {
    if (!configReceived) {
      log("WARN", "Closing: no config message received within timeout", { ip });
      ws.close(4001, "Config timeout");
    }
  }, 10000);

  ws.on("message", async (data: Buffer | ArrayBuffer | Buffer[]) => {
    // ── Handle JSON messages (config, etc.) ──
    if (typeof data === "string" || (Buffer.isBuffer(data) && !configReceived)) {
      try {
        const str = Buffer.isBuffer(data) ? data.toString("utf8") : (data as unknown as string);
        const msg = JSON.parse(str);

        if (msg.type === "config") {
          configReceived = true;
          clearTimeout(configTimeout);

          // SECURITY: Validate auth token
          const tokenResult = await validateSupabaseToken(msg.token ?? "");
          if (!tokenResult.valid) {
            log("WARN", "Auth failed", { ip, error: tokenResult.error });
            ws.send(JSON.stringify({
              type: "error",
              message: `Authentication required: ${tokenResult.error}`,
              code: "AUTH_REQUIRED",
            }));
            ws.close(4003, "Authentication required");
            return;
          }

          authenticated = true;
          userId = tokenResult.userId;
          log("INFO", "User authenticated", { ip, userId });

          // ── Set up Azure Speech SDK ──
          const speechConfig = sdk.SpeechConfig.fromSubscription(SPEECH_KEY, SPEECH_REGION);
          speechConfig.speechRecognitionLanguage = msg.locale ?? "ar-SA";
          speechConfig.outputFormat = sdk.OutputFormat.Detailed;
          speechConfig.setProfanity(sdk.ProfanityOption.Raw);
          speechConfig.requestWordLevelTimestamps();

          pushStream = sdk.AudioInputStream.createPushStream(
            sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1),
          );
          const audioConfig = sdk.AudioConfig.fromStreamInput(pushStream);
          recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

          // Interim hypothesis
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

              log("INFO", "Transcription", { userId, text: text.substring(0, 50), wordCount: words.length });
            } else if (e.result.reason === sdk.ResultReason.NoMatch) {
              log("INFO", "No speech recognized", { userId });
            }
          };

          recognizer.canceled = (_s, e) => {
            if (e.reason === sdk.CancellationReason.Error) {
              log("ERROR", "Azure Speech error", { userId, error: e.errorDetails });
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "error", message: "Speech recognition error" }));
              }
            }
          };

          recognizer.startContinuousRecognitionAsync(
            () => {
              log("INFO", "Azure recognizer started", { userId });
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "ready" }));
              }
            },
            (err) => {
              log("ERROR", "Failed to start recognizer", { userId, error: String(err) });
              ws.close(4500, "Recognizer start failed");
            },
          );

          return;
        }

        // Handle ref text updates mid-session
        if (msg.type === "updateRefText" && authenticated) {
          // For pronunciation assessment — currently not implemented in base Azure SDK streaming
          return;
        }
      } catch {
        // Not JSON — continue to binary audio handling below
      }
    }

    // ── Handle binary audio data ──
    if (!authenticated) {
      log("WARN", "Rejecting audio: not authenticated", { ip });
      ws.close(4003, "Authentication required");
      return;
    }

    if (!pushStream) return;

    // Check message rate limit
    if (!checkMessageRate(ip)) {
      log("WARN", "Rate limited", { ip, userId });
      ws.send(JSON.stringify({
        type: "error",
        message: "Rate limit exceeded. Please slow down.",
        code: "RATE_LIMITED",
      }));
      return;
    }

    try {
      let ab: ArrayBuffer;
      if (data instanceof ArrayBuffer) {
        ab = data;
      } else if (Buffer.isBuffer(data)) {
        ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
      } else {
        const merged = Buffer.concat(data as Buffer[]);
        ab = merged.buffer.slice(merged.byteOffset, merged.byteOffset + merged.byteLength) as ArrayBuffer;
      }
      pushStream.write(ab);
    } catch (err) {
      log("ERROR", "Audio write error", { userId, error: String(err) });
    }
  });

  ws.on("close", () => {
    clearTimeout(configTimeout);
    state.connections = Math.max(0, state.connections - 1);
    state.lastDisconnect = Date.now();

    log("INFO", "Client disconnected", { ip, userId, connections: state.connections });

    if (pushStream) {
      pushStream.close();
    }
    if (recognizer) {
      recognizer.stopContinuousRecognitionAsync(
        () => recognizer?.close(),
        (err) => {
          log("ERROR", "Recognizer stop error", { error: String(err) });
          recognizer?.close();
        },
      );
    }
  });

  ws.on("error", (err) => {
    log("ERROR", "WebSocket error", { ip, userId, error: String(err) });
  });
});

// ── Cleanup stale IP states every 5 minutes ───────────────────────────────────

setInterval(() => {
  const now = Date.now();
  for (const [ip, state] of ipStates.entries()) {
    if (state.connections === 0 && now - state.lastDisconnect > 300000) {
      ipStates.delete(ip);
    }
  }
}, 300000);

// ── Start ─────────────────────────────────────────────────────────────────────

httpServer.listen(PORT, () => {
  log("INFO", `Quran recitation backend started`, {
    port: PORT,
    region: SPEECH_REGION,
    model: "Azure Cognitive Services Speech SDK (ar-SA)",
    maxConnectionsPerIP: MAX_CONNECTIONS_PER_IP,
    rateLimitMessages: RATE_LIMIT_MESSAGES,
    rateLimitWindowMs: RATE_LIMIT_WINDOW_MS,
    sessionCooldownMs: SESSION_COOLDOWN_MS,
    authEnabled: !!SUPABASE_JWT_SECRET,
  });
  console.log(`   WebSocket:    ws://0.0.0.0:${PORT}/ws/transcribe`);
  console.log(`   Health:       http://0.0.0.0:${PORT}/health`);
  console.log(`   Architecture: http://0.0.0.0:${PORT}/architecture`);
});
