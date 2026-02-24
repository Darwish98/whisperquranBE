# 🕌 Quran Recitation App – Azure Setup Guide
## Achieving Tarteel-level Performance

---

## Architecture Overview

```
Browser Mic
    │  Raw PCM16 @ 16kHz (AudioWorklet)
    ▼
WebSocket ──────────────────────────────────────────────────────────────────
    │                                                            Azure App Service
    ▼                                                           (Node.js backend)
Azure Speech SDK ──► Azure Cognitive Services Speech
    │                  ar-SA, real-time streaming
    ▼                  ~100-200ms latency
Transcript (interim + final)
    │
    ▼
Frontend matching (phoneme-level Levenshtein)
    │
    ▼
Word highlight / advance
```

**Why this beats Web Speech API:**
| Feature | Web Speech API | Azure Streaming |
|---|---|---|
| Latency | 1–3 seconds | 100–300ms |
| Arabic accuracy | ~60-70% | ~90%+ |
| Interim results | Partial | Yes (per-phoneme) |
| Word confidence | No | Yes |
| Works offline | No | No |
| Cost | Free | ~$1 per 5hrs use |

---

## Step 1: Create Azure Cognitive Services Speech Resource

1. Go to [portal.azure.com](https://portal.azure.com)
2. Click **Create a resource** → search **"Speech"**
3. Select **Speech** (under Azure AI Services)
4. Fill in:
   - **Subscription**: Your subscription
   - **Resource group**: Create new → `quran-app-rg`
   - **Region**: `East US` (lowest latency for most users; pick closest to your users)
   - **Name**: `quran-speech-[yourname]`
   - **Pricing tier**: `S0` (Standard) — $1 per 5 hours of audio
5. Click **Review + Create** → **Create**
6. After deployment, go to resource → **Keys and Endpoint**
7. Copy **Key 1** and the **Location/Region** value

> 💡 **Free tier (F0)** gives 5 hours/month free. Use S0 for production.

---

## Step 2: Create Azure App Service for the Backend

1. In Azure Portal → **Create a resource** → **Web App**
2. Fill in:
   - **Resource group**: `quran-app-rg` (same as above)
   - **Name**: `quran-backend-[yourname]` (this becomes your URL)
   - **Publish**: Code
   - **Runtime stack**: `Node 20 LTS`
   - **OS**: Linux
   - **Region**: Same as your Speech resource (e.g., East US)
   - **Pricing plan**: **B1** (Basic, ~$13/month) minimum for WebSocket support
     > ⚠️ Free tier (F0/F1) does NOT support WebSockets. Use B1+.
3. Click **Review + Create** → **Create**

---

## Step 3: Configure App Service Environment Variables

1. Go to your App Service → **Configuration** → **Application Settings**
2. Add these settings (click **+ New application setting** for each):

   | Name | Value |
   |---|---|
   | `AZURE_SPEECH_KEY` | Your Key 1 from Step 1 |
   | `AZURE_SPEECH_REGION` | e.g. `eastus` |
   | `ALLOWED_ORIGINS` | `https://your-app.lovable.app` |
   | `PORT` | `8000` |

3. Click **Save**

---

## Step 4: Enable WebSocket Support in App Service

1. App Service → **Configuration** → **General settings**
2. Toggle **Web sockets** → **On**
3. Click **Save**

---

## Step 5: Deploy the Backend

### Option A: Deploy via GitHub (Recommended)

1. Push the `quran-backend/` folder to a GitHub repo
2. App Service → **Deployment Center** → **GitHub**
3. Connect your repo and branch
4. Azure will auto-build and deploy on every push

### Option B: Deploy via ZIP (Quick)

```bash
# In the quran-backend folder:
npm install
npm run build
zip -r deploy.zip dist/ package.json package-lock.json node_modules/

# Then upload via Azure CLI:
az webapp deployment source config-zip \
  --resource-group quran-app-rg \
  --name quran-backend-[yourname] \
  --src deploy.zip
```

### Option C: Azure CLI one-liner
```bash
# Install Azure CLI, then:
az login
az webapp up \
  --name quran-backend-[yourname] \
  --resource-group quran-app-rg \
  --runtime "NODE:20-lts" \
  --sku B1
```

---

## Step 6: Set Startup Command

1. App Service → **Configuration** → **General settings**
2. **Startup Command**: `node dist/server.js`
3. Save

---

## Step 7: Verify Backend is Running

Visit: `https://quran-backend-[yourname].azurewebsites.net/health`

You should see:
```json
{"status":"ok","region":"eastus"}
```

---

## Step 8: Configure the Frontend (Lovable)

1. In Lovable: **Settings** → **Environment Variables**
2. Add:
   - **Key**: `VITE_WS_URL`
   - **Value**: `wss://quran-backend-[yourname].azurewebsites.net/ws/transcribe`
3. Copy the files from this upgrade package into your Lovable project:
   - `useAzureSpeech.ts` → `src/hooks/useAzureSpeech.ts`
   - `arabicUtils.ts` → `src/lib/arabicUtils.ts` (replace existing)
4. Follow `INDEX_PATCH_INSTRUCTIONS.ts` to update `src/pages/Index.tsx`

---

## Step 9: Test

1. Open your Lovable app
2. You should see **"🟢 Azure Speech – real-time streaming active"**
3. Start reciting — words should highlight within 100-300ms

---

## Cost Estimate

| Service | Tier | Cost |
|---|---|---|
| Azure Speech | S0 | $1 per 5 hours of audio |
| Azure App Service | B1 | ~$13/month |
| **Total (light use)** | | ~$15-20/month |

For heavy use or global audience, consider:
- **App Service P1v3** ($70/month) for faster CPU + more connections
- **Azure CDN** for the static frontend
- **Azure Front Door** for global WebSocket routing

---

## Troubleshooting

**WebSocket won't connect:**
- Verify Web sockets is ON in App Service → Configuration → General settings
- Check ALLOWED_ORIGINS includes your exact frontend URL
- Use `wss://` not `ws://` for HTTPS frontends

**High latency (>500ms):**
- Ensure App Service region matches Speech resource region
- Upgrade from B1 to B2 or P1v3
- Check browser DevTools Network tab for WS frame timing

**Arabic recognition poor:**
- Ensure `ar-SA` locale is set in server.ts (already done)
- Check mic quality — use headset for best results
- The phoneme matching threshold can be lowered from 0.72 to 0.65 in Index.tsx

**CORS errors:**
- Double-check ALLOWED_ORIGINS env var exactly matches your Lovable app URL
- Include both with and without trailing slash

---

## Optional: Azure Static Web Apps for Frontend

Instead of Lovable hosting, deploy frontend to Azure for lower latency:

1. Build your Lovable app locally
2. Azure Portal → **Create resource** → **Static Web App**
3. Connect to GitHub, set build output to `dist/`
4. Automatic HTTPS + global CDN included

---

*This architecture is similar to how Tarteel.ai operates: continuous streaming speech recognition with phoneme-level matching, not batch recognition.*
