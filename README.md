---
title: HeatSentinel
emoji: 🌡️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🌡️ HeatSentinel

Urban heat risk assistant powered by FortyGuard sensor data. Ask about heat conditions, risk levels, and safe outdoor windows for any US city — HeatSentinel routes your question to the right FortyGuard tool (temperature stats, environmental parameters, satellite/street-view segmentation) via a LangChain + Groq agent, auto-chains follow-up calls when heat crosses risk thresholds, and explains the result in plain language.

**Examples:**
- "How hot is it in Phoenix right now?"
- "Give me a full heat briefing for Houston today"
- "Compare heat in Dallas vs Miami"

## How it works

`app.py` clones the private [`heatsentinel-fortyguard`](https://github.com/faresayman-ai/heatsentinel-fortyguard) repo at startup (for the `fortyguard` client library) and wraps its `UnifiedOrchestrator` in a Gradio chat UI.

## Required Space secrets

Set these in **Settings → Variables and secrets** before the Space will start:

| Secret | Purpose |
|---|---|
| `GITHUB_TOKEN` | Read access to clone the private `heatsentinel-fortyguard` repo |
| `FORTYGUARD_API_KEY` | Auth for the FortyGuard Temperature API |
| `GROQ_API_KEY` | Auth for the Groq-hosted LLM (`openai/gpt-oss-20b`) |

The app will raise a clear `RuntimeError` on startup if any of these are missing.

## Deploying

1. Create a new Space (SDK: **Gradio**).
2. Upload `app.py` and `requirements.txt` to the Space repo (this `README.md` can go too — its front matter configures the Space).
3. Add the three secrets above.
4. The Space builds and gives you a permanent URL: `https://huggingface.co/spaces/<your-username>/<space-name>`.

## Notes

- The `fortyguard` client isn't on PyPI — it's pulled from GitHub at container startup, not listed in `requirements.txt`.
- Location resolution is currently US-only; non-US queries return an `outside_us` error.
