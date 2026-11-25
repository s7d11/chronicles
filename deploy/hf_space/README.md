---
title: The Sebeni Chronicle — Live 13-Language Swarm Theater
emoji: 📰
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Live self-organizing 2-5 agent clusters & Newsprint 3D Theater across Mali's 13 languages.
---

# The Sebeni Chronicle — Live 13-Language Swarm Theater

Live online observatory for **Edition No. 2** of [*The Sebeni Chronicle*](https://s7d11.github.io/chronicles/editions/vol-01-no-02/index.html), streaming 9 autonomous 2–5 agent clusters coordinated by `nvidia/nemotron-3-ultra-550b-a55b` across all 13 Malian national languages (`MULTI13`).

- **Public Endpoints:** Read-only `/api/health`, `/api/runs`, `/api/stream?lang=<code>` (SSE), `/api/edition2.json`, `/api/edition2_events.json`.
- **State Persistence:** Checkpoints synced to [`s7d11/chronicles-edition2-state`](https://huggingface.co/datasets/s7d11/chronicles-edition2-state).
- **Credits:** Antigravity Team (`Antigravity for Science`, workspace only), Adaption Labs, NVIDIA Build, Hugging Face Spaces, and `Gemini 4 Neon-High-Orchestrator`.
