# 🐾 EverMate.AI (v1.0 coming soon)
Your Local AI Pet & Friend · Privacy-first · Offline whenever possible

[English (Canada)](README.en-CA.md) · [中文](README.zh-CN.md) · [Português (Brasil)](README.pt-BR.md) · [Français (Canada)](README.fr-CA.md) · [日本語](README.ja-JP.md)

## ✨ What it is
EverMate.AI makes **long‑term companionship chats** practical and under your control: runs locally by default, no uploads, and organizes your history with two signature ideas — **Three‑Tier Memory** + **Scalable Local Index**.

## 🍱 Signature features
### 1) Three‑Tier Memory (Core → Persona → Vault)
- **Core**: high‑frequency topics + style/response cues (e.g., call you “you/您”, include examples, keep a warm tone). Most questions hit Core first — faster and steadier.  
- **Persona**: communication preferences, info density, long‑term interests, do’s & don’ts — distilled into ≤8 bullets. Uses a local LLM via Ollama when available; otherwise falls back to heuristics.  
- **Vault**: everything else is stored **in chunks on disk** and retrieved on demand — no giant doc, just short, precise evidence.  
- **Conflict rule**: if history disagrees with your **current input**, current takes precedence.

### 2) Scalable Local Index (Large‑scale · Local · Fast)
- **Chunking**: ~2.8K chars per chunk (tweakable), built for millions of words.  
- **Inverted index**: SQLite tables for `terms/postings/chunks`, WAL mode for robust R/W.  
- **BM25 retrieval**: rank candidate chunks, then lift the **best original sentence** as supporting evidence.  
- **Streaming build & incremental updates**: drag‑and‑drop `.docx/.txt` to build as we read; new chats append per turn and refresh Core/Persona periodically.

## 🚀 Quick start
1. `python app.py` in the project directory.  
2. Two paths:  
   - **Import history**: drag `.docx/.txt` on the import screen, click “Build/Rebuild Memory”, then start chatting.  
   - **New friend**: just chat. Each turn (“you ask + AI answers”) is appended to the index; Core/Persona refresh after thresholds.  
3. Optional: configure **Ollama** (`OLLAMA_URL`, `OLLAMA_MODEL`) to refine Persona locally.

## 💼 How it works (in one line)
Start with **Core** for style & frequent topics → use **Persona** for your taste → pull **Vault** for citations. **Short, relevant context** wins.

## 🧲 Import vs. New (same engine)
- **Drag‑and‑drop**: `.docx` parsed via stdlib; `.txt` read as a stream.  
- **New friend**: `append_turn` writes incrementally; default refresh every **20** new chunks.

## 🔧 Tunables
`CHUNK_CHARS`=2800 · `CORE_TOP_TERMS`=50 · `PERSONA_MAX_BULLETS`=8 · `REFRESH_EVERY`=20 · `retrieve(query,k)`=4–8

## 🛡️ Privacy & local‑first
Runs and stores data locally; no uploads by default. If you switch to remote models, review your network/compliance posture. Encrypt & back up the `memory` folder.

## ❓FAQ (about the signature features)
- Persona hasn’t changed? Likely below the refresh threshold — keep chatting or rebuild; you can also lower `REFRESH_EVERY`.  
- Want the **original wording**? Yes — Vault returns the most relevant sentence/snippet.  
- Index keeps growing? Increase `CHUNK_CHARS`, archive older `chunks`, or split `memory` per friend.  
- No local LLM? Only Persona quality is affected; heuristics keep the flow stable.  
- Feeling “stuck” to history? Current input overrides; you can also tweak `01_core.md` / `02_persona.md` then rebuild.

## 🗺️ Roadmap
Hybrid retrieval (BM25 + local embeddings), topic buckets & timeline, explainability panel, memory decay, event cards.

## 📦 Layout
```
memory/
  index.sqlite    # inverted index & stats
  chunks/         # chunked text
  uploads/        # imported files
  buffer.txt      # incremental buffer
  01_core.md      # core memory (freq & style)
  02_persona.md   # persona bullets
  03_vault.md     # vault note (dynamic retrieval)
```

## 📜 License
MIT (see repository LICENSE)
