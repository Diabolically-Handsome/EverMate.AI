# Changelog

## 2.0.0 — 2026-06-11

A clean-break release: the memory engine was rebuilt from scratch, the GUI
was reworked end to end, and every published claim is now backed by
reproducible verification.

### The Honesty Contract (engine rewrite)
- The old engine (V1.x) hardcoded benchmark answers from one test corpus
  into its retrieval path; its published numbers were withdrawn. It is
  preserved unimported in `legacy_quanzhi_heuristics.py`.
- New `engine/` package: corpus-agnostic BM25 retrieval with best-sentence
  evidence, fenced prompts (instructions inside imported text are never
  followed), fallbacks only on empty model output — the model's reply is
  never silently replaced.
- Re-measured benchmarks on public-domain corpora (deterministic cloze, no
  LLM judge): 紅樓夢 906K chars — retrieval 99.50% / end-to-end 93.33%;
  Moby-Dick 1.22M chars — retrieval 100% / 98.33% (`gpt-oss:20b`).

### Companion voice 🗣️
- Importing history with an old AI friend now restores *how it spoke*, not
  just what it said: Analyze learns a style profile (tone, addressing,
  catchphrases, sign-offs — `04_voice.md`) from the corpus via the local
  LLM and injects it into every reply.

### GUI
- Real widget-based chat bubbles (rounded, width-capped, selectable,
  markdown-lite) — replaces the broken rich-text rendering.
- Streaming replies with typing dots; all engine/LLM work on a background
  worker (the UI never freezes); live progress (stage text + bar) for
  Build/Analyze.
- "Memory constellation" welcome page, staggered entrances, breathing busy
  pill, counting stats, memory-card glow, gradient buttons, dual themes.
- Forget controls: clear chat memory / delete imported documents / wipe all.
- Full zh/en localization including all dialogs and prompts.

### Reliability & privacy
- Single-instance lock, atomic state saves (debounced, not just on close),
  content-hash import dedup that survives rebuilds, schema versioning with
  migration, DB backup before rebuild, instance-safe shutdown.
- Streamed replies are sanitized of reasoning artifacts (`<think>`,
  gpt-oss channel markers).
- Unified per-user memory location: `~/Library/Application Support/
  EverMate/memory` (override with `MEMORY_DIR`).
- 183-test pytest suite (no network/Ollama needed) + GitHub Actions CI
  (Python 3.10–3.12, macOS + Ubuntu).

### Known limitations
- The macOS build is ad-hoc signed, not notarized (right-click → Open on
  first launch); the Developer ID path is documented in PACKAGING.md.
- Retrieval is lexical (BM25); hybrid vector retrieval is on the roadmap.
