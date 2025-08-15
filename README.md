# 🐾 EverMate.AI (v1.0 coming soon)
Your Local AI Pet & Friend · Privacy-first · Offline whenever possible

**Language · Idioma · Langue · 言語**  
[English (Canada)](README.en-CA.md) · [中文](README.zh-CN.md) · [Português (Brasil)](README.pt-BR.md) · [Français (Canada)](README.fr-CA.md) · [日本語](README.ja-JP.md)

---

## Welcome
This is the entry page. Pick your preferred language above to read the full README.

## How to use this on GitHub
- **Option A (recommended):** Keep this file as `README.md`. Visitors will see the language switch first.  
- **Option B:** Rename your preferred localized file to `README.md` (e.g., `README.en-CA.md` → `README.md`) so it becomes the default. You can then rename this entry to `README.index.md` or remove it.

## Quick start (English summary)
1. Run `python app.py` at the project root.  
2. Two ways to use:  
   - **Import history:** drag `.docx/.txt` on the import screen, then click “Build/Rebuild Memory”.  
   - **New friend:** just start chatting; each turn is indexed incrementally and Core/Persona refresh periodically.  
3. (Optional) Configure **Ollama** (`OLLAMA_URL`, `OLLAMA_MODEL`) to refine Persona locally.

## Signature features (one‑liners)
- **Three‑Tier Memory:** Core → Persona → Vault. Core handles frequent topics & style, Persona captures preferences, Vault fetches precise evidence.  
- **Scalable Local Index:** chunked storage + SQLite inverted index + BM25; fast, local, and ready for millions of words.

## License
MIT (see repository LICENSE)
