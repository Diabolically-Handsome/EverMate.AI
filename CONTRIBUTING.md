# Contributing to EverMate.AI

Thanks for your interest! This is a small project with a simple workflow.

## Getting started
```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q     # must pass before and after your change
python app.py                  # manual smoke test
```

## Ground rules
1. **The honesty contract is non-negotiable.** Retrieval may inject
   *evidence*; it must never inject expected answers, entity lists, or any
   heuristic tied to a specific corpus. The model's reply must never be
   silently replaced. (See `legacy_quanzhi_heuristics.py` for why this rule
   exists.)
2. **Local-first.** No new network calls in the app's runtime path other
   than the user-configured Ollama server. Anything that sends data off the
   machine must be opt-in, loudly documented, and off by default.
3. **Keep the UI thread free.** All engine and LLM work goes through the
   `EngineWorker` queue in `views/chat.py`.
4. **Tests:** new engine behavior needs a test in `tests/` (no network, no
   Ollama, no PySide6 required there).
5. **i18n:** user-visible strings go through `i18n_qt.tr()` with both `zh`
   and `en` entries.

## Benchmark contributions
Use a corpus you have the right to redistribute (public domain preferred).
Never commit corpora or generated reports — `reports/` and `*.docx` are
gitignored on purpose.

## Commit style
Short imperative subject, body explains the *why*. Run the tests first.
