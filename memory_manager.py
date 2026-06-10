"""memory_manager.py — compatibility shim.

The memory engine now lives in the `engine/` package:

- engine/textutil.py   tokenization, sentence splitting, query analysis
- engine/storage.py    SQLite index, chunk store, ingestion, uploads
- engine/retrieval.py  BM25 + best-sentence evidence
- engine/persona.py    Core / Persona refresh
- engine/manager.py    MemoryManager facade

Existing imports (`from memory_manager import MemoryConfig, MemoryManager`)
keep working through this module.

The pre-2026-06 engine, which was overfitted to a single benchmark corpus,
is preserved for reference in legacy_quanzhi_heuristics.py and is no longer
imported by the application.
"""

from engine.manager import MemoryConfig, MemoryManager

__all__ = ["MemoryConfig", "MemoryManager"]
