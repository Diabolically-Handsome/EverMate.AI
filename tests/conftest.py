"""Shared fixtures for the EverMate engine test suite.

Hard rules enforced here:
- never touch the real per-user memory dir (everything goes to tmp_path)
- never talk to a real Ollama server (OLLAMA_URL points at a closed port,
  so any accidental call fails immediately)
- no PySide6: only engine / infra modules are imported
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from engine.manager import MemoryConfig, MemoryManager
from engine.storage import MemoryStore


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch):
    """Sandbox every test: tmp memory dir, unreachable Ollama."""

    # Even code paths that fall back to MemoryConfig.from_env() must never
    # land in ~/Library (or the platform equivalent).
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path / "default-memory"))
    # http://127.0.0.1:1 refuses connections instantly, so anything that
    # tries to reach Ollama gets a fast, deterministic failure.
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)


@pytest.fixture
def memory_dir(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def manager(memory_dir, monkeypatch):
    """A fresh MemoryManager rooted in a tmp_path memory dir."""

    monkeypatch.setenv("MEMORY_DIR", memory_dir)
    m = MemoryManager(MemoryConfig(memory_dir=memory_dir))
    yield m
    m.close()


@pytest.fixture
def store(memory_dir):
    """A fresh MemoryStore rooted in a tmp_path memory dir."""

    s = MemoryStore(memory_dir)
    yield s
    s.close()
