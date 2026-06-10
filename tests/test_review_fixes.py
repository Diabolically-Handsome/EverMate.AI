"""Regression tests for the issues found by the 2026-06 adversarial review."""

import os
import sqlite3

import pytest

from engine.manager import MemoryConfig, MemoryManager
from engine.textutil import conflict_markers, looks_like_recall_query
from ollama_client import ThinkTagFilter


def _write_corpus(tmp_path, name="doc.txt", content="这是一个测试文档。" * 60):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestDedupSurvivesRebuild:
    def test_reimport_after_rebuild_is_still_deduped(self, manager, tmp_path):
        src = _write_corpus(tmp_path)
        stored = manager.import_files([src])
        assert len(stored) == 1
        manager.ingest_new_uploads(stored)

        manager.rebuild_memory()

        # The dedup ledger must survive the rebuild: re-importing identical
        # content used to silently double the corpus.
        assert manager.import_files([src]) == []

    def test_ingest_new_uploads_skips_already_ingested(self, manager, tmp_path):
        src = _write_corpus(tmp_path)
        stored = manager.import_files([src])
        manager.ingest_new_uploads(stored)
        before = manager.count_chunks()

        # No-arg form used to re-ingest every upload on each call.
        assert manager.ingest_new_uploads() == 0
        assert manager.count_chunks() == before


class TestForgetReallyForgets:
    def test_clear_chat_memory_resets_persona(self, manager):
        from engine.storage import write_text

        write_text(
            manager.persona_md_path,
            "# Persona (≤ 8 bullets)\n\n- 喜欢深夜聊哲学\n- 养了一只叫煤球的猫\n",
        )
        manager.append_turn("我养了一只叫煤球的猫", "记住啦")
        manager.clear_chat_memory()

        persona = open(manager.persona_md_path, encoding="utf-8").read()
        assert "煤球" not in persona
        assert "哲学" not in persona

    def test_delete_upload_rejects_foreign_paths(self, manager, tmp_path):
        outside = _write_corpus(tmp_path, name="outside.txt")
        with pytest.raises(ValueError):
            manager.delete_upload(outside)


class TestScriptCompat:
    def test_conn_is_rebindable_after_manual_close(self, manager):
        # validate_memory_accuracy.py does exactly this dance.
        manager.close()
        manager.conn = manager._open_db()
        assert manager.count_chunks() == 0
        manager.store.ingest_text("重新打开后的写入。", source="note")
        assert manager.count_chunks() == 1


class TestMigrationGating:
    def test_explicit_memory_dir_is_never_contaminated(self, tmp_path, monkeypatch):
        # A CWD with a legacy ./memory layout (private data)...
        cwd = tmp_path / "cwd"
        legacy = cwd / "memory"
        legacy.mkdir(parents=True)
        sqlite3.connect(legacy / "index.sqlite").close()
        (legacy / "chat_log.txt").write_text("私密聊天记录", encoding="utf-8")
        monkeypatch.chdir(cwd)

        # ...must not leak into an explicitly configured memory dir.
        target = tmp_path / "bench-memory"
        mm = MemoryManager(MemoryConfig(memory_dir=str(target)))
        try:
            assert not (target / "chat_log.txt").exists()
        finally:
            mm.close()


class TestThinkFilterUnicode:
    def test_lowercase_length_change_does_not_misalign(self):
        # 'İ'.lower() is two code points; index math on a lowered copy used
        # to mis-slice the original buffer around it.
        f = ThinkTagFilter()
        out = f.feed("İstanbul <think>secret</think> answer") + f.flush()
        assert "secret" not in out
        assert "İstanbul" in out and "answer" in out

    def test_case_insensitive_tags(self):
        f = ThinkTagFilter()
        out = f.feed("a<THINK>x</ThInK>b") + f.flush()
        assert out == "ab"


class TestEnvParsing:
    def test_malformed_num_ctx_falls_back(self, monkeypatch):
        monkeypatch.setenv("EVERMATE_NUM_CTX", "banana")
        import importlib

        import ollama_client as oc

        importlib.reload(oc)
        try:
            assert oc.DEFAULT_NUM_CTX == 8192
        finally:
            monkeypatch.delenv("EVERMATE_NUM_CTX")
            importlib.reload(oc)


class TestRecallPhrasings:
    @pytest.mark.parametrize(
        "query",
        [
            "我们昨天聊了什么？",
            "我告诉过你我的猫叫什么",
            "你知道我的生日吗",
            "what did we talk about yesterday",
            "remind me what I said about the trip",
            "do you know my favorite movie",
        ],
    )
    def test_common_memory_questions_trigger_recall(self, query):
        assert looks_like_recall_query(query)

    def test_ordinary_smalltalk_stays_chat(self):
        assert not looks_like_recall_query("今天天气真好")


class TestLangParity:
    def test_extractive_fallback_in_english(self, manager):
        manager.store.ingest_text("My favorite movie is Blade Runner.", source="note")
        out = manager.render_fact_answer("do you remember my favorite movie", lang="en")
        assert "Blade Runner" in out
        assert "根据" not in out

    def test_rebuild_analyze_uses_ui_lang(self, manager):
        manager.ui_lang = "en"
        manager.store.ingest_text("Some imported text for the core.", source="note")
        manager.rebuild_memory()
        core = open(manager.core_md_path, encoding="utf-8").read()
        assert "Frequent topics" in core


class TestConflictMarkers:
    def test_date_components_are_not_numeric_conflicts(self):
        snippets = ["2023年5月我们去了长岛。", "她在2021年加入研究所。"]
        markers = conflict_markers(snippets)
        assert "数值" not in markers
