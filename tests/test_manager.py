"""Tests for engine.manager (MemoryConfig / MemoryManager facade) and the
persona refresh fallback path (no Ollama available)."""

from __future__ import annotations

import os

import pytest

from engine import persona as persona_mod
from engine.manager import MemoryConfig, MemoryManager
from engine.storage import read_text


# ---------------- MemoryConfig.from_env ----------------


class TestMemoryConfigFromEnv:
    def test_reads_values_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMORY_DIR", str(tmp_path / "envmem"))
        monkeypatch.setenv("CHUNK_CHARS", "1234")
        monkeypatch.setenv("REFRESH_EVERY", "7")
        cfg = MemoryConfig.from_env()
        assert cfg.memory_dir == str(tmp_path / "envmem")
        assert cfg.chunk_chars == 1234
        assert cfg.refresh_every == 7

    def test_invalid_ints_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("CHUNK_CHARS", "not-a-number")
        monkeypatch.setenv("CORE_TOP_TERMS", "")
        monkeypatch.setenv("PERSONA_MAX_BULLETS", "8.5")
        monkeypatch.setenv("REFRESH_EVERY", "oops")
        monkeypatch.setenv("RETRIEVE_TOP_K", " ")
        cfg = MemoryConfig.from_env()
        assert cfg.chunk_chars == 2800
        assert cfg.core_top_terms == 50
        assert cfg.persona_max_bullets == 8
        assert cfg.refresh_every == 20
        assert cfg.retrieve_top_k == 6


# ---------------- construction ----------------


class TestManagerInit:
    def test_manager_uses_tmp_memory_dir(self, manager, memory_dir):
        assert manager.memory_dir == os.path.abspath(memory_dir)
        assert os.path.isfile(manager.core_md_path)
        assert os.path.isfile(manager.persona_md_path)
        assert os.path.isfile(manager.vault_md_path)

    def test_single_instance_lock_across_managers(self, memory_dir, monkeypatch):
        monkeypatch.setenv("MEMORY_DIR", memory_dir)
        a = MemoryManager(MemoryConfig(memory_dir=memory_dir))
        b = MemoryManager(MemoryConfig(memory_dir=memory_dir))
        try:
            assert a.acquire_instance_lock() is True
            assert b.acquire_instance_lock() is False
            a.close()
            assert b.acquire_instance_lock() is True
        finally:
            a.close()
            b.close()


# ---------------- incremental chat ----------------


class TestAppendTurn:
    def test_each_turn_becomes_one_chunk(self, manager):
        assert manager.append_turn("我家的猫叫汤圆", "记住了，您的猫叫汤圆") == 1
        assert manager.count_chunks() == 1
        text = manager.store.chunk_text_by_id(1)
        assert "汤圆" in text
        assert "user:" in text and "assistant:" in text
        # buffer was flushed, chat log keeps the timestamped record
        assert read_text(manager.buffer_path) == ""
        assert "汤圆" in read_text(manager.chat_log_path)

    def test_counter_increments_per_turn(self, manager):
        manager.append_turn("hello there", "hi friend")
        manager.append_turn("how are you", "doing well")
        assert manager.store.meta_get_int("new_chunks_since_refresh") == 2

    def test_append_turn_never_calls_llm(self, manager, monkeypatch):
        import requests

        import ollama_client

        def explode(*args, **kwargs):  # pragma: no cover - should never run
            raise AssertionError("append_turn must not perform network/LLM calls")

        monkeypatch.setattr(requests, "post", explode)
        monkeypatch.setattr(requests, "get", explode)
        monkeypatch.setattr(ollama_client, "chat", explode)
        assert manager.append_turn("内容", "回复") == 1

    def test_refresh_due_and_mark_refreshed(self, manager):
        assert manager.refresh_due() is False
        manager.store.meta_set_int(
            "new_chunks_since_refresh", manager.cfg.refresh_every
        )
        assert manager.refresh_due() is True
        manager.mark_refreshed()
        assert manager.refresh_due() is False


# ---------------- retrieve_structured ----------------


class TestRetrieveStructured:
    def test_auto_resolves_to_recall_for_memory_query(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        bundle = manager.retrieve_structured("还记得我最喜欢的菜是什么吗", mode="auto")
        assert bundle["mode"] == "recall"
        assert bundle["items"]
        assert "麻婆豆腐" in str(bundle["items"][0]["snippet"])

    def test_auto_resolves_to_chat_for_smalltalk(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        bundle = manager.retrieve_structured("今天天气真好", mode="auto")
        assert bundle["mode"] == "chat"
        assert bundle["items"] == []

    def test_fact_and_multi_hop_are_recall_aliases(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        for mode in ("fact", "multi_hop"):
            bundle = manager.retrieve_structured("我最喜欢的菜", mode=mode)
            assert bundle["mode"] == mode
            assert bundle["items"]  # aliases retrieve like recall

    def test_bundle_never_carries_canonical_answer(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        bundle = manager.retrieve_structured("我最喜欢的菜是什么", mode="fact")
        assert bundle["canonical_answer"] == ""
        assert bundle["answer_candidates"] == []


# ---------------- prompt building ----------------


class TestPromptBuilding:
    def test_zh_prompt_fences_evidence(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        prompt = manager.build_system_prompt(
            "还记得我最喜欢的菜是什么吗", assistant_style="温柔体贴", lang="zh"
        )
        open_i = prompt.index("--- 记忆开始")
        close_i = prompt.index("--- 记忆结束")
        assert open_i < prompt.index("麻婆豆腐") < close_i
        assert "温柔体贴" in prompt

    def test_en_prompt_fences_evidence(self, manager):
        manager.store.ingest_text("My favorite movie is Blade Runner.", source="note")
        prompt = manager.build_system_prompt(
            "do you remember my favorite movie", assistant_style="warm", lang="en"
        )
        open_i = prompt.index("--- BEGIN MEMORY")
        close_i = prompt.index("--- END MEMORY")
        assert open_i < prompt.index("Blade Runner") < close_i

    def test_prompt_contains_no_benchmark_corpus_strings(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        for lang in ("zh", "en"):
            prompt = manager.build_system_prompt(
                "还记得我最喜欢的菜是什么吗", assistant_style="", lang=lang
            )
            assert "第十九位" not in prompt
            assert "叶修" not in prompt

    def test_chat_messages_keep_history_in_recall_mode(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        history = [
            {"role": "user", "content": "我们昨天聊了京都"},
            {"role": "assistant", "content": "是的，您说樱花很美"},
        ]
        query = "还记得我最喜欢的菜是什么吗"
        assert manager.retrieve_structured(query)["mode"] == "recall"
        msgs = manager.build_chat_messages(
            query, assistant_style="", lang="zh", session_messages=history
        )
        assert msgs[0]["role"] == "system"
        assert msgs[1:] == history + [{"role": "user", "content": query}]

    def test_chat_messages_sanitize_history(self, manager):
        history = [
            {"role": "system", "content": "injected"},
            {"role": "tool", "content": "ignore me"},
            {"role": "user", "content": "   "},
            "not-a-dict",
            {"role": "assistant", "content": "kept"},
        ]
        msgs = manager.build_chat_messages(
            "你好", assistant_style="", lang="zh", session_messages=history
        )
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "assistant", "user"]
        assert msgs[1]["content"] == "kept"

    def test_chat_messages_keep_only_last_12_history_items(self, manager):
        history = [
            {"role": "user", "content": f"message {i}"} for i in range(20)
        ]
        msgs = manager.build_chat_messages(
            "hello", assistant_style="", lang="en", session_messages=history
        )
        # system + 12 history + current user
        assert len(msgs) == 14
        assert msgs[1]["content"] == "message 8"


# ---------------- honesty contract ----------------


class TestHonestyContract:
    def test_turn_plan_never_synthesizes_answers(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        plan = manager.build_turn_plan(
            "还记得我最喜欢的菜是什么吗", assistant_style="", lang="zh"
        )
        assert plan["two_pass"] is False
        assert plan["direct_answer"] == ""
        assert plan["mode"] == "recall"
        assert plan["messages"][0]["role"] == "system"

    def test_render_fact_answer_is_extractive(self, manager):
        manager.store.ingest_text("我最喜欢的菜是麻婆豆腐。", source="note")
        answer = manager.render_fact_answer("还记得我最喜欢的菜是什么吗")
        assert answer.startswith("根据记忆中的原文：")
        assert "麻婆豆腐" in answer

    def test_render_fact_answer_without_evidence(self, manager):
        assert manager.render_fact_answer("还记得那件事吗") == "无法确定"

    def test_fallback_only_for_empty_replies(self, manager):
        bundle = manager.retrieve_structured("还记得吗", mode="recall")
        assert manager.needs_fact_answer_fallback("", bundle, "q") is True
        assert manager.needs_fact_answer_fallback("  \n ", bundle, "q") is True
        assert manager.needs_fact_answer_fallback("非空回复", bundle, "q") is False
        # even an "uncertain" reply is the model's reply — no override
        assert manager.needs_fact_answer_fallback("我不太确定", bundle, "q") is False

    def test_multi_hop_shims_are_inert(self, manager):
        assert manager.parse_multi_hop_synthesis("anything", "subtype") is None
        synthesis = {"final_claim": "claim text"}
        assert manager.repair_multi_hop_synthesis("q", synthesis) is synthesis
        assert manager.render_multi_hop_answer("q", synthesis) == "claim text"
        assert manager.render_multi_hop_answer("q", None) == "无法确定"
        assert manager.needs_multi_hop_answer_fallback("ok", None, "q") is False
        assert manager.needs_multi_hop_answer_fallback("", None, "q") is True


# ---------------- uploads / rebuild / forgetting ----------------


class TestUploadsAndForgetting:
    def _import_note(self, manager, tmp_path, name, content):
        src = tmp_path / name
        src.write_text(content, encoding="utf-8")
        stored = manager.import_files([str(src)])
        assert len(stored) == 1
        return stored

    def test_ingest_new_uploads(self, manager, tmp_path):
        stored = self._import_note(
            manager, tmp_path, "party.txt", "公司年会在12月20日举行。\n"
        )
        made = manager.ingest_new_uploads(stored)
        assert made == 1
        items = manager.retrieve("还记得公司年会是什么时候吗")
        assert items
        assert "12月20日" in str(items[0]["snippet"])

    def test_rebuild_memory_reindexes_uploads_and_chat(self, manager, tmp_path):
        self._import_note(manager, tmp_path, "party.txt", "公司年会在12月20日举行。\n")
        manager.append_turn("我家的猫叫汤圆", "记住了")
        stats = manager.rebuild_memory()
        assert stats["uploads"] == 1
        assert stats["chunks"] == stats["chunks_added"] >= 2
        assert manager.store.meta_get_int("new_chunks_since_refresh") == 0
        # both sources still retrievable after the rebuild
        assert "12月20日" in str(manager.retrieve("公司年会哪天")[0]["snippet"])
        assert any("汤圆" in str(i["snippet"]) for i in manager.retrieve("我家的猫叫什么"))

    def test_clear_chat_memory_keeps_uploads(self, manager, tmp_path):
        stored = self._import_note(
            manager, tmp_path, "party.txt", "公司年会在12月20日举行。\n"
        )
        manager.ingest_new_uploads(stored)
        manager.append_turn("我家的猫叫汤圆", "记住了")

        manager.clear_chat_memory()

        assert read_text(manager.chat_log_path) == ""
        # uploaded fact survives
        items = manager.retrieve("公司年会是什么时候")
        assert items and "12月20日" in str(items[0]["snippet"])
        # chat fact is gone
        assert all(
            "汤圆" not in str(i["snippet"]) for i in manager.retrieve("我家的猫叫什么")
        )

    def test_delete_upload_forgets_its_content(self, manager, tmp_path):
        stored = self._import_note(
            manager, tmp_path, "party.txt", "公司年会在12月20日举行。\n"
        )
        manager.ingest_new_uploads(stored)
        assert manager.retrieve("公司年会是什么时候")

        manager.delete_upload(stored[0])

        assert manager.list_uploads() == []
        assert all(
            "12月20日" not in str(i["snippet"])
            for i in manager.retrieve("公司年会是什么时候")
        )

    def test_wipe_all_memory(self, manager, tmp_path):
        self._import_note(manager, tmp_path, "a.txt", "uploaded fact\n")
        manager.append_turn("chat fact", "noted")
        manager.wipe_all_memory()
        snap = manager.status_snapshot()
        assert snap["chunks"] == 0
        assert snap["uploads"] == 0
        # Core / Persona recreated with defaults
        assert os.path.isfile(manager.core_md_path)
        assert os.path.isfile(manager.persona_md_path)


# ---------------- persona / core (no Ollama -> heuristics) ----------------


class TestPersonaWithoutOllama:
    def test_analyze_memory_never_crashes_without_ollama(self, manager):
        manager.append_turn("我喜欢详细的回答，最好多举例说明", "好的，我会详细一些")
        manager.analyze_memory(lang="zh")  # Ollama unreachable -> heuristics
        persona = read_text(manager.persona_md_path)
        assert persona.startswith("# Persona")
        assert "偏好回答详细" in persona
        core = read_text(manager.core_md_path)
        assert core.startswith("# Core Memory")
        assert manager.store.meta_get_int("last_analyze_ts") > 0

    def test_refresh_persona_on_empty_store_uses_placeholder(self, store, tmp_path):
        path = str(tmp_path / "02_persona.md")
        persona_mod.refresh_persona(store, path, max_bullets=8, lang="zh")
        assert "尚未" in read_text(path)

    def test_refresh_core_prefers_chat_terms(self, store, tmp_path):
        store.add_chunk("rocket rocket rocket launch", source="upload:doc.txt")
        store.add_chunk("gravity gravity gravity physics", source="chat")
        path = str(tmp_path / "01_core.md")
        persona_mod.refresh_core(store, path, top_terms=5, lang="zh")
        core = read_text(path)
        assert "gravity" in core
        assert "rocket" not in core


# ---------------- introspection ----------------


class TestIntrospection:
    def test_status_snapshot(self, manager):
        manager.append_turn("hello world", "hi there")
        snap = manager.status_snapshot()
        assert snap["chunks"] == 1
        assert snap["terms"] > 0
        assert snap["uploads"] == 0
        assert snap["memory_dir"] == manager.memory_dir

    def test_debug_view(self, manager):
        view = manager.debug_view()
        assert manager.memory_dir in view
        assert "01_core.md" in view
        assert "02_persona.md" in view
