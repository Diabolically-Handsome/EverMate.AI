"""Tests for ollama_client. No real Ollama server is contacted: HTTP is
either monkeypatched or pointed at a closed loopback port."""

from __future__ import annotations

import json

import pytest
import requests

import ollama_client
from ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    ThinkTagFilter,
    chat,
    chat_stream,
    clean_response_text,
    list_models,
)


# ---------------- clean_response_text ----------------


class TestCleanResponseText:
    def test_plain_text_untouched(self):
        assert clean_response_text("Hello world") == "Hello world"

    def test_closed_think_block_stripped(self):
        assert clean_response_text("<think>reasoning</think>Hello") == "Hello"

    def test_multiple_think_blocks_stripped(self):
        assert clean_response_text("<think>a</think>one <think>b</think>two") == "one two"

    def test_unclosed_think_yields_empty(self):
        assert clean_response_text("<think>still reasoning, cut off") == ""

    def test_text_before_unclosed_think_kept(self):
        assert clean_response_text("Answer. <think>then cut off") == "Answer."

    def test_case_insensitive(self):
        assert clean_response_text("<THINK>x</THINK>ok") == "ok"

    def test_gpt_oss_channel_artifacts_stripped(self):
        assert clean_response_text("<|channel|>thought final answer") == "final answer"
        assert clean_response_text("<|channel>thought final answer") == "final answer"

    def test_empty_and_none(self):
        assert clean_response_text("") == ""
        assert clean_response_text(None) == ""


# ---------------- ThinkTagFilter ----------------


class TestThinkTagFilter:
    def test_passthrough(self):
        f = ThinkTagFilter()
        assert f.feed("plain text") == "plain text"
        assert f.flush() == ""

    def test_tag_split_across_deltas(self):
        f = ThinkTagFilter()
        out = f.feed("Hello <thi")
        out += f.feed("nk>secret</thi")
        out += f.feed("nk> world")
        out += f.flush()
        assert out == "Hello  world"

    def test_multiple_blocks_in_one_delta(self):
        f = ThinkTagFilter()
        assert f.feed("a<think>x</think>b<think>y</think>c") == "abc"

    def test_flush_discards_unclosed_think(self):
        f = ThinkTagFilter()
        assert f.feed("Hello <think>secret reasoning") == "Hello "
        assert f.flush() == ""

    def test_flush_emits_pending_partial_tag(self):
        # A suffix that *could* be an opening tag is withheld during feed()
        # but emitted on flush since the stream ended without completing it.
        f = ThinkTagFilter()
        assert f.feed("abc<thi") == "abc"
        assert f.flush() == "<thi"

    def test_case_insensitive_tags(self):
        f = ThinkTagFilter()
        assert f.feed("<THINK>x</THINK>done") == "done"


# ---------------- HTTP helpers ----------------


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class DummyStreamResponse:
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = lines
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        return iter(self._lines)

    def json(self):
        return {}


# ---------------- chat ----------------


class TestChat:
    def test_payload_includes_num_ctx_and_keep_alive(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["url"] = url
            captured["payload"] = json
            return DummyResponse(200, {"message": {"content": "<think>x</think>hi"}})

        monkeypatch.setattr(requests, "post", fake_post)
        out = chat(
            [{"role": "user", "content": "hello"}],
            model="m1",
            options={"temperature": 0.2},
        )
        assert out == "hi"  # think block cleaned
        assert captured["url"].endswith("/api/chat")
        payload = captured["payload"]
        assert payload["model"] == "m1"
        assert payload["stream"] is False
        assert payload["keep_alive"] == ollama_client.DEFAULT_KEEP_ALIVE
        assert payload["options"]["num_ctx"] == ollama_client.DEFAULT_NUM_CTX
        assert payload["options"]["temperature"] == 0.2

    def test_options_can_override_num_ctx(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, **kwargs):
            captured["payload"] = json
            return DummyResponse(200, {"message": {"content": "ok"}})

        monkeypatch.setattr(requests, "post", fake_post)
        chat([{"role": "user", "content": "hi"}], options={"num_ctx": 4096})
        assert captured["payload"]["options"]["num_ctx"] == 4096

    def test_model_not_found_raises_typed_error(self, monkeypatch):
        def fake_post(url, json=None, timeout=None, **kwargs):
            return DummyResponse(404, {"error": "model 'foo:1b' not found"})

        monkeypatch.setattr(requests, "post", fake_post)
        with pytest.raises(OllamaModelNotFoundError) as ei:
            chat([{"role": "user", "content": "hi"}], model="foo:1b")
        assert ei.value.model == "foo:1b"

    def test_connection_error_mapped(self, monkeypatch):
        def fake_post(url, json=None, timeout=None, **kwargs):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", fake_post)
        with pytest.raises(OllamaConnectionError):
            chat([{"role": "user", "content": "hi"}])

    def test_malformed_response_raises_ollama_error(self, monkeypatch):
        def fake_post(url, json=None, timeout=None, **kwargs):
            return DummyResponse(200, {"unexpected": "shape"})

        monkeypatch.setattr(requests, "post", fake_post)
        with pytest.raises(OllamaError):
            chat([{"role": "user", "content": "hi"}])


# ---------------- chat_stream ----------------


class TestChatStream:
    @staticmethod
    def _lines(*contents, done=True):
        lines = [
            json.dumps({"message": {"content": c}}).encode("utf-8") for c in contents
        ]
        if done:
            lines.append(json.dumps({"done": True}).encode("utf-8"))
        return lines

    def test_stream_payload_and_think_filtering(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None, stream=False, **kwargs):
            captured["payload"] = json
            captured["stream_kw"] = stream
            return DummyStreamResponse(
                self._lines("<th", "ink>secret</think>He", "llo")
            )

        monkeypatch.setattr(requests, "post", fake_post)
        chunks = list(chat_stream([{"role": "user", "content": "hi"}], model="m1"))
        assert "".join(chunks) == "Hello"
        assert captured["stream_kw"] is True
        payload = captured["payload"]
        assert payload["stream"] is True
        assert payload["model"] == "m1"
        assert payload["keep_alive"] == ollama_client.DEFAULT_KEEP_ALIVE
        assert payload["options"]["num_ctx"] == ollama_client.DEFAULT_NUM_CTX

    def test_unclosed_think_discarded_at_stream_end(self, monkeypatch):
        def fake_post(url, json=None, timeout=None, stream=False, **kwargs):
            return DummyStreamResponse(
                self._lines("Answer ", "<think>cut off mid-reason")
            )

        monkeypatch.setattr(requests, "post", fake_post)
        chunks = list(chat_stream([{"role": "user", "content": "hi"}]))
        assert "".join(chunks) == "Answer "

    def test_stream_error_line_raises(self, monkeypatch):
        error_line = json.dumps({"error": "boom"}).encode("utf-8")

        def fake_post(url, json=None, timeout=None, stream=False, **kwargs):
            return DummyStreamResponse([error_line])

        monkeypatch.setattr(requests, "post", fake_post)
        with pytest.raises(OllamaError):
            list(chat_stream([{"role": "user", "content": "hi"}]))


# ---------------- list_models ----------------


class TestListModels:
    def test_parses_model_names(self, monkeypatch):
        def fake_get(url, timeout=None, **kwargs):
            return DummyResponse(
                200, {"models": [{"name": "qwen3:30b"}, {"name": ""}, {}]}
            )

        monkeypatch.setattr(requests, "get", fake_get)
        assert list_models() == ["qwen3:30b"]

    def test_unreachable_server_raises_connection_error(self):
        # Port 1 on loopback is closed: the connection is refused immediately,
        # no real network involved.
        with pytest.raises(OllamaConnectionError):
            list_models(url="http://127.0.0.1:1", timeout=1.0)
