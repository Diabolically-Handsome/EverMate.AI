"""Tests for engine.retrieval: BM25 retrieval and evidence fragments."""

from __future__ import annotations

import pytest

from engine.retrieval import Retriever, best_evidence_fragment, fragment_score
from engine.textutil import query_flags, tokenize


FACTS = [
    "我们上次去京都旅行是2024年4月3日出发的。",
    "我最喜欢的菜是麻婆豆腐。",
    "My favorite movie is Blade Runner, I rewatch it every year.",
    "昨天的会议讨论了季度预算问题。",
    "I jog in the park every morning before work.",
]


@pytest.fixture
def loaded_store(store):
    for i, fact in enumerate(FACTS):
        store.ingest_text(fact, source=f"note{i}")
    return store


class TestRetrieve:
    def test_trip_date_query_wins_right_chunk(self, loaded_store):
        items = Retriever(loaded_store).retrieve("我们什么时候去京都旅行的？", k=3)
        assert items
        top = items[0]
        assert "京都" in str(top["snippet"])
        assert "4月3日" in str(top["snippet"])

    def test_favorite_dish_query_wins_right_chunk(self, loaded_store):
        items = Retriever(loaded_store).retrieve("我最喜欢的菜是什么？", k=3)
        assert items
        assert "麻婆豆腐" in str(items[0]["snippet"])

    def test_english_fact_query_wins_right_chunk(self, loaded_store):
        items = Retriever(loaded_store).retrieve("what is my favorite movie", k=3)
        assert items
        assert "Blade Runner" in str(items[0]["snippet"])

    def test_result_shape(self, loaded_store):
        items = Retriever(loaded_store).retrieve("京都旅行", k=2)
        assert items
        item = items[0]
        assert {"chunk_id", "score", "source", "created_at", "snippet"} <= set(item)
        assert item["source"].startswith("note")

    def test_k_limits_results(self, loaded_store):
        items = Retriever(loaded_store).retrieve("favorite morning 喜欢", k=2)
        assert len(items) <= 2

    def test_empty_query_returns_empty(self, loaded_store):
        assert Retriever(loaded_store).retrieve("", k=3) == []

    def test_empty_store_returns_empty(self, store):
        assert Retriever(store).retrieve("anything at all", k=3) == []

    def test_unknown_terms_return_empty(self, loaded_store):
        assert Retriever(loaded_store).retrieve("zzzqqqxxx", k=3) == []


class TestFragmentScore:
    def test_empty_fragment_is_minus_inf(self):
        assert fragment_score("", set(), {}) == float("-inf")

    def test_token_overlap_scores(self):
        q = set(tokenize("budget meeting"))
        assert fragment_score("the budget meeting went well", q, {}) >= 2.0

    def test_long_fragments_penalized(self):
        q = {"budget"}
        short = fragment_score("budget plan", q, {})
        long = fragment_score("budget " + "x" * 220, q, {})
        assert short > long

    def test_date_boost_when_asked(self):
        flags = query_flags("什么时候出发")
        assert flags["asks_date"]
        with_date = fragment_score("出发时间定在4月3日", {"出发"}, flags)
        without_date = fragment_score("出发前我们收拾了行李", {"出发"}, flags)
        assert with_date > without_date

    def test_duration_boost_when_asked(self):
        flags = query_flags("会议开了多久")
        assert flags["asks_duration"]
        with_duration = fragment_score("会议开了45分钟", {"会议"}, flags)
        without_duration = fragment_score("会议气氛很好", {"会议"}, flags)
        assert with_duration > without_duration


class TestBestEvidenceFragment:
    def test_picks_dated_sentence_for_date_question(self):
        text = "我们聊了很多。出发时间定在4月3日。天气不错。"
        q_set = set(tokenize("什么时候出发"))
        flags = query_flags("什么时候出发")
        snippet, score = best_evidence_fragment(text, q_set, flags)
        assert snippet == "出发时间定在4月3日。"
        assert score > 0

    def test_long_snippet_truncated(self):
        text = "x" * 400  # no sentence enders
        snippet, _ = best_evidence_fragment(text, {"x"}, {}, max_chars=240)
        assert snippet.endswith("…")
        assert len(snippet) <= 241

    def test_empty_text(self):
        snippet, score = best_evidence_fragment("", set(), {})
        assert snippet == ""
        assert score == 0.0
