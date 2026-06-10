"""Tests for engine.textutil: tokenization, sentence splitting, query analysis."""

from __future__ import annotations

from engine.textutil import (
    conflict_markers,
    is_cjk,
    looks_like_recall_query,
    query_flags,
    split_sentences,
    tokenize,
)


# ---------------- tokenize ----------------


class TestTokenize:
    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize(None) == []

    def test_english_words_lowercased(self):
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_english_stopwords_removed(self):
        tokens = tokenize("the cat and the dog")
        assert "the" not in tokens
        assert "and" not in tokens
        assert tokens == ["cat", "dog"]

    def test_numbers_kept(self):
        assert tokenize("Room 42") == ["room", "42"]

    def test_chinese_bigrams(self):
        # Contiguous CJK runs become character bigrams.
        assert tokenize("麻婆豆腐") == ["麻婆", "婆豆", "豆腐"]

    def test_chinese_stop_bigrams_removed(self):
        tokens = tokenize("我们买了一个苹果")
        assert "我们" not in tokens
        assert "一个" not in tokens
        assert "苹果" in tokens

    def test_single_cjk_char(self):
        assert tokenize("好") == ["好"]
        # Single-char stopword is dropped entirely.
        assert tokenize("的") == []

    def test_kana_bigrams(self):
        assert tokenize("ありがとう") == ["あり", "りが", "がと", "とう"]

    def test_hangul_bigrams(self):
        assert tokenize("김치찌개") == ["김치", "치찌", "찌개"]

    def test_mixed_english_and_cjk(self):
        tokens = tokenize("I like 寿司 sushi")
        assert "like" in tokens
        assert "寿司" in tokens
        assert "sushi" in tokens
        assert "i" not in tokens  # stopword


def test_is_cjk():
    assert is_cjk("漢字")
    assert is_cjk("カナ")
    assert is_cjk("한글")
    assert not is_cjk("abc")
    assert not is_cjk("漢a")
    assert not is_cjk("")


# ---------------- split_sentences ----------------


class TestSplitSentences:
    def test_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_cjk_enders_split_without_whitespace(self):
        assert split_sentences("今天很好。明天下雨！后天呢？") == [
            "今天很好。",
            "明天下雨！",
            "后天呢？",
        ]

    def test_cjk_semicolon(self):
        assert split_sentences("第一；第二") == ["第一；", "第二"]

    def test_ascii_enders_require_trailing_whitespace(self):
        assert split_sentences("Hello world. This is fine! Done? Yes.") == [
            "Hello world.",
            "This is fine!",
            "Done?",
            "Yes.",
        ]

    def test_decimals_survive(self):
        # "3.14" must not be split because the dot has no trailing whitespace.
        assert split_sentences("Pi is about 3.14 and tau is 6.28.") == [
            "Pi is about 3.14 and tau is 6.28."
        ]

    def test_newlines_split(self):
        assert split_sentences("line one\nline two") == ["line one", "line two"]


# ---------------- query_flags ----------------


class TestQueryFlags:
    def test_asks_date(self):
        assert query_flags("我们什么时候去的北京")["asks_date"]
        assert query_flags("when did we meet")["asks_date"]
        assert query_flags("what year did we meet")["asks_date"]

    def test_asks_duration(self):
        assert query_flags("我们走了多久")["asks_duration"]
        assert query_flags("how long did the meeting take")["asks_duration"]

    def test_asks_count(self):
        assert query_flags("买了多少个苹果")["asks_count"]
        assert query_flags("how many apples did I buy")["asks_count"]

    def test_asks_name(self):
        assert query_flags("我家的猫叫什么")["asks_name"]
        assert query_flags("who is my manager")["asks_name"]

    def test_asks_place(self):
        assert query_flags("我们在哪里吃的饭")["asks_place"]
        assert query_flags("where did we eat")["asks_place"]

    def test_neutral_query_has_no_flags(self):
        flags = query_flags("今天天气真好")
        assert not any(flags.values())

    def test_empty_query(self):
        flags = query_flags("")
        assert not any(flags.values())


# ---------------- looks_like_recall_query ----------------


class TestLooksLikeRecallQuery:
    def test_explicit_memory_reference_zh(self):
        assert looks_like_recall_query("还记得我们上次去哪了吗")
        assert looks_like_recall_query("我之前说过什么")

    def test_explicit_memory_reference_en(self):
        assert looks_like_recall_query("do you remember my birthday")
        assert looks_like_recall_query("did I mention my trip last time")

    def test_fact_shaped_question_counts(self):
        assert looks_like_recall_query("我家的猫叫什么")
        assert looks_like_recall_query("where did we eat")

    def test_smalltalk_is_not_recall(self):
        assert not looks_like_recall_query("你好呀")
        assert not looks_like_recall_query("hello there")
        assert not looks_like_recall_query("")


# ---------------- conflict_markers ----------------


class TestConflictMarkers:
    def test_two_dates_flag_date_dimension(self):
        markers = conflict_markers(["我们3月5日出发", "出发日期是4月2日"])
        assert markers == ["日期"]

    def test_many_numbers_flag_numeric_dimension(self):
        markers = conflict_markers(["速度是3和5", "另一边是7和9"])
        assert markers == ["数值"]

    def test_single_date_is_fine(self):
        assert conflict_markers(["我们3月5日出发了"]) == []

    def test_empty_inputs(self):
        assert conflict_markers([]) == []
        assert conflict_markers(["", None and "" or ""]) == []
