"""Pure text utilities: tokenization, sentence splitting, query analysis.

Everything here is corpus-agnostic by design. Heuristics tied to a specific
test corpus do not belong in this package (see legacy_quanzhi_heuristics.py
for the cautionary tale).
"""

from __future__ import annotations

import re
from typing import Dict, List

# Minimal stopword lists, kept small on purpose.
EN_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that",
    "i", "you", "we", "they", "he", "she",
}

ZH_STOP = {
    "的", "了", "呢", "吗", "啊", "吧", "和", "与", "及", "在", "对", "把", "被",
    "一个", "我们", "你们", "他们", "她们", "它们", "我", "你", "您",
}

# Han + Hiragana/Katakana + Hangul. The bigram strategy works for all of
# them; the old Han-only pattern silently dropped Japanese and Korean text.
_CJK_RANGES = r"぀-ヿ㐀-䶿一-鿿가-힯"
_TOKEN_RE = re.compile(rf"[a-z0-9]+|[{_CJK_RANGES}]+")
_CJK_ONLY_RE = re.compile(rf"[{_CJK_RANGES}]+")

# CJK sentence enders split unconditionally; ASCII enders need trailing
# whitespace so decimals and abbreviations survive.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?;])\s+|\n+")


def is_cjk(token: str) -> bool:
    return bool(_CJK_ONLY_RE.fullmatch(token))


def tokenize(text: str) -> List[str]:
    """Tokens for indexing / retrieval.

    - English: lowercased alphanumeric words
    - CJK (Han/Kana/Hangul): character bigrams over contiguous runs

    Not real segmentation, but good enough for BM25 matching without extra
    dependencies.
    """

    if not text:
        return []

    out: List[str] = []
    for part in _TOKEN_RE.findall(text.lower()):
        if is_cjk(part):
            if len(part) == 1:
                if part not in ZH_STOP:
                    out.append(part)
            else:
                for i in range(len(part) - 1):
                    bigram = part[i : i + 2]
                    if bigram not in ZH_STOP:
                        out.append(bigram)
        elif part not in EN_STOP:
            out.append(part)
    return out


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    raw = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in raw if s and s.strip()]


def query_flags(query: str) -> Dict[str, bool]:
    """Generic, language-level signals about what kind of answer is wanted."""

    q = (query or "").strip()
    lowered = q.lower()
    return {
        "asks_date": bool(
            re.search(r"什么时候|哪一天|哪天|几月|几号|哪[一]?年", q)
            or re.search(r"\bwhen\b|\bwhat (day|date|year)\b", lowered)
        ),
        "asks_duration": bool(
            re.search(r"多久|多长时间|几分钟|几小时|几天", q)
            or re.search(r"\bhow long\b", lowered)
        ),
        "asks_count": bool(
            re.search(r"多少|几个|几次|几位|第几", q)
            or re.search(r"\bhow (many|much)\b", lowered)
        ),
        "asks_name": bool(
            re.search(r"叫什么|是什么名|是谁|哪个人|谁的", q)
            or re.search(r"\bwho\b|\bwhat('s| is) the name\b", lowered)
        ),
        "asks_place": bool(
            re.search(r"在哪|哪里|什么地方", q) or re.search(r"\bwhere\b", lowered)
        ),
    }


_RECALL_PATTERNS = re.compile(
    r"还记得|记得|上次|上回|之前|那次|那天|当时|昨天.{0,12}[吗么?？]"
    r"|我说过|我提过|我提到过|我告诉过你|我们聊过|我们说过|我们谈过"
    r"|聊了什么|说了什么|谈了什么|讲了什么|说过什么"
    r"|你知道我|你了解我|我的(名字|生日|爱好|偏好|习惯)"
    r"|do you remember|did (i|we) (say|mention|tell|talk|discuss)"
    r"|what did (i|we) (say|talk|discuss|mention)|last time|previously"
    r"|remind me (what|about)|you know (me|my)|my (name|birthday|favorite|preference)",
    re.IGNORECASE,
)


def looks_like_recall_query(query: str) -> bool:
    """Should this turn lean on long-term memory retrieval?

    Either an explicit memory reference ("还记得/上次/我说过…") or a
    fact-shaped question (who/when/where/how many) counts.
    """

    q = (query or "").strip()
    if not q:
        return False
    if _RECALL_PATTERNS.search(q):
        return True
    flags = query_flags(q)
    return any(flags.values())


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_DATE_RE = re.compile(r"\d{1,4}\s*年|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}-\d{2}-\d{2}")


def conflict_markers(snippets: List[str]) -> List[str]:
    """Light-touch inconsistency hints across evidence snippets.

    Only flags dimensions a reader can verify (diverging numbers / dates);
    the system prompt asks the model to stay conservative when these fire.
    """

    markers: List[str] = []
    numbers = set()
    dates = set()
    for s in snippets:
        if not s:
            continue
        dates.update(m.group(0) for m in _DATE_RE.finditer(s))
        # Date components are not standalone numbers; strip them first so a
        # single "2023年5月" doesn't read as three conflicting values.
        without_dates = _DATE_RE.sub(" ", s)
        numbers.update(_NUM_RE.findall(without_dates))
    if len(dates) >= 2:
        markers.append("日期")
    elif len(numbers) >= 4:
        markers.append("数值")
    return markers
