"""BM25 retrieval over the inverted index, with best-sentence evidence.

Corpus-agnostic on purpose: scores come from term statistics and light,
generic signals (dates / numbers when the question asks for them) — never
from hardcoded entities or expected answers.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Tuple

from engine.storage import MemoryStore, read_text
from engine.textutil import query_flags, split_sentences, tokenize

_DURATION_RE = re.compile(r"(\d+\s*分钟|[一二三四五六七八九十两半]+\s*分钟|\d+\s*(minutes?|hours?|小时))")
_DATE_TEXT_RE = re.compile(r"\d+\s*月\s*\d+\s*日|\d{4}-\d{2}-\d{2}")

BM25_K1 = 1.5
BM25_B = 0.75


def _idf(df: int, n_docs: int) -> float:
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def fragment_score(fragment: str, query_token_set: set, flags: Dict[str, bool]) -> float:
    """Generic sentence relevance: token overlap + answer-shape hints."""

    frag = (fragment or "").strip()
    if not frag:
        return float("-inf")

    score = float(len(query_token_set & set(tokenize(frag))))

    if flags.get("asks_duration"):
        score += 4.0 if _DURATION_RE.search(frag) else -1.0
    if flags.get("asks_date"):
        score += 4.0 if _DATE_TEXT_RE.search(frag) else -1.0
    if flags.get("asks_count") and re.search(r"\d", frag):
        score += 2.0

    # Long fragments dilute the evidence; prefer focused sentences.
    if len(frag) > 160:
        score -= min(8.0, (len(frag) - 160) / 40.0)
    return score


def best_evidence_fragment(
    text: str,
    query_token_set: set,
    flags: Dict[str, bool],
    max_chars: int = 240,
) -> Tuple[str, float]:
    """The most query-relevant sentence inside a chunk."""

    fragments = split_sentences(text)
    if not fragments:
        s = text.strip().replace("\n", " ")
        return (s[:max_chars] + "…") if len(s) > max_chars else s, 0.0

    best = max(fragments, key=lambda f: (fragment_score(f, query_token_set, flags), -len(f)))
    score = fragment_score(best, query_token_set, flags)
    best = best.replace("\n", " ").strip()
    if len(best) > max_chars:
        best = best[:max_chars] + "…"
    return best, score


class Retriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, object]]:
        """Top-k chunks by BM25 with a best-sentence snippet each."""

        q_terms = tokenize(query)
        if not q_terms:
            return []
        q_set = set(q_terms)
        flags = query_flags(query)

        n_docs, avgdl = self.store.corpus_stats()
        if n_docs == 0:
            return []

        cur = self.store.conn.cursor()
        scores: Dict[int, float] = {}
        for term in q_set:
            row_df = cur.execute("SELECT df FROM terms WHERE term=?", (term,)).fetchone()
            if not row_df:
                continue
            df = int(row_df["df"])
            if df <= 0:
                continue
            idf = _idf(df, n_docs)
            rows = cur.execute(
                """
                SELECT p.chunk_id AS chunk_id, p.tf AS tf, c.doc_len AS dl
                FROM postings p JOIN chunks c ON c.id = p.chunk_id
                WHERE p.term = ?
                """,
                (term,),
            ).fetchall()
            for r in rows:
                cid = int(r["chunk_id"])
                tf = int(r["tf"])
                dl = int(r["dl"] or 1)
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
                scores[cid] = scores.get(cid, 0.0) + idf * (tf * (BM25_K1 + 1)) / (denom + 1e-9)

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(20, k)]

        out: List[Dict[str, object]] = []
        for cid, sc in top:
            row = self.store.chunk_row_by_id(cid)
            if not row:
                continue
            abs_path = os.path.join(self.store.memory_dir, str(row["path"]))
            if not os.path.exists(abs_path):
                continue
            text = read_text(abs_path)
            snippet, snippet_score = best_evidence_fragment(text, q_set, flags)
            out.append(
                {
                    "chunk_id": cid,
                    "score": float(sc) + 0.35 * float(snippet_score),
                    "source": str(row["source"] or ""),
                    "created_at": int(row["created_at"] or 0),
                    "snippet": snippet,
                }
            )

        out.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)

        # Prefer distinct snippets among the winners; fall back to duplicates
        # only when there are not enough unique ones.
        selected: List[Dict[str, object]] = []
        deferred: List[Dict[str, object]] = []
        seen = set()
        for item in out:
            sig = str(item.get("snippet", ""))[:80]
            if sig in seen:
                deferred.append(item)
                continue
            seen.add(sig)
            selected.append(item)
            if len(selected) >= k:
                break
        for item in deferred:
            if len(selected) >= k:
                break
            selected.append(item)
        return selected[:k]
