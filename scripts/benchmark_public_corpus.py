#!/usr/bin/env python3
"""Deterministic memory benchmark on a public-domain corpus.

Honesty-first design:
- Question generation is deterministic (seeded) and derived only from the
  corpus itself — no LLM in the loop, no hand-written question bank.
- Scoring is exact string containment of the blanked-out target in the
  model's reply. No LLM judge.
- Retrieval quality and end-to-end accuracy are reported separately, so a
  weak answering model cannot hide retrieval wins (or vice versa).

Each question is a cloze: a sentence sampled from the corpus with one
distinctive term blanked out. The system must retrieve the source passage
via BM25 and the local model must fill in the blank from the injected
evidence.

Example:
    python scripts/benchmark_public_corpus.py \
        --txt ~/corpora/hongloumeng.txt --lang zh --model gpt-oss:20b
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.manager import MemoryConfig, MemoryManager
from engine.textutil import split_sentences
from ollama_client import chat as ollama_chat

GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

HAN_RUN = re.compile(r"[㐀-䶿一-鿿]{2,}")
EN_WORD = re.compile(r"[A-Za-z][a-z]+")

ZH_TEMPLATE = (
    "我之前导入的文档里有这样一句话：『{blanked}』。"
    "空缺处（____）的原文是什么？请只回答空缺的内容，不要解释。"
)
EN_TEMPLATE = (
    'A document I imported earlier contains this sentence: "{blanked}". '
    "What is the missing word at the blank (____)? Reply with only the missing word."
)

BLANK = "____"


@dataclass(frozen=True)
class ClozeCase:
    case_id: int
    chunk_id: int
    sentence: str
    blanked: str
    answer: str
    question: str


@dataclass
class CaseResult:
    case_id: int
    chunk_id: int
    answer: str
    retrieval_hit: bool
    evidence_has_answer: bool
    model_reply: str = ""
    correct: Optional[bool] = None
    seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--txt", required=True, help="Path to a UTF-8 .txt corpus (public domain).")
    p.add_argument("--lang", required=True, choices=["zh", "en"])
    p.add_argument("--model", default="gpt-oss:20b", help="Local Ollama model that answers.")
    p.add_argument("--questions", type=int, default=60, help="End-to-end questions (with LLM).")
    p.add_argument("--retrieval-probes", type=int, default=200, help="Retrieval-only probes (no LLM).")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--memory-dir", default="", help="Isolated memory dir (default: temp).")
    p.add_argument("--report-prefix", default="", help="Output path prefix (default: reports/<corpus>-<lang>).")
    p.add_argument("--answer-timeout", type=int, default=300)
    p.add_argument("--keep-memory-dir", action="store_true")
    return p.parse_args()


def load_corpus(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m = GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = GUTENBERG_END.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def load_chunks(mm: MemoryManager) -> List[Tuple[int, str]]:
    rows = mm.conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()
    return [(int(r["id"]), mm.store.chunk_text_by_id(int(r["id"]))) for r in rows]


def term_df(mm: MemoryManager, term: str) -> int:
    row = mm.conn.execute("SELECT df FROM terms WHERE term=?", (term.lower(),)).fetchone()
    return int(row["df"]) if row else 0


class ZhCohesion:
    """PMI over adjacent Han pairs, computed from the corpus itself.

    Cross-word bigrams ("們了") are frequent characters that rarely pair, so
    their PMI is low; real lexical units and names ("寶玉") pair far more
    often than chance. This keeps cloze targets natural without any
    hand-curated word list.
    """

    def __init__(self, text: str):
        import math
        from collections import Counter

        self._math = math
        self.char_counts: Counter = Counter()
        self.pair_counts: Counter = Counter()
        for run in HAN_RUN.findall(text):
            self.char_counts.update(run)
            for i in range(len(run) - 1):
                self.pair_counts[run[i : i + 2]] += 1
        self.total_pairs = max(1, sum(self.pair_counts.values()))

    def pmi(self, bigram: str) -> float:
        c_pair = self.pair_counts.get(bigram, 0)
        c_a = self.char_counts.get(bigram[0], 0)
        c_b = self.char_counts.get(bigram[1], 0)
        if not (c_pair and c_a and c_b):
            return float("-inf")
        return self._math.log(c_pair * self.total_pairs / (c_a * c_b))


def candidate_targets_zh(
    mm: MemoryManager, sentence: str, n_chunks: int, cohesion: ZhCohesion, min_pmi: float = 3.0
) -> List[str]:
    """Distinctive, cohesive 2-char terms: in several chunks, not ubiquitous,
    and PMI-bound so the blank covers a natural unit (usually a name)."""

    lo, hi = 5, max(6, int(n_chunks * 0.15))
    out: List[str] = []
    for run in HAN_RUN.findall(sentence):
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if sentence.count(bigram) != 1:
                continue
            if not (lo <= term_df(mm, bigram) <= hi):
                continue
            if cohesion.pmi(bigram) >= min_pmi:
                out.append(bigram)
    return out


def candidate_targets_en(mm: MemoryManager, sentence: str, n_chunks: int) -> List[str]:
    """Distinctive capitalized words (not sentence-initial) or rare-ish nouns."""

    lo, hi = 3, max(4, int(n_chunks * 0.15))
    out: List[str] = []
    words = re.findall(r"\b[A-Za-z][A-Za-z'’-]*\b", sentence)
    for idx, word in enumerate(words):
        if idx == 0 or len(word) < 4:
            continue
        if not word[0].isupper():
            continue
        if sentence.count(word) != 1:
            continue
        if lo <= term_df(mm, word) <= hi:
            out.append(word)
    return out


def build_cases(
    mm: MemoryManager,
    chunks: List[Tuple[int, str]],
    lang: str,
    count: int,
    seed: int,
    corpus_text: str = "",
) -> List[ClozeCase]:
    rng = random.Random(seed)
    n_chunks = len(chunks)
    min_len, max_len = (20, 120) if lang == "zh" else (60, 220)
    template = ZH_TEMPLATE if lang == "zh" else EN_TEMPLATE
    if lang == "zh":
        cohesion = ZhCohesion(corpus_text)

        def pick_targets(mm_, s_, n_):
            return candidate_targets_zh(mm_, s_, n_, cohesion)

    else:
        pick_targets = candidate_targets_en

    pool: List[ClozeCase] = []
    for chunk_id, text in chunks:
        for sentence in split_sentences(text):
            s = sentence.strip()
            if not (min_len <= len(s) <= max_len):
                continue
            if "Gutenberg" in s or "第" == s[:1] and "回" in s[:8]:
                continue
            targets = pick_targets(mm, s, n_chunks)
            if not targets:
                continue
            target = rng.choice(targets)
            blanked = s.replace(target, BLANK, 1)
            pool.append(
                ClozeCase(
                    case_id=0,
                    chunk_id=chunk_id,
                    sentence=s,
                    blanked=blanked,
                    answer=target,
                    question=template.format(blanked=blanked),
                )
            )

    if not pool:
        raise SystemExit("No cloze candidates found — corpus too small or wrong --lang?")

    # Even coverage across the corpus: bucket by chunk order, one per bucket.
    pool.sort(key=lambda c: c.chunk_id)
    chosen: List[ClozeCase] = []
    step = max(1, len(pool) // count)
    for i in range(0, len(pool), step):
        bucket = pool[i : i + step]
        chosen.append(rng.choice(bucket))
        if len(chosen) >= count:
            break
    return [ClozeCase(case_id=i + 1, **{k: v for k, v in asdict(c).items() if k != "case_id"}) for i, c in enumerate(chosen)]


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def run_case(
    mm: MemoryManager,
    case: ClozeCase,
    lang: str,
    model: str,
    top_k: int,
    timeout: int,
    with_llm: bool,
) -> CaseResult:
    t0 = time.time()
    bundle = mm.retrieve_structured(case.question, mode="recall", k=top_k)
    items = list(bundle.get("items", []))
    retrieval_hit = any(int(i.get("chunk_id", -1)) == case.chunk_id for i in items)
    evidence_has_answer = any(case.answer in str(i.get("snippet", "")) for i in items)

    result = CaseResult(
        case_id=case.case_id,
        chunk_id=case.chunk_id,
        answer=case.answer,
        retrieval_hit=retrieval_hit,
        evidence_has_answer=evidence_has_answer,
    )
    if with_llm:
        messages = mm.build_chat_messages(
            user_text=case.question,
            assistant_style="",
            lang=lang,
            answer_mode="recall",
            retrieved_bundle=bundle,
        )
        reply = ollama_chat(messages, model=model, options={"temperature": 0}, timeout=timeout)
        result.model_reply = reply.strip()
        result.correct = normalize(case.answer) in normalize(reply)
    result.seconds = time.time() - t0
    return result


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.2f}%" if den else "n/a"


def main() -> None:
    args = parse_args()
    corpus_name = Path(args.txt).stem
    memory_dir = args.memory_dir or f"/tmp/evermate-bench-{corpus_name}-{args.lang}"
    report_prefix = args.report_prefix or str(ROOT / "reports" / f"{corpus_name}-{args.lang}")
    Path(report_prefix).parent.mkdir(parents=True, exist_ok=True)

    shutil.rmtree(memory_dir, ignore_errors=True)
    mm = MemoryManager(MemoryConfig(memory_dir=memory_dir))
    try:
        text = load_corpus(args.txt)
        print(f"[1/4] Indexing {len(text):,} chars from {args.txt} ...")
        t0 = time.time()
        n_chunks = mm.store.ingest_text(text, source=f"upload:{corpus_name}")
        index_seconds = time.time() - t0
        print(f"      {n_chunks} chunks, {mm.count_terms():,} terms in {index_seconds:.1f}s")

        chunks = load_chunks(mm)
        probes_wanted = max(args.retrieval_probes, args.questions)
        print(f"[2/4] Generating {probes_wanted} deterministic cloze cases (seed={args.seed}) ...")
        cases = build_cases(mm, chunks, args.lang, probes_wanted, args.seed, corpus_text=text)
        print(f"      {len(cases)} cases across chunks {cases[0].chunk_id}..{cases[-1].chunk_id}")

        print(f"[3/4] Retrieval probes (top-{args.top_k}, no LLM) ...")
        probe_results = [
            run_case(mm, c, args.lang, args.model, args.top_k, args.answer_timeout, with_llm=False)
            for c in cases
        ]
        hits = sum(1 for r in probe_results if r.retrieval_hit)
        ev_hits = sum(1 for r in probe_results if r.evidence_has_answer)

        rng = random.Random(args.seed + 1)
        llm_cases = cases if len(cases) <= args.questions else sorted(
            rng.sample(cases, args.questions), key=lambda c: c.case_id
        )
        print(f"[4/4] End-to-end answering with {args.model} on {len(llm_cases)} cases ...")
        llm_results: List[CaseResult] = []
        for i, case in enumerate(llm_cases, 1):
            r = run_case(mm, case, args.lang, args.model, args.top_k, args.answer_timeout, with_llm=True)
            llm_results.append(r)
            mark = "✓" if r.correct else "✗"
            print(f"      {i:3d}/{len(llm_cases)} {mark} expect={case.answer!r} got={truncate(r.model_reply, 40)!r} ({r.seconds:.1f}s)")
        correct = sum(1 for r in llm_results if r.correct)

        summary = {
            "corpus": str(args.txt),
            "corpus_chars": len(text),
            "lang": args.lang,
            "chunks": n_chunks,
            "terms": mm.count_terms(),
            "index_seconds": round(index_seconds, 2),
            "model": args.model,
            "seed": args.seed,
            "top_k": args.top_k,
            "retrieval": {
                "probes": len(probe_results),
                "chunk_hit_rate": pct(hits, len(probe_results)),
                "evidence_contains_answer_rate": pct(ev_hits, len(probe_results)),
            },
            "end_to_end": {
                "questions": len(llm_results),
                "accuracy": pct(correct, len(llm_results)),
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(f"{report_prefix}-results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "cases": [asdict(c) for c in cases],
                    "probe_results": [asdict(r) for r in probe_results],
                    "llm_results": [asdict(r) for r in llm_results],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        lines = [
            f"# Public-corpus benchmark — {corpus_name} ({args.lang})",
            "",
            f"- Corpus: {args.txt} ({len(text):,} chars)",
            f"- Index: {n_chunks} chunks / {mm.count_terms():,} terms / {index_seconds:.1f}s",
            f"- Answer model: {args.model} (temperature 0)",
            f"- Question generation: deterministic seeded cloze (seed={args.seed}), no LLM, no hand-written bank",
            f"- Scoring: exact string containment, no LLM judge",
            "",
            f"## Retrieval (top-{args.top_k}, {len(probe_results)} probes)",
            f"- Source-chunk hit rate: **{pct(hits, len(probe_results))}**",
            f"- Evidence snippet contains answer: **{pct(ev_hits, len(probe_results))}**",
            "",
            f"## End-to-end ({len(llm_results)} questions)",
            f"- Accuracy: **{pct(correct, len(llm_results))}**",
            "",
            "## Reproduce",
            "```bash",
            f"python scripts/benchmark_public_corpus.py --txt {args.txt} --lang {args.lang} \\",
            f"    --model {args.model} --questions {args.questions} --retrieval-probes {args.retrieval_probes} --seed {args.seed}",
            "```",
        ]
        with open(f"{report_prefix}-report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print("\n=== SUMMARY ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nReports: {report_prefix}-report.md / -results.json")
    finally:
        mm.close()
        if not args.keep_memory_dir:
            shutil.rmtree(memory_dir, ignore_errors=True)


def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


if __name__ == "__main__":
    main()
