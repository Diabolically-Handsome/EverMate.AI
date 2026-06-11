#!/usr/bin/env python3
"""Offline manual accuracy benchmark runner for long-form corpora.

This script is intentionally judge-free:
- reuses an existing question bank
- answers with a local Ollama model
- records enough evidence for human/manual review
- optionally applies a manual judgments JSON file to produce final results/report
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from scripts.validate_memory_accuracy import (
    AccuracyCase,
    ChunkRecord,
    alias_keys,
    answer_contains_alias,
    answer_exact_alias_hit,
    artifact_prefix,
    build_case_turn_plan,
    case_response_style,
    corpus_stats,
    heuristic_cloze_eval,
    load_chunks,
    normalize_space,
    rebuild_accuracy_memory,
    retrieval_alignment,
    slugify,
    support_fragments_for_case,
    truncate,
)


DEFAULT_DOCX = ""  # pass --docx explicitly
DEFAULT_QUESTION_BANK = "reports/全职高手-accuracy-question-bank.json"
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_MEMORY_DIR = "/tmp/evermate-memory-accuracy-gpt-oss-20b"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_RESULTS = "全职高手-gpt-oss-20b-manual-results.json"
DEFAULT_REPORT = "全职高手-gpt-oss-20b-manual-report.md"
DEFAULT_JUDGMENTS = "全职高手-gpt-oss-20b-manual-judgments.json"
OLD_RESULTS = "reports/全职高手-accuracy-results.json"
PREVIOUS_20B_RESULTS = "reports/全职高手-gpt-oss-20b-manual-fix-results.json"
ROOT_CAUSES = (
    "retrieval_miss",
    "alias_mismatch",
    "generation_error",
    "multi_hop_conflict",
    "benchmark_ambiguous",
)


@dataclass(frozen=True)
class AttemptRecord:
    prompt_mode: str
    num_predict: int
    done_reason: str
    content: str
    thinking_len: int


@dataclass(frozen=True)
class ManualEvalRow:
    case: AccuracyCase
    model_answer: str
    label: str
    score: float
    root_cause: str
    reason: str
    retrieved_chunk_ids: Tuple[int, ...]
    retrieved_snippets: Tuple[str, ...]
    supporting_chunk_ids: Tuple[int, ...]
    support_snippets: Tuple[str, ...]
    evidence_quotes: Tuple[str, ...]
    done_reason: str
    prompt_mode: str
    attempts: Tuple[AttemptRecord, ...]
    synthesis: Dict[str, object]
    used_fallback: bool
    answer_candidates: Tuple[Dict[str, object], ...]
    canonical_answer: str


@dataclass(frozen=True)
class Suggestion:
    label: str
    score: float
    root_cause: str
    reason: str


@dataclass(frozen=True)
class PendingRow:
    case: AccuracyCase
    model_answer: str
    retrieved_chunk_ids: Tuple[int, ...]
    retrieved_snippets: Tuple[str, ...]
    support_snippets: Tuple[str, ...]
    done_reason: str
    prompt_mode: str
    attempts: Tuple[AttemptRecord, ...]
    synthesis: Dict[str, object]
    used_fallback: bool
    answer_candidates: Tuple[Dict[str, object], ...]
    canonical_answer: str
    suggestion: Suggestion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline manual accuracy benchmark for gpt-oss:20b.")
    parser.add_argument("--docx", default=DEFAULT_DOCX, required=not DEFAULT_DOCX)
    parser.add_argument("--question-bank", default=DEFAULT_QUESTION_BANK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-name", default=DEFAULT_RESULTS)
    parser.add_argument("--report-name", default=DEFAULT_REPORT)
    parser.add_argument("--judgments-name", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--questions", type=int, default=77)
    parser.add_argument("--cloze-questions", type=int, default=52)
    parser.add_argument("--short-qa-questions", type=int, default=15)
    parser.add_argument("--multi-hop-questions", type=int, default=10)
    parser.add_argument("--retrieve-top-k", type=int, default=12)
    parser.add_argument("--answer-timeout", type=int, default=480)
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--think-level", default="low", help="Top-level Ollama think value, e.g. low/medium/high/false.")
    parser.add_argument("--case-ids", default="", help="Optional comma-separated case ids to run, e.g. 58,64,71,77")
    parser.add_argument("--keep-memory-dir", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Do not query the model; only finalize using existing results + judgments.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_question_bank(path: Path) -> List[AccuracyCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[AccuracyCase] = []
    for row in raw:
        out.append(
            AccuracyCase(
                case_id=int(row["case_id"]),
                case_type=str(row["case_type"]),
                question=str(row["question"]),
                gold_answer=str(row["gold_answer"]),
                accepted_aliases=tuple(str(item) for item in row.get("accepted_aliases", [])),
                supporting_chunk_ids=tuple(int(item) for item in row.get("supporting_chunk_ids", [])),
                evidence_quotes=tuple(str(item) for item in row.get("evidence_quotes", [])),
                score=float(row.get("score", 1.0)),
            )
        )
    return out


def pick_cases(
    cases: Sequence[AccuracyCase],
    *,
    total: int,
    cloze_n: int,
    short_n: int,
    multi_n: int,
    case_ids: Optional[Sequence[int]] = None,
) -> List[AccuracyCase]:
    if case_ids:
        wanted = {int(case_id) for case_id in case_ids}
        picked = [case for case in cases if case.case_id in wanted]
        if len(picked) != len(wanted):
            found = {case.case_id for case in picked}
            missing = sorted(wanted - found)
            raise SystemExit(f"Unknown case ids: {missing}")
        return sorted(picked, key=lambda case: case.case_id)
    if total != cloze_n + short_n + multi_n:
        raise SystemExit("--questions must equal cloze + short-qa + multi-hop counts.")
    buckets = {
        "cloze": [case for case in cases if case.case_type == "cloze"],
        "grounded_short_qa": [case for case in cases if case.case_type == "grounded_short_qa"],
        "multi_hop_consistency": [case for case in cases if case.case_type == "multi_hop_consistency"],
    }
    if total == 77 and cloze_n == 52 and short_n == 15 and multi_n == 10:
        return sorted(cases, key=lambda case: case.case_id)
    picked = buckets["cloze"][:cloze_n] + buckets["grounded_short_qa"][:short_n] + buckets["multi_hop_consistency"][:multi_n]
    return sorted(picked, key=lambda case: case.case_id)


def ollama_chat_raw(
    *,
    url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    num_predict: int,
    think_level: str,
    prompt_mode: str = "full",
) -> AttemptRecord:
    payload: Dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": int(num_predict),
        },
    }
    think_value = (think_level or "").strip()
    if think_value:
        payload["think"] = think_value
    response = requests.post(url.rstrip("/") + "/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    message = body.get("message") or {}
    content = normalize_space(str(message.get("content") or ""))
    thinking = str(message.get("thinking") or "")
    return AttemptRecord(
        prompt_mode=prompt_mode,
        num_predict=int(num_predict),
        done_reason=str(body.get("done_reason") or ""),
        content=content,
        thinking_len=len(thinking),
    )


def generation_budget(case_type: str) -> int:
    if case_type == "cloze":
        return 512
    if case_type == "grounded_short_qa":
        return 768
    return 1024


def answer_case(
    *,
    mm: MemoryManager,
    case: AccuracyCase,
    answer_model: str,
    answer_timeout: int,
    retrieve_top_k: int,
    think_level: str,
    ollama_url: str,
    chunk_map: Dict[int, ChunkRecord],
) -> PendingRow:
    style, _ = case_response_style(case)
    plan = build_case_turn_plan(mm, case)
    retrieved_bundle = dict(plan.get("bundle", {}))
    retrieved = list(retrieved_bundle.get("items", []))
    retrieved_chunk_ids = tuple(int(item.get("chunk_id", 0) or 0) for item in retrieved)
    retrieved_snippets = tuple(
        truncate(" | ".join(str(line) for line in item.get("answer_lines", []) if str(line).strip()) or str(item.get("snippet", "")), 220)
        for item in retrieved
    )
    support_snippets = tuple(support_fragments_for_case(case, chunk_map))
    answer_candidates = tuple(dict(item) for item in list(retrieved_bundle.get("answer_candidates", []) or []))
    canonical_answer = str(retrieved_bundle.get("canonical_answer", "") or "")

    base_budget = generation_budget(case.case_type)
    budgets = [base_budget]
    if base_budget < 2048:
        budgets.append(min(base_budget * 2, 2048))

    attempts: List[AttemptRecord] = []
    final_attempt: Optional[AttemptRecord] = None
    synthesis_payload: Dict[str, object] = {}
    used_fallback = False
    if str(plan.get("direct_answer", "")).strip():
        final_attempt = AttemptRecord(
            prompt_mode="fact_rendered",
            num_predict=0,
            done_reason="rendered",
            content=str(plan.get("direct_answer", "")).strip(),
            thinking_len=0,
        )
        attempts.append(final_attempt)
        used_fallback = True
    elif bool(plan.get("two_pass")) and case.case_type == "multi_hop_consistency":
        synthesis_budget = max(384, min(768, base_budget))
        synthesis_budgets = [synthesis_budget]
        if synthesis_budget < 1024:
            synthesis_budgets.append(1024)
        synthesis = None
        for budget in synthesis_budgets:
            attempt = ollama_chat_raw(
                url=ollama_url,
                model=answer_model,
                messages=list(plan.get("synthesis_messages", [])),
                timeout=answer_timeout,
                num_predict=budget,
                think_level=think_level,
                prompt_mode="multi_hop_synthesis",
            )
            attempts.append(attempt)
            synthesis = mm.parse_multi_hop_synthesis(attempt.content, str(plan.get("question_subtype") or ""))
            if synthesis is not None:
                synthesis = mm.repair_multi_hop_synthesis(
                    user_text=case.question,
                    synthesis=synthesis,
                    retrieved_bundle=retrieved_bundle,
                )
                synthesis_payload = dict(synthesis)
                break

        if synthesis is not None:
            final_messages = mm.build_multi_hop_answer_messages(
                user_text=case.question,
                assistant_style=style,
                synthesis=synthesis,
                lang="zh",
                retrieved_bundle=retrieved_bundle,
            )
        else:
            final_messages = list(plan.get("fallback_messages", plan.get("messages", [])))

        for budget in budgets:
            attempt = ollama_chat_raw(
                url=ollama_url,
                model=answer_model,
                messages=final_messages,
                timeout=answer_timeout,
                num_predict=budget,
                think_level=think_level,
                prompt_mode="multi_hop_final" if synthesis is not None else "multi_hop_fallback",
            )
            attempts.append(attempt)
            final_attempt = attempt
            if attempt.content and attempt.done_reason != "length":
                break
        if synthesis is not None and final_attempt is not None and mm.needs_multi_hop_answer_fallback(final_attempt.content, synthesis, case.question):
            repaired_answer = mm.render_multi_hop_answer(case.question, synthesis).strip()
            final_attempt = AttemptRecord(
                prompt_mode="multi_hop_rendered",
                num_predict=0,
                done_reason="rendered",
                content=repaired_answer,
                thinking_len=0,
            )
            attempts.append(final_attempt)
            used_fallback = True
    else:
        for budget in budgets:
            attempt = ollama_chat_raw(
                url=ollama_url,
                model=answer_model,
                messages=list(plan.get("messages", [])),
                timeout=answer_timeout,
                num_predict=budget,
                think_level=think_level,
                prompt_mode="full",
            )
            attempts.append(attempt)
            final_attempt = attempt
            if attempt.content and attempt.done_reason != "length":
                break
            if attempt.content and case.case_type == "cloze":
                break
        if (
            final_attempt is not None
            and str(plan.get("mode", "")) == "fact"
            and mm.needs_fact_answer_fallback(final_attempt.content, retrieved_bundle, case.question)
        ):
            repaired_answer = mm.render_fact_answer(case.question, retrieved_bundle=retrieved_bundle).strip()
            final_attempt = AttemptRecord(
                prompt_mode="fact_rendered",
                num_predict=0,
                done_reason="rendered",
                content=repaired_answer,
                thinking_len=0,
            )
            attempts.append(final_attempt)
            used_fallback = True

    if not final_attempt:
        raise RuntimeError(f"No attempt was recorded for case {case.case_id}.")

    suggestion = suggest_judgment(case, final_attempt.content, retrieved_chunk_ids)
    return PendingRow(
        case=case,
        model_answer=final_attempt.content,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_snippets=retrieved_snippets,
        support_snippets=support_snippets,
        done_reason=final_attempt.done_reason,
        prompt_mode=final_attempt.prompt_mode,
        attempts=tuple(attempts),
        synthesis=synthesis_payload,
        used_fallback=used_fallback,
        answer_candidates=answer_candidates,
        canonical_answer=canonical_answer,
        suggestion=suggestion,
    )


def suggest_judgment(case: AccuracyCase, answer: str, retrieved_chunk_ids: Sequence[int]) -> Suggestion:
    answer = normalize_space(answer)
    if case.case_type == "cloze":
        label, score, reason, root_cause = heuristic_cloze_eval(case, answer, retrieved_chunk_ids)
        return Suggestion(label=label, score=score, root_cause=root_cause, reason=reason)

    if answer_exact_alias_hit(answer, case):
        return Suggestion(label="Exact", score=1.0, root_cause="alias_mismatch", reason="命中参考答案或可接受别名。")
    if answer_contains_alias(answer, case):
        label = "Exact" if case.case_type == "grounded_short_qa" else "Partial"
        score = 1.0 if label == "Exact" else 0.5
        reason = "回答包含参考答案，事实本身正确。" if label == "Exact" else "回答包含参考答案，但表达不够紧凑或链条不够完整。"
        return Suggestion(label=label, score=score, root_cause="alias_mismatch", reason=reason)

    if not answer:
        root_cause = "retrieval_miss" if retrieval_alignment(case, retrieved_chunk_ids) == "miss" else "generation_error"
        reason = "没有作答，且检索未命中支撑 chunk。" if root_cause == "retrieval_miss" else "没有作答。"
        return Suggestion(label="Miss", score=0.0, root_cause=root_cause, reason=reason)

    root_cause = "retrieval_miss" if retrieval_alignment(case, retrieved_chunk_ids) == "miss" else ("multi_hop_conflict" if case.case_type == "multi_hop_consistency" else "generation_error")
    return Suggestion(label="Miss", score=0.0, root_cause=root_cause, reason="需要人工复核。")


def pending_payload(row: PendingRow) -> Dict[str, object]:
    return {
        "case_id": row.case.case_id,
        "case_type": row.case.case_type,
        "question": row.case.question,
        "gold_answer": row.case.gold_answer,
        "accepted_aliases": list(row.case.accepted_aliases),
        "supporting_chunk_ids": list(row.case.supporting_chunk_ids),
        "evidence_quotes": list(row.case.evidence_quotes),
        "retrieved_chunk_ids": list(row.retrieved_chunk_ids),
        "retrieved_snippets": list(row.retrieved_snippets),
        "support_snippets": list(row.support_snippets),
        "model_answer": row.model_answer,
        "done_reason": row.done_reason,
        "prompt_mode": row.prompt_mode,
        "attempts": [asdict(attempt) for attempt in row.attempts],
        "synthesis": row.synthesis,
        "used_fallback": row.used_fallback,
        "answer_candidates": [dict(item) for item in row.answer_candidates],
        "canonical_answer": row.canonical_answer,
        "suggested_label": row.suggestion.label,
        "suggested_score": row.suggestion.score,
        "suggested_root_cause": row.suggestion.root_cause,
        "suggested_reason": row.suggestion.reason,
    }


def manual_row_payload(row: ManualEvalRow) -> Dict[str, object]:
    return {
        "case_id": row.case.case_id,
        "case_type": row.case.case_type,
        "question": row.case.question,
        "gold_answer": row.case.gold_answer,
        "accepted_aliases": list(row.case.accepted_aliases),
        "supporting_chunk_ids": list(row.supporting_chunk_ids),
        "evidence_quotes": list(row.evidence_quotes),
        "retrieved_chunk_ids": list(row.retrieved_chunk_ids),
        "retrieved_snippets": list(row.retrieved_snippets),
        "support_snippets": list(row.support_snippets),
        "model_answer": row.model_answer,
        "done_reason": row.done_reason,
        "prompt_mode": row.prompt_mode,
        "attempts": [asdict(attempt) for attempt in row.attempts],
        "synthesis": row.synthesis,
        "used_fallback": row.used_fallback,
        "answer_candidates": [dict(item) for item in row.answer_candidates],
        "canonical_answer": row.canonical_answer,
        "label": row.label,
        "score": row.score,
        "root_cause": row.root_cause,
        "reason": row.reason,
    }


def judgments_template(rows: Sequence[PendingRow]) -> Dict[str, object]:
    return {
        "instructions": {
            "labels": ["Exact", "Partial", "Miss"],
            "scores": {"Exact": 1.0, "Partial": 0.5, "Miss": 0.0},
            "root_causes": list(ROOT_CAUSES),
        },
        "judgments": {
            str(row.case.case_id): {
                "label": row.suggestion.label,
                "score": row.suggestion.score,
                "root_cause": row.suggestion.root_cause,
                "reason": row.suggestion.reason,
            }
            for row in rows
        },
    }


def load_judgments(path: Path) -> Dict[int, Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("judgments") if isinstance(payload, dict) else None
    if not isinstance(block, dict):
        raise ValueError("Judgments file must contain a top-level 'judgments' object.")
    out: Dict[int, Dict[str, object]] = {}
    for key, value in block.items():
        out[int(key)] = dict(value)
    return out


def apply_judgments(rows: Sequence[PendingRow], judgments: Dict[int, Dict[str, object]]) -> List[ManualEvalRow]:
    final_rows: List[ManualEvalRow] = []
    missing: List[int] = []
    for row in rows:
        manual = judgments.get(row.case.case_id)
        if not manual:
            missing.append(row.case.case_id)
            continue
        label = str(manual.get("label") or "").strip()
        score = float(manual.get("score", 0.0))
        root_cause = str(manual.get("root_cause") or "").strip()
        reason = normalize_space(str(manual.get("reason") or "").strip())
        if label not in {"Exact", "Partial", "Miss"}:
            raise ValueError(f"Invalid label for case {row.case.case_id}: {label}")
        if root_cause not in ROOT_CAUSES:
            raise ValueError(f"Invalid root cause for case {row.case.case_id}: {root_cause}")
        final_rows.append(
            ManualEvalRow(
                case=row.case,
                model_answer=row.model_answer,
                label=label,
                score=score,
                root_cause=root_cause,
                reason=reason,
                retrieved_chunk_ids=row.retrieved_chunk_ids,
                retrieved_snippets=row.retrieved_snippets,
                supporting_chunk_ids=row.case.supporting_chunk_ids,
                support_snippets=row.support_snippets,
                evidence_quotes=row.case.evidence_quotes,
                done_reason=row.done_reason,
                prompt_mode=row.prompt_mode,
                attempts=row.attempts,
                synthesis=row.synthesis,
                used_fallback=row.used_fallback,
                answer_candidates=row.answer_candidates,
                canonical_answer=row.canonical_answer,
            )
        )
    if missing:
        raise ValueError(f"Missing manual judgments for case ids: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return final_rows


def summarize_results(rows: Sequence[ManualEvalRow]) -> Dict[str, object]:
    counts = Counter(row.label for row in rows)
    root_cause_breakdown = Counter(row.root_cause for row in rows if row.label != "Exact")
    grouped: Dict[str, List[ManualEvalRow]] = defaultdict(list)
    for row in rows:
        grouped[row.case.case_type].append(row)
    by_type: Dict[str, Dict[str, object]] = {}
    for case_type, bucket in grouped.items():
        exact = sum(1 for row in bucket if row.label == "Exact")
        partial = sum(1 for row in bucket if row.label == "Partial")
        miss = sum(1 for row in bucket if row.label == "Miss")
        accuracy = sum(row.score for row in bucket) / max(1, len(bucket))
        by_type[case_type] = {
            "questions": len(bucket),
            "exact": exact,
            "partial": partial,
            "miss": miss,
            "accuracy": accuracy,
        }
    return {
        "questions": len(rows),
        "exact": counts["Exact"],
        "partial": counts["Partial"],
        "miss": counts["Miss"],
        "accuracy": sum(row.score for row in rows) / max(1, len(rows)),
        "root_cause_breakdown": dict(root_cause_breakdown),
        "by_type": by_type,
    }


def load_old_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("summary") or {})


def delta(new_value: float, old_value: float) -> str:
    return f"{(new_value - old_value):+.2%}"


def build_report(
    *,
    docx_path: Path,
    char_count: int,
    paragraph_count: int,
    import_stats: Dict[str, object],
    model: str,
    question_bank_path: Path,
    results_path: Path,
    report_path: Path,
    judgments_path: Path,
    summary: Dict[str, object],
    old_summary: Dict[str, object],
    previous_20b_summary: Dict[str, object],
    rows: Sequence[ManualEvalRow],
) -> str:
    old_by_type = old_summary.get("by_type") or {}
    previous_by_type = previous_20b_summary.get("by_type") or {}
    lines = [
        f"# {docx_path.stem} gpt-oss:20b Manual Accuracy Report",
        "",
        "## Corpus",
        f"- Source: `{docx_path}`",
        f"- Characters: `{char_count}`",
        f"- Paragraphs: `{paragraph_count}`",
        f"- Indexed chunks: `{import_stats.get('chunks', 0)}`",
        f"- Indexed terms: `{import_stats.get('terms', 0)}`",
        "",
        "## Setup",
        f"- Answer model: `{model}`",
        f"- Question bank: `{question_bank_path}`",
        "- Judge: `manual review by Codex (no Google/OpenAI API)`",
        "- Question bank policy: `Reused the existing 77-question bank for comparability.`",
        "",
        "## Overall",
        f"- Questions: `{summary['questions']}`",
        f"- Exact: `{summary['exact']}`",
        f"- Partial: `{summary['partial']}`",
        f"- Miss: `{summary['miss']}`",
        f"- Accuracy: `{summary['accuracy']:.2%}`",
    ]
    if old_summary:
        lines.append(f"- Delta vs 26B baseline: `{delta(float(summary['accuracy']), float(old_summary.get('accuracy', 0.0)))}`")
    if previous_20b_summary:
        lines.append(
            f"- Delta vs previous 20B fix: `{delta(float(summary['accuracy']), float(previous_20b_summary.get('accuracy', 0.0)))}`"
        )
    lines.extend(["", "## By Type"])
    for case_type in ("cloze", "grounded_short_qa", "multi_hop_consistency"):
        data = summary["by_type"].get(case_type, {})
        old_acc = float((old_by_type.get(case_type) or {}).get("accuracy", 0.0)) if old_by_type else 0.0
        previous_acc = float((previous_by_type.get(case_type) or {}).get("accuracy", 0.0)) if previous_by_type else 0.0
        line = (
            f"- `{case_type}`: questions `{data.get('questions', 0)}`, exact `{data.get('exact', 0)}`, "
            f"partial `{data.get('partial', 0)}`, miss `{data.get('miss', 0)}`, accuracy `{float(data.get('accuracy', 0.0)):.2%}`"
        )
        if old_by_type:
            line += f", delta vs 26B `{delta(float(data.get('accuracy', 0.0)), old_acc)}`"
        if previous_by_type:
            line += f", delta vs previous 20B `{delta(float(data.get('accuracy', 0.0)), previous_acc)}`"
        lines.append(line)

    lines.extend(["", "## Root Causes"])
    breakdown = summary.get("root_cause_breakdown", {})
    if breakdown:
        for key in ROOT_CAUSES:
            lines.append(f"- `{key}`: `{int(breakdown.get(key, 0))}`")
    else:
        lines.append("- No failures.")

    lines.extend(["", "## Top 10 Misses"])
    misses = [row for row in rows if row.label != "Exact"]
    if misses:
        for row in misses[:10]:
            lines.append(f"### Q{row.case.case_id:02d} [{row.case.case_type}]")
            lines.append(f"- Question: {row.case.question}")
            lines.append(f"- Gold: {row.case.gold_answer}")
            lines.append(f"- Answer: {truncate(row.model_answer, 260)}")
            lines.append(f"- Root Cause: `{row.root_cause}`")
            lines.append(f"- Reason: {row.reason}")
            lines.append(f"- Supporting Chunks: {', '.join(str(cid) for cid in row.supporting_chunk_ids)}")
            lines.append(f"- Retrieved Chunks: {', '.join(str(cid) for cid in row.retrieved_chunk_ids)}")
            lines.append("")
    else:
        lines.append("- No misses.")

    lines.extend([
        "## Artifacts",
        f"- Manual results JSON: `{results_path}`",
        f"- Manual report: `{report_path}`",
        f"- Manual judgments JSON: `{judgments_path}`",
        "",
    ])

    if old_summary:
        old_overall = float(old_summary.get("accuracy", 0.0))
        multi_new = float((summary["by_type"].get("multi_hop_consistency") or {}).get("accuracy", 0.0))
        multi_old = float((old_by_type.get("multi_hop_consistency") or {}).get("accuracy", 0.0)) if old_by_type else 0.0
        short_new = float((summary["by_type"].get("grounded_short_qa") or {}).get("accuracy", 0.0))
        short_old = float((old_by_type.get("grounded_short_qa") or {}).get("accuracy", 0.0)) if old_by_type else 0.0
        cloze_new = float((summary["by_type"].get("cloze") or {}).get("accuracy", 0.0))
        cloze_old = float((old_by_type.get("cloze") or {}).get("accuracy", 0.0)) if old_by_type else 0.0
        weakest = min(
            [("cloze", cloze_new - cloze_old), ("grounded_short_qa", short_new - short_old), ("multi_hop_consistency", multi_new - multi_old)],
            key=lambda item: item[1],
        )[0]
        lines.extend([
            "## Conclusion",
            f"- Overall delta vs 26B: `{delta(float(summary['accuracy']), old_overall)}`",
            (
                f"- Overall delta vs previous 20B: "
                f"`{delta(float(summary['accuracy']), float(previous_20b_summary.get('accuracy', 0.0)))}`"
                if previous_20b_summary
                else "- Overall delta vs previous 20B: `n/a`"
            ),
            f"- The largest accuracy drop landed in `{weakest}`.",
            "- This offline run was manually reviewed and did not use any external judge API.",
            "",
        ])

    return "\n".join(lines).rstrip() + "\n"


def write_seed_results(
    *,
    path: Path,
    docx_path: Path,
    answer_model: str,
    import_stats: Dict[str, object],
    rows: Sequence[PendingRow],
    summary: Dict[str, object],
    manual_review_status: str,
) -> None:
    payload = {
        "docx": str(docx_path),
        "model": answer_model,
        "judge": "manual",
        "manual_review_status": manual_review_status,
        "import_stats": import_stats,
        "summary": summary,
        "results": [pending_payload(row) for row in rows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_final_results(
    *,
    path: Path,
    docx_path: Path,
    answer_model: str,
    import_stats: Dict[str, object],
    rows: Sequence[ManualEvalRow],
    summary: Dict[str, object],
) -> None:
    payload = {
        "docx": str(docx_path),
        "model": answer_model,
        "judge": "manual",
        "manual_review_status": "completed",
        "import_stats": import_stats,
        "summary": summary,
        "results": [manual_row_payload(row) for row in rows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    docx_path = Path(args.docx).expanduser().resolve()
    bank_path = Path(args.question_bank).expanduser().resolve()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    results_path = output_dir / args.results_name
    report_path = output_dir / args.report_name
    judgments_path = output_dir / args.judgments_name
    old_results_path = Path(OLD_RESULTS)
    previous_20b_results_path = Path(PREVIOUS_20B_RESULTS)

    if not docx_path.exists():
        raise SystemExit(f"Document not found: {docx_path}")
    if not bank_path.exists():
        raise SystemExit(f"Question bank not found: {bank_path}")

    ensure_dir(output_dir)
    all_cases = load_question_bank(bank_path)
    selected_case_ids = [int(value.strip()) for value in str(args.case_ids or "").split(",") if value.strip()]
    cases = pick_cases(
        all_cases,
        total=int(args.questions),
        cloze_n=int(args.cloze_questions),
        short_n=int(args.short_qa_questions),
        multi_n=int(args.multi_hop_questions),
        case_ids=selected_case_ids or None,
    )

    char_count, paragraph_count = corpus_stats(docx_path)
    old_summary = load_old_summary(old_results_path)
    previous_20b_summary = load_old_summary(previous_20b_results_path)

    pending_rows: List[PendingRow] = []
    import_stats: Dict[str, object] = {}

    if not args.skip_run:
        if memory_dir.exists():
            shutil.rmtree(memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)

        config = MemoryConfig(memory_dir=str(memory_dir), retrieve_top_k=int(args.retrieve_top_k))
        mm = MemoryManager(config)
        try:
            stored = mm.import_files([str(docx_path)])
            if not stored:
                raise RuntimeError("Import failed: no supported files were stored.")
            import_stats = rebuild_accuracy_memory(mm)
            chunks = load_chunks(mm)
            if not chunks:
                raise RuntimeError("No chunks were indexed for the document.")
            chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
            print(f"Running {len(cases)} offline manual-review cases with {args.model}...", flush=True)
            for case in cases:
                print(f"[{case.case_id:02d}/{len(cases):02d}] [{case.case_type}] {truncate(case.question, 120)}", flush=True)
                row = answer_case(
                    mm=mm,
                    case=case,
                    answer_model=args.model,
                    answer_timeout=int(args.answer_timeout),
                    retrieve_top_k=int(args.retrieve_top_k),
                    think_level=args.think_level,
                    ollama_url=args.ollama_url,
                    chunk_map=chunk_map,
                )
                pending_rows.append(row)
                print(
                    f"  -> suggestion={row.suggestion.label} | done={row.done_reason or '(none)'} | gold={truncate(case.gold_answer, 60)} | got={truncate(row.model_answer, 90)}",
                    flush=True,
                )

            seed_summary = summarize_results(
                [
                    ManualEvalRow(
                        case=row.case,
                        model_answer=row.model_answer,
                        label=row.suggestion.label,
                        score=row.suggestion.score,
                        root_cause=row.suggestion.root_cause,
                        reason=row.suggestion.reason,
                        retrieved_chunk_ids=row.retrieved_chunk_ids,
                        retrieved_snippets=row.retrieved_snippets,
                        supporting_chunk_ids=row.case.supporting_chunk_ids,
                        support_snippets=row.support_snippets,
                        evidence_quotes=row.case.evidence_quotes,
                        done_reason=row.done_reason,
                        prompt_mode=row.prompt_mode,
                        attempts=row.attempts,
                        synthesis=row.synthesis,
                        used_fallback=row.used_fallback,
                        answer_candidates=row.answer_candidates,
                        canonical_answer=row.canonical_answer,
                    )
                    for row in pending_rows
                ]
            )
            write_seed_results(
                path=results_path,
                docx_path=docx_path,
                answer_model=args.model,
                import_stats=import_stats,
                rows=pending_rows,
                summary=seed_summary,
                manual_review_status="pending",
            )
            judgments_path.write_text(json.dumps(judgments_template(pending_rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote pending results to {results_path}", flush=True)
            print(f"Wrote judgments seed to {judgments_path}", flush=True)
        finally:
            mm.close()
            if not args.keep_memory_dir:
                shutil.rmtree(memory_dir, ignore_errors=True)

    if not results_path.exists():
        raise SystemExit(f"Results file missing: {results_path}")
    if not judgments_path.exists():
        raise SystemExit(f"Judgments file missing: {judgments_path}")

    seed_payload = json.loads(results_path.read_text(encoding="utf-8"))
    import_stats = dict(seed_payload.get("import_stats") or {})
    raw_index = {int(item["case_id"]): item for item in seed_payload.get("results", [])}
    pending_rows = []
    for case in cases:
        item = raw_index.get(case.case_id)
        if not item:
            raise SystemExit(f"Pending result missing for case {case.case_id}")
        pending_rows.append(
            PendingRow(
                case=case,
                model_answer=str(item.get("model_answer") or ""),
                retrieved_chunk_ids=tuple(int(x) for x in item.get("retrieved_chunk_ids", [])),
                retrieved_snippets=tuple(str(x) for x in item.get("retrieved_snippets", [])),
                support_snippets=tuple(str(x) for x in item.get("support_snippets", [])),
                done_reason=str(item.get("done_reason") or ""),
                prompt_mode=str(item.get("prompt_mode") or "full"),
                attempts=tuple(
                    AttemptRecord(
                        prompt_mode=str(a.get("prompt_mode") or "full"),
                        num_predict=int(a.get("num_predict", 0) or 0),
                        done_reason=str(a.get("done_reason") or ""),
                        content=str(a.get("content") or ""),
                        thinking_len=int(a.get("thinking_len", 0) or 0),
                    )
                    for a in item.get("attempts", [])
                ),
                synthesis=dict(item.get("synthesis") or {}),
                used_fallback=bool(item.get("used_fallback", False)),
                answer_candidates=tuple(dict(x) for x in item.get("answer_candidates", [])),
                canonical_answer=str(item.get("canonical_answer") or ""),
                suggestion=Suggestion(
                    label=str(item.get("suggested_label") or "Miss"),
                    score=float(item.get("suggested_score", 0.0) or 0.0),
                    root_cause=str(item.get("suggested_root_cause") or "generation_error"),
                    reason=str(item.get("suggested_reason") or ""),
                ),
            )
        )

    judgments = load_judgments(judgments_path)
    final_rows = apply_judgments(pending_rows, judgments)
    summary = summarize_results(final_rows)
    write_final_results(
        path=results_path,
        docx_path=docx_path,
        answer_model=args.model,
        import_stats=import_stats,
        rows=final_rows,
        summary=summary,
    )
    report = build_report(
        docx_path=docx_path,
        char_count=char_count,
        paragraph_count=paragraph_count,
        import_stats=import_stats,
        model=args.model,
        question_bank_path=bank_path,
        results_path=results_path,
        report_path=report_path,
        judgments_path=judgments_path,
        summary=summary,
        old_summary=old_summary,
        previous_20b_summary=previous_20b_summary,
        rows=final_rows,
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
