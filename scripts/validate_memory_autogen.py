#!/usr/bin/env python3
"""Run an auto-generated memory recall evaluation against EverMate."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from ollama_client import chat as ollama_chat


DEFAULT_DOCX = "/Users/lawrencegrey/Desktop/Prompt 7 Mar.14-Apr.30.docx"
DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q8_0"
DEFAULT_QGEN_MODEL = "gpt-oss:20b"
DEFAULT_JUDGE_MODEL = "gpt-oss:20b"
DEFAULT_MEMORY_DIR = "/tmp/evermate-memory-autogen-validation"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    path: str
    source: str
    created_at: int
    text: str


@dataclass(frozen=True)
class AutoCase:
    case_id: int
    chunk_id: int
    question: str
    answer: str
    evidence: str


@dataclass(frozen=True)
class EvalRow:
    case: AutoCase
    answer: str
    label: str
    reason: str
    snippet: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EverMate memory recall with auto-generated questions.")
    parser.add_argument("--docx", default=DEFAULT_DOCX, help="Path to the source .docx file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Exact Ollama model name to answer with.")
    parser.add_argument("--question-model", default=DEFAULT_QGEN_MODEL, help="Model used to generate grounded questions.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Model used to judge answers.")
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR, help="Isolated memory directory.")
    parser.add_argument("--questions", type=int, default=77, help="Number of test questions to run.")
    parser.add_argument("--answer-timeout", type=int, default=300, help="Per-question answer timeout in seconds.")
    parser.add_argument("--gen-timeout", type=int, default=240, help="Question generation timeout in seconds.")
    parser.add_argument("--judge-timeout", type=int, default=240, help="Judge timeout in seconds.")
    parser.add_argument(
        "--keep-memory-dir",
        action="store_true",
        help="Keep the isolated memory directory after the run.",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def truncate(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def clean_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_array(text: str) -> List[Dict[str, object]]:
    cleaned = clean_json_text(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[\s*{.*}\s*]", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model output.")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("JSON payload is not a list.")
    return data


def load_chunks(mm: MemoryManager) -> List[ChunkRecord]:
    cur = mm.conn.cursor()
    rows = cur.execute(
        "SELECT id, path, source, created_at FROM chunks ORDER BY id ASC"
    ).fetchall()
    out: List[ChunkRecord] = []
    for row in rows:
        rel_path = str(row["path"])
        abs_path = Path(mm.memory_dir) / rel_path
        if not abs_path.exists():
            continue
        out.append(
            ChunkRecord(
                chunk_id=int(row["id"]),
                path=rel_path,
                source=str(row["source"] or ""),
                created_at=int(row["created_at"] or 0),
                text=abs_path.read_text(encoding="utf-8"),
            )
        )
    return out


def build_messages(mm: MemoryManager, question: str) -> List[Dict[str, str]]:
    style = "请只用一句中文直接回答问题，不要发挥，不要使用表情，不要使用项目符号。"
    system_prompt = mm.build_system_prompt(user_text=question, assistant_style=style, lang="zh")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def chunk_batches(items: List[ChunkRecord], batch_size: int) -> Iterable[List[ChunkRecord]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def sample_chunks(chunks: List[ChunkRecord], wanted: int) -> List[ChunkRecord]:
    if wanted >= len(chunks):
        return list(chunks)
    out: List[ChunkRecord] = []
    for i in range(wanted):
        pos = round(i * (len(chunks) - 1) / max(1, wanted - 1))
        out.append(chunks[pos])
    deduped: List[ChunkRecord] = []
    seen = set()
    for chunk in out:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append(chunk)
    if len(deduped) >= wanted:
        return deduped[:wanted]
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        deduped.append(chunk)
        seen.add(chunk.chunk_id)
        if len(deduped) >= wanted:
            break
    return deduped


def format_chunk_block(chunk: ChunkRecord, used_questions: List[str], used_answers: List[str]) -> str:
    lines = [f"[[CHUNK {chunk.chunk_id}]]"]
    if used_questions:
        lines.append("已用问题：")
        lines.extend(f"- {q}" for q in used_questions[:4])
    if used_answers:
        lines.append("已用答案：")
        lines.extend(f"- {a}" for a in used_answers[:4])
    lines.append(chunk.text)
    return "\n".join(lines)


def validate_case(
    raw: Dict[str, object],
    chunk_map: Dict[int, ChunkRecord],
    seen_questions: set[str],
    seen_answers: set[str],
) -> Optional[AutoCase]:
    try:
        chunk_id = int(raw.get("chunk_id", 0))
    except Exception:
        return None
    if chunk_id not in chunk_map:
        return None
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    evidence = str(raw.get("evidence", "")).strip()
    if not question or not answer or not evidence:
        return None
    n_question = normalize(question)
    n_answer = normalize(answer)
    if not n_question or not n_answer:
        return None
    if not contains_cjk(question):
        return None
    if n_question in seen_questions or n_answer in seen_answers:
        return None
    if normalize(answer) in normalize(question):
        return None
    if len(answer) < 2 or len(answer) > 28:
        return None
    if any(token in n_question for token in ("哪一句", "哪句", "according to", "which ", "what ", "who ", "when ", "where ")):
        return None
    if any(ch in answer for ch in "\n!?？！"):
        return None
    chunk_text = chunk_map[chunk_id].text
    if answer not in chunk_text or answer not in evidence:
        return None
    if len(evidence) > 180:
        return None
    seen_questions.add(n_question)
    seen_answers.add(n_answer)
    return AutoCase(case_id=0, chunk_id=chunk_id, question=question, answer=answer, evidence=evidence)


def generate_cases(
    chunks: List[ChunkRecord],
    target: int,
    model: str,
    timeout: int,
) -> List[AutoCase]:
    selected = sample_chunks(chunks, min(len(chunks), max(target, 24)))
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    per_chunk_questions: Dict[int, List[str]] = {}
    per_chunk_answers: Dict[int, List[str]] = {}
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    cases: List[AutoCase] = []

    def _request_once(batch: List[ChunkRecord], max_per_chunk: int) -> List[AutoCase]:
        blocks = [
            format_chunk_block(
                chunk,
                per_chunk_questions.get(chunk.chunk_id, []),
                per_chunk_answers.get(chunk.chunk_id, []),
            )
            for chunk in batch
        ]
        prompt = (
            "你是一个严格的中文事实题生成器。下面给你若干文档分块，请尽量为每个分块生成"
            f"{max_per_chunk}道不重复的单一事实题。\n"
            "要求：\n"
            "1. 每道题必须能仅凭对应分块回答。\n"
            "2. 答案必须是分块中连续出现的原文短语。\n"
            "3. 问题必须使用中文，不能输出英文问题。\n"
            "4. 问题里不能直接出现答案原词。\n"
            "5. 避免主观题、感受题、开放题。\n"
            "6. 优先选择名字、数字、日期、地点、课程、模型名、目标、步骤名、专有名词。\n"
            "7. 优先选择短而明确的答案，尽量控制在 2 到 16 个字，不要出需要逐字复述整句话的题。\n"
            "8. 如果分块不适合出题，可以跳过。\n"
            "9. 只返回 JSON 数组，不要解释。每项格式为 "
            '{"chunk_id": 1, "question": "...", "answer": "...", "evidence": "..."}。\n'
            "10. evidence 必须是分块中的原文短句，并且包含 answer。\n"
            "11. 不要重复已经列出的已用问题或已用答案。\n\n"
            + "\n\n".join(blocks)
        )
        response = ollama_chat(
            [
                {"role": "system", "content": "只返回合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            timeout=timeout,
            options={"temperature": 0, "num_predict": 1200},
        )
        rows = extract_json_array(response)
        out: List[AutoCase] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            case = validate_case(row, chunk_map, seen_questions, seen_answers)
            if not case:
                continue
            per_chunk_questions.setdefault(case.chunk_id, []).append(case.question)
            per_chunk_answers.setdefault(case.chunk_id, []).append(case.answer)
            out.append(case)
        return out

    def request_batch(batch: List[ChunkRecord], max_per_chunk: int) -> List[AutoCase]:
        try:
            return _request_once(batch, max_per_chunk=max_per_chunk)
        except Exception:
            if len(batch) <= 1:
                return []
            midpoint = max(1, len(batch) // 2)
            left = request_batch(batch[:midpoint], max_per_chunk=max_per_chunk)
            right = request_batch(batch[midpoint:], max_per_chunk=max_per_chunk)
            return left + right

    for batch in chunk_batches(selected, batch_size=6):
        for case in request_batch(batch, max_per_chunk=1):
            cases.append(case)
            if len(cases) >= target:
                break
        if len(cases) >= target:
            break

    if len(cases) < target:
        richest = sorted(chunks, key=lambda item: len(item.text), reverse=True)
        second_pass = sample_chunks(richest, min(len(richest), max(12, math.ceil((target - len(cases)) * 2))))
        for batch in chunk_batches(second_pass, batch_size=4):
            for case in request_batch(batch, max_per_chunk=2):
                cases.append(case)
                if len(cases) >= target:
                    break
            if len(cases) >= target:
                break

    numbered: List[AutoCase] = []
    for index, case in enumerate(cases[:target], start=1):
        numbered.append(
            AutoCase(
                case_id=index,
                chunk_id=case.chunk_id,
                question=case.question,
                answer=case.answer,
                evidence=case.evidence,
            )
        )
    return numbered


def judge_exact_heuristic(answer: str, gold: str) -> Optional[str]:
    n_answer = normalize(answer)
    n_gold = normalize(gold)
    if not n_answer:
        return "Miss"
    if n_gold and (n_gold in n_answer or n_answer in n_gold):
        return "Exact"
    return None


def judge_batch(rows: List[EvalRow], model: str, timeout: int) -> Dict[int, tuple[str, str]]:
    payload = []
    for row in rows:
        payload.append(
            {
                "id": row.case.case_id,
                "question": row.case.question,
                "gold_answer": row.case.answer,
                "evidence": row.case.evidence,
                "model_answer": row.answer,
            }
        )
    prompt = (
        "你是严格的中文问答评分器。请根据 gold_answer 和 evidence 判断 model_answer 是否回答正确。\n"
        "评分规则：\n"
        "- Exact: 语义正确，关键事实完整；允许少量措辞差异。\n"
        "- Partial: 方向对了但不完整，或只答对了一部分。\n"
        "- Miss: 事实错误、答非所问、缺失关键事实。\n"
        "只返回 JSON 数组。每项格式为 "
        '{"id": 1, "label": "Exact|Partial|Miss", "reason": "一句中文简短说明"}。'
    )
    response = ollama_chat(
        [
            {"role": "system", "content": "只返回合法 JSON。"},
            {"role": "user", "content": prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)},
        ],
        model=model,
        timeout=timeout,
        options={"temperature": 0, "num_predict": 1200},
    )
    parsed = extract_json_array(response)
    out: Dict[int, tuple[str, str]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            case_id = int(item.get("id", 0))
        except Exception:
            continue
        label = str(item.get("label", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if label not in {"Exact", "Partial", "Miss"}:
            continue
        out[case_id] = (label, reason or "模型未提供原因。")
    return out


def main() -> int:
    args = parse_args()
    docx_path = Path(args.docx).expanduser().resolve()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    report_path = memory_dir / "validation_report.txt"
    cases_path = memory_dir / "generated_cases.json"

    if not docx_path.exists():
        raise SystemExit(f"Document not found: {docx_path}")
    if args.questions < 1:
        raise SystemExit("--questions must be >= 1")

    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    mm = MemoryManager(MemoryConfig(memory_dir=str(memory_dir)))
    try:
        stored = mm.import_files([str(docx_path)])
        if not stored:
            raise RuntimeError("Import failed: no supported files were stored.")
        import_stats = mm.rebuild_memory()
        snapshot = mm.status_snapshot()

        chunks = load_chunks(mm)
        if not chunks:
            raise RuntimeError("No chunks were indexed for the document.")

        cases = generate_cases(
            chunks=chunks,
            target=args.questions,
            model=args.question_model,
            timeout=args.gen_timeout,
        )
        if len(cases) < args.questions:
            raise RuntimeError(f"Only generated {len(cases)} grounded questions, expected {args.questions}.")

        cases_path.write_text(
            json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"Generated {len(cases)} grounded questions.", flush=True)
        results: List[EvalRow] = []
        infra_errors: List[str] = []

        for case in cases:
            retrieval = mm.retrieve(case.question, k=3)
            snippet = truncate(retrieval[0]["snippet"], 180) if retrieval else "(no evidence)"
            print(f"[{case.case_id:02d}/{len(cases):02d}] {case.question}", flush=True)
            try:
                answer = ollama_chat(
                    build_messages(mm, case.question),
                    model=args.model,
                    timeout=args.answer_timeout,
                    options={"temperature": 0, "num_predict": 96},
                )
                heuristic = judge_exact_heuristic(answer, case.answer)
                if heuristic == "Exact":
                    label = "Exact"
                    reason = "命中参考答案。"
                else:
                    label = "Pending"
                    reason = ""
            except Exception as exc:
                answer = f"INFRA ERROR: {exc}"
                label = "Infra"
                reason = str(exc)
                infra_errors.append(f"Q{case.case_id}: {exc}")
            results.append(EvalRow(case=case, answer=answer, label=label, reason=reason, snippet=snippet))

        pending = [row for row in results if row.label == "Pending"]
        for batch in [pending[i : i + 10] for i in range(0, len(pending), 10)]:
            judged = judge_batch(batch, model=args.judge_model, timeout=args.judge_timeout)
            for row in batch:
                label, reason = judged.get(row.case.case_id, ("Miss", "评审输出缺失，按 Miss 处理。"))
                results[results.index(row)] = EvalRow(
                    case=row.case,
                    answer=row.answer,
                    label=label,
                    reason=reason,
                    snippet=row.snippet,
                )

        counts = Counter(row.label for row in results)
        exact = counts["Exact"]
        partial = counts["Partial"]
        miss = counts["Miss"]
        infra = counts["Infra"]
        scored = exact + partial + miss
        recall = ((exact + 0.5 * partial) / scored) if scored else 0.0

        report_lines: List[str] = []
        report_lines.append("EverMate Auto-Generated Memory Recall Validation")
        report_lines.append("=" * 48)
        report_lines.append(f"Answer Model: {args.model}")
        report_lines.append(f"Question Model: {args.question_model}")
        report_lines.append(f"Judge Model: {args.judge_model}")
        report_lines.append(f"Source: {docx_path}")
        report_lines.append(f"Memory Dir: {memory_dir}")
        report_lines.append("")
        report_lines.append("[Import Stats]")
        report_lines.append(f"Uploads: {import_stats.get('uploads', 0)}")
        report_lines.append(f"Chunks: {import_stats.get('chunks', 0)}")
        report_lines.append(f"Terms: {import_stats.get('terms', 0)}")
        report_lines.append(f"Chunks Added: {import_stats.get('chunks_added', 0)}")
        report_lines.append(f"Last Analyze Ts: {snapshot.get('last_analyze_ts', 0)}")
        report_lines.append("")
        report_lines.append("[Question Generation]")
        report_lines.append(f"Generated Cases: {len(cases)}")
        report_lines.append(f"Saved Cases: {cases_path}")
        report_lines.append("")
        report_lines.append("[Per Question]")
        for row in results:
            report_lines.append(f"{row.case.case_id:02d}. {row.label}")
            report_lines.append(f"Q: {row.case.question}")
            report_lines.append(f"Gold: {row.case.answer}")
            report_lines.append(f"A: {truncate(row.answer, 400)}")
            report_lines.append(f"Judge: {row.reason}")
            report_lines.append(f"Evidence: {truncate(row.case.evidence, 220)}")
            report_lines.append(f"Top Retrieval: {row.snippet}")
            report_lines.append("")
        report_lines.append("[Summary]")
        report_lines.append(f"Exact: {exact}")
        report_lines.append(f"Partial: {partial}")
        report_lines.append(f"Miss: {miss}")
        report_lines.append(f"Infra: {infra}")
        report_lines.append(f"Recall: {recall:.2%}")
        if infra_errors:
            report_lines.append("")
            report_lines.append("[Infrastructure Failures]")
            report_lines.extend(infra_errors)

        report = "\n".join(report_lines) + "\n"
        report_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"Report saved to: {report_path}")
        print(f"Cases saved to: {cases_path}")
        return 0
    finally:
        mm.close()
        if not args.keep_memory_dir:
            cleanup_targets = [memory_dir / "app_state.json"]
            for target in cleanup_targets:
                if target.exists():
                    target.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
