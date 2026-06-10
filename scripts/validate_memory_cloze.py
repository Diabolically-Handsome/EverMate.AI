#!/usr/bin/env python3
"""Run a grounded cloze-style memory recall evaluation against EverMate."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from ollama_client import chat as ollama_chat


DEFAULT_DOCX = "/Users/lawrencegrey/Desktop/Prompt 5 Feb.25-Mar.7.docx"
DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q8_0"
DEFAULT_MEMORY_DIR = "/tmp/evermate-memory-cloze-validation"
DEFAULT_OUTPUT_DIR = "/Users/lawrencegrey/Desktop/EverMate/reports"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    path: str
    text: str


@dataclass(frozen=True)
class ClozeCase:
    case_id: int
    chunk_id: int
    question: str
    answer: str
    evidence: str
    score: float


@dataclass(frozen=True)
class EvalRow:
    case: ClozeCase
    answer: str
    label: str
    reason: str
    top_retrieval: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EverMate memory recall with cloze questions.")
    parser.add_argument("--docx", default=DEFAULT_DOCX, help="Path to the source .docx file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Exact Ollama model name to use.")
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR, help="Isolated memory directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for reports and question bank.")
    parser.add_argument("--questions", type=int, default=77, help="Number of cloze questions to run.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-question timeout in seconds.")
    parser.add_argument(
        "--keep-memory-dir",
        action="store_true",
        help="Keep the isolated memory directory after the run.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_match(text: str) -> str:
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("“", "").replace("”", "").replace('"', "")
    cleaned = cleaned.replace("`", "").replace("*", "")
    cleaned = cleaned.replace("。", "").replace("，", "").replace(",", "")
    cleaned = cleaned.replace("！", "").replace("!", "").replace("？", "").replace("?", "")
    cleaned = cleaned.replace("：", "").replace(":", "").replace("；", "").replace(";", "")
    cleaned = cleaned.replace("（", "(").replace("）", ")")
    return cleaned


def truncate(text: str, limit: int = 220) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?])\s+|\n+", text) if p.strip()]
    return [part for part in parts if 10 <= len(part) <= 120]


def semantic_context_length(text: str) -> int:
    return len(re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", text))


def load_chunks(mm: MemoryManager) -> List[ChunkRecord]:
    cur = mm.conn.cursor()
    rows = cur.execute("SELECT id, path FROM chunks ORDER BY id ASC").fetchall()
    chunks: List[ChunkRecord] = []
    for row in rows:
        rel_path = str(row["path"])
        abs_path = Path(mm.memory_dir) / rel_path
        if not abs_path.exists():
            continue
        chunks.append(
            ChunkRecord(
                chunk_id=int(row["id"]),
                path=rel_path,
                text=abs_path.read_text(encoding="utf-8"),
            )
        )
    return chunks


def candidate_patterns() -> List[tuple[str, float]]:
    return [
        (r"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*(?:km/h|分钟|小时|%))", 12.0),
        (r"(\d+\s*月\s*\d+\s*日)", 12.0),
        (r"(第\d+天)", 12.0),
        (r"“([^”]{2,20})”", 11.0),
        (r"\"([^\"]{2,20})\"", 11.0),
        (r"([\u4e00-\u9fffA-Za-z0-9.+:_/-]{2,18}\+[\u4e00-\u9fffA-Za-z0-9.+:_/-]{2,24}(?:\+[\u4e00-\u9fffA-Za-z0-9.+:_/-]{2,24})*)", 10.5),
        (r"([A-Za-z][A-Za-z0-9.+:_/-]{1,30})", 9.5),
        (r"((?:4\.5|4\.0|4o|GTA5|C\+\+|Ontario|HT520|A9\s*级别财富|微积分部分|微积分考试|微积分|多伦多|加拿大|台湾|台灣))", 9.0),
        (r"(\d+(?:\.\d+)?\s*(?:%|km/h|分钟|小时|天|次|级))", 8.5),
        (r"(?:是|为|叫|叫做|名为|属于)([^，。！？\n]{2,18})", 7.5),
    ]


def answer_blacklist() -> set[str]:
    return {
        "主人",
        "胡桃",
        "小胡桃",
        "这个模型",
        "现实生活",
        "小说世界",
        "现实中",
        "男生",
        "女生",
        "第一印象",
        "方向盘",
        "警车",
        "NPC",
        "台湾女生",
        "主人和胡桃",
        "这件事",
        "这个世界",
        "这个人",
    }


def build_candidates(chunks: List[ChunkRecord]) -> List[Dict[str, object]]:
    patterns = candidate_patterns()
    blacklist = answer_blacklist()
    candidates: List[Dict[str, object]] = []

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            for pattern, base_score in patterns:
                for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                    answer = (match.group(1) if match.groups() else match.group(0)).strip(' ：:,.，。!！?？"“”()[]')
                    if not answer or len(answer) < 2 or len(answer) > 24:
                        continue
                    if answer in blacklist:
                        continue
                    if sentence.count(answer) != 1:
                        continue
                    question = sentence.replace(answer, "____", 1)
                    if question == sentence or question.count("____") != 1:
                        continue
                    if len(question.replace("____", "").strip()) < 8:
                        continue
                    if semantic_context_length(question.replace("____", "")) < 10:
                        continue
                    if "http://" in question.lower() or "https://" in question.lower():
                        continue
                    if any(token in question for token in ('model="____"', "model='____'", "____=", "={", "};")):
                        continue
                    if re.match(r"^[^A-Za-z0-9\u4e00-\u9fff]*____", question):
                        continue

                    score = float(base_score)
                    if re.search(r"\d", answer):
                        score += 2.0
                    if re.search(r"[A-Za-z]", answer):
                        score += 1.4
                    if "+" in answer or "-" in answer or "/" in answer:
                        score += 0.8
                    score += min(semantic_context_length(question.replace("____", "")) / 10.0, 2.0)
                    score -= max(0, len(answer) - 12) * 0.08

                    candidates.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "question": "请根据记忆填空，只填写空缺内容：" + question,
                            "answer": answer,
                            "evidence": sentence,
                            "score": score,
                        }
                    )

    answer_freq = Counter(str(candidate["answer"]) for candidate in candidates)
    for candidate in candidates:
        candidate["score"] = float(candidate["score"]) - min(answer_freq[str(candidate["answer"])], 8) * 0.35

    return candidates


def select_cases(candidates: List[Dict[str, object]], target: int) -> List[ClozeCase]:
    selected: List[Dict[str, object]] = []
    seen_chunks: set[int] = set()
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()

    def qkey(text: str) -> str:
        return normalize_match(text)

    ordered = sorted(candidates, key=lambda item: (-float(item["score"]), int(item["chunk_id"])))

    for candidate in ordered:
        chunk_id = int(candidate["chunk_id"])
        question = str(candidate["question"])
        answer = str(candidate["answer"])
        if chunk_id in seen_chunks:
            continue
        if qkey(question) in seen_questions or qkey(answer) in seen_answers:
            continue
        selected.append(candidate)
        seen_chunks.add(chunk_id)
        seen_questions.add(qkey(question))
        seen_answers.add(qkey(answer))
        if len(selected) >= target:
            break

    if len(selected) < target:
        for candidate in ordered:
            question = str(candidate["question"])
            answer = str(candidate["answer"])
            if qkey(question) in seen_questions or qkey(answer) in seen_answers:
                continue
            selected.append(candidate)
            seen_questions.add(qkey(question))
            seen_answers.add(qkey(answer))
            if len(selected) >= target:
                break

    out: List[ClozeCase] = []
    for index, candidate in enumerate(selected[:target], start=1):
        out.append(
            ClozeCase(
                case_id=index,
                chunk_id=int(candidate["chunk_id"]),
                question=str(candidate["question"]),
                answer=str(candidate["answer"]),
                evidence=str(candidate["evidence"]),
                score=float(candidate["score"]),
            )
        )
    return out


def build_messages(mm: MemoryManager, question: str) -> List[Dict[str, str]]:
    style = "请只填写空缺内容本身，不要重复整句，不要解释。"
    system_prompt = mm.build_system_prompt(user_text=question, assistant_style=style, lang="zh")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def classify_answer(answer: str, gold: str) -> tuple[str, str]:
    norm_answer = normalize_match(answer)
    norm_gold = normalize_match(gold)
    if not norm_answer:
        return "Miss", "未作答。"
    if norm_gold and norm_gold in norm_answer:
        return "Exact", "命中参考答案。"
    if norm_answer and norm_answer in norm_gold:
        ratio = len(norm_answer) / max(1, len(norm_gold))
        if ratio >= 0.9:
            return "Exact", "回答与参考答案仅有轻微格式差异。"
        if ratio >= 0.6:
            return "Partial", "回答命中了参考答案的一部分。"
    similarity = SequenceMatcher(None, norm_answer, norm_gold).ratio() if norm_answer and norm_gold else 0.0
    if similarity >= 0.72:
        return "Partial", "回答与参考答案接近，但不完全一致。"
    return "Miss", "回答未命中参考答案。"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")
    return stem or "document"


def write_question_bank(output_dir: Path, slug: str, cases: List[ClozeCase]) -> tuple[Path, Path]:
    json_path = output_dir / f"{slug}-77-question-bank.json"
    md_path = output_dir / f"{slug}-77-question-bank.md"

    json_path.write_text(
        json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {slug} 77题题库",
        "",
        f"- 题目数：{len(cases)}",
        "",
    ]
    for case in cases:
        lines.append(f"## {case.case_id:02d}")
        lines.append(f"- Question: {case.question}")
        lines.append(f"- Answer: {case.answer}")
        lines.append(f"- Chunk: {case.chunk_id}")
        lines.append(f"- Evidence: {case.evidence}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = parse_args()
    docx_path = Path(args.docx).expanduser().resolve()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    slug = slugify(docx_path)
    results_path = output_dir / f"{slug}-77-results.json"
    report_path = output_dir / f"{slug}-77-report.txt"

    if not docx_path.exists():
        raise SystemExit(f"Document not found: {docx_path}")
    if args.questions < 1:
        raise SystemExit("--questions must be >= 1")

    ensure_dir(output_dir)
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    mm = MemoryManager(MemoryConfig(memory_dir=str(memory_dir)))
    try:
        stored = mm.import_files([str(docx_path)])
        if not stored:
            raise RuntimeError("Import failed: no supported files were stored.")
        import_stats = mm.rebuild_memory()
        chunks = load_chunks(mm)
        if not chunks:
            raise RuntimeError("No chunks were indexed for the document.")

        candidates = build_candidates(chunks)
        cases = select_cases(candidates, target=args.questions)
        if len(cases) < args.questions:
            raise RuntimeError(f"Only built {len(cases)} cloze cases, expected {args.questions}.")

        question_bank_json, question_bank_md = write_question_bank(output_dir, slug, cases)

        results: List[EvalRow] = []
        counts = Counter()
        print(f"Running {len(cases)} cloze recall questions...", flush=True)
        for case in cases:
            retrieval = mm.retrieve(case.question, k=3)
            top_retrieval = truncate(str(retrieval[0]["snippet"])) if retrieval else "(no evidence)"
            print(f"[{case.case_id:02d}/{len(cases):02d}] {case.question}", flush=True)
            answer = ollama_chat(
                build_messages(mm, case.question),
                model=args.model,
                timeout=args.timeout,
                options={"temperature": 0, "num_predict": 48},
            ).strip()
            label, reason = classify_answer(answer, case.answer)
            counts[label] += 1
            results.append(
                EvalRow(
                    case=case,
                    answer=answer,
                    label=label,
                    reason=reason,
                    top_retrieval=top_retrieval,
                )
            )
            print(f"  -> {label} | gold={case.answer} | got={truncate(answer, 120)}", flush=True)

        exact = counts["Exact"]
        partial = counts["Partial"]
        miss = counts["Miss"]
        recall = (exact + 0.5 * partial) / len(cases)

        results_path.write_text(
            json.dumps(
                {
                    "docx": str(docx_path),
                    "model": args.model,
                    "import_stats": import_stats,
                    "summary": {
                        "questions": len(cases),
                        "exact": exact,
                        "partial": partial,
                        "miss": miss,
                        "recall": recall,
                    },
                    "results": [
                        {
                            "case": asdict(row.case),
                            "answer": row.answer,
                            "label": row.label,
                            "reason": row.reason,
                            "top_retrieval": row.top_retrieval,
                        }
                        for row in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        lines = [
            "EverMate Cloze Recall Validation",
            "=" * 32,
            f"Source: {docx_path}",
            f"Model: {args.model}",
            f"Memory Dir: {memory_dir}",
            "",
            "[Import Stats]",
            f"Uploads: {import_stats.get('uploads', 0)}",
            f"Chunks: {import_stats.get('chunks', 0)}",
            f"Terms: {import_stats.get('terms', 0)}",
            f"Chunks Added: {import_stats.get('chunks_added', 0)}",
            "",
            "[Summary]",
            f"Questions: {len(cases)}",
            f"Exact: {exact}",
            f"Partial: {partial}",
            f"Miss: {miss}",
            f"Recall: {recall:.2%}",
            "",
            "[Artifacts]",
            f"Question Bank JSON: {question_bank_json}",
            f"Question Bank MD: {question_bank_md}",
            f"Results JSON: {results_path}",
        ]

        misses = [row for row in results if row.label != "Exact"]
        lines.append("")
        lines.append("[Sample Misses]")
        if misses:
            for row in misses[:12]:
                lines.append(f"Q{row.case.case_id:02d}: {row.case.question}")
                lines.append(f"Gold: {row.case.answer}")
                lines.append(f"A: {row.answer}")
                lines.append(f"Reason: {row.reason}")
                lines.append("")
        else:
            lines.append("No misses.")

        report = "\n".join(lines) + "\n"
        report_path.write_text(report, encoding="utf-8")
        print(report)
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
