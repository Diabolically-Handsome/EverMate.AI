#!/usr/bin/env python3
"""EverMate system-level Needle in a Haystack benchmark.

This benchmark does not test raw long-context stuffing. It tests the full
EverMate system path:

ingest -> rebuild memory -> retrieve -> answer
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from ollama_client import chat as ollama_chat
from scripts.validate_memory_accuracy import normalize_match, normalize_space, slugify

try:
    from docx import Document
except Exception:  # pragma: no cover - optional dependency
    Document = None


DEFAULT_CONTEXT_LENGTHS = (8000, 32000, 128000, 512000, 1000000)
DEFAULT_DEPTHS = (10, 30, 50, 70, 90)
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_MEMORY_PREFIX = "/tmp/evermate-niah"


@dataclass(frozen=True)
class NIAHRun:
    context_length: int
    depth_percent: int
    repeat_index: int
    needle: str
    retrieval_question: str
    answer: str
    hit: bool
    root_cause: str
    retrieved_chunk_ids: Tuple[int, ...]
    needle_chunk_ids: Tuple[int, ...]
    answer_candidates: Tuple[Dict[str, object], ...]
    canonical_answer: str
    fact_strategy: str
    used_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an EverMate-system Needle in a Haystack benchmark.")
    parser.add_argument("--haystack-dir", required=True, help="Directory containing .txt/.md/.docx haystack files.")
    parser.add_argument("--needle", required=True, help="Needle sentence or paragraph to inject.")
    parser.add_argument("--retrieval-question", required=True, help="Question whose answer should be the injected needle.")
    parser.add_argument("--context-lengths", default=",".join(str(x) for x in DEFAULT_CONTEXT_LENGTHS))
    parser.add_argument("--document-depth-percents", default=",".join(str(x) for x in DEFAULT_DEPTHS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--memory-dir-prefix", default=DEFAULT_MEMORY_PREFIX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-name", default="")
    parser.add_argument("--report-name", default="")
    parser.add_argument("--heatmap-name", default="")
    parser.add_argument("--answer-timeout", type=int, default=240)
    parser.add_argument("--retrieve-top-k", type=int, default=12)
    return parser.parse_args()


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for part in str(raw or "").split(","):
        piece = part.strip()
        if not piece:
            continue
        values.append(int(piece))
    if not values:
        raise SystemExit("Expected at least one integer value.")
    return values


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        if Document is None:
            raise RuntimeError("python-docx is required to read .docx haystack files.")
        doc = Document(str(path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
    raise ValueError(f"Unsupported haystack file: {path}")


def load_haystack_corpus(haystack_dir: Path) -> str:
    files = sorted(
        path
        for path in haystack_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".docx"}
    )
    if not files:
        raise SystemExit(f"No supported haystack files found in {haystack_dir}")
    parts: List[str] = []
    for path in files:
        text = normalize_space(read_text_file(path))
        if text:
            parts.append(text)
    corpus = "\n\n".join(parts).strip()
    if not corpus:
        raise SystemExit(f"Haystack corpus is empty after reading {haystack_dir}")
    return corpus


def build_haystack_slice(corpus: str, target_length: int, repeat_index: int) -> str:
    if target_length <= 0:
        raise ValueError("context length must be positive")
    repeated = corpus
    while len(repeated) < target_length + len(corpus):
        repeated += "\n\n" + corpus
    stride = max(1, min(len(corpus), target_length // 3 or 1))
    max_start = max(0, len(repeated) - target_length)
    start = min((repeat_index * stride) % max(1, max_start + 1), max_start)
    chunk = repeated[start : start + target_length]
    if len(chunk) < target_length:
        chunk = (chunk + "\n\n" + repeated)[:target_length]
    return chunk


def inject_needle(haystack: str, needle: str, depth_percent: int) -> str:
    depth = max(0, min(100, int(depth_percent)))
    needle_block = "\n\n" + needle.strip() + "\n\n"
    insert_at = int(len(haystack) * (depth / 100.0))
    left_break = haystack.rfind("\n", 0, insert_at)
    right_break = haystack.find("\n", insert_at)
    if left_break >= 0:
        insert_at = left_break
    elif right_break >= 0:
        insert_at = right_break
    return haystack[:insert_at] + needle_block + haystack[insert_at:]


def find_needle_chunk_ids(mm: MemoryManager, needle: str) -> List[int]:
    needle_key = normalize_match(needle)
    out: List[int] = []
    cur = mm.conn.cursor()
    rows = cur.execute("SELECT id, path FROM chunks ORDER BY id ASC").fetchall()
    for row in rows:
        chunk_id = int(row["id"])
        path = Path(mm.memory_dir) / str(row["path"])
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if needle_key and needle_key in normalize_match(text):
            out.append(chunk_id)
    return out


def answer_hits_needle(answer: str, needle: str) -> bool:
    answer_key = normalize_match(answer)
    needle_key = normalize_match(needle)
    if not answer_key or not needle_key:
        return False
    return answer_key == needle_key or needle_key in answer_key or answer_key in needle_key


def summarize_runs(runs: Sequence[NIAHRun]) -> Dict[str, object]:
    total = len(runs)
    hits = sum(1 for run in runs if run.hit)
    by_context: Dict[int, Dict[str, object]] = {}
    by_depth: Dict[int, Dict[str, object]] = {}
    heatmap: Dict[Tuple[int, int], Dict[str, object]] = {}
    root_causes = Counter(run.root_cause for run in runs if not run.hit)
    for context_length in sorted({run.context_length for run in runs}):
        bucket = [run for run in runs if run.context_length == context_length]
        by_context[context_length] = {
            "runs": len(bucket),
            "hits": sum(1 for run in bucket if run.hit),
            "hit_rate": sum(1 for run in bucket if run.hit) / max(1, len(bucket)),
        }
    for depth in sorted({run.depth_percent for run in runs}):
        bucket = [run for run in runs if run.depth_percent == depth]
        by_depth[depth] = {
            "runs": len(bucket),
            "hits": sum(1 for run in bucket if run.hit),
            "hit_rate": sum(1 for run in bucket if run.hit) / max(1, len(bucket)),
        }
    for context_length in sorted({run.context_length for run in runs}):
        for depth in sorted({run.depth_percent for run in runs}):
            bucket = [run for run in runs if run.context_length == context_length and run.depth_percent == depth]
            heatmap[(context_length, depth)] = {
                "runs": len(bucket),
                "hits": sum(1 for run in bucket if run.hit),
                "hit_rate": sum(1 for run in bucket if run.hit) / max(1, len(bucket)),
            }
    return {
        "runs": total,
        "hits": hits,
        "misses": total - hits,
        "hit_rate": hits / max(1, total),
        "by_context_length": by_context,
        "by_depth_percent": by_depth,
        "heatmap": heatmap,
        "root_cause_breakdown": dict(root_causes),
    }


def write_heatmap_csv(path: Path, summary: Dict[str, object]) -> None:
    by_context = summary.get("by_context_length", {})
    by_depth = summary.get("by_depth_percent", {})
    context_lengths = sorted(int(key) for key in by_context.keys())
    depths = sorted(int(key) for key in by_depth.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["context_length"] + [str(depth) for depth in depths])
        for context_length in context_lengths:
            row = [str(context_length)]
            for depth in depths:
                cell = (summary.get("heatmap", {}) or {}).get((context_length, depth), {})
                row.append(f"{float(cell.get('hit_rate', 0.0)):.4f}")
            writer.writerow(row)


def write_results_json(path: Path, *, args: argparse.Namespace, summary: Dict[str, object], runs: Sequence[NIAHRun]) -> None:
    payload = {
        "benchmark": "evermate_system_niah",
        "model": args.model,
        "haystack_dir": str(Path(args.haystack_dir).expanduser().resolve()),
        "needle": args.needle,
        "retrieval_question": args.retrieval_question,
        "context_lengths": parse_int_list(args.context_lengths),
        "document_depth_percents": parse_int_list(args.document_depth_percents),
        "repeats": int(args.repeats),
        "summary": {
            **summary,
            "heatmap": {
                f"{context}:{depth}": value for (context, depth), value in (summary.get("heatmap", {}) or {}).items()
            },
        },
        "runs": [asdict(run) for run in runs],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, args: argparse.Namespace, summary: Dict[str, object], runs: Sequence[NIAHRun]) -> None:
    lines = [
        "# EverMate System NIAH Report",
        "",
        "## Setup",
        f"- Model: `{args.model}`",
        f"- Haystack Dir: `{Path(args.haystack_dir).expanduser().resolve()}`",
        f"- Needle: `{args.needle}`",
        f"- Retrieval Question: `{args.retrieval_question}`",
        f"- Context Lengths: `{', '.join(str(x) for x in parse_int_list(args.context_lengths))}`",
        f"- Depth Percents: `{', '.join(str(x) for x in parse_int_list(args.document_depth_percents))}`",
        f"- Repeats per grid point: `{int(args.repeats)}`",
        "",
        "## Overall",
        f"- Runs: `{summary['runs']}`",
        f"- Hits: `{summary['hits']}`",
        f"- Misses: `{summary['misses']}`",
        f"- Overall hit rate: `{float(summary['hit_rate']):.2%}`",
        "",
        "## By Context Length",
    ]
    for context_length, bucket in sorted((summary.get("by_context_length") or {}).items()):
        lines.append(
            f"- `{context_length}`: hits `{bucket['hits']}` / `{bucket['runs']}`, hit rate `{float(bucket['hit_rate']):.2%}`"
        )
    lines.extend(["", "## By Depth"])
    for depth, bucket in sorted((summary.get("by_depth_percent") or {}).items()):
        lines.append(
            f"- `{depth}%`: hits `{bucket['hits']}` / `{bucket['runs']}`, hit rate `{float(bucket['hit_rate']):.2%}`"
        )
    lines.extend(["", "## Root Causes"])
    breakdown = summary.get("root_cause_breakdown") or {}
    if breakdown:
        for key, value in sorted(breakdown.items()):
            lines.append(f"- `{key}`: `{int(value)}`")
    else:
        lines.append("- No misses.")

    lines.extend(["", "## Failed Examples"])
    failures = [run for run in runs if not run.hit]
    if failures:
        for run in failures[:10]:
            lines.append(
                f"- context `{run.context_length}`, depth `{run.depth_percent}%`, repeat `{run.repeat_index}`: "
                f"answer `{run.answer}`, root cause `{run.root_cause}`, needle chunks `{', '.join(str(cid) for cid in run.needle_chunk_ids) or 'n/a'}`, "
                f"retrieved `{', '.join(str(cid) for cid in run.retrieved_chunk_ids) or 'n/a'}`"
            )
    else:
        lines.append("- No failures.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    haystack_dir = Path(args.haystack_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not haystack_dir.exists():
        raise SystemExit(f"Haystack dir not found: {haystack_dir}")

    corpus = load_haystack_corpus(haystack_dir)
    slug = slugify(haystack_dir)
    results_path = output_dir / (args.results_name or f"{slug}-niah-results.json")
    report_path = output_dir / (args.report_name or f"{slug}-niah-report.md")
    heatmap_path = output_dir / (args.heatmap_name or f"{slug}-niah-heatmap.csv")

    context_lengths = parse_int_list(args.context_lengths)
    depths = parse_int_list(args.document_depth_percents)
    runs: List[NIAHRun] = []
    scratch_prefix = Path(args.memory_dir_prefix).expanduser()
    scratch_parent = scratch_prefix.parent if scratch_prefix.parent.exists() else Path(tempfile.gettempdir())
    scratch_name = scratch_prefix.name or "evermate-niah"
    scratch_parent.mkdir(parents=True, exist_ok=True)

    for context_length in context_lengths:
        for depth in depths:
            for repeat_index in range(1, int(args.repeats) + 1):
                haystack = build_haystack_slice(corpus, int(context_length), repeat_index - 1)
                injected = inject_needle(haystack, args.needle, depth)
                run_dir = Path(tempfile.mkdtemp(prefix=f"{scratch_name}-run-", dir=str(scratch_parent)))
                memory_dir = run_dir / "memory"
                source_path = run_dir / "haystack.txt"
                source_path.write_text(injected, encoding="utf-8")
                try:
                    mm = MemoryManager(MemoryConfig(memory_dir=str(memory_dir), retrieve_top_k=int(args.retrieve_top_k)))
                    stored = mm.import_files([str(source_path)])
                    if not stored:
                        raise RuntimeError("NIAH import failed: haystack source was not stored.")
                    mm.rebuild_memory()
                    plan = mm.build_turn_plan(
                        user_text=args.retrieval_question,
                        assistant_style="请只回答 needle 本身，不要解释，不要扩写。",
                        lang="zh",
                        answer_mode="fact",
                    )
                    bundle = dict(plan.get("bundle", {}))
                    if str(plan.get("direct_answer", "")).strip():
                        answer = str(plan.get("direct_answer", "")).strip()
                        used_fallback = True
                    else:
                        answer = ollama_chat(
                            list(plan.get("messages", [])),
                            model=args.model,
                            timeout=int(args.answer_timeout),
                            options={"temperature": 0, "num_predict": 192},
                        ).strip()
                        used_fallback = False
                        if mm.needs_fact_answer_fallback(answer, bundle, args.retrieval_question):
                            answer = mm.render_fact_answer(args.retrieval_question, retrieved_bundle=bundle).strip()
                            used_fallback = True
                    retrieved_chunk_ids = tuple(int(item.get("chunk_id", 0) or 0) for item in list(bundle.get("items", [])))
                    needle_chunk_ids = tuple(find_needle_chunk_ids(mm, args.needle))
                    answer_candidates = tuple(dict(item) for item in list(bundle.get("answer_candidates", []) or []))
                    canonical_answer = str(bundle.get("canonical_answer", "") or "")
                    fact_strategy = str(bundle.get("fact_strategy", "") or "")
                    hit = answer_hits_needle(answer, args.needle)
                    if hit:
                        root_cause = ""
                    elif not (set(needle_chunk_ids) & set(retrieved_chunk_ids)):
                        root_cause = "retrieval_miss"
                    elif canonical_answer and normalize_match(canonical_answer) != normalize_match(args.needle):
                        root_cause = "wrong_candidate_selection"
                    else:
                        root_cause = "answer_generation_error"
                    runs.append(
                        NIAHRun(
                            context_length=int(context_length),
                            depth_percent=int(depth),
                            repeat_index=int(repeat_index),
                            needle=args.needle,
                            retrieval_question=args.retrieval_question,
                            answer=answer,
                            hit=hit,
                            root_cause=root_cause,
                            retrieved_chunk_ids=retrieved_chunk_ids,
                            needle_chunk_ids=needle_chunk_ids,
                            answer_candidates=answer_candidates,
                            canonical_answer=canonical_answer,
                            fact_strategy=fact_strategy,
                            used_fallback=used_fallback,
                        )
                    )
                finally:
                    shutil.rmtree(run_dir, ignore_errors=True)

    summary = summarize_runs(runs)
    write_results_json(results_path, args=args, summary=summary, runs=runs)
    write_report(report_path, args=args, summary=summary, runs=runs)
    write_heatmap_csv(heatmap_path, summary)

    print(f"Wrote NIAH results to {results_path}")
    print(f"Wrote NIAH report to {report_path}")
    print(f"Wrote NIAH heatmap to {heatmap_path}")
    print(f"Overall hit rate: {float(summary['hit_rate']):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
