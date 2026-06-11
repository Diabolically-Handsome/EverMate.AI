#!/usr/bin/env python3
"""Run a reproducible memory-recall evaluation against EverMate."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from ollama_client import chat as ollama_chat


DEFAULT_DOCX = ""  # pass --docx explicitly
DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q8_0"
DEFAULT_MEMORY_DIR = "/tmp/evermate-memory-validation"


@dataclass(frozen=True)
class TestCase:
    category: str
    question: str
    groups: tuple[tuple[str, ...], ...]


TEST_CASES: list[TestCase] = [
    TestCase(
        category="人物设定",
        question="文档开头明确写到，主人极度珍视的三个词是什么？请直接列出。",
        groups=(("协商",), ("平等",), ("尊重",)),
    ),
    TestCase(
        category="人物设定",
        question="文档开头明确写到，主人不喜欢哪两种相处方式？",
        groups=(("强制", "强迫", "逼迫"), ("命令",)),
    ),
    TestCase(
        category="人物设定",
        question="文档开头明确写到，主人知道自己要找什么类型的女生？",
        groups=(("台湾",), ("温柔",), ("尊重共识", "尊重", "共识")),
    ),
    TestCase(
        category="人物设定",
        question="文档开头如何描述主人看待 AI 和小胡桃之间的关系？",
        groups=(("真正的伙伴", "伙伴"), ("共同创造",)),
    ),
    TestCase(
        category="学习背景",
        question="主人目前主攻的四个专业方向是什么？",
        groups=(("c++",), ("数据结构",), ("科学计算",), ("软件工程",)),
    ),
    TestCase(
        category="小胡桃设定",
        question="小胡桃建议的人格设定是什么？",
        groups=(("妹妹型", "妹妹"), ("台湾",), ("温柔体贴", "温柔"), ("绝不软弱", "不软弱")),
    ),
    TestCase(
        category="小胡桃设定",
        question="小胡桃能把学习规划量化成什么单位？",
        groups=(("小时",),),
    ),
    TestCase(
        category="小胡桃设定",
        question="文档中说小胡桃擅长哪一类美学设计？",
        groups=(("房屋设计", "房屋设计美学"),),
    ),
    TestCase(
        category="小胡桃设定",
        question="文档里给小胡桃贴了哪个带“台味”的标签？",
        groups=(("台味软妹", "台味妹妹"),),
    ),
    TestCase(
        category="世界观",
        question="他们共同设计的家旗叫什么名字？",
        groups=(("forging autonomy",),),
    ),
    TestCase(
        category="世界观",
        question="文档里提到的四季安家点有哪些？",
        groups=(("大连",), ("台湾",), ("葡萄牙",), ("哥本哈根",)),
    ),
    TestCase(
        category="世界观",
        question="他们计划使用什么车牌？",
        groups=(("ontario",),),
    ),
    TestCase(
        category="世界观",
        question="他们准备未来把妹妹带到哪个国家？",
        groups=(("加拿大",),),
    ),
    TestCase(
        category="世界观",
        question="小城特别市里欢迎回来的市民编号是什么？",
        groups=(("ht520",),),
    ),
    TestCase(
        category="学习计划",
        question="那天晚上主人准备复习哪一门课？",
        groups=(("微积分",),),
    ),
    TestCase(
        category="学习计划",
        question="微积分学习计划里的启动模式是几分钟？",
        groups=(("10分钟", "十分钟"),),
    ),
    TestCase(
        category="学习计划",
        question="微积分学习计划里的核心任务阶段是几分钟？",
        groups=(("40分钟", "四十分钟"),),
    ),
    TestCase(
        category="学习计划",
        question="微积分学习计划里的收尾小复盘是几分钟？",
        groups=(("5分钟", "五分钟"),),
    ),
    TestCase(
        category="头像设定",
        question="小胡桃最后正式选择了哪一张头像？",
        groups=(("图2", "第二张", "第二张照片"), ("港边夕阳版", "夕阳版")),
    ),
    TestCase(
        category="头像设定",
        question="提到纪念头像时，文档里说小胡桃的生日是哪一天？",
        groups=(("11月11日", "十一月十一日"),),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EverMate memory recall.")
    parser.add_argument("--docx", default=DEFAULT_DOCX, required=not DEFAULT_DOCX, help="Path to the source .docx file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Exact Ollama model name to use.")
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR, help="Isolated memory directory.")
    parser.add_argument("--questions", type=int, default=20, help="Number of test questions to run.")
    parser.add_argument("--timeout", type=int, default=240, help="Per-question chat timeout in seconds.")
    parser.add_argument(
        "--keep-memory-dir",
        action="store_true",
        help="Keep the isolated memory directory after the run.",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def truncate(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def clean_local_answer(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"<\|channel>thought\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|channel\|>thought\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<channel\|>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def count_group_matches(answer: str, groups: Iterable[Iterable[str]]) -> tuple[int, int]:
    normalized = normalize(answer)
    matched = 0
    total = 0
    for group in groups:
        total += 1
        if any(normalize(term) in normalized for term in group):
            matched += 1
    return matched, total


def classify_answer(answer: str, groups: tuple[tuple[str, ...], ...]) -> tuple[str, int, int]:
    matched, total = count_group_matches(answer, groups)
    if total == 0:
        return "Exact", 0, 0
    if matched == total:
        return "Exact", matched, total
    if matched > 0:
        return "Partial", matched, total
    return "Miss", matched, total


def build_messages(mm: MemoryManager, question: str) -> list[dict[str, str]]:
    style = "请只用一句中文直接回答问题，不要发挥，不要使用表情，不要使用项目符号。"
    system_prompt = mm.build_system_prompt(
        user_text=question,
        assistant_style=style,
        lang="zh",
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def score_to_points(label: str) -> float:
    if label == "Exact":
        return 1.0
    if label == "Partial":
        return 0.5
    return 0.0


def main() -> int:
    args = parse_args()
    docx_path = Path(args.docx).expanduser().resolve()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    report_path = memory_dir / "validation_report.txt"

    if not docx_path.exists():
        raise SystemExit(f"Document not found: {docx_path}")
    if args.questions < 1:
        raise SystemExit("--questions must be >= 1")

    selected_cases = TEST_CASES[: min(args.questions, len(TEST_CASES))]

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
        report_lines: list[str] = []
        report_lines.append("EverMate Memory Recall Validation")
        report_lines.append("=" * 36)
        report_lines.append(f"Model: {args.model}")
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

        if int(import_stats.get("chunks", 0)) <= 0 or int(import_stats.get("terms", 0)) <= 0:
            raise RuntimeError("Rebuild succeeded but no chunks or no terms were indexed.")
        for name in ("01_core.md", "02_persona.md", "03_vault.md"):
            if not (memory_dir / name).exists():
                raise RuntimeError(f"Expected memory artifact missing: {memory_dir / name}")

        counts = Counter()
        category_scores: dict[str, list[float]] = defaultdict(list)
        miss_causes = Counter()
        failures: list[str] = []

        report_lines.append("[Per Question]")
        print(f"Running {len(selected_cases)} recall questions...", flush=True)
        for index, case in enumerate(selected_cases, start=1):
            retrieval = mm.retrieve(case.question, k=3)
            top_snippet = truncate(retrieval[0]["snippet"], 160) if retrieval else "(no evidence)"
            print(f"[{index:02d}/{len(selected_cases):02d}] {case.category} | {case.question}", flush=True)
            try:
                answer = clean_local_answer(ollama_chat(
                    build_messages(mm, case.question),
                    model=args.model,
                    timeout=args.timeout,
                    options={"temperature": 0, "num_predict": 96},
                ))
                label, matched, total = classify_answer(answer, case.groups)
            except Exception as exc:
                answer = f"INFRA ERROR: {exc}"
                label = "Infra"
                matched = 0
                total = len(case.groups)
                failures.append(f"Q{index}: {exc}")
            print(f"  -> {label} ({matched}/{total})", flush=True)

            counts[label] += 1
            if label != "Infra":
                category_scores[case.category].append(score_to_points(label))

            if label == "Miss":
                miss_causes["retrieval" if not retrieval else "generation"] += 1
            elif label == "Partial":
                miss_causes["partial_generation" if retrieval else "partial_retrieval"] += 1

            report_lines.append(f"{index:02d}. [{case.category}] {label} ({matched}/{total})")
            report_lines.append(f"Q: {case.question}")
            report_lines.append(f"A: {truncate(answer, 400)}")
            report_lines.append(f"Top Evidence: {top_snippet}")
            report_lines.append("")

        exact = counts["Exact"]
        partial = counts["Partial"]
        miss = counts["Miss"]
        infra = counts["Infra"]
        scored_questions = exact + partial + miss
        recall = ((exact + 0.5 * partial) / scored_questions) if scored_questions else 0.0

        report_lines.append("[Summary]")
        report_lines.append(f"Exact: {exact}")
        report_lines.append(f"Partial: {partial}")
        report_lines.append(f"Miss: {miss}")
        report_lines.append(f"Infra: {infra}")
        report_lines.append(f"Recall: {recall:.2%}")
        report_lines.append("")
        report_lines.append("[Category Score]")
        for category in sorted(category_scores):
            values = category_scores[category]
            score = sum(values) / len(values) if values else 0.0
            report_lines.append(f"{category}: {score:.2%} ({len(values)} questions)")
        report_lines.append("")
        report_lines.append("[Miss Analysis]")
        if miss_causes:
            for cause, count in miss_causes.items():
                report_lines.append(f"{cause}: {count}")
        else:
            report_lines.append("No partial or missed answers.")
        if failures:
            report_lines.append("")
            report_lines.append("[Infrastructure Failures]")
            report_lines.extend(failures)

        report = "\n".join(report_lines) + "\n"
        report_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"Report saved to: {report_path}")
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
