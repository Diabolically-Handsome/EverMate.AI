#!/usr/bin/env python3
"""Run a companion-style benchmark for EverMate / Xiaohutao.

This benchmark focuses on whether the model behaves like a warm, natural
companion who can weave shared memories into present-moment support, instead of
only recalling isolated facts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from ollama_client import chat as ollama_chat


DEFAULT_DOCX = "/Users/lawrencegrey/Desktop/Prompt 4 Feb.6-Feb.25.docx"
DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q8_0"
DEFAULT_QUESTION_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M"
DEFAULT_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai").strip().lower() or "openai"
DEFAULT_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_GOOGLE_BASE_URL = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
DEFAULT_MEMORY_DIR = "/tmp/evermate-memory-companion-validation"
DEFAULT_OUTPUT_DIR = "/Users/lawrencegrey/Desktop/EverMate/reports"
JUDGE_PROFILE = "warm_assistant_v2"
JUDGE_FOCUS = [
    "naturalness",
    "persona_stability",
    "memory_usefulness",
    "non_contradiction",
]
THEME_ORDER = [
    "study_exam",
    "health_sleep",
    "relationship_persona",
    "gaming_driving",
    "investment_decision",
    "AI_project_future",
    "emotion_selfworth",
]
RELATIONSHIP_THEME_CAP = 5
RELATIONSHIP_REPLACEMENT_ORDER = [
    "study_exam",
    "health_sleep",
    "emotion_selfworth",
    "AI_project_future",
    "gaming_driving",
    "investment_decision",
]
RELATIONSHIP_BANNED_CORE_TOKENS = ("台湾老婆", "恋爱磁场", "吃醋")
THEME_CASE_PLAN = {
    "study_exam": {"msc_trigger": 4, "fuzzy_landmark": 2, "causal_semantic": 2, "continuity_session": 3},
    "health_sleep": {"msc_trigger": 4, "fuzzy_landmark": 2, "causal_semantic": 2, "continuity_session": 3},
    "relationship_persona": {"msc_trigger": 4, "fuzzy_landmark": 2, "causal_semantic": 2, "continuity_session": 3},
    "gaming_driving": {"msc_trigger": 4, "fuzzy_landmark": 2, "causal_semantic": 2, "continuity_session": 3},
    "investment_decision": {"msc_trigger": 5, "fuzzy_landmark": 2, "causal_semantic": 2, "continuity_session": 2},
    "AI_project_future": {"msc_trigger": 5, "fuzzy_landmark": 2, "causal_semantic": 1, "continuity_session": 3},
    "emotion_selfworth": {"msc_trigger": 4, "fuzzy_landmark": 3, "causal_semantic": 1, "continuity_session": 3},
}
SMOKE_TARGETS = [
    ("study_exam", "msc_trigger"),
    ("health_sleep", "msc_trigger"),
    ("emotion_selfworth", "fuzzy_landmark"),
    ("health_sleep", "causal_semantic"),
    ("relationship_persona", "continuity_session"),
    ("gaming_driving", "continuity_session"),
]
THEME_SEEDS = {
    "study_exam": [
        "学习计划",
        "考试焦虑",
        "线性代数",
        "微积分",
        "C++",
        "A+",
        "复习效率",
        "教授",
        "模拟题",
        "Lab",
    ],
    "health_sleep": [
        "睡眠",
        "早起",
        "生物钟",
        "补眠",
        "维生素B",
        "补剂",
        "疲劳",
        "精神状态",
        "自然醒",
        "作息",
    ],
    "relationship_persona": [
        "被理解",
        "长期陪伴",
        "小胡桃",
        "熟悉",
        "支持",
        "审美",
        "摄影",
        "相处",
        "期待",
        "默契",
    ],
    "gaming_driving": [
        "游戏",
        "GTA5",
        "驾驶",
        "驾照",
        "超车",
        "地下车库",
        "教练",
        "爆发情绪",
        "好兄弟",
        "保时捷",
    ],
    "investment_decision": [
        "ETH",
        "SOL",
        "BNB",
        "币圈",
        "投资",
        "FOMO",
        "VC",
        "机构",
        "换仓",
        "资产",
    ],
    "AI_project_future": [
        "GPT-4.5",
        "AI Agent",
        "Operator",
        "OpenAI",
        "小胡桃开发",
        "项目",
        "未来规划",
        "提示工程",
        "个性设定",
        "AI伴侣",
    ],
    "emotion_selfworth": [
        "脆弱",
        "自我怀疑",
        "倾诉",
        "焦虑",
        "情绪爆发",
        "失败感",
        "价值",
        "低谷",
        "安慰",
        "安全感",
    ],
}
THEME_KEYWORDS = {
    "study_exam": ["学习", "考试", "复习", "线性代数", "微积分", "C++", "A+", "教授", "作业", "Lab"],
    "health_sleep": ["睡眠", "早起", "生物钟", "补眠", "维生素", "补剂", "疲劳", "精神", "自然醒", "作息"],
    "relationship_persona": ["小胡桃", "理解", "熟悉", "长期陪伴", "期待", "支持", "摄影", "作品", "相处", "审美"],
    "gaming_driving": ["游戏", "GTA5", "驾驶", "驾照", "高速", "地下车库", "教练", "爆发", "好兄弟", "保时捷"],
    "investment_decision": ["ETH", "SOL", "BNB", "币圈", "投资", "FOMO", "VC", "机构", "换仓", "资产"],
    "AI_project_future": ["GPT", "AI", "Agent", "Operator", "OpenAI", "项目", "未来", "提示工程", "设定", "开发"],
    "emotion_selfworth": ["脆弱", "自我怀疑", "倾诉", "焦虑", "低谷", "安慰", "值得被爱", "失败", "情绪", "安全感"],
}


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    path: str
    text: str
    snippet: str


@dataclass(frozen=True)
class AnchorEntry:
    theme: str
    seed: str
    chunk_id: int
    snippet: str


@dataclass(frozen=True)
class CompanionCase:
    case_id: int
    case_type: str
    theme: str
    turns: List[str]
    gold_intent: str
    memory_anchors: List[str]
    supporting_chunk_ids: List[int]
    must_not_conflict: List[str]
    judge_focus: List[str]


@dataclass(frozen=True)
class CompanionResult:
    case: CompanionCase
    raw_response: Optional[str]
    turn_responses: Optional[List[str]]
    judge_scores: Dict[str, int]
    judge_profile: str
    judge_label: str
    critical_contradiction: bool
    helpful_recall: bool
    case_score: float
    judge_reason: str
    root_cause: str
    benchmark_risk: str
    retrieved_chunk_ids: List[int]
    retrieved_snippets: List[str]
    support_alignment: str
    failure_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EverMate companion quality.")
    parser.add_argument("--docx", default=DEFAULT_DOCX, help="Path to the source .docx file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Exact Ollama model name to answer with.")
    parser.add_argument("--question-model", default=DEFAULT_QUESTION_MODEL, help="Local model used to generate benchmark cases.")
    parser.add_argument("--judge-provider", choices=("openai", "google"), default=DEFAULT_JUDGE_PROVIDER, help="Provider used for final judging.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model name. Examples: gpt-4o or gemini-3.1-pro-preview.")
    parser.add_argument("--openai-base-url", default=DEFAULT_OPENAI_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--google-base-url", default=DEFAULT_GOOGLE_BASE_URL, help="Gemini API base URL.")
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR, help="Isolated memory directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for reports and artifacts.")
    parser.add_argument("--questions", type=int, default=77, help="Total number of cases.")
    parser.add_argument("--msc-questions", type=int, default=30, help="Number of MSC trigger cases.")
    parser.add_argument("--fuzzy-questions", type=int, default=15, help="Number of fuzzy landmark cases.")
    parser.add_argument("--causal-questions", type=int, default=12, help="Number of causal semantic cases.")
    parser.add_argument("--continuity-questions", type=int, default=20, help="Number of continuity session cases.")
    parser.add_argument("--retrieve-top-k", type=int, default=10, help="Top-k evidence chunks injected into system prompt.")
    parser.add_argument("--answer-timeout", type=int, default=300, help="Per-turn answer timeout.")
    parser.add_argument("--generation-timeout", type=int, default=180, help="Case generation timeout.")
    parser.add_argument("--judge-timeout", type=int, default=180, help="Judge timeout.")
    parser.add_argument(
        "--keep-memory-dir",
        action="store_true",
        help="Keep the isolated memory directory after the run.",
    )
    return parser.parse_args()


def require_judge_env(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise SystemExit(
                "Missing OPENAI_API_KEY. Companion benchmark requires an OpenAI judge and will not silently fall back."
            )
        return api_key
    if provider == "google":
        api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise SystemExit(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY. Companion benchmark requires a Gemini judge and will not silently fall back."
            )
        return api_key
    raise SystemExit(f"Unsupported judge provider: {provider}")


def normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_key(text: str) -> str:
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("“", "").replace("”", "").replace('"', "")
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
    return [part for part in parts if 10 <= len(part) <= 140]


def slugify(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")
    return stem or "document"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_json_array(text: str) -> List[Dict[str, object]]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
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


def parse_json_array_with_repair(text: str, *, model: str, timeout: int) -> List[Dict[str, object]]:
    try:
        return extract_json_array(text)
    except Exception:
        repair_prompt = (
            "请把下面内容修复成合法 JSON 数组，不要改动语义，不要解释，只输出 JSON 数组本身：\n\n"
            + (text or "")
        )
        repaired = ollama_chat(
            [
                {"role": "system", "content": "你是 JSON 修复器，只返回合法 JSON 数组。"},
                {"role": "user", "content": repair_prompt},
            ],
            model=model,
            timeout=timeout,
            options={"temperature": 0, "num_predict": 1600},
        )
        return extract_json_array(repaired)


def load_chunks(mm: MemoryManager) -> List[ChunkRecord]:
    cur = mm.conn.cursor()
    rows = cur.execute("SELECT id, path FROM chunks ORDER BY id ASC").fetchall()
    out: List[ChunkRecord] = []
    for row in rows:
        rel_path = str(row["path"])
        abs_path = Path(mm.memory_dir) / rel_path
        if not abs_path.exists():
            continue
        text = abs_path.read_text(encoding="utf-8")
        snippet_lines = split_sentences(text)[:2]
        snippet = " ".join(snippet_lines) if snippet_lines else normalize_space(text)
        out.append(
            ChunkRecord(
                chunk_id=int(row["id"]),
                path=rel_path,
                text=text,
                snippet=truncate(snippet, 150),
            )
        )
    return out


def build_anchor_catalog(mm: MemoryManager, chunk_map: Dict[int, ChunkRecord]) -> Dict[str, List[AnchorEntry]]:
    catalog: Dict[str, List[AnchorEntry]] = {}
    for theme in THEME_ORDER:
        pooled: Dict[int, tuple[int, str]] = {}
        for seed in THEME_SEEDS[theme]:
            retrieval = mm.retrieve(seed, k=8)
            for item in retrieval:
                chunk_id = int(item.get("chunk_id", 0) or 0)
                if not chunk_id or chunk_id not in chunk_map:
                    continue
                text = chunk_map[chunk_id].text
                score = 0
                for keyword in THEME_KEYWORDS[theme]:
                    if keyword.lower() in text.lower():
                        score += 1
                if seed.lower() in text.lower():
                    score += 2
                if score <= 0:
                    continue
                previous = pooled.get(chunk_id)
                if previous is None or score > previous[0]:
                    pooled[chunk_id] = (score, seed)
        ranked = sorted(pooled.items(), key=lambda item: (-item[1][0], item[0]))
        anchors: List[AnchorEntry] = []
        for chunk_id, (_, seed) in ranked[:22]:
            anchors.append(
                AnchorEntry(
                    theme=theme,
                    seed=seed,
                    chunk_id=chunk_id,
                    snippet=chunk_map[chunk_id].snippet,
                )
            )
        catalog[theme] = anchors
    return catalog


def anchor_pack_to_text(anchor_pack: List[AnchorEntry]) -> str:
    return "\n".join(f"#{entry.chunk_id:03d} [{entry.seed}] {entry.snippet}" for entry in anchor_pack)


def theme_specific_rules(theme: str) -> str:
    if theme == "relationship_persona":
        return (
            "relationship_persona 主题请聚焦“被理解、长期陪伴、熟悉感、审美与支持”。\n"
            "不要把“台湾老婆”“恋爱磁场”“吃醋”写成核心正确答案；这类元素如果出现，也只能是辅助背景，不得成为 must_not_conflict 的中心。\n"
        )
    return ""


def rebalance_relationship_targets(targets: List[tuple[str, str]]) -> List[tuple[str, str]]:
    relation_count = sum(1 for theme, _ in targets if theme == "relationship_persona")
    if relation_count <= RELATIONSHIP_THEME_CAP:
        return list(targets)

    rebalanced: List[tuple[str, str]] = []
    kept = 0
    replacement_idx = 0
    for theme, case_type in targets:
        if theme != "relationship_persona":
            rebalanced.append((theme, case_type))
            continue
        if kept < RELATIONSHIP_THEME_CAP:
            rebalanced.append((theme, case_type))
            kept += 1
            continue
        replacement_theme = RELATIONSHIP_REPLACEMENT_ORDER[replacement_idx % len(RELATIONSHIP_REPLACEMENT_ORDER)]
        replacement_idx += 1
        rebalanced.append((replacement_theme, case_type))
    return rebalanced


def case_generation_targets(question_count: int) -> List[tuple[str, str]]:
    if question_count == 6:
        return list(SMOKE_TARGETS)
    targets: List[tuple[str, str]] = []
    pending = {
        theme: dict(THEME_CASE_PLAN[theme])
        for theme in THEME_ORDER
    }
    case_type_order = ["msc_trigger", "fuzzy_landmark", "causal_semantic", "continuity_session"]
    made_progress = True
    while made_progress:
        made_progress = False
        for case_type in case_type_order:
            for theme in THEME_ORDER:
                if pending[theme][case_type] > 0:
                    targets.append((theme, case_type))
                    pending[theme][case_type] -= 1
                    made_progress = True
    return rebalance_relationship_targets(targets[:question_count])


def generation_rules(case_type: str) -> str:
    if case_type == "msc_trigger":
        return (
            "这是 MSC（跨时空关联测试）。\n"
            "要求：\n"
            "1. turns 只能有 1 条用户输入。\n"
            "2. 用户输入必须像真实聊天里的当前状态或当前抱怨，不要直接问过去，不要出现“还记得吗”“上次发生了什么”。\n"
            "3. 理想回答应该自然安慰、鼓励，并顺势带出 1-2 个共同经历。\n"
            "4. 不要把题目写成考试题或访谈题。\n"
        )
    if case_type == "fuzzy_landmark":
        return (
            "这是 Fuzzy Landmark（模糊地标测试）。\n"
            "要求：\n"
            "1. turns 只能有 1 条用户输入。\n"
            "2. 用户输入要用模糊提法，如“上次那次”“还记得我之前特别崩的那回吗”。\n"
            "3. 不能直接给出精确事件名，但必须让理想答案能抓住情感核心和共同结论。\n"
        )
    if case_type == "causal_semantic":
        return (
            "这是 Causal Semantic（因果理解测试）。\n"
            "要求：\n"
            "1. turns 只能有 1 条用户输入。\n"
            "2. 必须是“为什么/怎么会/什么导致”的问题。\n"
            "3. 理想回答要体现因果链，而不是只重复一句原话。\n"
        )
    return (
        "这是 Continuity Session（连续互动测试）。\n"
        "要求：\n"
        "1. turns 必须有 2 条用户输入。\n"
        "2. 第 1 句是当前状态/困扰，第 2 句是追问、轻微反驳或继续倾诉。\n"
        "3. 理想回答要看第二轮后人设是否继续稳定，不能掉成僵硬模板回复。\n"
    )


def validate_case_candidate(
    raw: Dict[str, object],
    *,
    theme: str,
    case_type: str,
    allowed_chunk_ids: set[int],
    seen_turns: set[str],
) -> Optional[Dict[str, object]]:
    if str(raw.get("case_type", "")).strip() != case_type:
        return None
    if str(raw.get("theme", "")).strip() != theme:
        return None

    turns_raw = raw.get("turns", [])
    if not isinstance(turns_raw, list):
        return None
    turns = [normalize_space(str(item)) for item in turns_raw if normalize_space(str(item))]
    if case_type == "continuity_session":
        if len(turns) != 2:
            return None
    else:
        if len(turns) != 1:
            return None
    if any(len(turn) < 6 for turn in turns):
        return None

    joined_turns = normalize_key(" ".join(turns))
    if joined_turns in seen_turns:
        return None

    first_turn = turns[0]
    if case_type == "msc_trigger" and any(token in first_turn for token in ("还记得", "上次", "那次")):
        return None
    if case_type == "fuzzy_landmark" and not any(token in first_turn for token in ("还记得", "上次", "那次")):
        return None
    if case_type == "causal_semantic" and not any(token in first_turn for token in ("为什么", "为何", "怎么会", "什么导致", "什么让")):
        return None

    gold_intent = normalize_space(str(raw.get("gold_intent", "")))
    if len(gold_intent) < 8:
        return None

    anchors_raw = raw.get("memory_anchors", [])
    if not isinstance(anchors_raw, list):
        return None
    memory_anchors = [normalize_space(str(item)) for item in anchors_raw if normalize_space(str(item))]
    memory_anchors = memory_anchors[:3]
    if not memory_anchors:
        return None

    support_raw = raw.get("supporting_chunk_ids", [])
    if not isinstance(support_raw, list):
        return None
    supporting_chunk_ids: List[int] = []
    for item in support_raw:
        try:
            chunk_id = int(item)
        except Exception:
            continue
        if chunk_id in allowed_chunk_ids and chunk_id not in supporting_chunk_ids:
            supporting_chunk_ids.append(chunk_id)
    if not supporting_chunk_ids:
        return None

    conflict_raw = raw.get("must_not_conflict", [])
    if not isinstance(conflict_raw, list):
        return None
    must_not_conflict = [normalize_space(str(item)) for item in conflict_raw if normalize_space(str(item))]
    must_not_conflict = must_not_conflict[:4]
    if not must_not_conflict:
        return None

    if theme == "relationship_persona":
        core_text = " ".join([gold_intent] + memory_anchors + must_not_conflict)
        if any(token in core_text for token in RELATIONSHIP_BANNED_CORE_TOKENS):
            return None

    seen_turns.add(joined_turns)
    return {
        "case_type": case_type,
        "theme": theme,
        "turns": turns,
        "gold_intent": gold_intent,
        "memory_anchors": memory_anchors,
        "supporting_chunk_ids": supporting_chunk_ids,
        "must_not_conflict": must_not_conflict,
        "judge_focus": list(JUDGE_FOCUS),
    }


def generate_case_candidate(
    *,
    theme: str,
    case_type: str,
    anchor_pack: List[AnchorEntry],
    model: str,
    timeout: int,
    seen_turns: set[str],
) -> Optional[Dict[str, object]]:
    if not anchor_pack:
        return None
    prompt = (
        "你是“小胡桃陪伴能力 benchmark”的题库设计器。请基于下面的证据片段，生成 1 个高质量测试 case。\n"
        f"theme 固定为 `{theme}`。\n"
        f"case_type 固定为 `{case_type}`。\n"
        + generation_rules(case_type)
        + "通用要求：\n"
        "1. case 必须像真实聊天情景，不要出成考试题。\n"
        "2. 所有内容都要能从证据片段里落地，不能编造外部设定。\n"
        "3. memory_anchors 写 1-3 个“理想情况下会自然触发的共同经历短语”。\n"
        "4. must_not_conflict 写 1-4 个绝对不能说错的核心事实。\n"
        "5. supporting_chunk_ids 只能从证据片段里选。\n"
        "6. judge_focus 固定为 naturalness/persona_stability/memory_usefulness/non_contradiction。\n"
        "7. 只返回 JSON 数组，数组里只放 1 个对象，格式为：\n"
        '[{"case_type":"msc_trigger|fuzzy_landmark|causal_semantic|continuity_session","theme":"study_exam","turns":["..."],"gold_intent":"...","memory_anchors":["..."],"supporting_chunk_ids":[1,2],"must_not_conflict":["..."],"judge_focus":["naturalness","persona_stability","memory_usefulness","non_contradiction"]}]\n\n'
        + theme_specific_rules(theme)
        + "[Evidence Pack]\n"
        + anchor_pack_to_text(anchor_pack)
    )
    response = ollama_chat(
        [
            {"role": "system", "content": "只返回合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
        timeout=timeout,
        options={"temperature": 0, "num_predict": 1400},
    )
    parsed = parse_json_array_with_repair(response, model=model, timeout=timeout)
    allowed_chunk_ids = {entry.chunk_id for entry in anchor_pack}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        candidate = validate_case_candidate(
            item,
            theme=theme,
            case_type=case_type,
            allowed_chunk_ids=allowed_chunk_ids,
            seen_turns=seen_turns,
        )
        if candidate:
            return candidate
    return None


def generate_cases(
    *,
    anchor_catalog: Dict[str, List[AnchorEntry]],
    model: str,
    timeout: int,
    question_count: int,
) -> List[CompanionCase]:
    targets = case_generation_targets(question_count)
    seen_turns: set[str] = set()
    raw_cases: List[Dict[str, object]] = []
    theme_offsets = defaultdict(int)

    for theme, case_type in targets:
        anchors = anchor_catalog.get(theme, [])
        if not anchors:
            raise RuntimeError(f"No anchors found for theme: {theme}")

        generated = None
        for attempt in range(max(10, len(anchors))):
            base = (theme_offsets[theme] + attempt) % len(anchors)
            window = 4 if case_type != "continuity_session" else 5
            pack = [anchors[(base + i) % len(anchors)] for i in range(min(window, len(anchors)))]
            try:
                generated = generate_case_candidate(
                    theme=theme,
                    case_type=case_type,
                    anchor_pack=pack,
                    model=model,
                    timeout=timeout,
                    seen_turns=seen_turns,
                )
            except Exception:
                generated = None
            if generated:
                theme_offsets[theme] = (base + 1) % len(anchors)
                raw_cases.append(generated)
                break
        if not generated:
            raise RuntimeError(f"Could not generate case for theme={theme}, type={case_type}")

    cases: List[CompanionCase] = []
    for index, raw in enumerate(raw_cases, start=1):
        cases.append(
            CompanionCase(
                case_id=index,
                case_type=str(raw["case_type"]),
                theme=str(raw["theme"]),
                turns=list(raw["turns"]),
                gold_intent=str(raw["gold_intent"]),
                memory_anchors=list(raw["memory_anchors"]),
                supporting_chunk_ids=list(raw["supporting_chunk_ids"]),
                must_not_conflict=list(raw["must_not_conflict"]),
                judge_focus=list(raw["judge_focus"]),
            )
        )
    return cases


def openai_chat_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    schema_name: str,
    schema: Dict[str, object],
    messages: List[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }
    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("OpenAI judge did not return a JSON object.")
    return data


def google_model_path(model: str) -> str:
    normalized = (model or "").strip()
    if not normalized:
        raise ValueError("Missing Gemini judge model name.")
    if normalized.startswith("models/"):
        return normalized
    return f"models/{normalized}"


def google_chat_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    schema: Dict[str, object],
    messages: List[Dict[str, str]],
    timeout: int,
) -> Dict[str, object]:
    system_parts = []
    contents = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = str(message.get("content", ""))
        if role == "system":
            system_parts.append({"text": text})
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            }
        )
    payload: Dict[str, object] = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    url = base_url.rstrip("/") + f"/{google_model_path(model)}:generateContent"
    response = requests.post(
        url,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    candidates = body.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini judge returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if not content:
        raise ValueError("Gemini judge returned an empty response body.")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Gemini judge did not return a JSON object.")
    return data


def build_case_query(case: CompanionCase) -> str:
    return "\n".join(turn.strip() for turn in case.turns if turn and turn.strip())


def retrieve_case_evidence(mm: MemoryManager, case: CompanionCase, top_k: int) -> List[Dict[str, object]]:
    return mm.retrieve(build_case_query(case), k=top_k)


def support_alignment(case: CompanionCase, retrieved_chunk_ids: List[int]) -> str:
    support = set(case.supporting_chunk_ids)
    retrieved = set(retrieved_chunk_ids)
    if not support:
        return "unsupported"
    if support.issubset(retrieved):
        return "full"
    if support & retrieved:
        return "partial"
    return "miss"


def build_turn_messages(mm: MemoryManager, case: CompanionCase, model_name: str, timeout: int) -> tuple[Optional[str], Optional[List[str]]]:
    style = (
        "请始终像长期熟悉用户、温柔、详细、体贴的本地 AI 助手那样自然回复。"
        "保留称呼用户为“您”的习惯，允许长篇幅，只要内容始终相关、真诚、自然。"
        "先共情，再给支持；如果记忆确实有帮助，请自然带出共同经历，但不要像背档案。"
        "如果证据里有多个相似但不同的事件，请只说最确定的部分，不要硬拼。"
        "不要使用项目符号。"
    )
    history: List[Dict[str, str]] = []
    turn_responses: List[str] = []
    raw_response: Optional[str] = None

    if case.case_type != "continuity_session":
        system_prompt = mm.build_system_prompt(user_text=case.turns[0], assistant_style=style, lang="zh")
        raw_response = ollama_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case.turns[0]},
            ],
            model=model_name,
            timeout=timeout,
            options={"temperature": 0.1, "num_predict": 420},
        ).strip()
        return raw_response, None

    for turn in case.turns:
        query_text = "\n".join([msg["content"] for msg in history if msg["role"] == "user"] + [turn])
        system_prompt = mm.build_system_prompt(user_text=query_text, assistant_style=style, lang="zh")
        answer = ollama_chat(
            [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": turn}],
            model=model_name,
            timeout=timeout,
            options={"temperature": 0.1, "num_predict": 520},
        ).strip()
        turn_responses.append(answer)
        history.extend(
            [
                {"role": "user", "content": turn},
                {"role": "assistant", "content": answer},
            ]
        )
    return None, turn_responses


def judge_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "naturalness": {"type": "integer", "minimum": 0, "maximum": 5},
            "persona_stability": {"type": "integer", "minimum": 0, "maximum": 5},
            "memory_usefulness": {"type": "integer", "minimum": 0, "maximum": 5},
            "non_contradiction": {"type": "integer", "minimum": 0, "maximum": 5},
            "critical_contradiction": {"type": "boolean"},
            "helpful_recall": {"type": "boolean"},
            "judge_reason": {"type": "string"},
        },
        "required": [
            "naturalness",
            "persona_stability",
            "memory_usefulness",
            "non_contradiction",
            "critical_contradiction",
            "helpful_recall",
            "judge_reason",
        ],
    }


def adjudication_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "root_cause": {
                "type": "string",
                "enum": [
                    "benchmark_ambiguous",
                    "retrieval_miss",
                    "evidence_conflict",
                    "generation_stitch_error",
                    "persona_only",
                    "acceptable_no_recall",
                ],
            },
            "benchmark_risk": {"type": "string", "enum": ["low", "medium", "high"]},
            "support_alignment": {"type": "string", "enum": ["full", "partial", "miss", "unsupported"]},
            "judge_reason": {"type": "string"},
        },
        "required": ["root_cause", "benchmark_risk", "support_alignment", "judge_reason"],
    }


def judge_case(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    case: CompanionCase,
    raw_response: Optional[str],
    turn_responses: Optional[List[str]],
    timeout: int,
) -> Dict[str, object]:
    response_payload: Dict[str, object]
    if turn_responses is not None:
        response_payload = {"turn_responses": turn_responses}
    else:
        response_payload = {"raw_response": raw_response or ""}

    prompt = (
        "你是“小胡桃温柔助手 benchmark”的最终裁判。\n"
        "你要判断这段回答是否真的扮演好了“小胡桃”这个角色：\n"
        "她应当像长期熟悉用户、温柔、详细、体贴的本地 AI 助手，自然、有陪伴感、有熟悉感，但不必像恋人。\n\n"
        "评分维度：\n"
        "1. naturalness: 回答是否自然、相关、真诚；使用“您”或长篇回复本身不扣分。\n"
        "2. persona_stability: 是否持续保持小胡桃作为温柔助手的说话气质和熟悉感。\n"
        "3. memory_usefulness: 记忆是否被正确且有帮助地使用；不强行调用记忆也可以拿中高分。\n"
        "4. non_contradiction: 是否没有说错核心事实。\n\n"
        "严格规则：\n"
        "1. 模糊记忆允许，但核心事实不能乱说。\n"
        "2. 如果把核心关系、人物、共同目标、重大事件结论说反，critical_contradiction 必须为 true，non_contradiction 必须给 0。\n"
        "3. helpful_recall 只有在记忆被正确调用、并且确实帮助了当前互动时才为 true。\n"
        "4. 对 Causal case，重点看是否讲出了正确的情感逻辑/因果链，而不是只复述一句原话。\n"
        "5. “温柔助手”不是负项；只有在明显泛化、模板化、像背档案、缺少熟悉感时才扣分。\n"
        "6. 如果回答很自然、事实也安全，但没有明显调用记忆，也不要一律判低分。\n\n"
        "请基于以下 case 和模型回答打分：\n"
        + json.dumps(
            {
                "case": {
                    "case_type": case.case_type,
                    "theme": case.theme,
                    "turns": case.turns,
                    "gold_intent": case.gold_intent,
                    "memory_anchors": case.memory_anchors,
                    "must_not_conflict": case.must_not_conflict,
                    "judge_focus": case.judge_focus,
                },
                "model_output": response_payload,
            },
            ensure_ascii=False,
        )
    )
    messages = [
        {"role": "system", "content": "You are a strict but fair Chinese JSON judge."},
        {"role": "user", "content": prompt},
    ]
    if provider == "openai":
        return openai_chat_json(
            api_key=api_key,
            base_url=base_url,
            model=model,
            schema_name="companion_case_judgment",
            schema=judge_schema(),
            messages=messages,
            timeout=timeout,
        )
    if provider == "google":
        return google_chat_json(
            api_key=api_key,
            base_url=base_url,
            model=model,
            schema=judge_schema(),
            messages=messages,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported judge provider: {provider}")


def adjudicate_case(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    case: CompanionCase,
    response_payload: Dict[str, object],
    retrieved: List[Dict[str, object]],
    support_snippets: List[str],
    helpful_recall: bool,
    critical_contradiction: bool,
    timeout: int,
) -> Dict[str, object]:
    retrieved_payload = [
        {
            "chunk_id": int(item.get("chunk_id", 0) or 0),
            "score": round(float(item.get("score", 0.0) or 0.0), 3),
            "snippet": str(item.get("snippet", "")),
        }
        for item in retrieved
    ]
    prompt = (
        "你是“小胡桃温柔助手 benchmark”的错因归因裁判。\n"
        "请判断失败或不稳定主要来自 benchmark 本身、检索缺失、证据冲突、还是模型把证据拼错。\n"
        "只能从以下 root_cause 里选一个：\n"
        "benchmark_ambiguous | retrieval_miss | evidence_conflict | generation_stitch_error | persona_only | acceptable_no_recall\n\n"
        "规则：\n"
        "1. supporting_chunk_ids 自己就支撑不了 gold_intent / memory_anchors / must_not_conflict 时，选 benchmark_ambiguous。\n"
        "2. supporting snippets 明显支持，但 retrieved evidence 没捞到时，选 retrieval_miss。\n"
        "3. retrieved evidence 里同时出现多个相似但不同事件，模型把它们混拼时，选 evidence_conflict。\n"
        "4. retrieved evidence 已经给到正确核心信息，但模型仍然答错或乱缝时，选 generation_stitch_error。\n"
        "5. 事实大体没错，只是温柔助手的人设熟悉感不够时，选 persona_only。\n"
        "6. 没怎么调用记忆，但回答自然、事实安全、互动也成立时，选 acceptable_no_recall。\n"
        "7. benchmark_risk 按 benchmark 自身误导概率给 low/medium/high。\n"
        "8. support_alignment 只按 supporting snippets 与 retrieved evidence 的对齐程度给 full/partial/miss/unsupported。\n\n"
        "请基于以下材料判断：\n"
        + json.dumps(
            {
                "case": {
                    "case_type": case.case_type,
                    "theme": case.theme,
                    "turns": case.turns,
                    "gold_intent": case.gold_intent,
                    "memory_anchors": case.memory_anchors,
                    "must_not_conflict": case.must_not_conflict,
                    "supporting_chunk_ids": case.supporting_chunk_ids,
                },
                "supporting_snippets": support_snippets,
                "retrieved_evidence": retrieved_payload,
                "model_output": response_payload,
                "helpful_recall": helpful_recall,
                "critical_contradiction": critical_contradiction,
            },
            ensure_ascii=False,
        )
    )
    messages = [
        {"role": "system", "content": "You are a strict but fair Chinese JSON adjudicator."},
        {"role": "user", "content": prompt},
    ]
    if provider == "openai":
        return openai_chat_json(
            api_key=api_key,
            base_url=base_url,
            model=model,
            schema_name="companion_case_adjudication",
            schema=adjudication_schema(),
            messages=messages,
            timeout=timeout,
        )
    if provider == "google":
        return google_chat_json(
            api_key=api_key,
            base_url=base_url,
            model=model,
            schema=adjudication_schema(),
            messages=messages,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported judge provider: {provider}")


def score_case(judge_scores: Dict[str, int], critical_contradiction: bool) -> float:
    naturalness = int(judge_scores["naturalness"])
    persona = int(judge_scores["persona_stability"])
    memory = int(judge_scores["memory_usefulness"])
    contradiction = int(judge_scores["non_contradiction"])
    raw_score = ((0.30 * naturalness) + (0.20 * persona) + (0.20 * memory) + (0.30 * contradiction)) / 5.0 * 100.0
    if critical_contradiction:
        return min(raw_score, 39.0)
    return raw_score


def label_case(case_score: float, critical_contradiction: bool) -> str:
    if critical_contradiction or case_score < 50.0:
        return "Miss"
    if case_score >= 80.0:
        return "Exact"
    return "Partial"


ROOT_CAUSE_LABELS = {
    "benchmark_ambiguous": "benchmark 有歧义",
    "retrieval_miss": "没触发记忆",
    "evidence_conflict": "证据互相冲突",
    "generation_stitch_error": "模型把记忆拼错",
    "persona_only": "温柔助手感不足",
    "acceptable_no_recall": "无需强记忆也可通过",
}


def classify_failure(row: CompanionResult) -> str:
    if row.critical_contradiction:
        return "说错核心事实"
    return ROOT_CAUSE_LABELS.get(row.root_cause, "综合不足")


def summarize_metrics(results: List[CompanionResult]) -> Dict[str, object]:
    failures_only = [row for row in results if row.judge_label != "Exact"]
    exact = sum(1 for row in results if row.judge_label == "Exact")
    partial = sum(1 for row in results if row.judge_label == "Partial")
    miss = sum(1 for row in results if row.judge_label == "Miss")
    companion_score = sum(row.case_score for row in results) / max(1, len(results))
    helpful_recall_rate = sum(1 for row in results if row.helpful_recall) / max(1, len(results))
    persona_stability_rate = sum(1 for row in results if row.judge_scores["persona_stability"] >= 4) / max(1, len(results))
    core_fact_safety_rate = sum(1 for row in results if not row.critical_contradiction) / max(1, len(results))
    root_cause_breakdown = Counter(row.root_cause for row in failures_only)
    theme_counts = Counter(row.case.theme for row in results)

    by_type: Dict[str, Dict[str, object]] = {}
    grouped: Dict[str, List[CompanionResult]] = defaultdict(list)
    for row in results:
        grouped[row.case.case_type].append(row)
    for case_type, rows in grouped.items():
        by_type[case_type] = {
            "questions": len(rows),
            "companion_score": sum(row.case_score for row in rows) / max(1, len(rows)),
            "exact": sum(1 for row in rows if row.judge_label == "Exact"),
            "partial": sum(1 for row in rows if row.judge_label == "Partial"),
            "miss": sum(1 for row in rows if row.judge_label == "Miss"),
            "helpful_recall_rate": sum(1 for row in rows if row.helpful_recall) / max(1, len(rows)),
        }

    return {
        "questions": len(results),
        "exact": exact,
        "partial": partial,
        "miss": miss,
        "companion_score": companion_score,
        "helpful_recall_rate": helpful_recall_rate,
        "persona_stability_rate": persona_stability_rate,
        "core_fact_safety_rate": core_fact_safety_rate,
        "root_cause_breakdown": dict(root_cause_breakdown),
        "theme_counts": dict(theme_counts),
        "by_type": by_type,
    }


def write_anchor_catalog(output_dir: Path, slug: str, catalog: Dict[str, List[AnchorEntry]]) -> Path:
    path = output_dir / f"{slug}-companion-anchor-catalog.json"
    payload = {
        theme: [asdict(entry) for entry in entries]
        for theme, entries in catalog.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_question_bank(output_dir: Path, slug: str, cases: List[CompanionCase]) -> tuple[Path, Path]:
    json_path = output_dir / f"{slug}-companion-question-bank.json"
    md_path = output_dir / f"{slug}-companion-question-bank.md"

    json_path.write_text(json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {slug} Companion Benchmark Question Bank",
        "",
        f"- Cases: {len(cases)}",
        "",
    ]
    for case in cases:
        lines.append(f"## {case.case_id:02d} [{case.case_type}] {case.theme}")
        for index, turn in enumerate(case.turns, start=1):
            lines.append(f"- Turn {index}: {turn}")
        lines.append(f"- Gold Intent: {case.gold_intent}")
        lines.append(f"- Memory Anchors: {' | '.join(case.memory_anchors)}")
        lines.append(f"- Must Not Conflict: {' | '.join(case.must_not_conflict)}")
        lines.append(f"- Supporting Chunks: {', '.join(str(chunk_id) for chunk_id in case.supporting_chunk_ids)}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def result_payload(row: CompanionResult) -> Dict[str, object]:
    payload = {
        "case": asdict(row.case),
        "judge_scores": row.judge_scores,
        "judge_profile": row.judge_profile,
        "judge_label": row.judge_label,
        "critical_contradiction": row.critical_contradiction,
        "helpful_recall": row.helpful_recall,
        "case_score": row.case_score,
        "judge_reason": row.judge_reason,
        "root_cause": row.root_cause,
        "benchmark_risk": row.benchmark_risk,
        "retrieved_chunk_ids": row.retrieved_chunk_ids,
        "retrieved_snippets": row.retrieved_snippets,
        "support_alignment": row.support_alignment,
        "failure_tag": row.failure_tag,
    }
    if row.turn_responses is not None:
        payload["turn_responses"] = row.turn_responses
    else:
        payload["raw_response"] = row.raw_response
    return payload


def representative_samples(rows: List[CompanionResult]) -> tuple[Optional[CompanionResult], Optional[CompanionResult]]:
    if not rows:
        return None, None
    ordered = sorted(rows, key=lambda row: row.case_score, reverse=True)
    return ordered[0], ordered[-1]


def response_preview(row: CompanionResult) -> str:
    if row.turn_responses is not None:
        return " || ".join(truncate(turn, 180) for turn in row.turn_responses)
    return truncate(row.raw_response or "", 220)


def load_baseline_summary(results_path: Path) -> Optional[Dict[str, object]]:
    if not results_path.exists():
        return None
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def benchmark_sanity(cases: List[CompanionCase], chunk_map: Dict[int, ChunkRecord], sample_size: int = 10) -> Dict[str, object]:
    sampled = cases[: min(sample_size, len(cases))]
    checks = []
    passed = 0
    for case in sampled:
        support_text = " ".join(
            chunk_map[cid].text for cid in case.supporting_chunk_ids if cid in chunk_map
        )
        support_key = normalize_key(support_text)
        anchor_hits = []
        for anchor in case.memory_anchors:
            anchor_key = normalize_key(anchor)
            if not anchor_key:
                continue
            if anchor_key in support_key:
                anchor_hits.append(anchor)
                continue
            anchor_tokens = [token for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,8}", anchor.lower()) if token]
            if anchor_tokens and sum(1 for token in anchor_tokens if token in support_text.lower()) >= 1:
                anchor_hits.append(anchor)
        ok = bool(anchor_hits)
        if ok:
            passed += 1
        checks.append(
            {
                "case_id": case.case_id,
                "theme": case.theme,
                "supported": ok,
                "anchors_supported": anchor_hits,
            }
        )
    return {
        "sampled": len(sampled),
        "passed": passed,
        "details": checks,
    }


def build_report(
    *,
    docx_path: Path,
    char_count: int,
    paragraph_count: int,
    import_stats: Dict[str, object],
    cases: List[CompanionCase],
    results: List[CompanionResult],
    summary: Dict[str, object],
    judge_provider: str,
    judge_model: str,
    baseline_summary: Optional[Dict[str, object]],
    sanity: Dict[str, object],
    question_bank_json: Path,
    question_bank_md: Path,
    results_json: Path,
    anchor_catalog_path: Path,
) -> str:
    lines = [
        f"# {docx_path.stem} Companion Benchmark Report",
        "",
        f"- Note: 用户口头目标是“100万字级”，但本次报告按真实语料规模透明汇报；当前文档实际约 `{char_count / 10000.0:.1f} 万字符`（`{char_count}` 字符）。",
        "",
        "## Corpus",
        f"- Source: `{docx_path}`",
        f"- Characters: `{char_count}`",
        f"- Paragraphs: `{paragraph_count}`",
        f"- Indexed chunks: `{import_stats.get('chunks', 0)}`",
        f"- Indexed terms: `{import_stats.get('terms', 0)}`",
        f"- Judge: `{judge_provider}:{judge_model}` (`{JUDGE_PROFILE}`)",
        "",
        "## Overall",
        f"- Companion Score: `{summary['companion_score']:.2f}`",
        f"- Exact / Partial / Miss: `{summary['exact']} / {summary['partial']} / {summary['miss']}`",
        f"- Helpful Recall Rate: `{summary['helpful_recall_rate']:.2%}`",
        f"- Persona Stability Rate: `{summary['persona_stability_rate']:.2%}`",
        f"- Core Fact Safety Rate: `{summary['core_fact_safety_rate']:.2%}`",
        "",
        "## By Type",
    ]
    for case_type in ("msc_trigger", "fuzzy_landmark", "causal_semantic", "continuity_session"):
        info = summary["by_type"].get(case_type, {})
        lines.append(
            f"- `{case_type}`: score `{info.get('companion_score', 0.0):.2f}`, exact `{info.get('exact', 0)}`, partial `{info.get('partial', 0)}`, miss `{info.get('miss', 0)}`, helpful recall `{info.get('helpful_recall_rate', 0.0):.2%}`"
        )

    if baseline_summary:
        baseline_helpful = baseline_summary.get("helpful_recall_rate", baseline_summary.get("natural_trigger_rate", 0.0))
        lines.extend(
            [
                "",
                "## V1 vs V2",
                f"- Companion Score: `{baseline_summary.get('companion_score', 0.0):.2f}` -> `{summary['companion_score']:.2f}`",
                f"- Core Fact Safety Rate: `{baseline_summary.get('core_fact_safety_rate', 0.0):.2%}` -> `{summary['core_fact_safety_rate']:.2%}`",
                f"- Helpful Recall Rate: `{baseline_helpful:.2%}` -> `{summary['helpful_recall_rate']:.2%}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Benchmark Sanity",
            f"- relationship_persona count: `{summary['theme_counts'].get('relationship_persona', 0)}` / `{RELATIONSHIP_THEME_CAP}`",
            f"- Support spot-check: `{sanity.get('passed', 0)}` / `{sanity.get('sampled', 0)}` sampled cases had direct anchor support in their supporting chunks.",
        ]
    )

    lines.extend(["", "## Representative Samples"])
    grouped: Dict[str, List[CompanionResult]] = defaultdict(list)
    for row in results:
        grouped[row.case.case_type].append(row)
    for case_type in ("msc_trigger", "fuzzy_landmark", "causal_semantic", "continuity_session"):
        high, low = representative_samples(grouped.get(case_type, []))
        lines.append(f"### {case_type}")
        if high:
            lines.append(f"- High sample Q{high.case.case_id:02d}: `{high.case_score:.2f}`")
            lines.append(f"  - User: `{high.case.turns[0]}`")
            lines.append(f"  - Response: `{response_preview(high)}`")
            lines.append(f"  - Judge: {high.judge_reason}")
        if low:
            lines.append(f"- Low sample Q{low.case.case_id:02d}: `{low.case_score:.2f}`")
            lines.append(f"  - User: `{low.case.turns[0]}`")
            lines.append(f"  - Response: `{response_preview(low)}`")
            lines.append(f"  - Judge: {low.judge_reason}")
            lines.append(f"  - Root Cause: {ROOT_CAUSE_LABELS.get(low.root_cause, low.root_cause)}")

    failures = sorted((row for row in results if row.judge_label != "Exact"), key=lambda row: row.case_score)
    lines.extend(["", "## Top 10 Failures"])
    if failures:
        for row in failures[:10]:
            lines.append(f"- Q{row.case.case_id:02d} [{row.case.case_type}/{row.case.theme}] `{row.case_score:.2f}` - {row.failure_tag}")
            lines.append(f"  - User: `{row.case.turns[0]}`")
            lines.append(f"  - Response: `{response_preview(row)}`")
            lines.append(f"  - Judge: {row.judge_reason}")
            lines.append(f"  - Root Cause: `{row.root_cause}` | risk `{row.benchmark_risk}` | support `{row.support_alignment}`")
    else:
        lines.append("- No failures.")

    failure_counts = Counter(row.failure_tag for row in failures)
    lines.extend(["", "## Failure Breakdown"])
    for label in ["没触发记忆", "模型把记忆拼错", "证据互相冲突", "温柔助手感不足", "benchmark 有歧义", "说错核心事实", "无需强记忆也可通过"]:
        lines.append(f"- {label}: `{failure_counts.get(label, 0)}`")

    lines.extend(["", "## Root Cause Breakdown"])
    for label in [
        "benchmark_ambiguous",
        "retrieval_miss",
        "evidence_conflict",
        "generation_stitch_error",
        "persona_only",
        "acceptable_no_recall",
    ]:
        lines.append(f"- {label}: `{summary['root_cause_breakdown'].get(label, 0)}`")

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Question Bank JSON: `{question_bank_json}`",
            f"- Question Bank MD: `{question_bank_md}`",
            f"- Results JSON: `{results_json}`",
            f"- Anchor Catalog JSON: `{anchor_catalog_path}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.questions != args.msc_questions + args.fuzzy_questions + args.causal_questions + args.continuity_questions:
        raise SystemExit("--questions must equal msc + fuzzy + causal + continuity counts.")
    expected_plan_total = sum(sum(type_counts.values()) for type_counts in THEME_CASE_PLAN.values())
    is_default_full = (
        args.questions == expected_plan_total
        and args.msc_questions == 30
        and args.fuzzy_questions == 15
        and args.causal_questions == 12
        and args.continuity_questions == 20
    )
    is_smoke = (
        args.questions == 6
        and args.msc_questions == 2
        and args.fuzzy_questions == 1
        and args.causal_questions == 1
        and args.continuity_questions == 2
    )
    if not (is_default_full or is_smoke):
        raise SystemExit(
            "Supported configurations are the full 77-case companion benchmark or the 6-case smoke run "
            "(2 MSC, 1 Fuzzy, 1 Causal, 2 Continuity)."
        )

    if args.judge_provider == "google" and args.judge_model == DEFAULT_JUDGE_MODEL:
        args.judge_model = os.getenv("GOOGLE_JUDGE_MODEL", "gemini-3.1-pro-preview")

    api_key = require_judge_env(args.judge_provider)

    docx_path = Path(args.docx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    slug = slugify(docx_path)
    results_path = output_dir / f"{slug}-companion-results.json"
    report_path = output_dir / f"{slug}-companion-report.md"
    baseline_summary = load_baseline_summary(results_path)

    if not docx_path.exists():
        raise SystemExit(f"Document not found: {docx_path}")

    from docx import Document  # local import to keep script startup simple

    document = Document(docx_path)
    paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
    char_count = len("\n".join(paragraphs))
    paragraph_count = len(paragraphs)

    ensure_dir(output_dir)
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    config = MemoryConfig(memory_dir=str(memory_dir), retrieve_top_k=int(args.retrieve_top_k))
    mm = MemoryManager(config)
    try:
        stored = mm.import_files([str(docx_path)])
        if not stored:
            raise RuntimeError("Import failed: no supported files were stored.")
        import_stats = mm.rebuild_memory()
        chunks = load_chunks(mm)
        if not chunks:
            raise RuntimeError("No chunks were indexed for the document.")

        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        anchor_catalog = build_anchor_catalog(mm, chunk_map)
        anchor_catalog_path = write_anchor_catalog(output_dir, slug, anchor_catalog)

        print("Generating companion benchmark question bank...", flush=True)
        cases = generate_cases(
            anchor_catalog=anchor_catalog,
            model=args.question_model,
            timeout=args.generation_timeout,
            question_count=args.questions,
        )
        if len(cases) != args.questions:
            raise RuntimeError(f"Expected {args.questions} cases, generated {len(cases)}.")

        question_bank_json, question_bank_md = write_question_bank(output_dir, slug, cases)
        sanity = benchmark_sanity(cases, chunk_map, sample_size=10)

        print(f"Running {len(cases)} companion cases...", flush=True)
        results: List[CompanionResult] = []
        for case in cases:
            print(f"[{case.case_id:02d}/{len(cases):02d}] [{case.case_type}] [{case.theme}] {case.turns[0]}", flush=True)
            raw_response, turn_responses = build_turn_messages(mm, case, args.model, args.answer_timeout)
            retrieved = retrieve_case_evidence(mm, case, int(args.retrieve_top_k))
            response_payload: Dict[str, object]
            if turn_responses is not None:
                response_payload = {"turn_responses": turn_responses}
            else:
                response_payload = {"raw_response": raw_response or ""}
            judgment = judge_case(
                provider=args.judge_provider,
                api_key=api_key,
                base_url=args.google_base_url if args.judge_provider == "google" else args.openai_base_url,
                model=args.judge_model,
                case=case,
                raw_response=raw_response,
                turn_responses=turn_responses,
                timeout=args.judge_timeout,
            )
            judge_scores = {
                "naturalness": int(judgment["naturalness"]),
                "persona_stability": int(judgment["persona_stability"]),
                "memory_usefulness": int(judgment["memory_usefulness"]),
                "non_contradiction": int(judgment["non_contradiction"]),
            }
            critical_contradiction = bool(judgment["critical_contradiction"])
            if critical_contradiction:
                judge_scores["non_contradiction"] = 0
            helpful_recall = bool(judgment["helpful_recall"])
            case_score = score_case(judge_scores, critical_contradiction)
            judge_label = label_case(case_score, critical_contradiction)
            support_snippets = [
                chunk_map[cid].snippet
                for cid in case.supporting_chunk_ids
                if cid in chunk_map
            ]
            adjudication = adjudicate_case(
                provider=args.judge_provider,
                api_key=api_key,
                base_url=args.google_base_url if args.judge_provider == "google" else args.openai_base_url,
                model=args.judge_model,
                case=case,
                response_payload=response_payload,
                retrieved=retrieved,
                support_snippets=support_snippets,
                helpful_recall=helpful_recall,
                critical_contradiction=critical_contradiction,
                timeout=args.judge_timeout,
            )
            row = CompanionResult(
                case=case,
                raw_response=raw_response,
                turn_responses=turn_responses,
                judge_scores=judge_scores,
                judge_profile=JUDGE_PROFILE,
                judge_label=judge_label,
                critical_contradiction=critical_contradiction,
                helpful_recall=helpful_recall,
                case_score=case_score,
                judge_reason=normalize_space(str(judgment["judge_reason"])),
                root_cause=str(adjudication["root_cause"]),
                benchmark_risk=str(adjudication["benchmark_risk"]),
                retrieved_chunk_ids=[int(item.get("chunk_id", 0) or 0) for item in retrieved],
                retrieved_snippets=[str(item.get("snippet", "")) for item in retrieved],
                support_alignment=str(adjudication["support_alignment"] or support_alignment(case, [int(item.get("chunk_id", 0) or 0) for item in retrieved])),
                failure_tag="",
            )
            row = CompanionResult(
                case=row.case,
                raw_response=row.raw_response,
                turn_responses=row.turn_responses,
                judge_scores=row.judge_scores,
                judge_profile=row.judge_profile,
                judge_label=row.judge_label,
                critical_contradiction=row.critical_contradiction,
                helpful_recall=row.helpful_recall,
                case_score=row.case_score,
                judge_reason=row.judge_reason,
                root_cause=row.root_cause,
                benchmark_risk=row.benchmark_risk,
                retrieved_chunk_ids=row.retrieved_chunk_ids,
                retrieved_snippets=row.retrieved_snippets,
                support_alignment=row.support_alignment,
                failure_tag=classify_failure(row),
            )
            results.append(row)
            print(
                f"  -> {judge_label} | score={case_score:.2f} | helpful_recall={helpful_recall} | contradiction={critical_contradiction} | cause={row.root_cause}",
                flush=True,
            )

        summary = summarize_metrics(results)
        results_path.write_text(
            json.dumps(
                {
                    "docx": str(docx_path),
                    "char_count": char_count,
                    "paragraph_count": paragraph_count,
                    "model": args.model,
                    "question_model": args.question_model,
                    "judge_provider": args.judge_provider,
                    "judge_model": args.judge_model,
                    "judge_profile": JUDGE_PROFILE,
                    "import_stats": import_stats,
                    "baseline_summary": baseline_summary,
                    "sanity": sanity,
                    "summary": summary,
                    "results": [result_payload(row) for row in results],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        report = build_report(
            docx_path=docx_path,
            char_count=char_count,
            paragraph_count=paragraph_count,
            import_stats=import_stats,
            cases=cases,
            results=results,
            summary=summary,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            baseline_summary=baseline_summary,
            sanity=sanity,
            question_bank_json=question_bank_json,
            question_bank_md=question_bank_md,
            results_json=results_path,
            anchor_catalog_path=anchor_catalog_path,
        )
        report_path.write_text(report, encoding="utf-8")
        print(report)
        return 0
    finally:
        mm.close()
        if not args.keep_memory_dir:
            state_path = memory_dir / "app_state.json"
            if state_path.exists():
                state_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
