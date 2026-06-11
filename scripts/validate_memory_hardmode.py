#!/usr/bin/env python3
"""Run a mixed hard-mode memory recall evaluation against EverMate.

This script upgrades the benchmark in three ways:
1. Keeps grounded cloze recall questions for precise single-fact retrieval.
2. Adds multi-hop questions that require combining multiple chunks.
3. Adds causal questions that test "why" understanding rather than pure memorization.

Scoring uses a hybrid strategy:
- fast heuristics for obvious exact cloze hits
- LLM-as-a-Judge for order-insensitive, paraphrased, multi-hop, and causal answers
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from ollama_client import chat as ollama_chat


DEFAULT_DOCX = ""  # pass --docx explicitly
DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q8_0"
DEFAULT_QUESTION_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M"
DEFAULT_JUDGE_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M"
DEFAULT_MEMORY_DIR = "/tmp/evermate-memory-hardmode-validation"
DEFAULT_OUTPUT_DIR = "reports"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    path: str
    text: str
    snippet: str


@dataclass(frozen=True)
class MixedCase:
    case_id: int
    case_type: str
    question: str
    gold_answer: str
    key_points: Tuple[str, ...]
    supporting_chunk_ids: Tuple[int, ...]
    evidence_quotes: Tuple[str, ...]
    score: float


@dataclass(frozen=True)
class EvalRow:
    case: MixedCase
    answer: str
    label: str
    reason: str
    top_retrieval: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EverMate memory recall with mixed hard-mode questions.")
    parser.add_argument("--docx", default=DEFAULT_DOCX, required=not DEFAULT_DOCX, help="Path to the source .docx file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Exact Ollama model name to answer with.")
    parser.add_argument("--question-model", default=DEFAULT_QUESTION_MODEL, help="Model used to generate reasoning questions.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Model used to judge answers.")
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR, help="Isolated memory directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for reports and question bank.")
    parser.add_argument("--questions", type=int, default=77, help="Total number of questions to run.")
    parser.add_argument("--cloze-questions", type=int, default=52, help="Number of grounded cloze questions.")
    parser.add_argument("--multi-hop-questions", type=int, default=13, help="Number of multi-hop synthesis questions.")
    parser.add_argument("--causal-questions", type=int, default=12, help="Number of causal why-questions.")
    parser.add_argument("--retrieve-top-k", type=int, default=12, help="Top-k evidence chunks injected into the prompt.")
    parser.add_argument("--answer-timeout", type=int, default=300, help="Per-question answer timeout in seconds.")
    parser.add_argument("--generation-timeout", type=int, default=360, help="Reasoning question generation timeout in seconds.")
    parser.add_argument("--judge-timeout", type=int, default=300, help="Judge timeout in seconds.")
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")
    return stem or "document"


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


def load_chunks(mm: MemoryManager) -> List[ChunkRecord]:
    cur = mm.conn.cursor()
    rows = cur.execute("SELECT id, path FROM chunks ORDER BY id ASC").fetchall()
    chunks: List[ChunkRecord] = []
    for row in rows:
        rel_path = str(row["path"])
        abs_path = Path(mm.memory_dir) / rel_path
        if not abs_path.exists():
            continue
        text = abs_path.read_text(encoding="utf-8")
        snippet_parts = split_sentences(text)[:2]
        if snippet_parts:
            snippet = " ".join(snippet_parts)
        else:
            snippet = normalize_space(text)[:160]
        chunks.append(
            ChunkRecord(
                chunk_id=int(row["id"]),
                path=rel_path,
                text=text,
                snippet=truncate(snippet, 110),
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


def build_cloze_candidates(chunks: List[ChunkRecord]) -> List[Dict[str, object]]:
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
                            "case_type": "cloze",
                            "chunk_id": chunk.chunk_id,
                            "question": "请根据记忆填空，只填写空缺内容：" + question,
                            "gold_answer": answer,
                            "key_points": split_answer_points(answer),
                            "supporting_chunk_ids": [chunk.chunk_id],
                            "evidence_quotes": [sentence],
                            "score": score,
                        }
                    )

    answer_freq = Counter(str(candidate["gold_answer"]) for candidate in candidates)
    for candidate in candidates:
        candidate["score"] = float(candidate["score"]) - min(answer_freq[str(candidate["gold_answer"])], 8) * 0.35

    return candidates


def split_answer_points(answer: str) -> Tuple[str, ...]:
    text = normalize_space(answer)
    if not text:
        return tuple()
    separators = ["+", "、", "/", "，", ",", "；", ";", "和", "及", "与"]
    parts = [text]
    for separator in separators:
        next_parts: List[str] = []
        for part in parts:
            if separator in part and len(part) >= 4:
                next_parts.extend(x.strip() for x in part.split(separator))
            else:
                next_parts.append(part.strip())
        parts = next_parts
    points = tuple(part for part in parts if part)
    return points or (text,)


def select_cases_from_candidates(candidates: List[Dict[str, object]], target: int) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    seen_chunks: set[int] = set()
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()

    ordered = sorted(candidates, key=lambda item: (-float(item["score"]), int(item["chunk_id"])))

    for candidate in ordered:
        chunk_id = int(candidate["chunk_id"])
        question_key = normalize_match(str(candidate["question"]))
        answer_key = normalize_match(str(candidate["gold_answer"]))
        if chunk_id in seen_chunks:
            continue
        if question_key in seen_questions or answer_key in seen_answers:
            continue
        selected.append(candidate)
        seen_chunks.add(chunk_id)
        seen_questions.add(question_key)
        seen_answers.add(answer_key)
        if len(selected) >= target:
            break

    if len(selected) < target:
        for candidate in ordered:
            question_key = normalize_match(str(candidate["question"]))
            answer_key = normalize_match(str(candidate["gold_answer"]))
            if question_key in seen_questions or answer_key in seen_answers:
                continue
            selected.append(candidate)
            seen_questions.add(question_key)
            seen_answers.add(answer_key)
            if len(selected) >= target:
                break

    return selected[:target]


def reasoning_seed_queries(case_type: str) -> List[str]:
    if case_type == "multi_hop":
        return [
            "学习计划",
            "考试",
            "微积分",
            "C++",
            "焦虑",
            "情绪",
            "台湾女生",
            "台湾老婆",
            "成长",
            "驾驶",
            "GTA5",
            "睡眠",
            "生物钟",
            "作息",
            "维生素B",
            "补剂",
            "健康",
            "投资",
            "币圈",
            "安全感",
            "亲密关系",
            "未来",
            "AI",
            "小胡桃",
            "计划",
            "好兄弟",
            "朋友",
            "老师",
            "Lab",
            "生活",
        ]
    return [
        "焦虑",
        "情绪",
        "学习",
        "考试",
        "睡眠",
        "生物钟",
        "作息",
        "维生素B",
        "补剂",
        "驾驶",
        "GTA5",
        "台湾女生",
        "台湾老婆",
        "安全感",
        "关系",
        "亲密",
        "未来",
        "成长",
        "投资",
        "AI",
        "生活",
        "健康",
        "好兄弟",
        "朋友",
        "老师",
        "Lab",
    ]


def validate_reasoning_candidate(
    raw: Dict[str, object],
    case_type: str,
    chunk_map: Dict[int, ChunkRecord],
    seen_questions: set[str],
    seen_answers: set[str],
) -> Optional[Dict[str, object]]:
    if str(raw.get("type", "")).strip() != case_type:
        return None

    question = normalize_space(str(raw.get("question", "")))
    gold_answer = normalize_space(str(raw.get("gold_answer", "")))
    if not question or not gold_answer:
        return None

    if case_type == "causal" and not any(token in question for token in ("为什么", "为何", "什么让", "什么导致", "怎么会")):
        return None
    if case_type == "multi_hop" and len(question) < 12:
        return None

    question_key = normalize_match(question)
    answer_key = normalize_match(gold_answer)
    if question_key in seen_questions or answer_key in seen_answers:
        return None

    key_points_raw = raw.get("key_points", [])
    if not isinstance(key_points_raw, list):
        return None
    key_points = [normalize_space(str(item)) for item in key_points_raw if normalize_space(str(item))]
    key_points = key_points[:4]
    if len(key_points) < 2:
        return None

    chunk_ids_raw = raw.get("supporting_chunk_ids", [])
    if not isinstance(chunk_ids_raw, list):
        return None
    chunk_ids: List[int] = []
    for item in chunk_ids_raw:
        try:
            cid = int(item)
        except Exception:
            continue
        if cid in chunk_map and cid not in chunk_ids:
            chunk_ids.append(cid)
    if case_type == "multi_hop" and len(chunk_ids) < 2:
        return None
    if case_type == "causal" and len(chunk_ids) < 1:
        return None

    evidence_raw = raw.get("evidence_quotes", [])
    if not isinstance(evidence_raw, list):
        return None
    evidence_quotes = [normalize_space(str(item)) for item in evidence_raw if normalize_space(str(item))]
    evidence_quotes = evidence_quotes[:4]
    if len(evidence_quotes) < 1:
        return None

    support_text = "\n".join(chunk_map[cid].text for cid in chunk_ids)
    matched_quotes = 0
    for quote in evidence_quotes:
        if quote in support_text:
            matched_quotes += 1
    if matched_quotes < max(1, len(evidence_quotes) // 2):
        return None

    score = float(len(chunk_ids) * 2 + len(key_points))
    if case_type == "multi_hop" and len(chunk_ids) >= 2:
        score += min((max(chunk_ids) - min(chunk_ids)) / 40.0, 6.0)
    if case_type == "causal":
        score += 1.5

    seen_questions.add(question_key)
    seen_answers.add(answer_key)
    return {
        "case_type": case_type,
        "question": question,
        "gold_answer": gold_answer,
        "key_points": key_points,
        "supporting_chunk_ids": chunk_ids,
        "evidence_quotes": evidence_quotes,
        "score": score,
    }


def generate_reasoning_candidates(
    *,
    case_type: str,
    target: int,
    mm: MemoryManager,
    chunk_map: Dict[int, ChunkRecord],
    model: str,
    timeout: int,
) -> List[Dict[str, object]]:
    if target <= 0:
        return []

    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    out: List[Dict[str, object]] = []
    seeds = reasoning_seed_queries(case_type)
    if case_type == "multi_hop":
        guidance = (
            "请生成 1 道 multi_hop 题，必须至少结合 2 条证据才能完整回答。"
            "优先做“跨阶段归纳”“时间跨度总结”“把多次建议合并总结”的题。"
        )
    else:
        guidance = (
            "请生成 1 道 causal 题，必须是为什么/因果链理解题。"
            "要测试行为、情绪、价值观或偏好背后的原因，而不是机械复述。"
        )

    for seed_index, seed in enumerate(seeds, start=1):
        if len(out) >= target:
            break
        retrieval = mm.retrieve(seed, k=max(6, mm.cfg.retrieve_top_k))
        unique_chunk_ids = []
        for item in retrieval:
            cid = int(item.get("chunk_id", 0) or 0)
            if cid and cid in chunk_map and cid not in unique_chunk_ids:
                unique_chunk_ids.append(cid)
        if case_type == "multi_hop" and len(unique_chunk_ids) < 2:
            continue
        if case_type == "causal" and len(unique_chunk_ids) < 1:
            continue

        evidence_lines = []
        for cid in unique_chunk_ids[:6]:
            evidence_lines.append(f"#{cid:03d}: {truncate(chunk_map[cid].snippet, 120)}")
        evidence_block = "\n".join(evidence_lines)
        print(f"Generating {case_type} candidate from seed {seed_index}/{len(seeds)}: {seed}", flush=True)

        prompt = (
            "你是一个高标准中文题库设计器。下面是一组围绕同一主题检索出的证据片段。\n"
            + guidance
            + "\n硬性要求：\n"
            "1. 题目必须可以由这些证据回答，不能凭空编造。\n"
            "2. gold_answer 用中文简洁作答，长度控制在 1 到 3 句内。\n"
            "3. key_points 提供 2 到 4 个判分关键点。\n"
            "4. supporting_chunk_ids 必须从给定证据片段的 chunk id 中选择。\n"
            "5. evidence_quotes 提供 2 到 4 条短证据，并且必须能在给定片段里找到原文。\n"
            "6. multi_hop 题至少给 2 个 supporting_chunk_ids，causal 题至少给 1 个。\n"
            "7. 只返回 JSON 数组，数组里只放 1 道题。格式为 "
            '{"type":"multi_hop|causal","question":"...","gold_answer":"...","key_points":["..."],"supporting_chunk_ids":[1,2],"evidence_quotes":["..."]}。\n\n'
            f"[Seed]\n{seed}\n\n"
            "[Evidence Pack]\n"
            + evidence_block
        )

        try:
            response = ollama_chat(
                [
                    {"role": "system", "content": "只返回合法 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                timeout=timeout,
                options={"temperature": 0, "num_predict": 1400},
            )
            parsed = extract_json_array(response)
        except Exception:
            print(f"  -> seed {seed} returned no usable JSON", flush=True)
            continue

        added = 0
        for row in parsed:
            if not isinstance(row, dict):
                continue
            candidate = validate_reasoning_candidate(
                row,
                case_type=case_type,
                chunk_map=chunk_map,
                seen_questions=seen_questions,
                seen_answers=seen_answers,
            )
            if candidate:
                out.append(candidate)
                added += 1
        print(f"  -> accumulated {len(out)} valid {case_type} candidates", flush=True)
    out.sort(key=lambda item: float(item["score"]), reverse=True)
    return out


def build_mixed_cases(
    *,
    mm: MemoryManager,
    chunks: List[ChunkRecord],
    cloze_questions: int,
    multi_hop_questions: int,
    causal_questions: int,
    question_model: str,
    timeout: int,
) -> List[MixedCase]:
    cloze_candidates = build_cloze_candidates(chunks)
    cloze_selected = select_cases_from_candidates(cloze_candidates, cloze_questions)
    print(f"Built {len(cloze_selected)} cloze cases.", flush=True)

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    multi_hop_candidates = generate_reasoning_candidates(
        case_type="multi_hop",
        target=multi_hop_questions,
        mm=mm,
        chunk_map=chunk_map,
        model=question_model,
        timeout=timeout,
    )
    print(f"Built {len(multi_hop_candidates)} multi-hop candidates.", flush=True)
    causal_candidates = generate_reasoning_candidates(
        case_type="causal",
        target=causal_questions,
        mm=mm,
        chunk_map=chunk_map,
        model=question_model,
        timeout=timeout,
    )
    print(f"Built {len(causal_candidates)} causal candidates.", flush=True)

    if len(multi_hop_candidates) < multi_hop_questions:
        raise RuntimeError(f"Only generated {len(multi_hop_candidates)} multi-hop questions, expected {multi_hop_questions}.")
    if len(causal_candidates) < causal_questions:
        raise RuntimeError(f"Only generated {len(causal_candidates)} causal questions, expected {causal_questions}.")

    raw_cases = cloze_selected + multi_hop_candidates[:multi_hop_questions] + causal_candidates[:causal_questions]
    mixed_cases: List[MixedCase] = []
    for index, case in enumerate(raw_cases, start=1):
        mixed_cases.append(
            MixedCase(
                case_id=index,
                case_type=str(case["case_type"]),
                question=str(case["question"]),
                gold_answer=str(case["gold_answer"]),
                key_points=tuple(str(item) for item in case["key_points"]),
                supporting_chunk_ids=tuple(int(item) for item in case["supporting_chunk_ids"]),
                evidence_quotes=tuple(str(item) for item in case["evidence_quotes"]),
                score=float(case["score"]),
            )
        )
    return mixed_cases


def build_messages(mm: MemoryManager, case: MixedCase) -> List[Dict[str, str]]:
    if case.case_type == "cloze":
        style = "请只填写空缺内容本身，不要重复整句，不要解释。"
    else:
        style = "请用中文直接回答。优先根据记忆证据作答，先给结论，再补关键点；不要编造，不要使用项目符号。"
    system_prompt = mm.build_system_prompt(user_text=case.question, assistant_style=style, lang="zh")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case.question},
    ]


def orderless_point_match(answer: str, key_points: Iterable[str]) -> bool:
    points = [normalize_match(point) for point in key_points if normalize_match(point)]
    if not points:
        return False
    normalized_answer = normalize_match(answer)
    if not normalized_answer:
        return False
    return all(point in normalized_answer for point in points)


def heuristic_label(case: MixedCase, answer: str) -> Optional[Tuple[str, str]]:
    normalized_answer = normalize_match(answer)
    normalized_gold = normalize_match(case.gold_answer)
    if not normalized_answer:
        return ("Miss", "未作答。")
    if normalized_gold and normalized_gold in normalized_answer:
        return ("Exact", "命中参考答案。")
    if normalized_answer and normalized_answer in normalized_gold:
        ratio = len(normalized_answer) / max(1, len(normalized_gold))
        if ratio >= 0.9:
            return ("Exact", "回答与参考答案仅有轻微格式差异。")
        if ratio >= 0.6:
            return ("Partial", "回答命中了参考答案的一部分。")

    if case.case_type == "cloze" and orderless_point_match(answer, case.key_points):
        return ("Exact", "关键点齐全，顺序或标点不同。")

    similarity = SequenceMatcher(None, normalized_answer, normalized_gold).ratio() if normalized_answer and normalized_gold else 0.0
    if case.case_type == "cloze" and similarity >= 0.88:
        return ("Exact", "回答与参考答案高度相似。")
    if case.case_type == "cloze" and similarity >= 0.72:
        return ("Partial", "回答与参考答案接近，但不完全一致。")
    return None


def judge_batch(rows: List[EvalRow], model: str, timeout: int) -> Dict[int, Tuple[str, str]]:
    if not rows:
        return {}

    payload = []
    for row in rows:
        payload.append(
            {
                "id": row.case.case_id,
                "type": row.case.case_type,
                "question": row.case.question,
                "gold_answer": row.case.gold_answer,
                "key_points": list(row.case.key_points),
                "evidence_quotes": list(row.case.evidence_quotes),
                "model_answer": row.answer,
            }
        )

    prompt = (
        "你是一个严格但公平的中文问答裁判。\n"
        "评分原则：\n"
        "1. 不要做脆弱的字符串逐字匹配；语义正确、顺序不同、标点不同、同义改写都算对。\n"
        "2. Exact：回答覆盖全部关键点，且不与证据冲突。\n"
        "3. Partial：回答方向对，但只覆盖部分关键点，或表述明显过于含糊。\n"
        "4. Miss：回答与证据冲突、遗漏大部分关键点、或答非所问。\n"
        "5. 对 cloze 题，只要填入内容语义等价即可判 Exact。\n"
        "6. 对 multi_hop / causal 题，必须根据 key_points 和 evidence_quotes 判断是否真的理解了跨 chunk 信息和因果链。\n"
        "只返回 JSON 数组，不要解释。格式为 "
        '[{"id":1,"label":"Exact|Partial|Miss","reason":"一句中文简短说明"}]。'
    )

    try:
        response = ollama_chat(
            [
                {"role": "system", "content": "只返回合法 JSON。"},
                {"role": "user", "content": prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)},
            ],
            model=model,
            timeout=timeout,
            options={"temperature": 0, "num_predict": 2200},
        )
        parsed = extract_json_array(response)
        out: Dict[int, Tuple[str, str]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                case_id = int(item.get("id", 0))
            except Exception:
                continue
            label = str(item.get("label", "")).strip()
            reason = normalize_space(str(item.get("reason", "")))
            if label not in {"Exact", "Partial", "Miss"}:
                continue
            out[case_id] = (label, reason or "模型未提供原因。")
        if out:
            return out
    except Exception:
        pass

    if len(rows) == 1:
        row = rows[0]
        return {row.case.case_id: fallback_semantic_judge(row)}

    midpoint = max(1, len(rows) // 2)
    left = judge_batch(rows[:midpoint], model=model, timeout=timeout)
    right = judge_batch(rows[midpoint:], model=model, timeout=timeout)
    merged = {}
    merged.update(left)
    merged.update(right)
    return merged


def fallback_semantic_judge(row: EvalRow) -> Tuple[str, str]:
    heuristic = heuristic_label(row.case, row.answer)
    if heuristic:
        return heuristic

    normalized_answer = normalize_match(row.answer)
    normalized_gold = normalize_match(row.case.gold_answer)
    if not normalized_answer:
        return ("Miss", "未作答。")

    coverage = 0
    total = 0
    for point in row.case.key_points:
        normalized_point = normalize_match(point)
        if not normalized_point:
            continue
        total += 1
        if normalized_point in normalized_answer:
            coverage += 1
    if total > 0 and coverage == total:
        return ("Exact", "关键点齐全。")
    if total > 0 and coverage >= max(1, total // 2):
        return ("Partial", "命中了一部分关键点。")

    similarity = SequenceMatcher(None, normalized_answer, normalized_gold).ratio() if normalized_answer and normalized_gold else 0.0
    if similarity >= 0.8:
        return ("Partial", "回答与参考答案较接近。")
    return ("Miss", "回答未覆盖足够的关键点。")


def write_question_bank(output_dir: Path, slug: str, cases: List[MixedCase]) -> Tuple[Path, Path]:
    json_path = output_dir / f"{slug}-77-hardmode-question-bank.json"
    md_path = output_dir / f"{slug}-77-hardmode-question-bank.md"

    json_path.write_text(
        json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    by_type = Counter(case.case_type for case in cases)
    lines = [
        f"# {slug} 77题魔鬼测试题库",
        "",
        f"- 题目总数：{len(cases)}",
        f"- Cloze：{by_type.get('cloze', 0)}",
        f"- Multi-hop：{by_type.get('multi_hop', 0)}",
        f"- Causal：{by_type.get('causal', 0)}",
        "",
    ]
    for case in cases:
        lines.append(f"## {case.case_id:02d} [{case.case_type}]")
        lines.append(f"- Question: {case.question}")
        lines.append(f"- Gold Answer: {case.gold_answer}")
        lines.append(f"- Key Points: {' | '.join(case.key_points)}")
        lines.append(f"- Supporting Chunks: {', '.join(str(cid) for cid in case.supporting_chunk_ids)}")
        lines.append(f"- Evidence: {' | '.join(case.evidence_quotes)}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    args = parse_args()
    if args.questions != args.cloze_questions + args.multi_hop_questions + args.causal_questions:
        raise SystemExit("--questions must equal cloze + multi-hop + causal counts.")

    docx_path = Path(args.docx).expanduser().resolve()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    slug = slugify(docx_path)
    results_path = output_dir / f"{slug}-77-hardmode-results.json"
    report_path = output_dir / f"{slug}-77-hardmode-report.txt"

    if not docx_path.exists():
        raise SystemExit(f"Document not found: {docx_path}")

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

        cases = build_mixed_cases(
            mm=mm,
            chunks=chunks,
            cloze_questions=args.cloze_questions,
            multi_hop_questions=args.multi_hop_questions,
            causal_questions=args.causal_questions,
            question_model=args.question_model,
            timeout=args.generation_timeout,
        )
        question_bank_json, question_bank_md = write_question_bank(output_dir, slug, cases)

        results: List[EvalRow] = []
        pending_for_judge: List[EvalRow] = []
        type_counts = Counter(case.case_type for case in cases)

        print(f"Running {len(cases)} hard-mode questions...", flush=True)
        for case in cases:
            retrieval = mm.retrieve(case.question, k=args.retrieve_top_k)
            top_retrieval = truncate(str(retrieval[0]["snippet"])) if retrieval else "(no evidence)"
            print(f"[{case.case_id:02d}/{len(cases):02d}] [{case.case_type}] {case.question}", flush=True)
            answer = ollama_chat(
                build_messages(mm, case),
                model=args.model,
                timeout=args.answer_timeout,
                options={"temperature": 0, "num_predict": 120 if case.case_type != 'cloze' else 48},
            ).strip()
            heuristic = heuristic_label(case, answer)
            if heuristic:
                label, reason = heuristic
                row = EvalRow(case=case, answer=answer, label=label, reason=reason, top_retrieval=top_retrieval)
                results.append(row)
                print(f"  -> {label} | gold={truncate(case.gold_answer, 80)} | got={truncate(answer, 120)}", flush=True)
            else:
                row = EvalRow(case=case, answer=answer, label="Pending", reason="", top_retrieval=top_retrieval)
                results.append(row)
                pending_for_judge.append(row)
                print(f"  -> Pending Judge | gold={truncate(case.gold_answer, 80)} | got={truncate(answer, 120)}", flush=True)

        if pending_for_judge:
            for batch_start in range(0, len(pending_for_judge), 8):
                batch = pending_for_judge[batch_start : batch_start + 8]
                judged = judge_batch(batch, model=args.judge_model, timeout=args.judge_timeout)
                for pending_row in batch:
                    label, reason = judged.get(pending_row.case.case_id, ("Miss", "裁判未返回结果，按 Miss 处理。"))
                    idx = results.index(pending_row)
                    results[idx] = EvalRow(
                        case=pending_row.case,
                        answer=pending_row.answer,
                        label=label,
                        reason=reason,
                        top_retrieval=pending_row.top_retrieval,
                    )

        counts = Counter(row.label for row in results)
        by_type = defaultdict(Counter)
        for row in results:
            by_type[row.case.case_type][row.label] += 1

        exact = counts["Exact"]
        partial = counts["Partial"]
        miss = counts["Miss"]
        recall = (exact + 0.5 * partial) / len(cases)

        results_path.write_text(
            json.dumps(
                {
                    "docx": str(docx_path),
                    "model": args.model,
                    "question_model": args.question_model,
                    "judge_model": args.judge_model,
                    "import_stats": import_stats,
                    "summary": {
                        "questions": len(cases),
                        "exact": exact,
                        "partial": partial,
                        "miss": miss,
                        "recall": recall,
                        "by_type": {
                            case_type: {
                                "questions": type_counts[case_type],
                                "exact": counter["Exact"],
                                "partial": counter["Partial"],
                                "miss": counter["Miss"],
                                "recall": (counter["Exact"] + 0.5 * counter["Partial"]) / max(1, type_counts[case_type]),
                            }
                            for case_type, counter in by_type.items()
                        },
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
            "EverMate Hard-Mode Recall Validation",
            "=" * 36,
            f"Source: {docx_path}",
            f"Answer Model: {args.model}",
            f"Question Model: {args.question_model}",
            f"Judge Model: {args.judge_model}",
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
            "[By Type]",
        ]
        for case_type in ("cloze", "multi_hop", "causal"):
            counter = by_type.get(case_type, Counter())
            question_count = type_counts.get(case_type, 0)
            type_recall = (counter["Exact"] + 0.5 * counter["Partial"]) / max(1, question_count)
            lines.append(
                f"{case_type}: questions={question_count}, exact={counter['Exact']}, partial={counter['Partial']}, miss={counter['Miss']}, recall={type_recall:.2%}"
            )
        lines.extend(
            [
                "",
                "[Artifacts]",
                f"Question Bank JSON: {question_bank_json}",
                f"Question Bank MD: {question_bank_md}",
                f"Results JSON: {results_path}",
                "",
                "[Sample Misses]",
            ]
        )

        misses = [row for row in results if row.label != "Exact"]
        if misses:
            for row in misses[:12]:
                lines.append(f"Q{row.case.case_id:02d} [{row.case.case_type}] {row.case.question}")
                lines.append(f"Gold: {row.case.gold_answer}")
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
