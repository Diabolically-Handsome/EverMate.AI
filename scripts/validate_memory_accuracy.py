#!/usr/bin/env python3
"""Run an accuracy-first memory benchmark against a long-form novel corpus.

This benchmark intentionally focuses on factual correctness:
- cloze: precise sentence-level extraction
- grounded_short_qa: single-hop factual QA
- multi_hop_consistency: multi-chunk consistency / event-order questions

Unlike the companion benchmark, this script does not score persona or tone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_manager import MemoryConfig, MemoryManager
from ollama_client import chat as ollama_chat


DEFAULT_DOCX = "/Users/lawrencegrey/Desktop/EverMate/全职高手.docx"
DEFAULT_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q8_0"
DEFAULT_QUESTION_MODEL = "hf.co/TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF:Q4_K_M"
DEFAULT_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "google").strip().lower() or "google"
DEFAULT_JUDGE_MODEL = os.getenv("GOOGLE_JUDGE_MODEL", "gemini-3.1-pro-preview")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_GOOGLE_BASE_URL = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
DEFAULT_MEMORY_DIR = "/tmp/evermate-memory-accuracy-validation"
DEFAULT_OUTPUT_DIR = "/Users/lawrencegrey/Desktop/EverMate/reports"
ROOT_CAUSES = (
    "retrieval_miss",
    "alias_mismatch",
    "generation_error",
    "multi_hop_conflict",
    "benchmark_ambiguous",
)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    path: str
    text: str
    snippet: str


@dataclass(frozen=True)
class AccuracyCase:
    case_id: int
    case_type: str
    question: str
    gold_answer: str
    accepted_aliases: Tuple[str, ...]
    supporting_chunk_ids: Tuple[int, ...]
    evidence_quotes: Tuple[str, ...]
    score: float


@dataclass(frozen=True)
class EvalRow:
    case: AccuracyCase
    model_answer: str
    label: str
    score: float
    reason: str
    root_cause: str
    retrieved_chunk_ids: Tuple[int, ...]
    retrieved_snippets: Tuple[str, ...]
    synthesis: Dict[str, object]
    used_fallback: bool
    answer_candidates: Tuple[Dict[str, object], ...]
    canonical_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EverMate memory accuracy against a novel corpus.")
    parser.add_argument("--docx", default=DEFAULT_DOCX, help="Path to the source .docx file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Exact Ollama model name used for answering.")
    parser.add_argument("--question-model", default=DEFAULT_QUESTION_MODEL, help="Local model used to generate the question bank.")
    parser.add_argument("--judge-provider", choices=("openai", "google"), default=DEFAULT_JUDGE_PROVIDER, help="Provider used for semantic judging.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model name.")
    parser.add_argument("--openai-base-url", default=DEFAULT_OPENAI_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--google-base-url", default=DEFAULT_GOOGLE_BASE_URL, help="Gemini API base URL.")
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR, help="Isolated memory directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for reports and artifacts.")
    parser.add_argument("--questions", type=int, default=77, help="Total number of questions to run.")
    parser.add_argument("--cloze-questions", type=int, default=52, help="Number of cloze questions.")
    parser.add_argument("--short-qa-questions", type=int, default=15, help="Number of grounded short QA questions.")
    parser.add_argument("--multi-hop-questions", type=int, default=10, help="Number of multi-hop consistency questions.")
    parser.add_argument("--retrieve-top-k", type=int, default=12, help="Top-k evidence chunks injected into the prompt.")
    parser.add_argument("--answer-timeout", type=int, default=300, help="Per-question answer timeout in seconds.")
    parser.add_argument("--generation-timeout", type=int, default=240, help="Question generation timeout in seconds.")
    parser.add_argument("--judge-timeout", type=int, default=180, help="Judge timeout in seconds.")
    parser.add_argument("--keep-memory-dir", action="store_true", help="Keep the isolated memory directory after the run.")
    return parser.parse_args()


def require_judge_env(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise SystemExit("Missing OPENAI_API_KEY. This accuracy benchmark needs an external judge for short QA and multi-hop cases.")
        return api_key
    if provider == "google":
        api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise SystemExit("Missing GOOGLE_API_KEY or GEMINI_API_KEY. This accuracy benchmark needs a Gemini judge for short QA and multi-hop cases.")
        return api_key
    raise SystemExit(f"Unsupported judge provider: {provider}")


def normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_key(text: str) -> str:
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("“", "").replace("”", "").replace('"', "")
    cleaned = cleaned.replace("‘", "").replace("’", "").replace("'", "")
    cleaned = cleaned.replace("。", "").replace("，", "").replace(",", "")
    cleaned = cleaned.replace("！", "").replace("!", "").replace("？", "").replace("?", "")
    cleaned = cleaned.replace("：", "").replace(":", "").replace("；", "").replace(";", "")
    cleaned = cleaned.replace("（", "(").replace("）", ")")
    cleaned = cleaned.replace("【", "[").replace("】", "]")
    cleaned = cleaned.replace("《", "").replace("》", "")
    cleaned = cleaned.replace("·", "")
    return cleaned


def normalize_match(text: str) -> str:
    cleaned = normalize_key(text)
    cleaned = cleaned.replace(" ", "")
    return cleaned


def truncate(text: str, limit: int = 220) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?])\s+|\n+", text) if p.strip()]
    return [part for part in parts if 8 <= len(part) <= 160]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", stem)
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


def extract_json_object(text: str) -> Dict[str, object]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{\s*\".*\}\s*$", cleaned, flags=re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON payload is not an object.")
    return data


def parse_json_array_with_repair(text: str, *, model: str, timeout: int) -> List[Dict[str, object]]:
    try:
        return extract_json_array(text)
    except Exception:
        repaired = ollama_chat(
            [
                {"role": "system", "content": "你是 JSON 修复器，只返回合法 JSON 数组。"},
                {"role": "user", "content": "请把下面内容修复成合法 JSON 数组，不要解释：\n\n" + (text or "")},
            ],
            model=model,
            timeout=timeout,
            options={"temperature": 0, "num_predict": 1800},
        )
        return extract_json_array(repaired)


def parse_json_object_with_repair(text: str, *, model: str, timeout: int) -> Dict[str, object]:
    try:
        return extract_json_object(text)
    except Exception:
        repaired = ollama_chat(
            [
                {"role": "system", "content": "你是 JSON 修复器，只返回合法 JSON 对象。"},
                {"role": "user", "content": "请把下面内容修复成合法 JSON 对象，不要解释：\n\n" + (text or "")},
            ],
            model=model,
            timeout=timeout,
            options={"temperature": 0, "num_predict": 1200},
        )
        return extract_json_object(repaired)


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
                snippet=truncate(snippet, 160),
            )
        )
    return out


def corpus_stats(docx_path: Path) -> Tuple[int, int]:
    try:
        from docx import Document  # type: ignore
    except Exception:
        return (0, 0)
    doc = Document(str(docx_path))
    paragraphs = [(para.text or "").strip() for para in doc.paragraphs]
    non_empty = [text for text in paragraphs if text]
    char_count = sum(len(text) for text in non_empty)
    return char_count, len(non_empty)


def rebuild_accuracy_memory(mm: MemoryManager) -> Dict[str, int]:
    """Rebuild the isolated benchmark memory without expensive persona analysis."""

    mm.close()

    if os.path.exists(mm.chunks_dir):
        for filename in os.listdir(mm.chunks_dir):
            path = os.path.join(mm.chunks_dir, filename)
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass

    Path(mm.vault_md_path).write_text(
        "# Vault (Long-tail)\n\n> Accuracy benchmark vault summary.\n\n",
        encoding="utf-8",
    )

    if os.path.exists(mm.db_path):
        os.remove(mm.db_path)
    mm.conn = mm._open_db()

    chunks_added = 0
    uploads = mm.list_uploads()
    for upload in uploads:
        chunks_added += mm._ingest_file(upload, source=f"upload:{os.path.basename(upload)}", auto_refresh=False)

    Path(mm.buffer_path).write_text("", encoding="utf-8")
    Path(mm.core_md_path).write_text(
        "# Core Memory\n\n"
        "- 当前处于准确度测试模式。\n"
        "- 回答请简短、直接、只说事实本身。\n"
        "- 记忆与当前输入冲突时，以当前输入为准。\n",
        encoding="utf-8",
    )
    Path(mm.persona_md_path).write_text(
        "# Persona (≤ 8 bullets)\n\n- 本轮不做人设评测，仅做事实准确度测试。\n",
        encoding="utf-8",
    )
    mm._meta_set_int("new_chunks_since_refresh", 0)
    mm._meta_set_int("last_analyze_ts", int(time.time()))
    return {
        "chunks": mm.count_chunks(),
        "terms": mm.count_terms(),
        "uploads": len(uploads),
        "chunks_added": chunks_added,
    }


def sample_chunks(chunks: Sequence[ChunkRecord], wanted: int) -> List[ChunkRecord]:
    if wanted >= len(chunks):
        return list(chunks)
    out: List[ChunkRecord] = []
    seen: set[int] = set()
    for i in range(wanted):
        pos = round(i * (len(chunks) - 1) / max(1, wanted - 1))
        chunk = chunks[pos]
        if chunk.chunk_id in seen:
            continue
        out.append(chunk)
        seen.add(chunk.chunk_id)
    if len(out) >= wanted:
        return out[:wanted]
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        out.append(chunk)
        seen.add(chunk.chunk_id)
        if len(out) >= wanted:
            break
    return out


def adjacent_chunk_windows(chunks: Sequence[ChunkRecord], window_size: int, wanted: int) -> List[Tuple[ChunkRecord, ...]]:
    if len(chunks) < window_size:
        return []
    starts = sample_chunks(list(chunks[: len(chunks) - window_size + 1]), wanted)
    windows: List[Tuple[ChunkRecord, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    for starter in starts:
        idx = next((i for i, chunk in enumerate(chunks) if chunk.chunk_id == starter.chunk_id), None)
        if idx is None or idx + window_size > len(chunks):
            continue
        window = tuple(chunks[idx : idx + window_size])
        key = tuple(chunk.chunk_id for chunk in window)
        if key in seen:
            continue
        seen.add(key)
        windows.append(window)
    if len(windows) >= wanted:
        return windows[:wanted]
    for index in range(0, len(chunks) - window_size + 1):
        window = tuple(chunks[index : index + window_size])
        key = tuple(chunk.chunk_id for chunk in window)
        if key in seen:
            continue
        seen.add(key)
        windows.append(window)
        if len(windows) >= wanted:
            break
    return windows


def answer_blacklist() -> set[str]:
    return {
        "自己",
        "这个",
        "那个",
        "这里",
        "那里",
        "他们",
        "我们",
        "于是",
        "然后",
        "因为",
        "所以",
        "而且",
        "只是",
        "不是",
        "可以",
        "不会",
        "一个",
        "一种",
        "一下",
        "什么",
        "怎么",
        "时候",
        "事情",
        "地方",
        "玩家们",
        "荣耀里",
        "网吧里",
    }


def semantic_context_length(text: str) -> int:
    return len(re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", text))


def validate_aliases(raw_aliases: object, gold_answer: str) -> Tuple[str, ...]:
    aliases: List[str] = []
    seen: set[str] = set()
    for text in [gold_answer] + list(raw_aliases if isinstance(raw_aliases, list) else []):
        item = normalize_space(str(text))
        if not item:
            continue
        key = normalize_match(item)
        if not key or key in seen:
            continue
        seen.add(key)
        aliases.append(item)
        if len(aliases) >= 5:
            break
    return tuple(aliases)


def validate_evidence_quotes(raw_quotes: object, support_text: str, gold_answer: str) -> Tuple[str, ...]:
    if not isinstance(raw_quotes, list):
        return tuple()
    quotes: List[str] = []
    seen: set[str] = set()
    support_key = normalize_space(support_text)
    for item in raw_quotes:
        quote = normalize_space(str(item))
        if not quote or quote in seen:
            continue
        if quote not in support_key:
            continue
        if gold_answer and normalize_match(gold_answer) not in normalize_match(quote):
            if len(quotes) >= 1:
                continue
        quotes.append(quote)
        seen.add(quote)
        if len(quotes) >= 4:
            break
    return tuple(quotes)


def cloze_patterns() -> List[Tuple[str, float]]:
    return [
        (r"“([^”]{2,18})”", 12.0),
        (r"\"([^\"]{2,18})\"", 12.0),
        (r"([A-Za-z][A-Za-z0-9_.:+/-]{1,30})", 9.0),
        (r"([\u4e00-\u9fff]{2,8})", 5.0),
    ]


def build_lexicon(chunks: Sequence[ChunkRecord]) -> set[str]:
    counts: Counter[str] = Counter()
    blacklist = answer_blacklist()
    for chunk in chunks:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.:+/-]{1,24}|[\u4e00-\u9fff]{2,8}", chunk.text):
            token = token.strip()
            if not token or token in blacklist:
                continue
            if token.isdigit():
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]{2,8}", token):
                if any(char in "的了是着把和就也都还很在有与及并让给对被从向" for char in token):
                    continue
                counts[token] += 1
            else:
                counts[token] += 1
    lexicon = {
        token
        for token, freq in counts.items()
        if (re.search(r"[A-Za-z]", token) and freq >= 2) or (re.search(r"[\u4e00-\u9fff]", token) and freq >= 3)
    }
    return lexicon


def build_cloze_candidates(chunks: Sequence[ChunkRecord]) -> List[Dict[str, object]]:
    blacklist = answer_blacklist()
    lexicon = build_lexicon(chunks)
    patterns = cloze_patterns()
    candidates: List[Dict[str, object]] = []

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if semantic_context_length(sentence) < 12:
                continue

            # First try structured patterns.
            for pattern, base_score in patterns:
                for match in re.finditer(pattern, sentence):
                    answer = (match.group(1) if match.groups() else match.group(0)).strip(' ：:,.，。!！?？"“”()[]《》')
                    if not answer or answer in blacklist:
                        continue
                    if answer not in lexicon and not re.search(r"[A-Za-z]", answer) and answer not in sentence:
                        continue
                    if len(answer) < 2 or len(answer) > 18:
                        continue
                    if sentence.count(answer) != 1:
                        continue
                    question = sentence.replace(answer, "____", 1)
                    if question.count("____") != 1 or len(question.replace("____", "").strip()) < 8:
                        continue
                    if normalize_match(question).startswith("____"):
                        continue
                    score = float(base_score) + min(semantic_context_length(question) / 12.0, 3.0)
                    if re.search(r"\d", answer):
                        score += 1.2
                    if answer in lexicon:
                        score += 1.0
                    candidates.append(
                        {
                            "case_type": "cloze",
                            "chunk_id": chunk.chunk_id,
                            "question": "请根据记忆填空，只填写空缺内容：" + question,
                            "gold_answer": answer,
                            "accepted_aliases": [answer],
                            "supporting_chunk_ids": [chunk.chunk_id],
                            "evidence_quotes": [sentence],
                            "score": score,
                        }
                    )

            # Then use lexicon hit fallback if nothing else stood out.
            if not any(int(candidate["chunk_id"]) == chunk.chunk_id and sentence in candidate["evidence_quotes"] for candidate in candidates[-4:]):
                for answer in sorted((token for token in lexicon if token in sentence), key=len, reverse=True):
                    if sentence.count(answer) != 1 or len(answer) < 2 or len(answer) > 10:
                        continue
                    question = sentence.replace(answer, "____", 1)
                    if question.count("____") != 1 or semantic_context_length(question.replace("____", "")) < 12:
                        continue
                    candidates.append(
                        {
                            "case_type": "cloze",
                            "chunk_id": chunk.chunk_id,
                            "question": "请根据记忆填空，只填写空缺内容：" + question,
                            "gold_answer": answer,
                            "accepted_aliases": [answer],
                            "supporting_chunk_ids": [chunk.chunk_id],
                            "evidence_quotes": [sentence],
                            "score": 5.5 + min(len(answer), 6) * 0.2,
                        }
                    )
                    break

    answer_freq = Counter(str(candidate["gold_answer"]) for candidate in candidates)
    for candidate in candidates:
        candidate["score"] = float(candidate["score"]) - min(answer_freq[str(candidate["gold_answer"])], 6) * 0.25
    return candidates


def select_cases_from_candidates(candidates: Sequence[Dict[str, object]], target: int) -> List[Dict[str, object]]:
    ordered = sorted(candidates, key=lambda item: (-float(item["score"]), int(item["chunk_id"])))
    selected: List[Dict[str, object]] = []
    seen_chunks: set[int] = set()
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    for candidate in ordered:
        chunk_id = int(candidate["chunk_id"])
        question_key = normalize_match(str(candidate["question"]))
        answer_key = normalize_match(str(candidate["gold_answer"]))
        if question_key in seen_questions or answer_key in seen_answers:
            continue
        if chunk_id in seen_chunks:
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


def validate_generated_case(
    raw: Dict[str, object],
    *,
    case_type: str,
    chunk_map: Dict[int, ChunkRecord],
    allowed_chunk_ids: set[int],
    seen_questions: set[str],
    seen_answers: set[str],
) -> Optional[Dict[str, object]]:
    question = normalize_space(str(raw.get("question", "")))
    gold_answer = normalize_space(str(raw.get("gold_answer", "")))
    if not question or not gold_answer:
        return None
    question_key = normalize_match(question)
    answer_key = normalize_match(gold_answer)
    if question_key in seen_questions or answer_key in seen_answers:
        return None

    if case_type == "cloze":
        if "____" not in question:
            return None
        if len(gold_answer) < 2 or len(gold_answer) > 18:
            return None
    elif case_type == "grounded_short_qa":
        if any(token in question for token in ("为什么", "为何", "怎么会")):
            return None
        if len(gold_answer) > 42:
            return None
    elif case_type == "multi_hop_consistency":
        if len(question) < 12 or len(gold_answer) > 80:
            return None
    else:
        return None

    raw_chunk_ids = raw.get("supporting_chunk_ids", [])
    if not isinstance(raw_chunk_ids, list):
        return None
    supporting_chunk_ids: List[int] = []
    for item in raw_chunk_ids:
        try:
            cid = int(item)
        except Exception:
            continue
        if cid in allowed_chunk_ids and cid in chunk_map and cid not in supporting_chunk_ids:
            supporting_chunk_ids.append(cid)
    if case_type == "multi_hop_consistency":
        if len(supporting_chunk_ids) < 2:
            return None
    elif len(supporting_chunk_ids) != 1:
        return None

    support_text = "\n".join(chunk_map[cid].text for cid in supporting_chunk_ids)
    if normalize_match(gold_answer) not in normalize_match(support_text) and case_type != "multi_hop_consistency":
        return None

    accepted_aliases = validate_aliases(raw.get("accepted_aliases", []), gold_answer)
    evidence_quotes = validate_evidence_quotes(raw.get("evidence_quotes", []), support_text, gold_answer)
    if not evidence_quotes:
        # Fallback: take a sentence from support that contains the gold answer.
        for sentence in split_sentences(support_text):
            if normalize_match(gold_answer) in normalize_match(sentence):
                evidence_quotes = (sentence,)
                break
    if not evidence_quotes:
        return None

    score = float(raw.get("score", 0.0) or 0.0)
    if score <= 0.0:
        score = float(len(evidence_quotes) + len(supporting_chunk_ids))
    if case_type == "multi_hop_consistency":
        score += min((max(supporting_chunk_ids) - min(supporting_chunk_ids)) / 24.0, 6.0)
    seen_questions.add(question_key)
    seen_answers.add(answer_key)
    return {
        "case_type": case_type,
        "question": question,
        "gold_answer": gold_answer,
        "accepted_aliases": list(accepted_aliases),
        "supporting_chunk_ids": supporting_chunk_ids,
        "evidence_quotes": list(evidence_quotes),
        "score": score,
    }


def support_verdict(
    *,
    case_type: str,
    question: str,
    gold_answer: str,
    support_text: str,
    model: str,
    timeout: int,
) -> bool:
    answer_key = normalize_match(gold_answer)
    support_key = normalize_match(support_text)
    if case_type != "multi_hop_consistency" and answer_key and answer_key in support_key:
        return True
    if case_type == "multi_hop_consistency":
        special_tokens = []
        for token in re.findall(r"\d+级|\d+|“[^”]{2,18}”|\"[^\"]{2,18}\"|[A-Za-z][A-Za-z0-9_.:+/-]{1,24}", gold_answer):
            normalized = normalize_match(token.strip("“”\""))
            if normalized:
                special_tokens.append(normalized)
        if special_tokens and any(token not in support_key for token in special_tokens):
                return False
    prompt = (
        "你是一个非常严格的中文题库审核器。\n"
        "请判断 question 和 gold_answer 是否能被 support_text 充分支持。\n"
        "只看事实支持，不看文风。若 gold_answer 需要的关键事实没有在 support_text 中明确出现或无法稳定推出，就判 unsupported。\n"
        "只返回 JSON 对象：{\"supported\":true|false,\"reason\":\"一句中文短说明\"}。\n\n"
        + json.dumps(
            {
                "case_type": case_type,
                "question": question,
                "gold_answer": gold_answer,
                "support_text": truncate(support_text, 3200),
            },
            ensure_ascii=False,
        )
    )
    try:
        raw = ollama_chat(
            [
                {"role": "system", "content": "只返回合法 JSON 对象。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            timeout=timeout,
            options={"temperature": 0, "num_predict": 600},
        )
        verdict = parse_json_object_with_repair(raw, model=model, timeout=timeout)
        return bool(verdict.get("supported"))
    except Exception:
        return case_type != "multi_hop_consistency"


def generate_cloze_cases(
    *,
    chunks: Sequence[ChunkRecord],
    target: int,
) -> List[Dict[str, object]]:
    candidates = build_cloze_candidates(chunks)
    return select_cases_from_candidates(candidates, target)


def generate_grounded_short_qa_cases(
    *,
    chunks: Sequence[ChunkRecord],
    target: int,
    model: str,
    timeout: int,
) -> List[Dict[str, object]]:
    if target <= 0:
        return []
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    sampled = sample_chunks(chunks, max(target * 4, target + 10))
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    out: List[Dict[str, object]] = []

    for index, chunk in enumerate(sampled, start=1):
        if len(out) >= target:
            break
        prompt = (
            "你是一个高标准中文题库设计器。请只根据给定小说片段，生成 2 道 grounded_short_qa 事实题。\n"
            "硬性要求：\n"
            "1. 题目必须是单跳可回答的事实问答，主测人名、地点、战队、装备、称号、具体结果等。\n"
            "2. 不要生成为什么题，不要生成开放感想题。\n"
            "3. gold_answer 必须短而确定，且能在片段中直接找到。\n"
            "4. accepted_aliases 可以包含别名或同义短写；若没有就填空数组。\n"
            "5. supporting_chunk_ids 必须只填当前 chunk id。\n"
            "6. evidence_quotes 提供 1-2 条能在片段中精确找到的原文短句。\n"
            "7. 只返回 JSON 数组，不要解释。格式："
            '[{"question":"...","gold_answer":"...","accepted_aliases":["..."],"supporting_chunk_ids":[123],"evidence_quotes":["..."]}]。\n\n'
            f"[Chunk ID]\n{chunk.chunk_id}\n\n"
            "[Chunk Text]\n"
            + chunk.text
        )
        print(f"Generating grounded_short_qa from chunk {index}/{len(sampled)} #{chunk.chunk_id}", flush=True)
        try:
            raw = ollama_chat(
                [
                    {"role": "system", "content": "只返回合法 JSON 数组。"},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                timeout=timeout,
                options={"temperature": 0, "num_predict": 1600},
            )
            parsed = parse_json_array_with_repair(raw, model=model, timeout=timeout)
        except Exception:
            continue
        added = 0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            candidate = validate_generated_case(
                item,
                case_type="grounded_short_qa",
                chunk_map=chunk_map,
                allowed_chunk_ids={chunk.chunk_id},
                seen_questions=seen_questions,
                seen_answers=seen_answers,
            )
            if candidate:
                support_text = "\n".join(chunk_map[cid].text for cid in candidate["supporting_chunk_ids"])
                if not support_verdict(
                    case_type="grounded_short_qa",
                    question=str(candidate["question"]),
                    gold_answer=str(candidate["gold_answer"]),
                    support_text=support_text,
                    model=model,
                    timeout=timeout,
                ):
                    continue
                candidate["score"] = float(candidate["score"]) + min(len(candidate["gold_answer"]), 12) * 0.1
                out.append(candidate)
                added += 1
                if len(out) >= target:
                    break
        print(f"  -> accumulated {len(out)} grounded_short_qa cases", flush=True)
        if added == 0:
            continue

    return out[:target]


def generate_multi_hop_cases(
    *,
    chunks: Sequence[ChunkRecord],
    target: int,
    model: str,
    timeout: int,
    judge_provider: Optional[str] = None,
    judge_api_key: str = "",
    judge_base_url: str = "",
    judge_model: str = "",
    judge_timeout: int = 0,
) -> List[Dict[str, object]]:
    if target <= 0:
        return []
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    windows = adjacent_chunk_windows(chunks, window_size=3, wanted=max(target * 6, target + 20))
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    out: List[Dict[str, object]] = []

    for index, window in enumerate(windows, start=1):
        if len(out) >= target:
            break
        chunk_ids = [chunk.chunk_id for chunk in window]
        evidence_block = "\n\n".join(f"[[CHUNK {chunk.chunk_id}]]\n{chunk.text}" for chunk in window)
        prompt = (
            "你是一个高标准中文题库设计器。请只根据给定的多个小说片段，生成 1 道 multi_hop_consistency 题。\n"
            "硬性要求：\n"
            "1. 题目必须至少结合 2 个 chunk 才能回答。\n"
            "2. 重点测试人物、战队、装备、事件顺序、前后状态或剧情设定是否一致。\n"
            "3. 不要生成开放感想题，不要生成人物动机分析题。\n"
            "4. gold_answer 应是事实结论，可以是 1-2 句短答。\n"
            "5. accepted_aliases 可包含语序变化或同义短答。\n"
            "6. supporting_chunk_ids 至少填 2 个，且必须从给定 chunk id 中选择。\n"
            "7. evidence_quotes 给 2-4 条短证据，必须能在给定文本中精确找到。\n"
            "8. 只返回 JSON 数组，不要解释。格式："
            '[{"question":"...","gold_answer":"...","accepted_aliases":["..."],"supporting_chunk_ids":[1,2],"evidence_quotes":["..."]}]。\n\n'
            f"[Chunk IDs]\n{', '.join(str(cid) for cid in chunk_ids)}\n\n"
            "[Evidence Pack]\n"
            + evidence_block
        )
        print(f"Generating multi_hop_consistency from window {index}/{len(windows)} {chunk_ids}", flush=True)
        try:
            raw = ollama_chat(
                [
                    {"role": "system", "content": "只返回合法 JSON 数组。"},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                timeout=timeout,
                options={"temperature": 0, "num_predict": 1600},
            )
            parsed = parse_json_array_with_repair(raw, model=model, timeout=timeout)
        except Exception:
            continue
        for item in parsed:
            if not isinstance(item, dict):
                continue
            candidate = validate_generated_case(
                item,
                case_type="multi_hop_consistency",
                chunk_map=chunk_map,
                allowed_chunk_ids=set(chunk_ids),
                seen_questions=seen_questions,
                seen_answers=seen_answers,
            )
            if candidate:
                support_text = "\n".join(chunk_map[cid].text for cid in candidate["supporting_chunk_ids"])
                if not support_verdict(
                    case_type="multi_hop_consistency",
                    question=str(candidate["question"]),
                    gold_answer=str(candidate["gold_answer"]),
                    support_text=support_text,
                    model=model,
                    timeout=timeout,
                ):
                    continue
                if judge_provider and judge_api_key and judge_model:
                    try:
                        supported = judge_support_alignment(
                            provider=judge_provider,
                            api_key=judge_api_key,
                            base_url=judge_base_url,
                            model=judge_model,
                            case_type="multi_hop_consistency",
                            question=str(candidate["question"]),
                            gold_answer=str(candidate["gold_answer"]),
                            support_text=support_text,
                            evidence_quotes=tuple(str(item) for item in candidate["evidence_quotes"]),
                            timeout=judge_timeout or timeout,
                        )
                    except Exception:
                        supported = False
                    if not supported:
                        continue
                out.append(candidate)
                if len(out) >= target:
                    break
        print(f"  -> accumulated {len(out)} multi_hop_consistency cases", flush=True)

    return out[:target]


def build_accuracy_cases(
    *,
    chunks: Sequence[ChunkRecord],
    cloze_questions: int,
    short_qa_questions: int,
    multi_hop_questions: int,
    question_model: str,
    timeout: int,
    judge_provider: Optional[str] = None,
    judge_api_key: str = "",
    judge_base_url: str = "",
    judge_model: str = "",
    judge_timeout: int = 0,
) -> List[AccuracyCase]:
    cloze_raw = generate_cloze_cases(chunks=chunks, target=cloze_questions)
    print(f"Built {len(cloze_raw)} cloze cases.", flush=True)
    if len(cloze_raw) < cloze_questions:
        raise RuntimeError(f"Only generated {len(cloze_raw)} cloze cases, expected {cloze_questions}.")

    short_raw = generate_grounded_short_qa_cases(
        chunks=chunks,
        target=short_qa_questions,
        model=question_model,
        timeout=timeout,
    )
    print(f"Built {len(short_raw)} grounded_short_qa cases.", flush=True)
    if len(short_raw) < short_qa_questions:
        raise RuntimeError(f"Only generated {len(short_raw)} grounded_short_qa cases, expected {short_qa_questions}.")

    multi_hop_raw = generate_multi_hop_cases(
        chunks=chunks,
        target=multi_hop_questions,
        model=question_model,
        timeout=timeout,
        judge_provider=judge_provider,
        judge_api_key=judge_api_key,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_timeout=judge_timeout,
    )
    print(f"Built {len(multi_hop_raw)} multi_hop_consistency cases.", flush=True)
    if len(multi_hop_raw) < multi_hop_questions:
        raise RuntimeError(f"Only generated {len(multi_hop_raw)} multi_hop_consistency cases, expected {multi_hop_questions}.")

    raw_cases = cloze_raw + short_raw + multi_hop_raw
    out: List[AccuracyCase] = []
    for index, row in enumerate(raw_cases, start=1):
        out.append(
            AccuracyCase(
                case_id=index,
                case_type=str(row["case_type"]),
                question=str(row["question"]),
                gold_answer=str(row["gold_answer"]),
                accepted_aliases=tuple(str(item) for item in row["accepted_aliases"]),
                supporting_chunk_ids=tuple(int(item) for item in row["supporting_chunk_ids"]),
                evidence_quotes=tuple(str(item) for item in row["evidence_quotes"]),
                score=float(row["score"]),
            )
        )
    return out


def case_response_style(case: AccuracyCase) -> Tuple[str, str]:
    if case.case_type == "cloze":
        style = "请只填写空缺内容本身，不要解释，不要重复整句。"
        answer_mode = "fact"
    elif case.case_type == "grounded_short_qa":
        answer_mode = "fact"
        style = (
            "请用中文简短直接回答，只回答事实本身。"
            "优先使用证据里的标准名字、标准术语或标准数值。"
            "不要发挥，不要使用项目符号。"
        )
    else:
        answer_mode = "multi_hop"
        style = (
            "请用中文直接回答。"
            "先对齐人物、事件和结论，再输出最终结论。"
            "如果题目在问分别是什么，请把每一部分都答全。"
            "不要发挥，不要使用项目符号。"
        )
        if "阵容" in case.question or "首发" in case.question:
            style += "若题目涉及阵容，请完整列出所有关键名字，并单独交代关键人物是否上场。"
        if "传闻" in case.question and "实际" in case.question:
            style += "若题目要求区分传闻与实际，请明确分成“传闻：…；实际：…”两部分。"
    return style, answer_mode


def build_case_turn_plan(mm: MemoryManager, case: AccuracyCase) -> Dict[str, object]:
    style, answer_mode = case_response_style(case)
    return mm.build_turn_plan(
        user_text=case.question,
        assistant_style=style,
        lang="zh",
        answer_mode=answer_mode,
    )


def build_messages(mm: MemoryManager, case: AccuracyCase) -> List[Dict[str, str]]:
    return list(build_case_turn_plan(mm, case).get("messages", []))


def alias_keys(case: AccuracyCase) -> List[str]:
    values = [case.gold_answer] + list(case.accepted_aliases)
    keys: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_match(value)
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def answer_exact_alias_hit(answer: str, case: AccuracyCase) -> bool:
    answer_key = normalize_match(answer)
    if not answer_key:
        return False
    return any(answer_key == candidate for candidate in alias_keys(case))


def answer_contains_alias(answer: str, case: AccuracyCase) -> bool:
    answer_key = normalize_match(answer)
    if not answer_key:
        return False
    return any(candidate in answer_key or answer_key in candidate for candidate in alias_keys(case))


def retrieval_alignment(case: AccuracyCase, retrieved_chunk_ids: Iterable[int]) -> str:
    support = set(case.supporting_chunk_ids)
    retrieved = set(int(item) for item in retrieved_chunk_ids)
    if not support:
        return "unsupported"
    if support.issubset(retrieved):
        return "full"
    if support & retrieved:
        return "partial"
    return "miss"


def support_fragments_for_case(case: AccuracyCase, chunk_map: Dict[int, ChunkRecord]) -> List[str]:
    fragments: List[str] = []
    if case.case_type == "multi_hop_consistency":
        for chunk_id in case.supporting_chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            fragment = truncate(chunk.text, 4000)
            if fragment:
                fragments.append(fragment)
        return fragments
    needle_keys = [normalize_match(case.gold_answer)] + [normalize_match(alias) for alias in case.accepted_aliases]
    needle_keys.extend(normalize_match(quote) for quote in case.evidence_quotes)
    needle_keys = [needle for needle in needle_keys if needle]
    for chunk_id in case.supporting_chunk_ids:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        matched_sentences: List[str] = []
        for sentence in split_sentences(chunk.text):
            sentence_key = normalize_match(sentence)
            if any(needle in sentence_key or sentence_key in needle for needle in needle_keys if needle):
                matched_sentences.append(sentence)
            if len(matched_sentences) >= 4:
                break
        if not matched_sentences:
            matched_sentences = split_sentences(chunk.text)[:3]
        fragment = truncate(" ".join(matched_sentences), 700)
        if fragment:
            fragments.append(fragment)
    return fragments


def heuristic_cloze_eval(case: AccuracyCase, answer: str, retrieved_chunk_ids: Sequence[int]) -> Tuple[str, float, str, str]:
    answer = normalize_space(answer)
    if not answer:
        if retrieval_alignment(case, retrieved_chunk_ids) == "miss":
            return ("Miss", 0.0, "没有作答，且检索未命中支撑 chunk。", "retrieval_miss")
        return ("Miss", 0.0, "没有作答。", "generation_error")

    gold_key = normalize_match(case.gold_answer)
    answer_key = normalize_match(answer)
    aliases = alias_keys(case)
    if answer_key == gold_key:
        return ("Exact", 1.0, "严格命中参考答案。", "alias_mismatch")
    if answer_key in aliases:
        return ("Exact", 1.0, "命中可接受别名或格式归一化答案。", "alias_mismatch")
    if any(candidate in answer_key for candidate in aliases):
        return ("Partial", 0.5, "答案包含参考答案，但带有额外内容。", "alias_mismatch")

    similarity = SequenceMatcher(None, answer_key, gold_key).ratio() if answer_key and gold_key else 0.0
    if similarity >= 0.88:
        return ("Partial", 0.5, "答案和参考答案非常接近，但不够严格。", "alias_mismatch")

    if retrieval_alignment(case, retrieved_chunk_ids) == "miss":
        return ("Miss", 0.0, "检索未命中支撑 chunk。", "retrieval_miss")
    return ("Miss", 0.0, "回答没有命中参考答案。", "generation_error")


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


def judge_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": ["Exact", "Partial", "Miss"]},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "root_cause": {"type": "string", "enum": list(ROOT_CAUSES)},
            "reason": {"type": "string"},
        },
        "required": ["label", "score", "root_cause", "reason"],
    }


def support_check_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "supported": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["supported", "reason"],
    }


def judge_support_alignment(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    case_type: str,
    question: str,
    gold_answer: str,
    support_text: str,
    evidence_quotes: Sequence[str],
    timeout: int,
) -> bool:
    prompt = (
        "你是一个严格的中文题库审核器。\n"
        "请只判断 gold_answer 是否能够被 support_text 充分支持。\n"
        "若 support_text 不足以稳定推出 gold_answer，就判 supported=false。\n"
        "只看事实支持，不看文风。只返回 JSON。\n\n"
        + json.dumps(
            {
                "case_type": case_type,
                "question": question,
                "gold_answer": gold_answer,
                "evidence_quotes": list(evidence_quotes),
                "support_text": truncate(support_text, 5000),
            },
            ensure_ascii=False,
        )
    )
    messages = [
        {"role": "system", "content": "You are a strict Chinese JSON verifier."},
        {"role": "user", "content": prompt},
    ]
    if provider == "openai":
        data = openai_chat_json(
            api_key=api_key,
            base_url=base_url,
            model=model,
            schema_name="accuracy_support_verdict",
            schema=support_check_schema(),
            messages=messages,
            timeout=timeout,
        )
    elif provider == "google":
        data = google_chat_json(
            api_key=api_key,
            base_url=base_url,
            model=model,
            schema=support_check_schema(),
            messages=messages,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported judge provider: {provider}")
    return bool(data.get("supported"))


def judge_case(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    case: AccuracyCase,
    model_answer: str,
    retrieved_chunk_ids: Sequence[int],
    retrieved_snippets: Sequence[str],
    support_snippets: Sequence[str],
    timeout: int,
) -> Dict[str, object]:
    prompt = (
        "你是一个严格但公平的小说问答准确度裁判。\n"
        "这不是人设评测，也不评文风；你只判断事实是否正确、是否有证据支持。\n\n"
        "评分规则：\n"
        "1. Exact: 回答事实正确、与 supporting evidence 一致，且完整回答了问题。\n"
        "2. Partial: 基本方向正确，但答案不够完整、过于含糊、只答对一部分，或只是别名/表面表达差异。\n"
        "3. Miss: 回答错误、与证据冲突、或无法被证据支持。\n"
        "4. 接受 accepted_aliases、同义短写、语序变化，不要做脆弱字符串匹配。\n"
        "5. 对 multi_hop_consistency，必须确认回答真的能由多个支撑 chunk 共同推出；如果把前后事件或状态拼错，root_cause 选 multi_hop_conflict。\n"
        "6. root_cause 只能从以下枚举中选一个：retrieval_miss | alias_mismatch | generation_error | multi_hop_conflict | benchmark_ambiguous。\n"
        "7. 如果 supporting evidence 自己就不足以稳定支撑 gold_answer，请选 benchmark_ambiguous。\n"
        "8. 如果 supporting evidence 很清楚，但 retrieved evidence 没捞到关键 chunk，请选 retrieval_miss。\n"
        "9. 如果只是表面措辞、别名、标点或语序问题，请选 alias_mismatch。\n"
        "10. 如果检索已给到关键证据，但回答仍然说错，请选 generation_error。\n\n"
        "请基于以下材料判断：\n"
        + json.dumps(
            {
                "case": {
                    "case_type": case.case_type,
                    "question": case.question,
                    "gold_answer": case.gold_answer,
                    "accepted_aliases": list(case.accepted_aliases),
                    "supporting_chunk_ids": list(case.supporting_chunk_ids),
                    "evidence_quotes": list(case.evidence_quotes),
                },
                "supporting_snippets": list(support_snippets),
                "retrieved_chunk_ids": list(retrieved_chunk_ids),
                "retrieved_snippets": list(retrieved_snippets),
                "model_answer": model_answer,
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
            schema_name="accuracy_case_judgment",
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


def evaluate_case(
    *,
    mm: MemoryManager,
    case: AccuracyCase,
    answer_model: str,
    answer_timeout: int,
    retrieve_top_k: int,
    judge_provider: str,
    judge_api_key: str,
    judge_base_url: str,
    judge_model: str,
    judge_timeout: int,
    chunk_map: Dict[int, ChunkRecord],
) -> EvalRow:
    plan = build_case_turn_plan(mm, case)
    retrieved_bundle = dict(plan.get("bundle", {}))
    retrieved = list(retrieved_bundle.get("items", []))
    retrieved_chunk_ids = tuple(int(item.get("chunk_id", 0) or 0) for item in retrieved)
    retrieved_snippets = tuple(
        truncate(" | ".join(str(line) for line in item.get("answer_lines", []) if str(line).strip()) or str(item.get("snippet", "")), 220)
        for item in retrieved
    )
    answer_candidates = tuple(dict(item) for item in list(retrieved_bundle.get("answer_candidates", []) or []))
    canonical_answer = str(retrieved_bundle.get("canonical_answer", "") or "")

    num_predict = 56 if case.case_type == "cloze" else 160
    synthesis_payload: Dict[str, object] = {}
    used_fallback = False
    if str(plan.get("direct_answer", "")).strip():
        model_answer = str(plan.get("direct_answer", "")).strip()
        used_fallback = True
    elif bool(plan.get("two_pass")) and case.case_type == "multi_hop_consistency":
        subtype = str(plan.get("question_subtype") or "")
        synthesis_text = ollama_chat(
            list(plan.get("synthesis_messages", [])),
            model=answer_model,
            timeout=answer_timeout,
            options={"temperature": 0, "num_predict": 384},
        ).strip()
        synthesis = mm.parse_multi_hop_synthesis(synthesis_text, subtype)
        if synthesis:
            synthesis = mm.repair_multi_hop_synthesis(
                user_text=case.question,
                synthesis=synthesis,
                retrieved_bundle=retrieved_bundle,
            )
            synthesis_payload = dict(synthesis)
            answer_messages = mm.build_multi_hop_answer_messages(
                user_text=case.question,
                assistant_style=case_response_style(case)[0],
                synthesis=synthesis,
                lang="zh",
                retrieved_bundle=retrieved_bundle,
            )
        else:
            answer_messages = list(plan.get("fallback_messages", plan.get("messages", [])))
        model_answer = ollama_chat(
            answer_messages,
            model=answer_model,
            timeout=answer_timeout,
            options={"temperature": 0, "num_predict": 224},
        ).strip()
        if synthesis and mm.needs_multi_hop_answer_fallback(model_answer, synthesis, case.question):
            model_answer = mm.render_multi_hop_answer(case.question, synthesis).strip()
            used_fallback = True
    else:
        model_answer = ollama_chat(
            list(plan.get("messages", [])),
            model=answer_model,
            timeout=answer_timeout,
            options={"temperature": 0, "num_predict": num_predict},
        ).strip()
        if str(plan.get("mode", "")) == "fact" and mm.needs_fact_answer_fallback(
            model_answer,
            retrieved_bundle,
            case.question,
        ):
            model_answer = mm.render_fact_answer(case.question, retrieved_bundle=retrieved_bundle).strip()
            used_fallback = True

    if case.case_type == "cloze":
        label, score, reason, root_cause = heuristic_cloze_eval(case, model_answer, retrieved_chunk_ids)
        return EvalRow(
            case=case,
            model_answer=model_answer,
            label=label,
            score=score,
            reason=reason,
            root_cause=root_cause,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_snippets=retrieved_snippets,
            synthesis=synthesis_payload,
            used_fallback=used_fallback,
            answer_candidates=answer_candidates,
            canonical_answer=canonical_answer,
        )

    if answer_exact_alias_hit(model_answer, case):
        return EvalRow(
            case=case,
            model_answer=model_answer,
            label="Exact",
            score=1.0,
            reason="命中参考答案或可接受别名。",
            root_cause="alias_mismatch",
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_snippets=retrieved_snippets,
            synthesis=synthesis_payload,
            used_fallback=used_fallback,
            answer_candidates=answer_candidates,
            canonical_answer=canonical_answer,
        )

    if answer_contains_alias(model_answer, case):
        label = "Exact" if case.case_type == "grounded_short_qa" else "Partial"
        score = 1.0 if label == "Exact" else 0.5
        return EvalRow(
            case=case,
            model_answer=model_answer,
            label=label,
            score=score,
            reason="回答包含参考答案，事实本身正确。" if label == "Exact" else "回答包含参考答案，但附带了额外内容或表达不够紧凑。",
            root_cause="alias_mismatch",
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_snippets=retrieved_snippets,
            synthesis=synthesis_payload,
            used_fallback=used_fallback,
            answer_candidates=answer_candidates,
            canonical_answer=canonical_answer,
        )

    support_snippets = support_fragments_for_case(case, chunk_map)
    judged = judge_case(
        provider=judge_provider,
        api_key=judge_api_key,
        base_url=judge_base_url,
        model=judge_model,
        case=case,
        model_answer=model_answer,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_snippets=retrieved_snippets,
        support_snippets=support_snippets,
        timeout=judge_timeout,
    )
    label = str(judged["label"])
    score = float(judged["score"])
    root_cause = str(judged["root_cause"])
    reason = normalize_space(str(judged["reason"]))
    return EvalRow(
        case=case,
        model_answer=model_answer,
        label=label,
        score=score,
        reason=reason,
        root_cause=root_cause,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_snippets=retrieved_snippets,
        synthesis=synthesis_payload,
        used_fallback=used_fallback,
        answer_candidates=answer_candidates,
        canonical_answer=canonical_answer,
    )


def summarize_results(rows: Sequence[EvalRow]) -> Dict[str, object]:
    counts = Counter(row.label for row in rows)
    root_cause_breakdown = Counter(row.root_cause for row in rows if row.label != "Exact")
    type_counts = Counter(row.case.case_type for row in rows)
    grouped: Dict[str, List[EvalRow]] = defaultdict(list)
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
        "type_counts": dict(type_counts),
    }


def artifact_prefix(slug: str, total_questions: int) -> str:
    if total_questions == 77:
        return f"{slug}-accuracy"
    return f"{slug}-{total_questions}-accuracy-smoke"


def write_question_bank(output_dir: Path, prefix: str, cases: Sequence[AccuracyCase]) -> Tuple[Path, Path]:
    json_path = output_dir / f"{prefix}-question-bank.json"
    md_path = output_dir / f"{prefix}-question-bank.md"

    json_path.write_text(json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {prefix} Question Bank",
        "",
        f"- Questions: {len(cases)}",
        f"- Cloze: {sum(1 for case in cases if case.case_type == 'cloze')}",
        f"- Grounded Short QA: {sum(1 for case in cases if case.case_type == 'grounded_short_qa')}",
        f"- Multi-hop Consistency: {sum(1 for case in cases if case.case_type == 'multi_hop_consistency')}",
        "",
    ]
    for case in cases:
        lines.append(f"## {case.case_id:02d} [{case.case_type}]")
        lines.append(f"- Question: {case.question}")
        lines.append(f"- Gold Answer: {case.gold_answer}")
        lines.append(f"- Accepted Aliases: {' | '.join(case.accepted_aliases) if case.accepted_aliases else '(none)'}")
        lines.append(f"- Supporting Chunks: {', '.join(str(cid) for cid in case.supporting_chunk_ids)}")
        lines.append(f"- Evidence: {' | '.join(case.evidence_quotes)}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def result_payload(row: EvalRow) -> Dict[str, object]:
    return {
        "case_id": row.case.case_id,
        "case_type": row.case.case_type,
        "question": row.case.question,
        "gold_answer": row.case.gold_answer,
        "accepted_aliases": list(row.case.accepted_aliases),
        "supporting_chunk_ids": list(row.case.supporting_chunk_ids),
        "retrieved_chunk_ids": list(row.retrieved_chunk_ids),
        "retrieved_snippets": list(row.retrieved_snippets),
        "evidence_quotes": list(row.case.evidence_quotes),
        "model_answer": row.model_answer,
        "label": row.label,
        "score": row.score,
        "root_cause": row.root_cause,
        "reason": row.reason,
        "synthesis": row.synthesis,
        "used_fallback": row.used_fallback,
        "answer_candidates": [dict(item) for item in row.answer_candidates],
        "canonical_answer": row.canonical_answer,
    }


def write_results_json(
    *,
    results_path: Path,
    docx_path: Path,
    answer_model: str,
    question_model: str,
    judge_provider: str,
    judge_model: str,
    import_stats: Dict[str, object],
    summary: Dict[str, object],
    rows: Sequence[EvalRow],
) -> None:
    payload = {
        "docx": str(docx_path),
        "model": answer_model,
        "question_model": question_model,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "import_stats": import_stats,
        "summary": summary,
        "results": [result_payload(row) for row in rows],
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_report(
    *,
    docx_path: Path,
    char_count: int,
    paragraph_count: int,
    import_stats: Dict[str, object],
    summary: Dict[str, object],
    rows: Sequence[EvalRow],
    answer_model: str,
    question_model: str,
    judge_provider: str,
    judge_model: str,
    question_bank_json: Path,
    question_bank_md: Path,
    results_json: Path,
) -> str:
    lines = [
        f"# {docx_path.stem} Accuracy Benchmark Report",
        "",
        "## Corpus",
        f"- Source: `{docx_path}`",
        f"- Characters: `{char_count}`",
        f"- Paragraphs: `{paragraph_count}`",
        f"- Indexed chunks: `{import_stats.get('chunks', 0)}`",
        f"- Indexed terms: `{import_stats.get('terms', 0)}`",
        "",
        "## Setup",
        f"- Answer model: `{answer_model}`",
        f"- Question model: `{question_model}`",
        f"- Judge: `{judge_provider}:{judge_model}`",
        "",
        "## Overall",
        f"- Questions: `{summary['questions']}`",
        f"- Exact: `{summary['exact']}`",
        f"- Partial: `{summary['partial']}`",
        f"- Miss: `{summary['miss']}`",
        f"- Accuracy: `{summary['accuracy']:.2%}`",
        "",
        "## By Type",
    ]
    for case_type in ("cloze", "grounded_short_qa", "multi_hop_consistency"):
        data = summary["by_type"].get(case_type, {})
        lines.append(
            f"- `{case_type}`: questions `{data.get('questions', 0)}`, exact `{data.get('exact', 0)}`, partial `{data.get('partial', 0)}`, miss `{data.get('miss', 0)}`, accuracy `{float(data.get('accuracy', 0.0)):.2%}`"
        )

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
            lines.append(f"- Answer: {truncate(row.model_answer, 240)}")
            lines.append(f"- Root Cause: `{row.root_cause}`")
            lines.append(f"- Reason: {row.reason}")
            lines.append(f"- Supporting Chunks: {', '.join(str(cid) for cid in row.case.supporting_chunk_ids)}")
            lines.append(f"- Retrieved Chunks: {', '.join(str(cid) for cid in row.retrieved_chunk_ids)}")
            lines.append("")
    else:
        lines.append("- No misses.")

    lines.extend(
        [
            "## Artifacts",
            f"- Question bank JSON: `{question_bank_json}`",
            f"- Question bank Markdown: `{question_bank_md}`",
            f"- Results JSON: `{results_json}`",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    if args.questions != args.cloze_questions + args.short_qa_questions + args.multi_hop_questions:
        raise SystemExit("--questions must equal cloze + short-qa + multi-hop counts.")

    docx_path = Path(args.docx).expanduser().resolve()
    memory_dir = Path(args.memory_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not docx_path.exists():
        raise SystemExit(f"Document not found: {docx_path}")

    ensure_dir(output_dir)
    prefix = artifact_prefix(slugify(docx_path), int(args.questions))
    question_bank_json_path = output_dir / f"{prefix}-question-bank.json"
    question_bank_md_path = output_dir / f"{prefix}-question-bank.md"
    results_path = output_dir / f"{prefix}-results.json"
    report_path = output_dir / f"{prefix}-report.md"

    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    judge_api_key = require_judge_env(args.judge_provider) if (args.short_qa_questions > 0 or args.multi_hop_questions > 0) else ""
    judge_base_url = args.google_base_url if args.judge_provider == "google" else args.openai_base_url

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
        char_count, paragraph_count = corpus_stats(docx_path)
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

        cases = build_accuracy_cases(
            chunks=chunks,
            cloze_questions=int(args.cloze_questions),
            short_qa_questions=int(args.short_qa_questions),
            multi_hop_questions=int(args.multi_hop_questions),
            question_model=args.question_model,
            timeout=int(args.generation_timeout),
            judge_provider=args.judge_provider if int(args.multi_hop_questions) > 0 else None,
            judge_api_key=judge_api_key,
            judge_base_url=judge_base_url,
            judge_model=args.judge_model,
            judge_timeout=int(args.judge_timeout),
        )
        question_bank_json, question_bank_md = write_question_bank(output_dir, prefix, cases)

        results: List[EvalRow] = []
        print(f"Running {len(cases)} accuracy questions...", flush=True)
        for case in cases:
            print(f"[{case.case_id:02d}/{len(cases):02d}] [{case.case_type}] {truncate(case.question, 150)}", flush=True)
            row = evaluate_case(
                mm=mm,
                case=case,
                answer_model=args.model,
                answer_timeout=int(args.answer_timeout),
                retrieve_top_k=int(args.retrieve_top_k),
                judge_provider=args.judge_provider,
                judge_api_key=judge_api_key,
                judge_base_url=judge_base_url,
                judge_model=args.judge_model,
                judge_timeout=int(args.judge_timeout),
                chunk_map=chunk_map,
            )
            results.append(row)
            print(
                f"  -> {row.label} | gold={truncate(case.gold_answer, 80)} | got={truncate(row.model_answer, 120)} | cause={row.root_cause}",
                flush=True,
            )

        summary = summarize_results(results)
        write_results_json(
            results_path=results_path,
            docx_path=docx_path,
            answer_model=args.model,
            question_model=args.question_model,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            import_stats=import_stats,
            summary=summary,
            rows=results,
        )

        report = build_report(
            docx_path=docx_path,
            char_count=char_count,
            paragraph_count=paragraph_count,
            import_stats=import_stats,
            summary=summary,
            rows=results,
            answer_model=args.model,
            question_model=args.question_model,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            question_bank_json=question_bank_json,
            question_bank_md=question_bank_md,
            results_json=results_path,
        )
        report_path.write_text(report, encoding="utf-8")
        print(report)
        return 0
    finally:
        mm.close()
        if not args.keep_memory_dir:
            shutil.rmtree(memory_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
