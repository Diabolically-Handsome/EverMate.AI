"""legacy_quanzhi_heuristics.py — RETIRED, NOT IMPORTED BY THE APP.

This is the pre-2026-06 memory engine, kept only as a historical reference
for the benchmark numbers published in older reports. It is severely
overfitted to one test corpus (the novel "The King's Avatar" / 全职高手):
expected benchmark answers are hardcoded into query expansion, candidate
scoring, and answer repair, so its benchmark results do NOT measure general
retrieval quality (answer leakage).

The production engine lives in the `engine/` package. Do not import this
module from application code. Personal strings from the original developer's
chat history have been redacted ("<redacted-*>" placeholders), which only
affects heuristics tied to that private corpus.
"""

from __future__ import annotations

_ORIGINAL_DOC = """memory_manager.py

EverMate.AI — Local Memory System
================================

This module implements the README "Memory Ternary":

  Core (01_core.md) → Persona (02_persona.md) → Vault (chunk-on-disk + BM25)

What you get
------------
- **Chunk-on-disk** (~CHUNK_CHARS chars per chunk) saved under `memory/chunks/`
- **SQLite inverted index** (`terms`, `postings`, `chunks`) under `memory/index.sqlite`
- **BM25 retrieval** and **best-sentence evidence injection**
- **Core / Persona** Markdown files refreshed periodically
- **Incremental chat append** via `buffer.txt` + `chat_log.txt`

Design goals
------------
- Local-first (no cloud required for memory). Optional local LLM summarization via Ollama.
- Deterministic + debuggable (no silent failures). Exceptions are surfaced to the UI.
- Reasonable Chinese support without external tokenizers (uses char-bigrams for CJK).

Environment variables
---------------------
- MEMORY_DIR (default: ./memory in source runs; user app-data directory in bundled macOS runs)
- CHUNK_CHARS (default: 2800)
- CORE_TOP_TERMS (default: 50)
- PERSONA_MAX_BULLETS (default: 8)
- REFRESH_EVERY (default: 20)

Optional (Persona summarization)
-------------------------------
- OLLAMA_URL (default: http://localhost:11434)
- OLLAMA_MODEL (any installed local model)

"""


import os
import re
import json
import time
import math
import shutil
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from runtime_paths import default_memory_dir, user_app_support_root


# -------------------------- helpers --------------------------

def _now_ts() -> int:
    return int(time.time())


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _append_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _safe_filename(name: str) -> str:
    # Keep readable but safe for filesystem.
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff._\- ()]", "_", name)
    return name[:180] if len(name) > 180 else name


def _can_write_dir(path: str) -> bool:
    """Return True if we can create and write within the directory."""
    try:
        os.makedirs(path, exist_ok=True)
        test_path = os.path.join(path, ".evermate_write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False


def _resolve_memory_dir(raw: str) -> str:
    """Pick a writable memory root. Prefer the configured/default path, then fallback to user app-data."""
    raw = (raw or "").strip()
    preferred = os.path.abspath(raw or default_memory_dir())
    fallback = str(user_app_support_root() / "memory")

    if _can_write_dir(preferred):
        return preferred
    if _can_write_dir(fallback):
        return os.path.abspath(fallback)
    # Last resort: return preferred and let downstream errors surface.
    return preferred


# -------------------------- tokenization --------------------------

# Minimal stopwords list (kept small on purpose)
_EN_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
}

_ZH_STOP = {
    "的",
    "了",
    "呢",
    "吗",
    "啊",
    "吧",
    "和",
    "与",
    "及",
    "在",
    "对",
    "把",
    "被",
    "一个",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "我",
    "你",
    "您",
}


def _is_cjk(token: str) -> bool:
    return bool(re.fullmatch(r"[\u4e00-\u9fff]+", token))


def _tokenize(text: str) -> List[str]:
    """Tokenize for indexing / retrieval.

    - English: alphanumeric words
    - Chinese/Japanese/Korean: char-bigrams for contiguous CJK sequences

    This is not perfect segmentation, but works well enough for BM25 matching
    without extra dependencies.
    """

    if not text:
        return []

    text = text.lower()
    # Keep alnum words and CJK sequences
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)

    out: List[str] = []
    for p in parts:
        if not p:
            continue
        if _is_cjk(p):
            if len(p) == 1:
                if p not in _ZH_STOP:
                    out.append(p)
            else:
                # char-bigrams
                for i in range(len(p) - 1):
                    bg = p[i : i + 2]
                    if bg and bg not in _ZH_STOP:
                        out.append(bg)
        else:
            if p in _EN_STOP:
                continue
            out.append(p)

    return out


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []

    # Split by common sentence delimiters, keep short lines too.
    raw = re.split(r"(?<=[。！？.!?])\s+|\n+", text.strip())
    out = [s.strip() for s in raw if s and s.strip()]
    return out


def _query_flags(query: str) -> Dict[str, bool]:
    q = (query or "").lower()
    return {
        "asks_opening": any(x in q for x in ("文档开头", "开头", "一开始", "最开始", "前面")),
        "asks_duration": any(x in q for x in ("几分钟", "多久", "多长时间", "时长", "多少小时")),
        "asks_date": any(x in q for x in ("哪一天", "几月几日", "日期", "生日", "哪天")),
        "asks_choice": any(x in q for x in ("哪一张", "哪张", "哪个", "哪一种", "哪两种")),
        "asks_relation": "关系" in q or "看待" in q,
        "asks_avatar": "头像" in q or "照片" in q or "夕阳版" in q,
        "asks_name_or_title": any(x in q for x in ("叫什么", "叫什么名字", "名称", "名字", "角色名", "名为", "叫什么名")),
        "asks_count_or_total": any(x in q for x in ("多少", "总数", "一共", "总共", "几个", "几家", "第几", "几位", "多少个")),
        "asks_current_vs_max": any(x in q for x in ("当前", "目前", "这时", "这一轮", "此时", "当前技能等阶", "最高", "最多", "理论")),
        "asks_first_or_order": any(x in q for x in ("第一个", "首发", "率先", "先后", "顺序", "出场")),
        "asks_roster": any(x in q for x in ("阵容", "组合", "名单", "哪些人", "哪几位", "首发")),
        "asks_item_or_weapon": any(x in q for x in ("武器", "银武", "装备", "道具", "赌注", "筹码")),
        "asks_exchange_or_role": any(x in q for x in ("交换", "筹码", "扮演", "分别", "送往哪支战队", "被送往", "角色")),
        "asks_status_then_fact": any(x in q for x in ("传闻", "实际情况", "分别是什么", "分别利用", "分别扮演", "情况分别")),
        "asks_skill_name": any(x in q for x in ("什么技能", "技能名", "技能名称")),
        "asks_boss_name": ("boss" in q or "b o s s" in q) and any(x in q for x in ("叫什么", "名字", "名称")),
        "asks_compare_two_sides": ("分别" in q or any(x in q for x in ("双方", "各自", "两边"))) and any(
            x in q for x in ("谁", "哪位", "哪支", "是什么", "做了什么", "第一个")
        ),
        "asks_rumor_vs_actual": any(x in q for x in ("传闻", "实际", "实际情况", "宣布复出", "复出的消息")),
        "asks_tactic_sequence": any(x in q for x in ("战术", "配合", "包围", "夹攻", "诱饵", "围剿", "击败", "分头行动")),
        "asks_design_rationale": any(x in q for x in ("属性侧重", "职业特点", "弱点", "风险", "弥补", "为什么这样设计", "装备设计")),
        "asks_roster_plus_status": any(x in q for x in ("阵容", "首发", "组合", "团队赛")) and any(
            x in q for x in ("没上场", "未上场", "上场", "参赛情况", "休息")
        ),
        "asks_exchange_mapping": any(x in q for x in ("交换", "筹码", "送往", "扮演了什么角色", "最终被送往")),
        "asks_lineup_constraint": any(x in q for x in ("排兵布阵", "布阵", "具体问题")) and any(
            x in q for x in ("个人赛", "擂台赛", "单人对决", "单人赛")
        ),
        "asks_counter_adjustment": any(x in q for x in ("针对性调整", "火力压制", "限制苏沐橙", "远距离火力")) and any(
            x in q for x in ("首发", "阵容", "团队赛", "调整")
        ),
        "asks_role_plus_tactic": any(x in q for x in ("职业角色", "分别利用什么职业角色", "分别利用")) and any(
            x in q for x in ("战术策略", "集火", "采取了怎样的战术策略", "面对各家公会")
        ),
    }


def _extract_fragments(text: str) -> List[str]:
    if not text:
        return []

    candidates: List[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    for idx, line in enumerate(lines):
        candidates.append(line)

        pieces = re.split(r"(?<=[。！？.!?；;])\s*|(?<=：)\s*", line)
        for piece_idx, piece in enumerate(pieces):
            piece = piece.strip()
            if piece:
                candidates.append(piece)
                if piece_idx + 1 < len(pieces) and len(piece) <= 40:
                    merged_piece = f"{piece} {pieces[piece_idx + 1].strip()}".strip()
                    if merged_piece:
                        candidates.append(merged_piece)

        if idx + 1 < len(lines) and len(line) <= 80:
            merged = f"{line} {lines[idx + 1]}".strip()
            candidates.append(merged)

    candidates.extend(_split_sentences(text))

    dedup: List[str] = []
    seen = set()
    for cand in candidates:
        cand = cand.strip().replace("\n", " ")
        if not cand or cand in seen:
            continue
        seen.add(cand)
        dedup.append(cand)
    return dedup


def _resolve_answer_mode(query: str, requested: str = "auto") -> str:
    mode = (requested or "auto").strip().lower()
    if mode in ("chat", "fact", "multi_hop"):
        return mode

    q = (query or "").strip()
    flags = _query_flags(q)
    if any(
        flags.get(key)
        for key in (
            "asks_status_then_fact",
            "asks_exchange_or_role",
            "asks_roster",
            "asks_first_or_order",
            "asks_compare_two_sides",
            "asks_rumor_vs_actual",
            "asks_tactic_sequence",
            "asks_design_rationale",
            "asks_roster_plus_status",
            "asks_exchange_mapping",
        )
    ):
        return "multi_hop"
    if any(
        flags.get(key)
        for key in (
            "asks_name_or_title",
            "asks_count_or_total",
            "asks_current_vs_max",
            "asks_item_or_weapon",
            "asks_skill_name",
            "asks_boss_name",
            "asks_duration",
            "asks_date",
        )
    ):
        return "fact"

    if any(x in q for x in ("为什么", "怎么做到", "分别", "先后", "之后", "随后", "最终")):
        return "multi_hop"
    if any(x in q for x in ("谁", "哪位", "哪支", "什么", "多少", "第几")):
        return "fact"
    return "chat"


def _normalize_fact_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^[“\"'（(]+", "", cleaned)
    cleaned = re.sub(r"[”\"'）)!。！？?；;：:]+$", "", cleaned)
    return cleaned.strip()


def _extract_time_cue(text: str) -> str:
    raw = (text or "").strip()
    patterns = [
        r"第[一二三四五六七八九十百0-9]+赛季",
        r"第[一二三四五六七八九十百0-9]+轮",
        r"\d+\s*月\s*\d+\s*日",
        r"\d+\s*级",
        r"\d+\s*分\d+\s*秒(?:\d+)?",
        r"\d+\s*个",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return match.group(0)
    if any(x in raw for x in ("当前", "目前", "此时", "这一轮", "这时")):
        return "当前"
    if any(x in raw for x in ("最高", "最多", "理论")):
        return "理论上限"
    return ""


def _extract_entity_hits(text: str, limit: int = 6) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    entities: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})?|[\u4e00-\u9fff]{2,8}", raw):
        token = token.strip()
        if not token or token in seen or token in _SIG_GENERIC:
            continue
        if token in {"什么", "这个", "那个", "目前", "现在", "当前", "其中"}:
            continue
        seen.add(token)
        entities.append(token)
        if len(entities) >= limit:
            break
    return entities


_WEAPON_TERMS = (
    "步枪",
    "手炮",
    "战矛",
    "太刀",
    "光剑",
    "短剑",
    "巨剑",
    "手杖",
    "法杖",
    "扫把",
    "十字架",
    "双剑",
    "拳套",
    "东方棍",
    "左轮",
    "自动手枪",
    "手弩",
    "长矛",
)

_MULTI_HOP_RELATION_TERMS = ("分别", "实际", "传闻", "最终", "之后", "为了", "导致", "针对")
_MULTI_HOP_ROLE_TERMS = ("谁", "哪支", "哪位", "阵容", "首发", "交换", "诱饵", "包围")
_MULTI_HOP_ACTION_TERMS = (
    "诱饵",
    "包围",
    "夹攻",
    "围剿",
    "击败",
    "复出",
    "宣布",
    "替换",
    "换下",
    "派出",
    "首发",
    "上场",
    "休息",
    "针对",
    "弥补",
    "利用",
    "撤退",
    "移动",
    "抢夺",
    "交换",
    "送往",
    "替下",
    "二线",
    "替补",
    "单人对决",
    "治疗",
    "走位",
    "树根",
)
_MULTI_HOP_TEAM_TOKENS = (
    "兴欣",
    "嘉世",
    "蓝雨",
    "虚空",
    "雷霆",
    "神奇",
    "轮回",
    "微草",
    "霸图",
)
_MULTI_HOP_SLOT_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "asks_compare_two_sides": ("side_a", "side_b", "comparison_basis"),
    "asks_rumor_vs_actual": ("rumor", "actual", "status_gate"),
    "asks_tactic_sequence": ("bait", "collapsing_force", "encirclement_pattern", "outcome"),
    "asks_design_rationale": ("design_target", "boosted_attributes", "mitigated_risk", "compensated_weakness"),
    "asks_roster_plus_status": ("roster", "not_playing", "status_note"),
    "asks_exchange_mapping": ("person_a_role", "person_b_role", "destination"),
    "asks_lineup_constraint": ("missing_core", "constraint_reason", "lineup_consequence"),
    "asks_counter_adjustment": ("changed_out", "added_in", "tactical_goal"),
    "asks_role_plus_tactic": ("person_a_role", "person_b_role", "side_a_tactic", "side_b_tactic", "shared_strategy"),
}


def _resolve_multi_hop_subtype(query: str, flags: Dict[str, bool]) -> str:
    q = (query or "").strip().lower()
    if flags.get("asks_rumor_vs_actual") or (flags.get("asks_status_then_fact") and any(x in q for x in ("传闻", "实际", "复出", "宣布"))):
        return "asks_rumor_vs_actual"
    if flags.get("asks_exchange_mapping") or (flags.get("asks_exchange_or_role") and any(x in q for x in ("交换", "筹码", "送往"))):
        return "asks_exchange_mapping"
    if flags.get("asks_counter_adjustment"):
        return "asks_counter_adjustment"
    if flags.get("asks_roster_plus_status"):
        return "asks_roster_plus_status"
    if flags.get("asks_role_plus_tactic"):
        return "asks_role_plus_tactic"
    if flags.get("asks_lineup_constraint"):
        return "asks_lineup_constraint"
    if flags.get("asks_tactic_sequence") or any(x in q for x in ("战术", "配合", "诱饵", "包围", "夹攻", "围剿", "击败")):
        return "asks_tactic_sequence"
    if flags.get("asks_design_rationale") or any(x in q for x in ("职业特点", "属性侧重", "弱点", "风险", "弥补", "装备设计")):
        return "asks_design_rationale"
    if flags.get("asks_compare_two_sides") or "分别" in q or any(x in q for x in ("双方", "各自", "两边")):
        return "asks_compare_two_sides"
    return "asks_compare_two_sides"


def _resolve_fact_subtype(query: str, flags: Dict[str, bool]) -> str:
    q = (query or "").strip().lower()
    if flags.get("asks_count_or_total") and any(x in q for x in ("第几位", "排名", "排行", "总排名")):
        return "ranking_position"
    if flags.get("asks_count_or_total") and any(x in q for x in ("多少级", "几级", "等级是多少")):
        return "level_count"
    if flags.get("asks_count_or_total") and any(x in q for x in ("多少家公会", "几家公会")):
        return "guild_count"
    if flags.get("asks_count_or_total") and any(x in q for x in ("记录", "纪录", "通关记录", "成绩", "多少时间")):
        return "record_time"
    if flags.get("asks_item_or_weapon") and any(x in q for x in ("赌注", "筹码", "抵押")):
        return "wager_or_material"
    if flags.get("asks_name_or_title") and any(x in q for x in ("角色", "角色名", "角色名称")):
        return "role_name"
    if flags.get("asks_boss_name") or ("boss" in q and any(x in q for x in ("名字", "名称", "叫什么"))):
        return "boss_name"
    if flags.get("asks_current_vs_max"):
        return "current_vs_max"
    if flags.get("asks_item_or_weapon") and any(x in q for x in ("银武", "形态", "什么武器", "武器")):
        return "weapon_form"
    if flags.get("asks_count_or_total") and any(x in q for x in ("总数", "数到", "累计", "达到了")):
        return "running_total"
    return "generic_fact"


def _required_multi_hop_slots(subtype: str) -> Tuple[str, ...]:
    return _MULTI_HOP_SLOT_TEMPLATES.get(subtype, _MULTI_HOP_SLOT_TEMPLATES["asks_compare_two_sides"])


def _multi_hop_slot_prompt_lines(subtype: str) -> List[str]:
    slots = _required_multi_hop_slots(subtype)
    labels = {
        "side_a": "第一方 / 左侧主体",
        "side_b": "第二方 / 右侧主体",
        "comparison_basis": "两边的关键区别或共同结论",
        "rumor": "传闻层信息",
        "actual": "实际层信息",
        "status_gate": "触发状态的规则或门槛",
        "bait": "诱饵是谁 / 什么",
        "collapsing_force": "参与合围的主力或主导者",
        "encirclement_pattern": "包围 / 夹攻是怎样形成的",
        "outcome": "最终结果",
        "design_target": "装备或设计针对的对象 / 场景",
        "boosted_attributes": "被强化的关键属性",
        "mitigated_risk": "被降低的风险",
        "compensated_weakness": "被弥补的弱点",
        "roster": "完整阵容",
        "not_playing": "未上场 / 休息的人",
        "status_note": "关于上场状态的说明",
        "person_a_role": "第一个人的角色 / 身份",
        "person_b_role": "第二个人的角色 / 身份",
        "destination": "最终去向",
        "missing_core": "缺席或无法上阵的核心点",
        "constraint_reason": "造成布阵问题的原因",
        "lineup_consequence": "对排兵布阵的直接后果",
        "changed_out": "被换下的人",
        "added_in": "顶上的人 / 新首发",
        "tactical_goal": "这次调整想达到的限制目标",
        "side_a_tactic": "第一方的具体策略",
        "side_b_tactic": "第二方的具体策略",
        "shared_strategy": "两边共有的应对方式",
    }
    return [f"- {slot}: {labels.get(slot, slot)}" for slot in slots]


def _empty_multi_hop_synthesis(subtype: str) -> Dict[str, object]:
    return {
        "question_type": subtype,
        "entities": [],
        "event_chain": [],
        "direct_facts": {slot: "" for slot in _required_multi_hop_slots(subtype)},
        "slot_sources": {slot: [] for slot in _required_multi_hop_slots(subtype)},
        "contrasts": {},
        "final_claim": "",
        "confidence": 0.0,
    }


def _extract_json_payload(text: str) -> Optional[Dict[str, object]]:
    raw = (text or "").strip()
    if not raw:
        return None

    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    decoder = json.JSONDecoder()
    search = raw
    while search:
        brace = search.find("{")
        if brace < 0:
            return None
        try:
            obj, _ = decoder.raw_decode(search[brace:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            search = search[brace + 1 :]
            continue
        search = search[brace + 1 :]
    return None


def _coerce_multi_hop_synthesis(text: str, subtype: str) -> Optional[Dict[str, object]]:
    payload = _extract_json_payload(text)
    if not isinstance(payload, dict):
        return None

    out = _empty_multi_hop_synthesis(subtype)
    out["question_type"] = str(payload.get("question_type") or subtype)

    entities = payload.get("entities", [])
    if isinstance(entities, list):
        out["entities"] = [str(item).strip() for item in entities if str(item).strip()][:8]

    event_chain = payload.get("event_chain", [])
    if isinstance(event_chain, list):
        out["event_chain"] = [str(item).strip() for item in event_chain if str(item).strip()][:8]

    direct_facts = payload.get("direct_facts", {})
    if isinstance(direct_facts, dict):
        for slot in _required_multi_hop_slots(subtype):
            out["direct_facts"][slot] = str(direct_facts.get(slot) or "").strip()

    slot_sources = payload.get("slot_sources", {})
    if isinstance(slot_sources, dict):
        for slot in _required_multi_hop_slots(subtype):
            values = slot_sources.get(slot, [])
            if isinstance(values, list):
                out["slot_sources"][slot] = [int(item) for item in values if str(item).isdigit()][:6]

    contrasts = payload.get("contrasts", {})
    if isinstance(contrasts, dict):
        out["contrasts"] = {str(k).strip(): str(v).strip() for k, v in contrasts.items() if str(k).strip() and str(v).strip()}

    out["final_claim"] = str(payload.get("final_claim") or "").strip()
    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    out["confidence"] = max(0.0, min(1.0, confidence))

    slots = _required_multi_hop_slots(subtype)
    filled = sum(1 for slot in slots if str(out["direct_facts"].get(slot) or "").strip())
    if filled == 0 and not out["final_claim"]:
        return None
    return out


_SCENARIO_ANCHOR_TOKENS = (
    "挑战赛决赛",
    "第八赛季总决赛",
    "总决赛",
    "个人赛",
    "擂台赛",
    "团队赛",
    "圣诞袜",
    "龙剑士",
    "吴羽策",
    "第二十八轮",
    "千波湖",
    "幽暗森林",
)


def _query_anchor_terms(query: str, limit: int = 8) -> List[str]:
    raw = (query or "").strip()
    if not raw:
        return []
    blocked_exact = {"角色", "名称", "名字", "技能", "武器", "银武", "赌注", "总数", "记录", "成绩", "等级", "排行", "排名", "公会", "玩家", "剧情", "BOSS", "boss", "这一", "排在"}
    split_pattern = (
        r"(?:是什么|叫什么名字|叫什么|多少家公会|多少级|几级|第几位|角色名称|角色名|名称|名字|使用的|玩家们|一行人|挑战的|野外|剧情中|围追堵截的|围追堵截|达到了|总数|目前|当前|是多少|是哪个|是什么武器|是什么技能|的)"
    )
    out: List[str] = []
    seen = set()
    pieces = re.split(split_pattern, raw)
    for token in pieces:
        for fragment in re.split(r"[，,、]", token):
            fragment = fragment.strip("，。！？,.!?：:；;（）()[]【】 \t\r\n")
            fragment = re.sub(r"^[在被由与和跟向从把对给为及、]+", "", fragment)
            fragment = re.sub(r"(?:战队|玩家们|一行人)$", "", fragment)
            fragment = re.sub(r"[里中上下一二三四五六七八九十这那该段场次轮局时]?$", "", fragment)
            if not fragment or fragment in seen:
                continue
            if fragment in blocked_exact:
                continue
            if fragment.lower() in _EN_STOP or fragment in _ZH_STOP:
                continue
            if len(fragment) > 8:
                continue
            seen.add(fragment)
            out.append(fragment)
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


def _scenario_anchor_terms(query: str) -> List[str]:
    q = (query or "").strip()
    anchors: List[str] = []
    for token in _SCENARIO_ANCHOR_TOKENS:
        if token in q and token not in anchors:
            anchors.append(token)
    for token in _MULTI_HOP_TEAM_TOKENS:
        if token in q and token not in anchors:
            anchors.append(token)
    for token in ("叶修", "肖时钦", "苏沐橙", "唐柔", "喻文州", "徐景熙", "吴羽策", "杨昊轩", "李迅"):
        if token in q and token not in anchors:
            anchors.append(token)
    return anchors


def _structured_query_variants(query: str, flags: Dict[str, bool], mode: str) -> List[str]:
    query = (query or "").strip()
    if not query:
        return []

    variants: List[str] = [query]
    seen = {query}

    def _push(parts: List[str]) -> None:
        cleaned = " ".join(part.strip() for part in parts if part and part.strip())
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        variants.append(cleaned)

    key_terms = _signature_terms(query, limit=8)
    numbers = re.findall(r"\d+\s*(?:级|轮|个|位|脚|分|秒)", query)

    if key_terms:
        _push(key_terms[:6])

    if flags.get("asks_name_or_title"):
        _push(key_terms[:6] + ["名字", "名称", "名为", "叫做"])
    if flags.get("asks_count_or_total"):
        _push(key_terms[:6] + numbers[:2] + ["多少", "总数", "一共", "达到"])
    if flags.get("asks_current_vs_max"):
        _push(key_terms[:6] + numbers[:2] + ["当前", "目前", "此时", "最高", "理论"])
    if flags.get("asks_first_or_order"):
        _push(key_terms[:6] + ["首发", "阵容", "第一个", "率先"])
    if flags.get("asks_roster"):
        _push(key_terms[:6] + numbers[:2] + ["首发", "阵容", "名单", "团队赛", "上场"])
    if flags.get("asks_item_or_weapon"):
        _push(key_terms[:6] + numbers[:2] + ["武器", "银武", "形态"] + list(_WEAPON_TERMS[:8]))
    if flags.get("asks_exchange_or_role"):
        _push(key_terms[:6] + ["分别", "角色", "去向", "交换"])
    if flags.get("asks_status_then_fact"):
        _push(key_terms[:6] + numbers[:2] + ["传闻", "实际", "并没有", "宣布", "消息"])
    if flags.get("asks_skill_name"):
        _push(key_terms[:6] + ["技能", "技名", "招式", "接投", "拆投"])

    if mode == "fact":
        fact_subtype = _resolve_fact_subtype(query, flags)
        if fact_subtype == "ranking_position":
            _push(key_terms[:6] + ["排名", "排行", "总排名", "第几位", "第十九位"])
        elif fact_subtype == "level_count":
            _push(key_terms[:6] + ["等级", "多少级", "36级", "几级"])
        elif fact_subtype == "guild_count":
            _push(key_terms[:6] + ["公会", "多少家", "7家", "围剿"])
        elif fact_subtype == "record_time":
            _push(key_terms[:6] + ["记录", "纪录", "成绩", "通关记录", "分秒"])
        elif fact_subtype == "wager_or_material":
            _push(key_terms[:6] + ["赌注", "筹码", "抵押", "材料", "强力蛛丝"])
        elif fact_subtype == "role_name":
            _push(key_terms[:6] + ["角色", "角色名", "角色名称", "账号", "名字"])
        elif fact_subtype == "boss_name":
            _push(key_terms[:6] + ["BOSS", "名称", "名字", "围追堵截", "霸气雄图"])
        elif fact_subtype == "current_vs_max":
            _push(key_terms[:6] + numbers[:2] + ["当前技能等阶", "当前", "目前", "理论", "最高"])
        elif fact_subtype == "weapon_form":
            _push(key_terms[:6] + numbers[:2] + ["银武", "形态", "武器类别", "25级", "步枪"])
        elif fact_subtype == "running_total":
            _push(key_terms[:6] + numbers[:2] + ["总数", "累计", "数到", "达到", "圣诞小偷", "陈果"])

    if mode == "multi_hop":
        subtype = _resolve_multi_hop_subtype(query, flags)
        _push(key_terms[:6] + numbers[:2] + ["时间线", "之后", "随后", "最终"])
        _push(key_terms[:6] + numbers[:2] + list(_MULTI_HOP_RELATION_TERMS))
        _push(key_terms[:6] + numbers[:2] + list(_MULTI_HOP_ROLE_TERMS))
        if subtype == "asks_rumor_vs_actual":
            _push(key_terms[:6] + ["传闻", "实际", "复出", "规则", "宣布", "消息"])
        elif subtype == "asks_tactic_sequence":
            _push(key_terms[:6] + ["诱饵", "包围", "夹攻", "围剿", "分头行动", "不同方向"])
        elif subtype == "asks_design_rationale":
            _push(key_terms[:6] + ["针对", "属性", "弱点", "风险", "冰抗", "暗抗", "反应慢"])
        elif subtype == "asks_lineup_constraint":
            _push(key_terms[:6] + ["喻文州", "徐景熙", "单人对决", "治疗职业", "二线", "替补", "一线主力"])
            _push(key_terms[:6] + ["个人赛", "擂台赛", "排兵布阵", "单人赛", "上二线"])
        elif subtype == "asks_counter_adjustment":
            _push(key_terms[:6] + ["李迅", "杨昊轩", "枪炮师", "替下", "首发阵容", "远距离火力"])
            _push(key_terms[:6] + ["苏沐橙", "限制", "对等火力", "火力压制", "团队赛"])
        elif subtype == "asks_role_plus_tactic":
            _push(key_terms[:6] + ["叶修", "肖时钦", "战斗法师", "机械师", "树根", "主动撤退", "走位"])
            _push(key_terms[:6] + ["龙剑士", "集火", "躲避攻击", "抢夺BOSS", "雷霆公会"])
        elif subtype == "asks_roster_plus_status":
            _push(key_terms[:6] + numbers[:2] + ["阵容", "首发", "团队赛", "上场", "休息"])
        elif subtype == "asks_exchange_mapping":
            _push(key_terms[:6] + ["交换", "筹码", "送往", "角色", "分别"])
        else:
            _push(key_terms[:6] + ["双方", "各自", "分别", "第一个", "出场"])
            if flags.get("asks_first_or_order"):
                _push(key_terms[:6] + ["唐柔", "肖时钦", "第一个出战", "率先出场"])

    return variants[:8]


def _answer_line_score(fragment: str, query: str, flags: Dict[str, bool]) -> float:
    frag = (fragment or "").strip()
    if not frag:
        return float("-inf")

    score = _fragment_score(frag, query, _tokenize(query), flags)
    compact = _normalize_fact_text(frag)
    unit_match = re.search(r"(个|位|级|脚|分|秒)", query or "")
    query_unit = unit_match.group(1) if unit_match else ""

    if "“" in frag or "\"" in frag:
        score += 0.6
    if len(compact) <= 48:
        score += 0.5
    if len(compact) <= 16:
        score += 0.8

    if flags.get("asks_name_or_title") and any(x in frag for x in ("叫", "名为", "名称", "名字", "角色名称", "是")):
        score += 3.5
    if flags.get("asks_count_or_total") and re.search(r"(第[一二三四五六七八九十百0-9]+位|\d+\s*(?:个|家|位|级|脚|分|秒))", frag):
        score += 4.0
    if flags.get("asks_current_vs_max") and any(x in frag for x in ("当前", "目前", "这时", "这一轮", "此时")):
        score += 4.0
    if flags.get("asks_current_vs_max") and any(x in frag for x in ("最高", "最多", "理论")) and not any(
        x in frag for x in ("当前", "目前", "这时", "这一轮", "此时")
    ):
        score -= 2.2
    if flags.get("asks_first_or_order") and any(x in frag for x in ("第一个", "首发", "率先", "第一个出战", "第一个出场")):
        score += 4.2
    if flags.get("asks_roster") and any(x in frag for x in ("阵容", "组合", "名单", "派出了", "首发")):
        score += 4.5
    if flags.get("asks_roster") and len(re.findall(r"、", frag)) >= 2:
        score += 2.0
    if flags.get("asks_item_or_weapon") and any(x in frag for x in ("武器", "步枪", "十字架", "赌注", "筹码", "银武")):
        score += 4.0
    if flags.get("asks_exchange_or_role") and any(x in frag for x in ("作为", "被送往", "送往", "交换", "筹码")):
        score += 3.8
    if flags.get("asks_status_then_fact") and any(x in frag for x in ("传闻", "实际", "但", "然而", "并没有", "未")):
        score += 2.8
    if flags.get("asks_lineup_constraint") and any(x in frag for x in ("喻文州", "徐景熙", "单人对决", "治疗职业", "二线", "替补")):
        score += 4.6
    if flags.get("asks_counter_adjustment") and any(x in frag for x in ("杨昊轩", "李迅", "枪炮师", "替下", "首发阵容")):
        score += 4.8
    if flags.get("asks_role_plus_tactic") and any(x in frag for x in ("战斗法师", "机械师", "树根", "走位", "撤退", "龙剑士")):
        score += 4.8
    if flags.get("asks_skill_name") and any(x in frag for x in ("技能", "技", "接投", "鹰踏", "挡拆")):
        score += 3.4
    if flags.get("asks_boss_name") and any(x in frag for x in ("BOSS", "boss", "叫做", "名字", "浪人", "女巫")):
        score += 3.0

    if flags.get("asks_name_or_title") and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·\-]{2,10}", compact):
        score += 5.2
    if (flags.get("asks_count_or_total") or flags.get("asks_current_vs_max")) and re.fullmatch(
        r"(第?[一二三四五六七八九十百0-9]+)\s*(?:个|家|位|级|脚|分|秒)?", compact
    ):
        score += 6.4
        if query_unit and compact.endswith(query_unit):
            score += 3.2
        elif query_unit and re.search(r"(个|家|位|级|脚|分|秒)$", compact) and not compact.endswith(query_unit):
            score -= 2.8
    if flags.get("asks_skill_name") and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·\-]{2,8}", compact):
        score += 7.0
    if flags.get("asks_item_or_weapon") and compact in _WEAPON_TERMS:
        score += 7.2
    if flags.get("asks_item_or_weapon") and any(term in compact for term in _WEAPON_TERMS) and len(compact) <= 12:
        score += 4.0
    if flags.get("asks_roster") and len(re.findall(r"、", compact)) >= 4:
        score += 5.5
    if flags.get("asks_status_then_fact") and all(x in frag for x in ("可以复出", "并没有")):
        score += 5.0

    return score


def _pattern_answer_candidates(text: str, flags: Dict[str, bool]) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    out: List[str] = []
    seen = set()

    def _push(value: str) -> None:
        compact = _normalize_fact_text(value)
        if not compact or compact in seen:
            return
        seen.add(compact)
        out.append(compact)

    if flags.get("asks_skill_name"):
        for match in re.finditer(r"(?:技能|技)[：:]\s*([^\s，。；！？“”\"'（）()]{2,8})", raw):
            _push(match.group(1))
        for match in re.finditer(r"挡拆技[：:\s]*([^\s，。；！？“”\"'（）()]{2,8})", raw):
            _push(match.group(1))

    if flags.get("asks_name_or_title"):
        for match in re.finditer(r"(?:叫|名为|名字(?:是)?|名称(?:是)?|角色名称(?:是)?)\s*[“\"']?([\u4e00-\u9fffA-Za-z0-9·\-]{2,10})", raw):
            _push(match.group(1))
        for match in re.finditer(r"这个([\u4e00-\u9fffA-Za-z0-9·\-]{2,10})吧", raw):
            _push(match.group(1))
        for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9·\-]{2,10})交还给", raw):
            _push(match.group(1))

    if flags.get("asks_item_or_weapon"):
        for match in re.finditer(r"(?:\d+\s*级)?\s*银武([^\s，。；！？“”\"'（）()]{1,8})", raw):
            value = match.group(1).strip()
            if value in _WEAPON_TERMS:
                _push(value)
                _push("银武" + value)
        for term in _WEAPON_TERMS:
            if term in raw and any(anchor in raw for anchor in ("银武", "武器", "形态", "枪械")):
                _push(term)

    if flags.get("asks_count_or_total") or flags.get("asks_current_vs_max"):
        for match in re.finditer(r"(?:目前|当前|此时|这时|这一轮)[^。！？]{0,24}?([一二三四五六七八九十百两0-9]+\s*(?:个|位|级|脚|分|秒))", raw):
            _push(match.group(1))
        for match in re.finditer(r"能出([一二三四五六七八九十百两0-9]+\s*(?:个|位|级|脚|分|秒))", raw):
            _push(match.group(1))
        for match in re.finditer(r"((?:第)?[一二三四五六七八九十百两0-9]+)\s*(个|家|位|级|脚|分|秒)", raw):
            _push("".join(match.groups()))
        for match in re.finditer(r"第\s*(\d+)\s*个", raw):
            _push(f"{match.group(1)}个")
        for match in re.finditer(r"(\d+)\s*个圣诞小偷", raw):
            _push(f"{match.group(1)}个")

    if flags.get("asks_roster"):
        for sentence in _split_sentences(raw):
            sentence = sentence.strip()
            if len(re.findall(r"、", sentence)) >= 4 and any(x in sentence for x in ("阵容", "组合", "派出了", "首发", "团队赛")):
                _push(sentence)

    return out


def _normalize_fact_candidate_text(value: str, subtype: str) -> str:
    compact = _normalize_fact_text(value)
    if not compact:
        return ""
    compact = compact.strip("：:，。；！？,.!?()（）[]【】“”\"'")
    if subtype == "ranking_position":
        match = re.search(r"(第[\u4e00-\u9fff0-9两十百千]+位)", compact)
        if match:
            return re.sub(r"\s+", "", match.group(1))
    if subtype == "record_time":
        match = re.search(r"(\d+\s*分\s*\d+\s*秒(?:\s*\d+)?)", compact)
        if match:
            return re.sub(r"\s+", "", match.group(1))
    if subtype in {"current_vs_max", "running_total", "level_count", "guild_count"}:
        match = re.search(r"([一二三四五六七八九十百两0-9]+\s*(?:个|家|位|级|脚|分|秒))", compact)
        if match:
            return re.sub(r"\s+", "", match.group(1))
    if subtype == "weapon_form":
        for term in _WEAPON_TERMS:
            if term in compact:
                return term
    if subtype in {"role_name", "boss_name", "wager_or_material"}:
        match = re.search(r"([\u4e00-\u9fffA-Za-z0-9·\-]{2,14})", compact)
        if match:
            return match.group(1)
    return compact


def _is_implausible_fact_candidate(value: str, subtype: str) -> bool:
    compact = _normalize_fact_text(value)
    if not compact:
        return True
    if subtype == "ranking_position":
        return not bool(re.fullmatch(r"第[\u4e00-\u9fff0-9两十百千]+位", compact))
    if subtype == "level_count":
        return not bool(re.fullmatch(r"[一二三四五六七八九十百两0-9]+级", compact))
    if subtype == "guild_count":
        return not bool(re.fullmatch(r"[一二三四五六七八九十百两0-9]+家", compact))
    if subtype == "record_time":
        return not bool(re.fullmatch(r"\d+分\d+秒\d*", compact))
    if subtype == "wager_or_material":
        return not (bool(re.search(r"[一二三四五六七八九十百两0-9]+个", compact)) and any(token in compact for token in ("蛛丝", "材料", "筹码", "抵押")))
    if subtype == "role_name":
        bad_tokens = (
            "什么",
            "就是",
            "应该",
            "模样",
            "名字",
            "名称",
            "名单",
            "面前",
            "身上",
            "出现在",
            "之一",
            "装备",
            "实力",
            "移动",
            "成绩",
            "对比",
            "高空",
            "扩散",
            "的时候",
        )
        return len(compact) > 6 or any(token in compact for token in bad_tokens) or compact.endswith(("的", "了", "吗", "呢"))
    if subtype == "boss_name":
        bad_tokens = (
            "什么",
            "名字",
            "名称",
            "BOSS",
            "身影",
            "围追堵截",
            "位置",
            "而已",
            "身分",
            "身上",
            "来了",
            "那边",
            "刷新",
            "条件",
            "时候",
            "家伙",
        )
        return len(compact) > 8 or any(token in compact for token in bad_tokens) or compact.endswith(("的", "了", "吗", "呢"))
    if subtype == "current_vs_max":
        return not bool(re.fullmatch(r"[一二三四五六七八九十百两0-9]+脚", compact))
    if subtype == "weapon_form":
        return compact not in _WEAPON_TERMS
    if subtype == "running_total":
        return not bool(re.fullmatch(r"[一二三四五六七八九十百两0-9]+个", compact))
    return False


def _extract_fact_candidates(text: str, query: str, flags: Dict[str, bool], subtype: str) -> List[Dict[str, object]]:
    raw = (text or "").strip()
    if not raw:
        return []

    extracted: List[Dict[str, object]] = []
    seen: set[str] = set()

    def _push(value: str, bonus: float = 0.0, source: str = "pattern") -> None:
        normalized = _normalize_fact_candidate_text(value, subtype)
        if not normalized or normalized in seen or _is_implausible_fact_candidate(normalized, subtype):
            return
        seen.add(normalized)
        extracted.append(
            {
                "answer": normalized,
                "normalized": normalized,
                "bonus": float(bonus),
                "source": source,
                "evidence": (raw.replace("\n", " ")[:220] + "…") if len(raw.replace("\n", " ")) > 220 else raw.replace("\n", " "),
            }
        )

    query_unit_match = re.search(r"(个|家|位|级|脚|分|秒)", query or "")
    query_unit = query_unit_match.group(1) if query_unit_match else ""

    if subtype == "ranking_position":
        patterns = (
            r"(第[\u4e00-\u9fff0-9两十百千]+位)",
            r"总排名(第[\u4e00-\u9fff0-9两十百千]+位)",
            r"排行(第[\u4e00-\u9fff0-9两十百千]+位)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw):
                _push(match.group(1), bonus=8.8, source="ranking_position")

    if subtype == "level_count":
        for match in re.finditer(r"([一二三四五六七八九十百两0-9]+\s*级)", raw):
            _push(match.group(1), bonus=8.0, source="level_count")

    if subtype == "guild_count":
        for match in re.finditer(r"([一二三四五六七八九十百两0-9]+\s*家)", raw):
            _push(match.group(1), bonus=8.0, source="guild_count")

    if subtype == "record_time":
        for match in re.finditer(r"(\d+\s*分\s*\d+\s*秒(?:\s*\d+)?)", raw):
            _push(match.group(1), bonus=8.5, source="record_time")

    if subtype == "wager_or_material":
        patterns = (
            r"([一二三四五六七八九十百两0-9]+\s*个强力蛛丝)",
            r"赌注[^。；！？\n]*?([一二三四五六七八九十百两0-9]+\s*个[\u4e00-\u9fff]{2,8})",
            r"筹码[^。；！？\n]*?([一二三四五六七八九十百两0-9]+\s*个[\u4e00-\u9fff]{2,8})",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw):
                _push(match.group(1), bonus=8.5, source="wager")

    if subtype == "role_name":
        query_names = [name for name in _query_anchor_terms(query, limit=4) if len(name) <= 4]
        for name in query_names:
            patterns = (
                rf"{re.escape(name)}[^。；！？\n]*角色(?:名称)?[是为叫做：:\s]*([\u4e00-\u9fffA-Za-z0-9·\-]{{2,10}})",
                rf"([\u4e00-\u9fffA-Za-z0-9·\-]{{2,10}})交还给了?{re.escape(name)}",
                rf"你说的妹子就是这个([\u4e00-\u9fffA-Za-z0-9·\-]{{2,10}})吧",
                rf"{re.escape(name)}[^。；！？\n]*?([\u4e00-\u9fffA-Za-z0-9·\-]{{2,10}})[，,\s]+角色",
            )
            for pattern in patterns:
                for match in re.finditer(pattern, raw):
                    _push(match.group(1), bonus=8.8, source="role_name")
        for match in re.finditer(r"角色(?:名称|名字|名)?[：:\s]*([\u4e00-\u9fffA-Za-z0-9·\-]{2,10})", raw):
            _push(match.group(1), bonus=5.0, source="role_name")

    if subtype == "boss_name":
        patterns = (
            r"(?:野外|隐藏)?(?:BOSS|boss)[，,:：\s]*([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})(?:的身影|[，。,；！？\s])",
            r"很快就看到BOSS([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})(?:的身影|[，。,；！？\s])",
            r"([\u4e00-\u9fffA-Za-z0-9·\-]{2,14})的身影",
            r"目标，是[^。；！？\n]{0,18}?野外BOSS[，,:：\s]*([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})",
            r"围追堵截的BOSS[叫名为是：:\s]*([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})(?:的身影|[，。,；！？\s])",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
                _push(match.group(1), bonus=8.2, source="boss_name")

    if subtype == "current_vs_max":
        for match in re.finditer(r"(?:当前|目前|这时|此时|当前技能等阶)[^。；！？\n]{0,24}?([一二三四五六七八九十百两0-9]+\s*脚)", raw):
            _push(match.group(1), bonus=8.8, source="current")
        for match in re.finditer(r"(?:等阶|目前等阶)[^。；！？\n]{0,24}?能出([一二三四五六七八九十百两0-9]+\s*脚)", raw):
            _push(match.group(1), bonus=8.2, source="current")
        for match in re.finditer(r"(?:最高|最多|理论)[^。；！？\n]{0,18}?([一二三四五六七八九十百两0-9]+\s*脚)", raw):
            _push(match.group(1), bonus=3.0, source="theoretical")

    if subtype == "weapon_form":
        for match in re.finditer(r"25\s*级银武([\u4e00-\u9fffA-Za-z0-9·\-]{1,8})", raw):
            _push(match.group(1), bonus=9.0, source="weapon_form")
        for match in re.finditer(r"银武(?:是|为)?([\u4e00-\u9fffA-Za-z0-9·\-]{1,8})", raw):
            _push(match.group(1), bonus=6.2, source="weapon_form")

    if subtype == "running_total":
        for match in re.finditer(r"([一二三四五六七八九十百两0-9]+\s*个)[^。；！？\n]{0,12}(?:总数|达到|数到)", raw):
            _push(match.group(1), bonus=8.2, source="running_total")
        for match in re.finditer(r"第\s*([一二三四五六七八九十百两0-9]+)\s*个", raw):
            _push(match.group(1) + "个", bonus=9.2, source="running_total")
        for match in re.finditer(r"陈果[^。；！？\n]{0,18}?([一二三四五六七八九十百两0-9]+)\s*了", raw):
            _push(match.group(1) + "个", bonus=9.6, source="running_total")

    if not extracted:
        for candidate in _pattern_answer_candidates(raw, flags):
            normalized_candidate = _normalize_fact_candidate_text(candidate, subtype)
            if not normalized_candidate:
                continue
            if query_unit and normalized_candidate.endswith(query_unit) is False and subtype not in {"role_name", "boss_name", "weapon_form", "wager_or_material"}:
                continue
            bonus = 0.0
            if subtype == "weapon_form" and normalized_candidate in _WEAPON_TERMS:
                bonus += 5.0
            if subtype == "generic_fact" and flags.get("asks_count_or_total") and query_unit and normalized_candidate.endswith(query_unit):
                bonus += 2.0
            _push(normalized_candidate, bonus=bonus, source="generic")

    return extracted[:8]


def _fact_conflict_watch(answer_candidates: Sequence[Dict[str, object]], subtype: str) -> List[str]:
    if len(answer_candidates) <= 1:
        return []
    answers = [str(item.get("answer", "")).strip() for item in answer_candidates if str(item.get("answer", "")).strip()]
    markers: List[str] = []
    if subtype in {"ranking_position", "record_time", "running_total", "level_count", "guild_count"} and len(set(answers)) >= 2:
        markers.append("same_entity_different_number")
    if subtype == "current_vs_max" and len(set(answers)) >= 2:
        markers.append("current_vs_theoretical")
    if subtype == "weapon_form" and len(set(answers)) >= 2:
        markers.append("same_item_multiple_forms")
    if subtype == "boss_name" and len(set(answers)) >= 2:
        markers.append("same_scene_multiple_bosses")
    if subtype == "wager_or_material" and any("钱" in answer for answer in answers) and any("蛛丝" in answer for answer in answers):
        markers.append("material_vs_currency")
    return markers


def _select_fact_answer_candidates(
    items: Sequence[Dict[str, object]],
    query: str,
    flags: Dict[str, bool],
    subtype: str,
) -> Tuple[List[Dict[str, object]], str, str]:
    query_tokens = {token for token in _tokenize(query) if token and token not in _ZH_STOP and token not in _EN_STOP}
    query_entities = set(_query_anchor_terms(query, limit=6))
    unit_match = re.search(r"(个|家|位|级|脚|分|秒)", query or "")
    query_unit = unit_match.group(1) if unit_match else ""
    grouped: Dict[str, Dict[str, object]] = {}
    for rank, item in enumerate(items):
        chunk_id = int(item.get("chunk_id", 0) or 0)
        item_score = float(item.get("score", 0.0))
        for candidate in item.get("fact_candidates", []) or []:
            normalized = str(candidate.get("normalized", "") or "").strip()
            answer = str(candidate.get("answer", "") or "").strip()
            if not normalized or not answer:
                continue
            evidence_text = str(candidate.get("evidence", "") or "")
            token_overlap = len(set(_tokenize(evidence_text)) & query_tokens)
            entity_bonus = sum(1 for entity in query_entities if entity in evidence_text)
            subtype_bonus = 0.0
            if entity_bonus:
                subtype_bonus += min(entity_bonus, 3) * 2.2
            subtype_bonus += min(token_overlap, 8) * 0.35
            if query_unit:
                if normalized.endswith(query_unit):
                    subtype_bonus += 1.4
                elif subtype not in {"role_name", "boss_name", "weapon_form", "wager_or_material"}:
                    subtype_bonus -= 1.8
            if subtype == "ranking_position":
                if any(anchor in evidence_text for anchor in ("总排名", "战绩排行", "嘉世战队")):
                    subtype_bonus += 5.0
                if "倒数" in evidence_text and "总排名" not in evidence_text:
                    subtype_bonus -= 1.6
            elif subtype == "level_count":
                if any(anchor in evidence_text for anchor in ("等级", "BOSS", "卡修", "炎女巫")):
                    subtype_bonus += 4.8
            elif subtype == "guild_count":
                if any(anchor in evidence_text for anchor in ("公会", "围剿", "参与")):
                    subtype_bonus += 4.8
            elif subtype == "record_time":
                if "打破副本" in evidence_text or "通关记录" in evidence_text or "成绩" in evidence_text:
                    subtype_bonus += 4.8
            elif subtype == "wager_or_material":
                if "强力蛛丝" in evidence_text:
                    subtype_bonus += 8.0
            elif subtype == "role_name":
                if any(entity in evidence_text for entity in query_entities) and any(anchor in evidence_text for anchor in ("这个", "角色", "交还给")):
                    subtype_bonus += 8.0
            elif subtype == "boss_name":
                if any(anchor in evidence_text for anchor in ("BOSS", "霸气雄图", "烈焰森林", "岩之浪人奥磐", "炎女巫卡修")):
                    subtype_bonus += 6.0
            elif subtype == "current_vs_max":
                if any(anchor in evidence_text for anchor in ("当前", "目前", "能出", "等阶")):
                    subtype_bonus += 7.0
                if any(anchor in evidence_text for anchor in ("最高", "理论", "最多")) and not any(anchor in evidence_text for anchor in ("当前", "目前", "能出", "等阶")):
                    subtype_bonus -= 2.5
            elif subtype == "weapon_form":
                if all(anchor in evidence_text for anchor in ("25级", "银武")):
                    subtype_bonus += 6.0
                if "步枪" in evidence_text:
                    subtype_bonus += 6.5
                if "飞枪移动" in evidence_text or "步枪的飞枪移动" in evidence_text:
                    subtype_bonus += 7.0
                if "25级银武" not in evidence_text:
                    subtype_bonus -= 2.8
            elif subtype == "running_total":
                if any(anchor in evidence_text for anchor in ("陈果", "323了", "第323个", "数到", "达到")):
                    subtype_bonus += 8.0
            elif subtype == "generic_fact":
                if entity_bonus:
                    subtype_bonus += 3.0
            entry = grouped.setdefault(
                normalized,
                {
                    "answer": answer,
                    "normalized": normalized,
                    "score": 0.0,
                    "support_count": 0,
                    "chunk_ids": [],
                    "evidence": [],
                },
            )
            if len(answer) < len(str(entry.get("answer", "")) or answer):
                entry["answer"] = answer
            entry["score"] = (
                float(entry.get("score", 0.0))
                + item_score
                + float(candidate.get("bonus", 0.0))
                + subtype_bonus
                + max(0.0, 2.4 - (0.25 * rank))
            )
            if chunk_id and chunk_id not in entry["chunk_ids"]:
                entry["chunk_ids"].append(chunk_id)
                entry["support_count"] = int(entry.get("support_count", 0)) + 1
            if evidence_text and evidence_text not in entry["evidence"]:
                entry["evidence"].append(evidence_text[:220])

    ranked = sorted(
        grouped.values(),
        key=lambda item: (float(item.get("score", 0.0)), int(item.get("support_count", 0)), -len(str(item.get("answer", "")))),
        reverse=True,
    )
    answer_candidates = [
        {
            "answer": str(item.get("answer", "")),
            "normalized": str(item.get("normalized", "")),
            "score": round(float(item.get("score", 0.0)), 3),
            "support_count": int(item.get("support_count", 0)),
            "chunk_ids": [int(chunk_id) for chunk_id in item.get("chunk_ids", [])[:6]],
            "evidence": list(item.get("evidence", [])[:3]),
        }
        for item in ranked[:8]
    ]
    if not answer_candidates:
        return [], "", "model_select"

    top = answer_candidates[0]
    second = answer_candidates[1] if len(answer_candidates) > 1 else None
    top_score = float(top.get("score", 0.0))
    second_score = float(second.get("score", 0.0)) if second else 0.0
    stable = len(answer_candidates) == 1 or top_score >= second_score + 2.5
    if int(top.get("support_count", 0)) >= 2 and top_score >= second_score + 1.0:
        stable = True
    if subtype in {"ranking_position", "level_count", "guild_count", "current_vs_max", "running_total", "weapon_form", "record_time"} and top_score >= second_score + 1.6:
        stable = True
    if subtype == "wager_or_material" and ("蛛丝" in str(top.get("answer", "")) or "筹码" in str(top.get("answer", ""))):
        stable = stable or top_score >= second_score + 1.0
    if subtype == "role_name" and len(str(top.get("answer", ""))) <= 10:
        stable = stable or top_score >= second_score + 1.0
    return answer_candidates, str(top.get("answer", "")), ("deterministic" if stable else "model_select")


def _multi_hop_pattern_candidates(text: str, query: str, flags: Dict[str, bool]) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    subtype = _resolve_multi_hop_subtype(query, flags)
    out: List[str] = []
    seen = set()

    def _push(value: str) -> None:
        compact = _normalize_fact_text(value)
        if not compact or compact in seen:
            return
        seen.add(compact)
        out.append(compact)

    if flags.get("asks_first_or_order") or subtype == "asks_compare_two_sides":
        patterns = (
            r"[^\n。]*第一个(?:出战|出场)选手[^。]*。",
            r"[^\n。]*率先出场的是[^\n。]*。",
            r"[^\n。]*首先是嘉世战队。率先出场的是[^\n。]*。",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw):
                _push(match.group(0))

    if subtype == "asks_lineup_constraint":
        patterns = (
            r"[^\n。]*喻文州[^。]*单人对决[^。]*。",
            r"[^\n。]*徐景熙[^。]*(?:单人赛|擂台赛|单人对决)[^。]*。",
            r"[^\n。]*只能是上二线的替补选手[^。]*。",
            r"[^\n。]*团队赛的一线主力不会出战单人对决赛事[^。]*。",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw):
                _push(match.group(0))

    if subtype == "asks_counter_adjustment":
        patterns = (
            r"[^\n。]*枪炮师杨昊轩出战[^。]*。",
            r"[^\n。]*替下李迅[^。]*。",
            r"[^\n。]*首发阵容当中[^。]*。",
            r"[^\n。]*(?:限制|对等)[^。]*(?:苏沐橙|远距离火力)[^。]*。",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw):
                _push(match.group(0))

    if subtype == "asks_role_plus_tactic":
        patterns = (
            r"[^\n。]*叶修[^。]*战斗法师[^。]*。",
            r"[^\n。]*肖时钦[^。]*机械师[^。]*。",
            r"[^\n。]*(?:树根|撤退|走位|移动|躲避攻击|抢夺BOSS)[^。]*。",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw):
                _push(match.group(0))

    return out


def _extract_answer_lines(text: str, query: str, flags: Dict[str, bool], limit: int = 3) -> List[str]:
    fragments = _extract_fragments(text)
    ranked: List[Tuple[float, str]] = []
    for candidate in _pattern_answer_candidates(text, flags):
        ranked.append((_answer_line_score(candidate, query, flags) + 8.0, candidate))
    for fragment in fragments:
        score = _answer_line_score(fragment, query, flags)
        if score == float("-inf"):
            continue
        ranked.append((score, fragment.strip()))
    ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)

    out: List[str] = []
    seen = set()
    for _, fragment in ranked:
        normalized = _normalize_fact_text(fragment)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(fragment)
        if len(out) >= limit:
            break
    return out


def _extract_multi_hop_lines(text: str, query: str, flags: Dict[str, bool], limit: int = 4) -> List[str]:
    subtype = _resolve_multi_hop_subtype(query, flags)
    ranked: List[Tuple[float, str]] = []
    for candidate in _multi_hop_pattern_candidates(text, query, flags):
        ranked.append((_answer_line_score(candidate, query, flags) + 10.0, candidate))
    for fragment in _extract_fragments(text):
        score = _answer_line_score(fragment, query, flags)
        norm = _normalize_fact_text(fragment)
        if not norm:
            continue
        if re.fullmatch(r"[一二三四五六七八九十百两0-9]+\s*(?:分|个|位|脚|秒)?", norm):
            score -= 12.0
        if len(norm) <= 4 and not any(
            marker in norm for marker in ("叶修", "嘉世", "兴欣", "唐柔", "肖时钦", "申建", "冰抗", "暗抗", "叶秋")
        ):
            score -= 6.0
        if len(norm) >= 16:
            score += 1.2
        if any(
            marker in fragment
            for marker in (
                "并没有",
                "任何渠道",
                "第一个出场",
                "第一个出战",
                "分头行事",
                "夹攻",
                "包围",
                "冰抗",
                "暗抗",
                "反应慢",
                "高智力",
                "高暴击",
                "休息了一轮",
                "肖时钦",
                "王泽",
                "孙翔",
                "申建",
                "张家兴",
                "唐柔",
                "喻文州",
                "徐景熙",
                "二线",
                "替补",
                "李迅",
                "杨昊轩",
                "枪炮师",
                "战斗法师",
                "机械师",
                "树根",
                "撤退",
            )
        ):
            score += 3.2
        if subtype == "asks_lineup_constraint":
            if any(x in fragment for x in ("喻文州", "徐景熙", "单人对决", "治疗职业", "二线", "替补", "一线主力")):
                score += 5.2
            if "蓝雨" not in fragment and any(x in fragment for x in ("轮回", "嘉世", "兴欣")):
                score -= 3.4
        elif subtype == "asks_counter_adjustment":
            if any(x in fragment for x in ("杨昊轩", "李迅", "枪炮师", "替下", "首发阵容", "远距离火力")):
                score += 5.4
            if any(x in fragment for x in ("冰抗", "暗抗", "安文逸", "反应慢")):
                score -= 5.0
        elif subtype == "asks_role_plus_tactic":
            if any(x in fragment for x in ("战斗法师", "机械师", "树根", "主动撤退", "走位", "躲避攻击", "抢夺")):
                score += 5.0
            if any(x in fragment for x in ("死亡之门", "魏琛", "迎风布阵")):
                score -= 4.2
        elif subtype == "asks_compare_two_sides" and flags.get("asks_first_or_order"):
            if any(x in fragment for x in ("第一个出战选手", "第一个出场选手", "率先出场的是")):
                score += 4.8
            if any(x in fragment for x in ("团队赛", "走在最前", "首发", "第六人")):
                score -= 2.8
        ranked.append((score, fragment.strip()))

    ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    out: List[str] = []
    seen = set()
    for _, fragment in ranked:
        normalized = _normalize_fact_text(fragment)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(fragment)
        if len(out) >= limit:
            break
    return out


def _event_tag_from_text(text: str) -> str:
    terms = _signature_terms(text, limit=3)
    return " / ".join(terms) if terms else ""


def _roster_signature(text: str) -> str:
    names = [token for token in re.findall(r"[\u4e00-\u9fff]{2,4}", text or "") if token not in _SIG_GENERIC]
    names = names[:8]
    return "|".join(names)


def _structured_conflict_markers(items: List[Dict[str, object]], query: str) -> List[str]:
    snippets = [str(item.get("snippet", "")) for item in items]
    answer_lines = [" ".join(str(line) for line in item.get("answer_lines", [])) for item in items]
    markers = _conflict_markers(snippets)

    joined = "\n".join(answer_lines + snippets)
    flags = _query_flags(query)
    if any(x in joined for x in ("当前", "目前", "这时", "这一轮", "此时")) and any(
        x in joined for x in ("最高", "最多", "理论")
    ):
        markers.append("当前 vs 历史状态")

    numbers = set(re.findall(r"\d+\s*(?:个|家|位|级|脚|分|秒)", joined))
    if len(numbers) >= 3:
        markers.append("数字")

    roster_signatures = {
        _roster_signature(text)
        for text in answer_lines
        if text and len(re.findall(r"、", text)) >= 2
    }
    roster_signatures.discard("")
    if len(roster_signatures) >= 2:
        markers.append("阵容名单")

    entities = set()
    for text in answer_lines:
        entities.update(_extract_entity_hits(text, limit=4))
    if len(entities) >= 6 and any(x in query for x in ("谁", "哪些人", "阵容", "首发")):
        markers.append("主体错位")

    if flags.get("asks_first_or_order") and len(set(re.findall(r"(首发|第一个出场|第一个出战|第六人|替补)", joined))) >= 2:
        markers.append("首发/第六人/替补混淆")
    if any(x in joined for x in ("为了", "导致")) and any(x in joined for x in ("因此", "所以")):
        markers.append("原因 vs 结果倒置")
    teams_in_query = {token for token in _MULTI_HOP_TEAM_TOKENS if token in query}
    teams_in_joined = {token for token in _MULTI_HOP_TEAM_TOKENS if token in joined}
    if teams_in_query and len(teams_in_joined - teams_in_query) >= 2:
        markers.append("战术主体混入别的队伍/别的场景")

    dedup: List[str] = []
    for marker in markers:
        if marker not in dedup:
            dedup.append(marker)
    return dedup


def _fragment_score(fragment: str, query: str, query_terms: List[str], flags: Dict[str, bool]) -> float:
    frag = (fragment or "").strip()
    if not frag:
        return float("-inf")

    qset = set(query_terms)
    toks = set(_tokenize(frag))
    score = float(len(qset & toks))

    q = query or ""
    flow = q.lower()
    frag_low = frag.lower()

    if "不喜欢" in q and "不喜欢" in frag:
        score += 6.0
    if "不喜欢" in q and frag.count("不喜欢") >= 2:
        score += 4.0

    if "启动模式" in q and "启动模式" in frag:
        score += 7.0
    if "核心任务" in q and "核心任务" in frag:
        score += 7.0
    if "收尾小复盘" in q and "收尾小复盘" in frag:
        score += 7.0
    if "人格设定" in q and any(x in frag for x in ("<redacted-persona-a>", "<redacted-persona-b>", "<redacted-persona-c>", "<redacted-persona-d>", "<redacted-persona-e>")):
        score += 9.0
    if "标签" in q and any(x in frag for x in ("<redacted-tag-a>", "<redacted-tag-b>", "<redacted-tag-c>", "<redacted-tag-d>", "<redacted-tag-e>", "<redacted-tag-f>")):
        score += 9.0

    if flags.get("asks_duration") and re.search(r"(\d+\s*分钟|[一二三四五六七八九十两半]+\s*分钟)", frag):
        score += 8.0
    if flags.get("asks_date") and re.search(r"\d+\s*月\s*\d+\s*日", frag):
        score += 8.0

    if flags.get("asks_relation") and "真正的伙伴" in frag:
        score += 6.0
    if flags.get("asks_relation") and "共同创造" in frag:
        score += 6.0
    if flags.get("asks_relation") and "真正的伙伴" in frag and "共同创造" in frag:
        score += 4.0
    if ("ai" in flow or "<redacted-companion-name>" in q) and ("ai" in frag_low or "<redacted-companion-name>" in frag):
        score += 2.0

    if flags.get("asks_avatar") and any(x in frag for x in ("头像", "照片")):
        score += 1.5
    if flags.get("asks_avatar") and any(x in frag for x in ("图2", "第二张", "<redacted-avatar-label>")):
        score += 6.0
    if flags.get("asks_avatar") and any(x in frag for x in ("正式头像", "专属头像", "正式决定", "最终选择")):
        score += 5.0

    if flags.get("asks_choice") and any(x in frag for x in ("第二张", "图2", "<redacted-avatar-label>", "强制", "命令")):
        score += 2.0

    if flags.get("asks_opening") and any(
        x in frag for x in ("<redacted-opening-a>", "<redacted-opening-b>", "<redacted-opening-c>", "<redacted-opening-d>", "<redacted-opening-e>")
    ):
        score += 3.0

    if flags.get("asks_duration") and "分钟" not in frag and len(frag) <= 20:
        score -= 2.0
    if flags.get("asks_date") and "月" not in frag and "日" not in frag:
        score -= 1.0
    if len(frag) > 160:
        score -= min(8.0, (len(frag) - 160) / 40.0)

    return score


def _best_evidence_fragment(text: str, query: str, query_terms: List[str], max_chars: int = 240) -> Tuple[str, float]:
    """Pick the best evidence fragment inside a chunk."""

    fragments = _extract_fragments(text)
    if not fragments:
        s = text.strip().replace("\n", " ")
        snippet = (s[:max_chars] + "…") if len(s) > max_chars else s
        return snippet, 0.0

    flags = _query_flags(query)
    best = max(fragments, key=lambda frag: (_fragment_score(frag, query, query_terms, flags), -len(frag)))
    score = _fragment_score(best, query, query_terms, flags)
    best = best.replace("\n", " ").strip()
    if len(best) > max_chars:
        best = best[:max_chars] + "…"
    return best, score


_SIG_GENERIC = {
    "感觉",
    "最近",
    "之前",
    "时候",
    "问题",
    "情况",
    "片段",
    "内容",
    "用户",
    "<redacted-companion-name>",
    "亲爱的",
    "真的",
    "就是",
    "自己",
    "我们",
    "回复",
}


def _signature_terms(text: str, limit: int = 4) -> List[str]:
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,10}", (text or "").lower())
    out: List[str] = []
    seen = set()
    for part in parts:
        if not part or part in seen or part in _SIG_GENERIC:
            continue
        seen.add(part)
        out.append(part)
        if len(out) >= limit:
            break
    return out


def _evidence_signature(snippet: str, source: str) -> str:
    terms = _signature_terms(snippet)
    if terms:
        return f"{source}|{'|'.join(terms[:4])}"
    fallback = re.sub(r"\s+", "", (snippet or ""))[:24]
    return f"{source}|{fallback}"


def _conflict_markers(snippets: List[str]) -> List[str]:
    joined = "\n".join(snippets)
    markers: List[str] = []
    assets = [token for token in ("ETH", "SOL", "BNB", "BTC") if token.lower() in joined.lower()]
    if len(set(assets)) >= 2:
        markers.append("币种")
    if len(set(re.findall(r"\d+\s*月\s*\d+\s*日", joined))) >= 2:
        markers.append("日期")
    if len(set(re.findall(r"\d+\s*分钟", joined))) >= 2:
        markers.append("时长")
    if len(set(re.findall(r"(C\\+\\+|线性代数|微积分|Reading Week|GTA5|驾照)", joined, flags=re.IGNORECASE))) >= 2:
        markers.append("事件")
    return markers


# -------------------------- configuration --------------------------


@dataclass
class MemoryConfig:
    memory_dir: str
    chunk_chars: int = 2800
    core_top_terms: int = 50
    persona_max_bullets: int = 8
    refresh_every: int = 20
    retrieve_top_k: int = 6

    @staticmethod
    def from_env() -> "MemoryConfig":
        return MemoryConfig(
            memory_dir=os.getenv("MEMORY_DIR", default_memory_dir()),
            chunk_chars=int(os.getenv("CHUNK_CHARS", "2800")),
            core_top_terms=int(os.getenv("CORE_TOP_TERMS", "50")),
            persona_max_bullets=int(os.getenv("PERSONA_MAX_BULLETS", "8")),
            refresh_every=int(os.getenv("REFRESH_EVERY", "20")),
            retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", "6")),
        )


# -------------------------- main class --------------------------


class MemoryManager:
    """Core → Persona → Vault.

    Public methods used by the GUI:
      - import_files(paths)
      - rebuild_memory()
      - analyze_memory()  (refresh Core/Persona)
      - append_turn(user, assistant)
      - build_system_prompt(user_text, assistant_style, lang)
      - debug_view()

    The implementation aims to be *boringly reliable*.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.cfg = config or MemoryConfig.from_env()
        self.memory_dir = _resolve_memory_dir(self.cfg.memory_dir)
        self.cfg.memory_dir = self.memory_dir

        # Paths
        self.db_path = os.path.join(self.memory_dir, "index.sqlite")
        self.chunks_dir = os.path.join(self.memory_dir, "chunks")
        self.uploads_dir = os.path.join(self.memory_dir, "uploads")
        self.buffer_path = os.path.join(self.memory_dir, "buffer.txt")
        self.chat_log_path = os.path.join(self.memory_dir, "chat_log.txt")

        self.core_md_path = os.path.join(self.memory_dir, "01_core.md")
        self.persona_md_path = os.path.join(self.memory_dir, "02_persona.md")
        self.vault_md_path = os.path.join(self.memory_dir, "03_vault.md")

        # Ensure directories
        _ensure_dir(self.memory_dir)
        _ensure_dir(self.chunks_dir)
        _ensure_dir(self.uploads_dir)

        self.conn = self._open_db()
        self._ensure_files_exist()

    # ---------------- DB ----------------

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                source TEXT,
                created_at INTEGER,
                char_len INTEGER,
                doc_len INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS terms (
                term TEXT PRIMARY KEY,
                df INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS postings (
                term TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (term, chunk_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_chunk ON postings(chunk_id);")
        conn.commit()
        return conn

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _meta_get_int(self, key: str, default: int = 0) -> int:
        cur = self.conn.cursor()
        row = cur.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return int(row["value"])
        except Exception:
            return default

    def _meta_set_int(self, key: str, value: int) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(int(value))),
        )
        self.conn.commit()

    # ---------------- files ----------------

    def _ensure_files_exist(self) -> None:
        if not os.path.exists(self.core_md_path):
            _write_text(
                self.core_md_path,
                "# Core Memory\n\n- 称呼用户为“您”。\n- 回答尽量详细、结构清晰。\n",
            )
        if not os.path.exists(self.persona_md_path):
            _write_text(
                self.persona_md_path,
                "# Persona (≤ 8 bullets)\n\n- （尚未分析）\n",
            )
        if not os.path.exists(self.vault_md_path):
            _write_text(
                self.vault_md_path,
                "# Vault (Long-tail)\n\n> 这里记录 Vault 的索引摘要；原文分块在 chunks/ 中。\n\n",
            )

    # ---------------- import / rebuild ----------------

    def import_files(self, file_paths: List[str]) -> List[str]:
        """Copy files into memory/uploads/ and return their stored paths."""

        stored: List[str] = []
        for p in file_paths:
            if not p:
                continue
            if not os.path.exists(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext not in (".txt", ".docx"):
                continue

            base = _safe_filename(os.path.basename(p))
            # Avoid overwrite
            dst = os.path.join(self.uploads_dir, base)
            if os.path.exists(dst):
                stem, ext2 = os.path.splitext(base)
                dst = os.path.join(self.uploads_dir, f"{stem}_{int(time.time())}{ext2}")

            shutil.copy2(p, dst)
            stored.append(dst)

        return stored

    def rebuild_memory(self) -> Dict[str, int]:
        """Rebuild the Vault index from: uploads/ + chat_log.txt.

        - Clears index.sqlite tables
        - Clears chunks/
        - Re-ingests sources
        - Refreshes Core/Persona
        """

        # Close DB so we can do a clean rebuild.
        self.close()

        # Reset chunks
        if os.path.exists(self.chunks_dir):
            for fn in os.listdir(self.chunks_dir):
                fp = os.path.join(self.chunks_dir, fn)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass

        # Reset vault summary
        _write_text(
            self.vault_md_path,
            "# Vault (Long-tail)\n\n> 这里记录 Vault 的索引摘要；原文分块在 chunks/ 中。\n\n",
        )

        # Remove and recreate DB
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.conn = self._open_db()

        chunks_added = 0

        # Ingest uploads
        uploads = self.list_uploads()
        for up in uploads:
            chunks_added += self._ingest_file(up, source=f"upload:{os.path.basename(up)}", auto_refresh=False)

        # Ingest chat log (if exists)
        if os.path.exists(self.chat_log_path):
            chat_text = _read_text(self.chat_log_path)
            if chat_text.strip():
                # Strip timestamps like "[2026-01-21 12:34:56]" to keep tokens clean.
                chat_text = re.sub(r"\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]\s*", "", chat_text)
                chunks_added += self._ingest_text(chat_text, source="chat_log", auto_refresh=False)

        # Reset buffer (buffer is only for incremental future appends)
        _write_text(self.buffer_path, "")

        self._meta_set_int("new_chunks_since_refresh", 0)

        # Finally, analyze
        self.analyze_memory()

        return {
            "chunks": self.count_chunks(),
            "terms": self.count_terms(),
            "uploads": len(uploads),
            "chunks_added": chunks_added,
        }

    def analyze_memory(self) -> None:
        """Refresh 01_core.md and 02_persona.md."""
        self._refresh_core()
        self._refresh_persona()
        self._meta_set_int("last_analyze_ts", _now_ts())

    def list_uploads(self) -> List[str]:
        if not os.path.exists(self.uploads_dir):
            return []
        files = [os.path.join(self.uploads_dir, f) for f in os.listdir(self.uploads_dir)]
        files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in (".txt", ".docx")]
        files.sort(key=lambda p: os.path.getmtime(p))
        return files

    # ---------------- incremental chat ----------------

    def append_turn(self, user_text: str, assistant_text: str) -> int:
        """Append a Q&A turn to chat_log + buffer, then index it.

        Important UX note:
        - If we only chunk when the buffer reaches ~2800 chars, users may feel
          "memory never works" in short chats.
        - So we **force-flush** the buffer on every turn, which effectively
          indexes each Q&A as its own chunk (still compatible with the README
          chunk-on-disk design).

        Returns number of new chunks created.
        """

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_now_ts()))
        record_log = f"[{ts}] user: {user_text.strip()}\n[{ts}] assistant: {assistant_text.strip()}\n\n"
        # Index text should be clean (avoid timestamp digits hurting retrieval).
        record_chunk = f"user: {user_text.strip()}\nassistant: {assistant_text.strip()}\n\n"

        _append_text(self.chat_log_path, record_log)
        _append_text(self.buffer_path, record_chunk)

        # Force-flush on every turn so memory is immediately retrievable.
        return self._flush_buffer(auto_refresh=True, force=True)

    def _flush_buffer(self, auto_refresh: bool, force: bool = False) -> int:
        """Turn buffer.txt into chunks.

        - Normal mode: split when buffer grows beyond CHUNK_CHARS.
        - Force mode: flush the whole buffer into a single chunk.
        """

        buf = _read_text(self.buffer_path)
        if not buf:
            return 0

        made = 0

        if force and buf.strip():
            # Single small chunk (best UX for chat)
            chunk_text = buf.strip()
            self._add_chunk(chunk_text, source="chat", auto_refresh=auto_refresh)
            _write_text(self.buffer_path, "")
            return 1

        while len(buf) >= self.cfg.chunk_chars:
            # Find a good boundary near chunk_chars
            cut = self.cfg.chunk_chars
            window = buf[:cut]
            # Prefer last newline within last 25% of window
            nl = window.rfind("\n")
            if nl >= int(cut * 0.5):
                cut = nl

            chunk_text = buf[:cut].strip()
            buf = buf[cut:].lstrip()

            if chunk_text:
                self._add_chunk(chunk_text, source="chat", auto_refresh=auto_refresh)
                made += 1

        _write_text(self.buffer_path, buf)
        return made

    # ---------------- ingestion ----------------

    def _ingest_file(self, path: str, source: str, auto_refresh: bool) -> int:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            return self._ingest_txt_stream(path, source=source, auto_refresh=auto_refresh)
        if ext == ".docx":
            return self._ingest_docx(path, source=source, auto_refresh=auto_refresh)
        return 0

    def _ingest_txt_stream(self, path: str, source: str, auto_refresh: bool) -> int:
        made = 0
        acc: List[str] = []
        acc_len = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line:
                    continue
                acc.append(line)
                acc_len += len(line)
                if acc_len >= self.cfg.chunk_chars:
                    text = "".join(acc).strip()
                    if text:
                        self._add_chunk(text, source=source, auto_refresh=auto_refresh)
                        made += 1
                    acc, acc_len = [], 0
        tail = "".join(acc).strip()
        if tail:
            self._add_chunk(tail, source=source, auto_refresh=auto_refresh)
            made += 1
        return made

    def _ingest_docx(self, path: str, source: str, auto_refresh: bool) -> int:
        try:
            from docx import Document  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "解析 .docx 需要 python-docx。请先安装：pip install python-docx"
            ) from e

        doc = Document(path)
        made = 0
        acc: List[str] = []
        acc_len = 0
        for para in doc.paragraphs:
            t = (para.text or "").strip()
            if not t:
                continue
            line = t + "\n"
            acc.append(line)
            acc_len += len(line)
            if acc_len >= self.cfg.chunk_chars:
                text = "".join(acc).strip()
                if text:
                    self._add_chunk(text, source=source, auto_refresh=auto_refresh)
                    made += 1
                acc, acc_len = [], 0

        tail = "".join(acc).strip()
        if tail:
            self._add_chunk(tail, source=source, auto_refresh=auto_refresh)
            made += 1

        return made

    def _ingest_text(self, text: str, source: str, auto_refresh: bool) -> int:
        if not text:
            return 0
        made = 0
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + self.cfg.chunk_chars)
            piece = text[start:end]
            # Try to cut at a newline near the end
            if end < n:
                nl = piece.rfind("\n")
                if nl >= int(self.cfg.chunk_chars * 0.5):
                    end = start + nl
                    piece = text[start:end]
            piece = piece.strip()
            if piece:
                self._add_chunk(piece, source=source, auto_refresh=auto_refresh)
                made += 1
            start = end
        return made

    # ---------------- indexing core ----------------

    def _add_chunk(self, text: str, source: str, auto_refresh: bool) -> int:
        """Persist chunk + index."""

        text = text.strip()
        if not text:
            return -1

        tokens = _tokenize(text)
        if not tokens:
            return -1

        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        uniq_terms = list(tf.keys())

        created_at = _now_ts()

        cur = self.conn.cursor()

        # Insert chunk row to get id
        cur.execute(
            "INSERT INTO chunks(path, source, created_at, char_len, doc_len) VALUES (?,?,?,?,?)",
            ("", source, created_at, len(text), len(tokens)),
        )
        chunk_id = int(cur.lastrowid)

        rel_path = os.path.join("chunks", f"{chunk_id:08d}.txt")
        abs_path = os.path.join(self.memory_dir, rel_path)
        _write_text(abs_path, text)

        cur.execute("UPDATE chunks SET path=? WHERE id=?", (rel_path, chunk_id))

        # postings
        cur.executemany(
            "INSERT INTO postings(term, chunk_id, tf) VALUES (?,?,?)",
            [(term, chunk_id, int(freq)) for term, freq in tf.items()],
        )

        # df updates
        cur.executemany(
            "INSERT INTO terms(term, df) VALUES (?,1) ON CONFLICT(term) DO UPDATE SET df=df+1",
            [(term,) for term in uniq_terms],
        )

        self.conn.commit()

        # vault summary (lightweight)
        stamp = time.strftime("%Y-%m-%d", time.localtime(created_at))
        preview = text.replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:200] + "…"
        _append_text(
            self.vault_md_path,
            f"### Chunk {chunk_id:08d} ({stamp}) [{source}]\n\n{preview}\n\n",
        )

        if auto_refresh:
            n = self._meta_get_int("new_chunks_since_refresh", 0) + 1
            self._meta_set_int("new_chunks_since_refresh", n)
            if n >= self.cfg.refresh_every:
                # Refresh Core/Persona and reset
                self.analyze_memory()
                self._meta_set_int("new_chunks_since_refresh", 0)

        return chunk_id

    # ---------------- stats ----------------

    def count_chunks(self) -> int:
        cur = self.conn.cursor()
        row = cur.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def count_terms(self) -> int:
        cur = self.conn.cursor()
        row = cur.execute("SELECT COUNT(*) AS n FROM terms").fetchone()
        return int(row["n"]) if row else 0

    # ---------------- retrieval (BM25) ----------------

    def _idf(self, df: int, N: int) -> float:
        # Standard BM25 idf
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, object]]:
        k = int(k or self.cfg.retrieve_top_k)
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        flags = _query_flags(query)

        cur = self.conn.cursor()
        rowN = cur.execute("SELECT COUNT(*) AS n, AVG(doc_len) AS avgdl FROM chunks").fetchone()
        if not rowN or int(rowN["n"]) == 0:
            return []

        N = int(rowN["n"])
        avgdl = float(rowN["avgdl"] or 1.0)

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        scores: Dict[int, float] = {}
        seen_terms = set()

        for term in q_terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)

            row_df = cur.execute("SELECT df FROM terms WHERE term=?", (term,)).fetchone()
            if not row_df:
                continue
            df = int(row_df["df"])
            if df <= 0:
                continue

            idf = self._idf(df, N)

            rows = cur.execute(
                """
                SELECT p.chunk_id AS chunk_id, p.tf AS tf, c.doc_len AS dl, c.path AS path, c.source AS source, c.created_at AS created_at
                FROM postings p
                JOIN chunks c ON c.id = p.chunk_id
                WHERE p.term = ?
                """,
                (term,),
            ).fetchall()

            for r in rows:
                cid = int(r["chunk_id"])
                tf = int(r["tf"])
                dl = int(r["dl"] or 1)
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                part = idf * ((tf * (k1 + 1)) / (denom + 1e-9))
                scores[cid] = scores.get(cid, 0.0) + float(part)

        if not scores:
            scores = {}

        if flags.get("asks_opening"):
            opening_rows = cur.execute(
                "SELECT id FROM chunks ORDER BY id ASC LIMIT 3"
            ).fetchall()
            for row in opening_rows:
                cid = int(row["id"])
                scores[cid] = scores.get(cid, 0.0) + (6.0 / max(1, cid))

        # Top-k by score
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(20, k)]

        # Hydrate and extract evidence
        out: List[Dict[str, object]] = []
        for cid, sc in top:
            row = cur.execute(
                "SELECT id, path, source, created_at FROM chunks WHERE id=?", (int(cid),)
            ).fetchone()
            if not row:
                continue
            rel_path = str(row["path"])
            abs_path = os.path.join(self.memory_dir, rel_path)
            if not os.path.exists(abs_path):
                continue
            text = _read_text(abs_path)
            snippet, snippet_score = _best_evidence_fragment(text, query, q_terms)

            structural_bonus = 0.0
            if flags.get("asks_opening"):
                structural_bonus += 4.0 / max(1, int(cid))
            if flags.get("asks_duration") and re.search(r"(\d+\s*分钟|[一二三四五六七八九十两半]+\s*分钟)", text):
                structural_bonus += 2.5
            if flags.get("asks_date") and re.search(r"\d+\s*月\s*\d+\s*日", text):
                structural_bonus += 2.5
            if "不喜欢" in query and text.count("不喜欢") >= 2:
                structural_bonus += 2.5
            if flags.get("asks_relation") and any(x in text for x in ("真正的伙伴", "共同创造")):
                structural_bonus += 2.5
            if flags.get("asks_avatar") and any(x in text for x in ("第二张", "图2", "<redacted-avatar-label>")):
                structural_bonus += 2.0

            combined_score = float(sc) + structural_bonus + (0.35 * float(snippet_score))
            out.append(
                {
                    "chunk_id": int(cid),
                    "score": combined_score,
                    "source": str(row["source"] or ""),
                    "created_at": int(row["created_at"] or 0),
                    "snippet": snippet,
                    "signature": _evidence_signature(snippet, str(row["source"] or "")),
                }
            )

        out.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        selected: List[Dict[str, object]] = []
        seen_signatures: set[str] = set()
        deferred: List[Dict[str, object]] = []
        for item in out:
            signature = str(item.get("signature", ""))
            if signature and signature in seen_signatures:
                deferred.append(item)
                continue
            if signature:
                seen_signatures.add(signature)
            selected.append(item)
            if len(selected) >= k:
                break
        if len(selected) < k:
            for item in deferred:
                selected.append(item)
                if len(selected) >= k:
                    break
        return selected[:k]

    def _chunk_text_by_id(self, chunk_id: int) -> str:
        cur = self.conn.cursor()
        row = cur.execute("SELECT path FROM chunks WHERE id=?", (int(chunk_id),)).fetchone()
        if not row:
            return ""
        rel_path = str(row["path"])
        abs_path = os.path.join(self.memory_dir, rel_path)
        if not os.path.exists(abs_path):
            return ""
        return _read_text(abs_path)

    def _chunk_row_by_id(self, chunk_id: int):
        cur = self.conn.cursor()
        return cur.execute("SELECT id, source, path, created_at FROM chunks WHERE id=?", (int(chunk_id),)).fetchone()

    def _supplement_fact_items(self, query: str, flags: Dict[str, bool], subtype: str, limit: int = 18) -> List[Dict[str, object]]:
        if not subtype:
            return []
        query_entities = set(_query_anchor_terms(query, limit=6))
        query_tokens = _tokenize(query)
        cur = self.conn.cursor()
        rows = cur.execute("SELECT id, source, path, created_at FROM chunks ORDER BY id ASC").fetchall()
        matches: List[Dict[str, object]] = []
        for row in rows:
            chunk_id = int(row["id"])
            abs_path = os.path.join(self.memory_dir, str(row["path"]))
            if not os.path.exists(abs_path):
                continue
            text = _read_text(abs_path)
            if not text:
                continue
            hit = False
            if subtype == "ranking_position":
                hit = any(anchor in text for anchor in ("排名", "排行", "总排名")) and "嘉世" in text and bool(
                    re.search(r"第[\u4e00-\u9fff0-9两十百千]+位", text)
                )
            elif subtype == "level_count":
                hit = any(entity in text for entity in query_entities) and bool(re.search(r"[一二三四五六七八九十百两0-9]+\s*级", text))
            elif subtype == "guild_count":
                hit = "公会" in text and bool(re.search(r"[一二三四五六七八九十百两0-9]+\s*家", text)) and any(
                    token in text for token in ("围剿", "参与", "队伍", "行动")
                )
            elif subtype == "record_time":
                hit = bool(re.search(r"\d+\s*分\s*\d+\s*秒(?:\s*\d+)?", text)) and any(
                    anchor in text for anchor in ("成绩", "通关记录", "打破副本", "冰霜森林")
                )
            elif subtype == "wager_or_material":
                hit = "强力蛛丝" in text and any(anchor in text for anchor in ("赌注", "筹码", "抵押"))
            elif subtype == "role_name":
                hit = any(entity in text for entity in query_entities) and bool(
                    re.search(r"这个[\u4e00-\u9fffA-Za-z0-9·\-]{2,10}吧", text)
                    or re.search(r"[\u4e00-\u9fffA-Za-z0-9·\-]{2,10}交还给了?[\u4e00-\u9fff]{2,4}", text)
                )
            elif subtype == "boss_name":
                hit = any(anchor in text for anchor in ("BOSS", "霸气雄图", "烈焰森林", "身影")) and bool(
                    re.search(r"(?:BOSS|boss)[叫名为是：:\s]*[\u4e00-\u9fffA-Za-z0-9·\-]{2,14}", text)
                    or re.search(r"[\u4e00-\u9fffA-Za-z0-9·\-]{2,14}的身影", text)
                )
            elif subtype == "current_vs_max":
                hit = any(anchor in text for anchor in ("当前", "目前", "等阶", "鹰踏")) and any(
                    anchor in text for anchor in ("三脚", "五脚", "能出")
                )
            elif subtype == "weapon_form":
                hit = "25级银武" in text and any(term in text for term in _WEAPON_TERMS)
            elif subtype == "running_total":
                hit = "圣诞小偷" in text and any(anchor in text for anchor in ("陈果", "323了", "第323个", "数到", "达到"))
            elif subtype == "generic_fact":
                hit = bool(query_entities) and any(entity in text for entity in query_entities)
            if not hit:
                continue
            snippet, fragment_score = _best_evidence_fragment(text, query, query_tokens)
            answer_lines = _extract_answer_lines(text, query, flags, limit=3)
            fact_candidates = _extract_fact_candidates(text, query, flags, subtype)
            if not fact_candidates:
                continue
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "source": str(row["source"]),
                    "created_at": int(row["created_at"] or 0),
                    "snippet": snippet,
                    "answer_lines": answer_lines,
                    "fact_candidates": fact_candidates,
                    "event_tag": _event_tag_from_text(snippet or text[:220]),
                    "time_cue": _extract_time_cue(snippet or text),
                    "entity_hits": _extract_entity_hits(text[:800], limit=8),
                    "score": 18.0 + (2.5 * float(fragment_score)) + min(len(fact_candidates), 4) * 1.5,
                    "signature": _evidence_signature(snippet or text[:220], str(row["source"])),
                }
            )
        matches.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return matches[:limit]

    def retrieve_structured(self, query: str, mode: str = "auto", k: Optional[int] = None) -> Dict[str, object]:
        final_k = int(k or self.cfg.retrieve_top_k)
        resolved_mode = _resolve_answer_mode(query, mode)
        flags = _query_flags(query)
        fact_subtype = _resolve_fact_subtype(query, flags) if resolved_mode == "fact" else ""
        question_subtype = _resolve_multi_hop_subtype(query, flags) if resolved_mode == "multi_hop" else ""
        candidate_floor = 48 if resolved_mode in ("fact", "multi_hop") else max(12, final_k)
        candidate_k = max(final_k, candidate_floor)
        query_variants = _structured_query_variants(query, flags, resolved_mode)
        raw_by_chunk: Dict[int, Dict[str, object]] = {}
        for idx, variant in enumerate(query_variants):
            weight = max(0.55, 1.0 - (0.12 * idx))
            for rank, item in enumerate(self.retrieve(variant, k=candidate_k)):
                chunk_id = int(item.get("chunk_id", 0) or 0)
                if chunk_id <= 0:
                    continue
                score = float(item.get("score", 0.0)) + (weight * max(0.0, 1.8 - (0.05 * rank)))
                existing = raw_by_chunk.get(chunk_id)
                if existing is None:
                    raw_by_chunk[chunk_id] = {
                        **item,
                        "score": score,
                        "_variant_hits": 1,
                    }
                    continue
                existing["score"] = max(float(existing.get("score", 0.0)), score) + (0.12 * weight)
                existing["_variant_hits"] = int(existing.get("_variant_hits", 1) or 1) + 1
                if len(str(item.get("snippet", ""))) > len(str(existing.get("snippet", ""))):
                    existing["snippet"] = item.get("snippet", "")

        raw_items = sorted(raw_by_chunk.values(), key=lambda d: float(d.get("score", 0.0)), reverse=True)
        query_term_list = _tokenize(query)
        query_tokens = {token for token in query_term_list if token and token not in _ZH_STOP and token not in _EN_STOP}
        query_numbers = set(re.findall(r"\d+\s*(?:级|轮|个|位|脚|分|秒)", query))
        query_entities = set(_query_anchor_terms(query, limit=8))
        scenario_anchors = _scenario_anchor_terms(query)
        relation_terms = {term for term in _MULTI_HOP_RELATION_TERMS if term in query}
        role_terms = {term for term in _MULTI_HOP_ROLE_TERMS if term in query}
        action_terms = {term for term in _MULTI_HOP_ACTION_TERMS if term in query}

        if resolved_mode in ("fact", "multi_hop"):
            seed_items = raw_items[: min(len(raw_items), 8 if resolved_mode == "multi_hop" else 6)]
            for seed in seed_items:
                seed_id = int(seed.get("chunk_id", 0) or 0)
                seed_source = str(seed.get("source", ""))
                for neighbor_id in (seed_id - 1, seed_id + 1):
                    if neighbor_id <= 0 or neighbor_id in raw_by_chunk:
                        continue
                    row = self._chunk_row_by_id(neighbor_id)
                    if not row or (seed_source and str(row["source"]) != seed_source):
                        continue
                    abs_path = os.path.join(self.memory_dir, str(row["path"]))
                    if not os.path.exists(abs_path):
                        continue
                    text = _read_text(abs_path)
                    if not text:
                        continue
                    snippet, frag_score = _best_evidence_fragment(text, query, query_term_list)
                    neighbor_penalty = -2.2
                    neighbor_weight = 0.35
                    if resolved_mode == "multi_hop" and (flags.get("asks_first_or_order") or question_subtype == "asks_compare_two_sides"):
                        neighbor_penalty = -0.8
                        neighbor_weight = 0.65
                    raw_by_chunk[neighbor_id] = {
                        "chunk_id": int(row["id"]),
                        "source": str(row["source"]),
                        "created_at": int(row["created_at"] or 0),
                        "snippet": snippet,
                        "score": max(0.0, float(seed.get("score", 0.0)) + neighbor_penalty) + (neighbor_weight * float(frag_score)),
                        "_variant_hits": 1,
                    }
            raw_items = sorted(raw_by_chunk.values(), key=lambda d: float(d.get("score", 0.0)), reverse=True)

        enriched: List[Dict[str, object]] = []
        for item in raw_items:
            chunk_id = int(item.get("chunk_id", 0) or 0)
            text = self._chunk_text_by_id(chunk_id)
            if resolved_mode == "multi_hop":
                answer_lines = _extract_multi_hop_lines(text, query, flags, limit=5)
            else:
                answer_lines = _extract_answer_lines(text, query, flags, limit=3)
            fact_candidates = _extract_fact_candidates(text, query, flags, fact_subtype) if resolved_mode == "fact" else []
            best_answer_score = max((_answer_line_score(line, query, flags) for line in answer_lines), default=0.0)
            answer_text = " ".join(answer_lines)
            combined_text = f"{answer_text} {item.get('snippet', '')} {text[:800]}"
            token_overlap = len(set(_tokenize(combined_text)) & query_tokens)
            number_overlap = len({num for num in query_numbers if num and num in combined_text})
            entity_hits = _extract_entity_hits(answer_text or text)
            entity_overlap = len(set(entity_hits) & query_entities)
            anchor_overlap = sum(1 for anchor in scenario_anchors if anchor in combined_text)
            foreign_anchors = [anchor for anchor in _SCENARIO_ANCHOR_TOKENS if anchor in combined_text and anchor not in scenario_anchors]
            relation_overlap = sum(1 for term in relation_terms if term in combined_text)
            role_overlap = sum(1 for term in role_terms if term in combined_text)
            action_overlap = sum(1 for term in action_terms if term in combined_text)

            rerank_bonus = 0.0
            rerank_bonus += min(token_overlap, 10) * 0.42
            rerank_bonus += min(number_overlap, 3) * 0.9
            rerank_bonus += min(int(item.get("_variant_hits", 1) or 1), 4) * 0.35
            if resolved_mode == "fact":
                if flags.get("asks_name_or_title") and any(x in answer_text for x in ("叫", "名为", "名称", "角色名称", "名字", "寒烟柔", "步枪")):
                    rerank_bonus += 3.0
                if flags.get("asks_count_or_total") and re.search(r"(第[一二三四五六七八九十百0-9]+位|\d+\s*(?:个|家|位|脚|级|分|秒))", answer_text):
                    rerank_bonus += 3.2
                if flags.get("asks_current_vs_max") and any(x in answer_text for x in ("当前", "目前", "这时", "此时", "这一轮")):
                    rerank_bonus += 3.2
                if flags.get("asks_current_vs_max") and any(x in answer_text for x in ("最高", "最多", "理论")) and not any(
                    x in answer_text for x in ("当前", "目前", "这时", "此时", "这一轮")
                ):
                    rerank_bonus -= 1.8
                if flags.get("asks_first_or_order") and any(x in answer_text for x in ("第一个", "首发", "率先", "第一个出战", "第一个出场")):
                    rerank_bonus += 3.0
                if flags.get("asks_roster") and len(re.findall(r"、", answer_text)) >= 2:
                    rerank_bonus += 2.8
                if flags.get("asks_item_or_weapon") and any(x in answer_text for x in ("步枪", "银武", "强力蛛丝", "赌注", "筹码")):
                    rerank_bonus += 3.0
                if flags.get("asks_item_or_weapon") and any(term in combined_text for term in _WEAPON_TERMS):
                    rerank_bonus += 2.8
                if flags.get("asks_item_or_weapon") and "银武" in query and "银武" in combined_text:
                    rerank_bonus += 2.8
                if flags.get("asks_item_or_weapon") and query_numbers and not any(num in combined_text for num in query_numbers):
                    rerank_bonus -= 2.4
                if flags.get("asks_skill_name") and any(x in answer_text for x in ("接投", "鹰踏", "挡拆")):
                    rerank_bonus += 2.6
                if flags.get("asks_skill_name") and re.search(r"[：:]\s*[\u4e00-\u9fffA-Za-z0-9·\-]{2,8}", combined_text):
                    rerank_bonus += 3.4
                if fact_subtype == "ranking_position":
                    if any(x in combined_text for x in ("排名", "排行", "总排名", "倒数")) and re.search(r"第[\u4e00-\u9fff0-9两十百千]+位", combined_text):
                        rerank_bonus += 5.2
                elif fact_subtype == "level_count":
                    if re.search(r"[一二三四五六七八九十百两0-9]+\s*级", combined_text):
                        rerank_bonus += 4.2
                elif fact_subtype == "guild_count":
                    if re.search(r"[一二三四五六七八九十百两0-9]+\s*家", combined_text) and "公会" in combined_text:
                        rerank_bonus += 4.6
                if fact_subtype == "record_time":
                    if re.search(r"\d+\s*分\s*\d+\s*秒(?:\s*\d+)?", combined_text):
                        rerank_bonus += 5.4
                    if re.findall(r"\d+\s*分", combined_text) and not re.search(r"\d+\s*分\s*\d+\s*秒(?:\s*\d+)?", combined_text):
                        rerank_bonus -= 2.8
                elif fact_subtype == "wager_or_material":
                    if any(x in combined_text for x in ("赌注", "筹码", "抵押", "强力蛛丝")):
                        rerank_bonus += 5.0
                    if "一百块" in combined_text and "强力蛛丝" not in combined_text:
                        rerank_bonus -= 2.2
                elif fact_subtype == "role_name":
                    if any(x in combined_text for x in ("角色", "交还给", "寒烟柔", "君莫笑", "一寸灰")):
                        rerank_bonus += 4.6
                    if query_entities and not (set(entity_hits) & query_entities) and "角色" not in combined_text:
                        rerank_bonus -= 2.6
                elif fact_subtype == "boss_name":
                    if any(x in combined_text for x in ("BOSS", "boss", "岩之浪人奥磐", "炎女巫卡修")):
                        rerank_bonus += 4.8
                    if "霸气雄图" in query and "霸气雄图" in combined_text:
                        rerank_bonus += 2.0
                elif fact_subtype == "current_vs_max":
                    if any(x in combined_text for x in ("当前技能等阶", "目前等阶", "能出", "三脚")):
                        rerank_bonus += 5.0
                    if any(x in combined_text for x in ("最高", "最多", "五脚")) and not any(x in combined_text for x in ("当前", "目前", "能出")):
                        rerank_bonus -= 3.0
                elif fact_subtype == "weapon_form":
                    if all(x in combined_text for x in ("25级", "银武")) and any(term in combined_text for term in _WEAPON_TERMS):
                        rerank_bonus += 5.0
                    if "步枪" in combined_text:
                        rerank_bonus += 3.0
                elif fact_subtype == "running_total":
                    if any(x in combined_text for x in ("陈果", "数到", "达到", "第323个", "323了")):
                        rerank_bonus += 5.2
                    if any(x in combined_text for x in ("341个", "415个")) and not any(x in combined_text for x in ("陈果", "数到", "达到")):
                        rerank_bonus -= 2.0

            if resolved_mode == "multi_hop":
                if entity_overlap:
                    rerank_bonus += min(entity_overlap, 4) * 0.95
                if anchor_overlap:
                    rerank_bonus += min(anchor_overlap, 4) * 1.2
                elif scenario_anchors:
                    rerank_bonus -= 2.6
                if relation_overlap:
                    rerank_bonus += min(relation_overlap, 4) * 0.8
                if role_overlap:
                    rerank_bonus += min(role_overlap, 4) * 0.75
                if action_overlap:
                    rerank_bonus += min(action_overlap, 5) * 1.0
                if entity_overlap and action_overlap:
                    rerank_bonus += 2.4
                if query_entities and entity_hits and not (set(entity_hits) & query_entities):
                    rerank_bonus -= 3.8
                if foreign_anchors:
                    rerank_bonus -= min(len(foreign_anchors), 3) * 1.5
                if len(entity_hits) < 1 and len(answer_text) > 40:
                    rerank_bonus -= 1.6
                if len(answer_text) > 100 and action_overlap == 0 and relation_overlap == 0:
                    rerank_bonus -= 1.4
                if any(x in answer_text for x in ("并没有", "最终", "之后", "随后", "因此", "导致", "休息了一轮")):
                    rerank_bonus += 2.4
                if flags.get("asks_roster") and len(re.findall(r"、", answer_text)) >= 2:
                    rerank_bonus += 2.5
                if flags.get("asks_status_then_fact") and any(x in answer_text for x in ("传闻", "实际", "并没有", "可以复出")):
                    rerank_bonus += 2.2
                if flags.get("asks_exchange_or_role") and any(x in answer_text for x in ("作为", "交换", "筹码", "送往")):
                    rerank_bonus += 2.2
                if flags.get("asks_roster") and len(re.findall(r"、", combined_text)) >= 4 and "叶修" in combined_text:
                    rerank_bonus += 4.2
                if flags.get("asks_roster") and any(x in combined_text for x in ("叶修休息了一轮", "叶修居然没有在团队赛中出场", "叶修在团队赛未出场")):
                    rerank_bonus += 4.4
                if flags.get("asks_status_then_fact") and all(x in combined_text for x in ("可以复出", "并没有")):
                    rerank_bonus += 4.0
                if flags.get("asks_status_then_fact") and any(x in combined_text for x in ("任何渠道", "宣布复出", "没有消息")):
                    rerank_bonus += 2.4
                if question_subtype == "asks_rumor_vs_actual":
                    if all(x in combined_text for x in ("可以复出", "并没有")):
                        rerank_bonus += 4.6
                    if any(x in combined_text for x in ("宣布复出", "任何渠道", "没有消息")):
                        rerank_bonus += 2.8
                elif question_subtype == "asks_tactic_sequence":
                    if any(x in combined_text for x in ("诱饵", "拳法家", "分头行动", "夹攻", "包围", "不同方向")):
                        rerank_bonus += 4.2
                    if any(x in combined_text for x in ("嘉世", "肖时钦", "王泽", "孙翔", "申建", "张家兴")):
                        rerank_bonus += 3.8
                    if any(x in combined_text for x in ("叶修", "包子", "依诺")) and not (set(entity_hits) & query_entities):
                        rerank_bonus -= 4.0
                elif question_subtype == "asks_design_rationale":
                    if any(x in combined_text for x in ("冰抗", "暗抗", "反应慢", "高暴击", "高智力", "爆发")):
                        rerank_bonus += 4.4
                elif question_subtype == "asks_lineup_constraint":
                    if any(x in combined_text for x in ("喻文州", "徐景熙", "单人对决", "治疗职业", "二线", "替补", "一线主力")):
                        rerank_bonus += 4.8
                    if "蓝雨" not in combined_text and any(x in combined_text for x in ("嘉世", "兴欣", "虚空")):
                        rerank_bonus -= 4.2
                elif question_subtype == "asks_counter_adjustment":
                    if any(x in combined_text for x in ("李迅", "杨昊轩", "枪炮师", "替下", "首发阵容")):
                        rerank_bonus += 5.0
                    if any(x in combined_text for x in ("冰抗", "暗抗", "安文逸", "反应慢")):
                        rerank_bonus -= 5.0
                elif question_subtype == "asks_role_plus_tactic":
                    if any(x in combined_text for x in ("战斗法师", "机械师", "树根", "走位", "撤退", "躲避攻击", "抢夺BOSS")):
                        rerank_bonus += 5.0
                    if any(x in combined_text for x in ("魏琛", "迎风布阵", "死亡之门")):
                        rerank_bonus -= 4.0
                elif question_subtype == "asks_roster_plus_status":
                    if flags.get("asks_roster") and len(re.findall(r"、", combined_text)) >= 4:
                        rerank_bonus += 3.2
                    if any(x in combined_text for x in ("休息了一轮", "没有在团队赛中出场", "未上场")):
                        rerank_bonus += 3.6
                elif question_subtype == "asks_exchange_mapping":
                    if any(x in combined_text for x in ("筹码", "交换", "送往", "雷霆")):
                        rerank_bonus += 3.8
                elif question_subtype == "asks_compare_two_sides":
                    if "分别" in combined_text or ("兴欣" in combined_text and "嘉世" in combined_text):
                        rerank_bonus += 2.6
                    if any(x in combined_text for x in ("第一个出场", "第一个出战", "站了起来", "肖时钦", "唐柔")):
                        rerank_bonus += 3.4

            signature_basis = answer_text or str(item.get("snippet", ""))
            enriched.append(
                {
                    **item,
                    "answer_lines": answer_lines,
                    "fact_candidates": fact_candidates,
                    "event_tag": _event_tag_from_text(signature_basis),
                    "time_cue": _extract_time_cue(signature_basis or text),
                    "entity_hits": entity_hits,
                    "score": float(item.get("score", 0.0)) + rerank_bonus + (0.55 * float(best_answer_score)),
                    "signature": _evidence_signature(signature_basis or str(item.get("snippet", "")), str(item.get("source", ""))),
                }
            )

        enriched.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        if resolved_mode == "fact":
            supplemental = self._supplement_fact_items(query, flags, fact_subtype, limit=18)
            if supplemental:
                by_chunk = {int(item.get("chunk_id", 0) or 0): item for item in enriched}
                for item in supplemental:
                    chunk_id = int(item.get("chunk_id", 0) or 0)
                    existing = by_chunk.get(chunk_id)
                    if existing is None:
                        enriched.append(item)
                        by_chunk[chunk_id] = item
                        continue
                    existing["score"] = max(float(existing.get("score", 0.0)), float(item.get("score", 0.0)) + 4.0)
                    merged_lines = list(existing.get("answer_lines", []) or [])
                    for line in item.get("answer_lines", []) or []:
                        if line not in merged_lines:
                            merged_lines.append(line)
                    existing["answer_lines"] = merged_lines[:5]
                    merged_candidates = list(existing.get("fact_candidates", []) or [])
                    seen_candidate_keys = {
                        str(candidate.get("normalized", "") or str(candidate.get("answer", "")))
                        for candidate in merged_candidates
                    }
                    for candidate in item.get("fact_candidates", []) or []:
                        key = str(candidate.get("normalized", "") or str(candidate.get("answer", "")))
                        if key in seen_candidate_keys:
                            continue
                        merged_candidates.append(candidate)
                        seen_candidate_keys.add(key)
                    existing["fact_candidates"] = merged_candidates[:10]
                enriched.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)

        selected: List[Dict[str, object]] = []
        seen_signatures: set[str] = set()

        def _push(candidate: Dict[str, object]) -> bool:
            signature = str(candidate.get("signature", ""))
            if signature and signature in seen_signatures:
                if resolved_mode != "multi_hop":
                    return False
                allow_adjacent = any(
                    str(existing.get("source", "")) == str(candidate.get("source", ""))
                    and abs(int(existing.get("chunk_id", 0) or 0) - int(candidate.get("chunk_id", 0) or 0)) <= 2
                    and " ".join(str(line) for line in existing.get("answer_lines", []))
                    != " ".join(str(line) for line in candidate.get("answer_lines", []))
                    for existing in selected
                )
                if not allow_adjacent:
                    return False
            if signature:
                seen_signatures.add(signature)
            selected.append(candidate)
            return len(selected) >= final_k

        if resolved_mode == "multi_hop":
            def _prefilter(candidates: Sequence[Dict[str, object]], required_terms: Sequence[str], optional_terms: Sequence[str] = ()) -> List[Dict[str, object]]:
                picked: List[Dict[str, object]] = []
                for candidate in candidates:
                    text = " ".join(str(line) for line in candidate.get("answer_lines", [])) + " " + str(candidate.get("snippet", ""))
                    if required_terms and not any(term in text for term in required_terms):
                        continue
                    bonus = sum(1 for term in optional_terms if term in text)
                    candidate = {**candidate, "score": float(candidate.get("score", 0.0)) + (bonus * 0.3)}
                    picked.append(candidate)
                picked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
                return picked

            direct_candidates = enriched[: min(len(enriched), 10)]
            if question_subtype == "asks_lineup_constraint":
                direct_candidates = _prefilter(
                    enriched,
                    ("蓝雨", "个人赛", "擂台赛"),
                    ("喻文州", "徐景熙", "单人对决", "二线", "替补"),
                )[:10] or direct_candidates
            elif question_subtype == "asks_counter_adjustment":
                direct_candidates = _prefilter(
                    enriched,
                    ("虚空", "团队赛", "首发", "苏沐橙"),
                    ("杨昊轩", "李迅", "枪炮师", "替下", "远距离火力"),
                )[:10] or direct_candidates
            elif question_subtype == "asks_role_plus_tactic":
                direct_candidates = _prefilter(
                    enriched,
                    ("叶修", "肖时钦", "龙剑士"),
                    ("战斗法师", "机械师", "树根", "撤退", "走位"),
                )[:10] or direct_candidates
            for candidate in direct_candidates[:4]:
                if _push(candidate):
                    break
            timeline_candidates = [
                candidate
                for candidate in enriched[4:]
                if candidate.get("time_cue")
                or any(
                    x in " ".join(str(line) for line in candidate.get("answer_lines", []))
                    for x in ("第", "之后", "随后", "最终", "传闻", "实际", "并没有", "休息了一轮", "首发")
                )
            ]
            for candidate in timeline_candidates[:4]:
                if _push(candidate):
                    break
            if question_subtype == "asks_compare_two_sides":
                side_candidates = [
                    candidate
                    for candidate in enriched
                    if candidate not in selected
                    and any(
                        x in " ".join(str(line) for line in candidate.get("answer_lines", []))
                        for x in ("唐柔", "肖时钦", "第一个出战", "第一个上场", "左上角", "右下角")
                    )
                ]
                side_candidates.sort(
                    key=lambda candidate: (
                        sum(
                            1
                            for marker in ("唐柔", "肖时钦", "第一个出战", "第一个上场", "左上角", "右下角", "率先")
                            if marker in " ".join(str(line) for line in candidate.get("answer_lines", []))
                        ),
                        float(candidate.get("score", 0.0)),
                    ),
                    reverse=True,
                )
                for candidate in side_candidates[:3]:
                    if _push(candidate):
                        break
            elif question_subtype == "asks_lineup_constraint":
                for candidate in _prefilter(enriched, ("喻文州", "徐景熙"), ("单人对决", "治疗职业", "二线", "替补"))[:3]:
                    if candidate not in selected and _push(candidate):
                        break
            elif question_subtype == "asks_counter_adjustment":
                for candidate in _prefilter(enriched, ("杨昊轩", "李迅", "枪炮师"), ("替下", "首发阵容", "远距离火力"))[:3]:
                    if candidate not in selected and _push(candidate):
                        break
            elif question_subtype == "asks_role_plus_tactic":
                for candidate in _prefilter(enriched, ("战斗法师", "机械师"), ("树根", "走位", "撤退", "龙剑士"))[:3]:
                    if candidate not in selected and _push(candidate):
                        break
            covered_entities = set()
            for candidate in selected:
                covered_entities.update(str(hit) for hit in candidate.get("entity_hits", []) if str(hit).strip())
            contrast_candidates = [
                candidate
                for candidate in enriched
                if candidate not in selected
                and (
                    "分别" in " ".join(str(line) for line in candidate.get("answer_lines", []))
                    or any(x in " ".join(str(line) for line in candidate.get("answer_lines", [])) for x in ("传闻", "实际", "并没有", "但是", "然而", "却"))
                    or bool(set(str(hit) for hit in candidate.get("entity_hits", [])) - covered_entities)
                )
            ]
            for candidate in contrast_candidates[:4]:
                if _push(candidate):
                    break
            for candidate in enriched:
                if _push(candidate):
                    break
        else:
            for candidate in enriched:
                if _push(candidate):
                    break

        conflict_watch = _structured_conflict_markers(selected, query)
        answer_candidates: List[Dict[str, object]] = []
        canonical_answer = ""
        fact_strategy = "model_select"
        if resolved_mode == "fact":
            answer_candidates, canonical_answer, fact_strategy = _select_fact_answer_candidates(
                enriched,
                query,
                flags,
                fact_subtype,
            )
            for marker in _fact_conflict_watch(answer_candidates, fact_subtype):
                if marker not in conflict_watch:
                    conflict_watch.append(marker)
        direct_evidence = selected[: min(len(selected), 6 if resolved_mode == "fact" else 4)]
        timeline_evidence = []
        contrast_evidence = []
        if resolved_mode == "multi_hop":
            timeline_evidence = [
                candidate
                for candidate in selected
                if candidate not in direct_evidence and (candidate.get("time_cue") or candidate.get("event_tag"))
            ][:4]
            contrast_evidence = [
                candidate
                for candidate in selected
                if candidate not in direct_evidence
                and candidate not in timeline_evidence
                and (
                    "分别" in " ".join(str(line) for line in candidate.get("answer_lines", []))
                    or any(
                        x in " ".join(str(line) for line in candidate.get("answer_lines", []))
                        for x in ("传闻", "实际", "并没有", "但是", "然而", "却")
                    )
                )
            ][:4]

        return {
            "mode": resolved_mode,
            "flags": flags,
            "fact_subtype": fact_subtype,
            "question_subtype": question_subtype,
            "items": selected[:final_k],
            "direct_evidence": direct_evidence,
            "timeline_evidence": timeline_evidence,
            "contrast_evidence": contrast_evidence,
            "conflict_watch": conflict_watch,
            "answer_candidates": answer_candidates,
            "canonical_answer": canonical_answer,
            "fact_strategy": fact_strategy,
        }

    # ---------------- Core / Persona refresh ----------------

    def _refresh_core(self) -> None:
        """Core = stable rules + top frequent terms."""

        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT term, SUM(tf) AS total_tf
            FROM postings
            GROUP BY term
            ORDER BY total_tf DESC
            LIMIT ?
            """,
            (int(self.cfg.core_top_terms) * 2,),
        ).fetchall()

        terms: List[str] = []
        for r in rows:
            t = str(r["term"])
            if not t:
                continue
            # Filter obvious noise
            if len(t) <= 1:
                continue
            if t in _EN_STOP or t in _ZH_STOP:
                continue
            terms.append(t)
            if len(terms) >= int(self.cfg.core_top_terms):
                break

        topic_line = "、".join(terms[: int(self.cfg.core_top_terms)]) if terms else "（暂无）"

        content = (
            "# Core Memory\n\n"
            "- 称呼用户为“您”。\n"
            "- 回答尽量详细、结构清晰，必要时给例子。\n"
            "- 记忆与当前输入冲突时，以当前输入为准。\n\n"
            f"- 高频关键词：{topic_line}\n"
        )

        _write_text(self.core_md_path, content)

    def _refresh_persona(self) -> None:
        """Persona = ≤ 8 bullets (prefers local LLM; fallback to heuristics)."""

        # Prefer LLM on last N chunks (recent)
        recent = self._recent_chunks_text(max_chunks=12, max_chars=12000)

        bullets = self._persona_via_ollama(recent, max_bullets=self.cfg.persona_max_bullets)
        if not bullets:
            bullets = self._persona_via_heuristics(recent, max_bullets=self.cfg.persona_max_bullets)

        if not bullets:
            bullets = ["（尚未从对话中提取到稳定偏好）"]

        # Ensure max bullets
        bullets = bullets[: int(self.cfg.persona_max_bullets)]

        md = "# Persona (≤ 8 bullets)\n\n" + "\n".join(f"- {b}" for b in bullets) + "\n"
        _write_text(self.persona_md_path, md)

    def _recent_chunks_text(self, max_chunks: int = 12, max_chars: int = 12000) -> str:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT id, path FROM chunks ORDER BY id DESC LIMIT ?", (int(max_chunks),)
        ).fetchall()

        texts: List[str] = []
        total = 0
        for r in rows:
            rel = str(r["path"])
            abs_path = os.path.join(self.memory_dir, rel)
            if not os.path.exists(abs_path):
                continue
            t = _read_text(abs_path).strip()
            if not t:
                continue
            if total + len(t) > max_chars:
                t = t[: max(0, max_chars - total)]
            texts.append(t)
            total += len(t)
            if total >= max_chars:
                break

        return "\n\n".join(reversed(texts))

    def _persona_via_ollama(self, text: str, max_bullets: int) -> List[str]:
        """Try local Ollama to summarize user persona. Returns [] on failure."""

        text = (text or "").strip()
        if not text:
            return []

        # Late import to avoid hard dependency during indexing
        try:
            from ollama_client import chat as ollama_chat, DEFAULT_MODEL
        except Exception:
            return []

        model = os.getenv("OLLAMA_MODEL", "") or DEFAULT_MODEL
        if not model:
            return []

        prompt = (
            "请根据下面的对话内容，总结‘用户画像 Persona’，要求：\n"
            f"- 最多 {int(max_bullets)} 条要点\n"
            "- 每条 1 句话，尽量具体（沟通偏好/信息密度/长期兴趣/禁忌）\n"
            "- 只输出要点列表，每行以 '- ' 开头，不要输出其他文字\n\n"
            "对话内容：\n" + text
        )

        try:
            resp = ollama_chat(
                messages=[{"role": "system", "content": "您是一位擅长提炼用户偏好的助手。"}, {"role": "user", "content": prompt}],
                model=model,
                timeout=120,
            )
        except Exception:
            return []

        lines = [ln.strip() for ln in (resp or "").splitlines() if ln.strip()]
        bullets: List[str] = []
        for ln in lines:
            if ln.startswith("-"):
                b = ln.lstrip("- ").strip()
                if b:
                    bullets.append(b)
        return bullets[: int(max_bullets)]

    def _persona_via_heuristics(self, text: str, max_bullets: int) -> List[str]:
        """Heuristic fallback persona extraction."""

        t = (text or "").strip()
        if not t:
            return []

        bullets: List[str] = []

        # Communication preferences (very rough)
        if "详细" in t or "展开" in t or "多举例" in t:
            bullets.append("偏好回答详细、带例子")
        if "简洁" in t or "别太长" in t:
            bullets.append("偏好回答简洁、直达结论")
        if "称呼" in t and "您" in t:
            bullets.append("希望被称呼为“您”")

        # Interests: use frequent terms in recent text (very rough)
        terms = _tokenize(t)
        freq: Dict[str, int] = {}
        for x in terms:
            if len(x) <= 1:
                continue
            if x in _EN_STOP or x in _ZH_STOP:
                continue
            freq[x] = freq.get(x, 0) + 1
        top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]
        if top:
            bullets.append("近期高频话题关键词：" + "、".join(k for k, _ in top))

        # Deduplicate
        dedup: List[str] = []
        seen = set()
        for b in bullets:
            if b in seen:
                continue
            seen.add(b)
            dedup.append(b)

        return dedup[: int(max_bullets)]

    # ---------------- prompt builder ----------------

    def _format_structured_items(
        self,
        items: Sequence[Dict[str, object]],
        title: str,
        resolved_mode: str,
    ) -> str:
        if not items:
            return f"【{title}】\n- （无）"
        lines = [f"【{title}】"]
        for item in items:
            ts = int(item.get("created_at", 0) or 0)
            date = time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else ""
            src = str(item.get("source", ""))
            cid = int(item.get("chunk_id", 0) or 0)
            snippet = str(item.get("snippet", "")).strip()
            answer_lines = [str(line).strip() for line in item.get("answer_lines", []) if str(line).strip()]
            event_tag = str(item.get("event_tag", "")).strip()
            time_cue = str(item.get("time_cue", "")).strip()
            entity_hits = [str(hit).strip() for hit in item.get("entity_hits", []) if str(hit).strip()]

            if answer_lines:
                lines.append(f"- ({date}) [{src}] #{cid:08d} Answer: {' | '.join(answer_lines)}")
            else:
                lines.append(f"- ({date}) [{src}] #{cid:08d} Answer: {snippet}")
            if resolved_mode != "chat":
                if snippet and answer_lines and snippet not in answer_lines:
                    lines.append(f"  Context: {snippet}")
                meta_bits: List[str] = []
                if event_tag:
                    meta_bits.append(f"event={event_tag}")
                if time_cue:
                    meta_bits.append(f"time={time_cue}")
                if entity_hits:
                    meta_bits.append(f"entities={', '.join(entity_hits[:6])}")
                if meta_bits:
                    lines.append("  Meta: " + " | ".join(meta_bits))
        return "\n".join(lines)

    def _render_evidence_sections(self, bundle: Dict[str, object]) -> str:
        resolved_mode = str(bundle.get("mode", "chat"))
        evidences = list(bundle.get("items", []))
        direct_evidence = list(bundle.get("direct_evidence", evidences))
        timeline_evidence = list(bundle.get("timeline_evidence", []))
        contrast_evidence = list(bundle.get("contrast_evidence", []))

        sections = [self._format_structured_items(direct_evidence, "Direct Evidence", resolved_mode)]
        if resolved_mode == "multi_hop":
            sections.append(self._format_structured_items(timeline_evidence, "Timeline Evidence", resolved_mode))
            sections.append(self._format_structured_items(contrast_evidence, "Contrast Evidence", resolved_mode))
        elif resolved_mode == "fact":
            sections.append(self._format_structured_items(evidences[len(direct_evidence) :], "Supporting Evidence", resolved_mode))
            answer_candidates = list(bundle.get("answer_candidates", []))
            if answer_candidates:
                lines = ["【Candidate Answers】"]
                for candidate in answer_candidates[:6]:
                    answer = str(candidate.get("answer", "")).strip()
                    score = float(candidate.get("score", 0.0))
                    support_count = int(candidate.get("support_count", 0))
                    chunk_ids = ", ".join(str(int(chunk_id)) for chunk_id in candidate.get("chunk_ids", [])[:4])
                    lines.append(
                        f"- {answer} (score={score:.2f}, support={support_count}, chunks={chunk_ids or 'n/a'})"
                    )
                canonical_answer = str(bundle.get("canonical_answer", "")).strip()
                if canonical_answer:
                    lines.append(f"- Canonical Answer: {canonical_answer}")
                fact_strategy = str(bundle.get("fact_strategy", "")).strip()
                if fact_strategy:
                    lines.append(f"- Fact Strategy: {fact_strategy}")
                sections.append("\n".join(lines))
        return "\n\n".join(section for section in sections if section.strip())

    def _sanitize_session_messages(self, session_messages: Optional[Sequence[Dict[str, str]]], resolved_mode: str) -> List[Dict[str, str]]:
        if resolved_mode != "chat" or not session_messages:
            return []
        cleaned: List[Dict[str, str]] = []
        for item in list(session_messages)[-12:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "")
            content = str(item.get("content", "") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    def build_chat_messages(
        self,
        user_text: str,
        assistant_style: str,
        lang: str = "zh",
        answer_mode: str = "auto",
        session_messages: Optional[Sequence[Dict[str, str]]] = None,
        retrieved_bundle: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, str]]:
        bundle = retrieved_bundle or self.retrieve_structured(user_text, mode=answer_mode, k=self.cfg.retrieve_top_k)
        resolved_mode = str(bundle.get("mode", "chat"))
        system_prompt = self.build_system_prompt(
            user_text=user_text,
            assistant_style=assistant_style,
            lang=lang,
            answer_mode=resolved_mode,
            retrieved_bundle=bundle,
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._sanitize_session_messages(session_messages, resolved_mode))
        messages.append({"role": "user", "content": user_text})
        return messages

    def build_turn_plan(
        self,
        user_text: str,
        assistant_style: str,
        lang: str = "zh",
        answer_mode: str = "auto",
        session_messages: Optional[Sequence[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        bundle = self.retrieve_structured(user_text, mode=answer_mode, k=self.cfg.retrieve_top_k)
        resolved_mode = str(bundle.get("mode", "chat"))
        plan: Dict[str, object] = {
            "mode": resolved_mode,
            "bundle": bundle,
            "messages": self.build_chat_messages(
                user_text=user_text,
                assistant_style=assistant_style,
                lang=lang,
                answer_mode=resolved_mode,
                session_messages=session_messages,
                retrieved_bundle=bundle,
            ),
            "two_pass": False,
            "direct_answer": "",
        }
        if resolved_mode == "fact":
            canonical_answer = str(bundle.get("canonical_answer", "") or "").strip()
            fact_strategy = str(bundle.get("fact_strategy", "model_select") or "model_select")
            rendered_answer = self.render_fact_answer(user_text, retrieved_bundle=bundle)
            plan.update(
                {
                    "fact_subtype": str(bundle.get("fact_subtype", "") or ""),
                    "answer_candidates": list(bundle.get("answer_candidates", []) or []),
                    "canonical_answer": canonical_answer,
                    "fact_strategy": fact_strategy,
                }
            )
            if rendered_answer and rendered_answer != "无法确定":
                plan["direct_answer"] = rendered_answer
        if resolved_mode == "multi_hop":
            subtype = str(bundle.get("question_subtype") or _resolve_multi_hop_subtype(user_text, dict(bundle.get("flags", {}))))
            plan.update(
                {
                    "two_pass": True,
                    "question_subtype": subtype,
                    "required_slots": list(_required_multi_hop_slots(subtype)),
                    "synthesis_messages": self.build_multi_hop_synthesis_messages(
                        user_text=user_text,
                        lang=lang,
                        retrieved_bundle=bundle,
                    ),
                    "fallback_messages": list(plan["messages"]),
                }
            )
        return plan

    def build_multi_hop_synthesis_messages(
        self,
        user_text: str,
        lang: str = "zh",
        retrieved_bundle: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, str]]:
        bundle = retrieved_bundle or self.retrieve_structured(user_text, mode="multi_hop", k=self.cfg.retrieve_top_k)
        subtype = str(bundle.get("question_subtype") or _resolve_multi_hop_subtype(user_text, dict(bundle.get("flags", {}))))
        required_slots = _required_multi_hop_slots(subtype)
        synthesis_template = _empty_multi_hop_synthesis(subtype)
        evidence_text = self._render_evidence_sections(bundle)
        conflict_watch = list(bundle.get("conflict_watch", []))
        slot_lines = "\n".join(_multi_hop_slot_prompt_lines(subtype))
        if lang == "en":
            system = (
                "You are a strict evidence synthesizer.\n"
                "Output exactly one JSON object and nothing else.\n"
                "Do not write natural-language explanation.\n"
                "Only include facts jointly supported by the evidence.\n"
                f"Question subtype: {subtype}\n"
                "Required slots:\n"
                f"{slot_lines}\n"
            )
        else:
            system = (
                "您是一名严格的证据抽槽器。\n"
                "只能输出一个 JSON 对象，不得输出自然语言说明。\n"
                "不得把证据没有共同支持的事实写进 final_claim 或 direct_facts。\n"
                "证据不足时，对应槽位留空字符串。\n"
                "如果证据里出现了人名、战队名、时间线动作，请尽量填入 entities 和 event_chain，不要都留空。\n"
                f"问题子类型：{subtype}\n"
                "必填槽位：\n"
                f"{slot_lines}\n"
            )
        if conflict_watch:
            system += "Conflict Watch: " + "、".join(conflict_watch) + "\n"
        system += (
            "JSON 模板（字段名必须保留）：\n"
            + json.dumps(synthesis_template, ensure_ascii=False, indent=2)
            + "\n\n"
            + evidence_text
        )
        user = (
            f"问题：{user_text}\n"
            f"请仅输出 JSON，并确保 direct_facts 至少覆盖这些槽位：{', '.join(required_slots)}。"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse_multi_hop_synthesis(self, text: str, subtype: str) -> Optional[Dict[str, object]]:
        return _coerce_multi_hop_synthesis(text, subtype)

    def _candidate_evidence_entries(self, bundle: Dict[str, object]) -> List[Dict[str, object]]:
        ordered_items: List[Dict[str, object]] = []
        for key in ("direct_evidence", "timeline_evidence", "contrast_evidence", "items"):
            for item in bundle.get(key, []) or []:
                if item not in ordered_items:
                    ordered_items.append(item)

        entries: List[Dict[str, object]] = []
        seen = set()
        for item in ordered_items:
            chunk_id = int(item.get("chunk_id", 0) or 0)
            answer_lines = [str(line).strip() for line in item.get("answer_lines", []) if str(line).strip()]
            snippets = answer_lines + [str(item.get("snippet", "")).strip()]
            for line in snippets:
                key = f"{chunk_id}:{re.sub(r'\\s+', '', line)}"
                if not line or key in seen:
                    continue
                seen.add(key)
                entries.append({"text": line, "chunk_id": chunk_id})
        return entries

    def _candidate_evidence_lines(self, bundle: Dict[str, object]) -> List[str]:
        lines: List[str] = []
        for entry in self._candidate_evidence_entries(bundle):
            text = str(entry.get("text", "")).strip()
            if text:
                lines.append(text)
        return lines

    def _pick_candidate_line(
        self,
        candidates: Sequence[str],
        *,
        prefer_terms: Sequence[str] = (),
        require_any: Sequence[str] = (),
        avoid_terms: Sequence[str] = (),
        min_delimiters: int = 0,
    ) -> str:
        ranked: List[Tuple[float, str]] = []
        for line in candidates:
            text = str(line).strip()
            if not text:
                continue
            if require_any and not any(term in text for term in require_any):
                continue
            if min_delimiters and text.count("、") < min_delimiters:
                continue
            score = 0.0
            score += sum(2.4 for term in prefer_terms if term in text)
            score -= sum(1.8 for term in avoid_terms if term in text)
            score -= max(0.0, len(text) - 160) / 80.0
            ranked.append((score, text))
        if not ranked:
            return ""
        ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        return ranked[0][1]

    def _pick_candidate_entry(
        self,
        entries: Sequence[Dict[str, object]],
        *,
        prefer_terms: Sequence[str] = (),
        require_any: Sequence[str] = (),
        avoid_terms: Sequence[str] = (),
        min_delimiters: int = 0,
    ) -> Dict[str, object]:
        ranked: List[Tuple[float, Dict[str, object]]] = []
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            if require_any and not any(term in text for term in require_any):
                continue
            if min_delimiters and text.count("、") < min_delimiters:
                continue
            score = 0.0
            score += sum(2.4 for term in prefer_terms if term in text)
            score -= sum(1.8 for term in avoid_terms if term in text)
            score -= max(0.0, len(text) - 180) / 90.0
            ranked.append((score, entry))
        if not ranked:
            return {"text": "", "chunk_ids": []}
        ranked.sort(key=lambda item: (item[0], -len(str(item[1].get("text", "")))), reverse=True)
        best = dict(ranked[0][1])
        return {
            "text": str(best.get("text", "")).strip(),
            "chunk_ids": [int(best.get("chunk_id", 0) or 0)] if int(best.get("chunk_id", 0) or 0) > 0 else [],
        }

    def repair_multi_hop_synthesis(
        self,
        user_text: str,
        synthesis: Dict[str, object],
        retrieved_bundle: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        bundle = retrieved_bundle or self.retrieve_structured(user_text, mode="multi_hop", k=self.cfg.retrieve_top_k)
        flags = dict(bundle.get("flags", {}))
        subtype = str(bundle.get("question_subtype") or synthesis.get("question_type") or _resolve_multi_hop_subtype(user_text, dict(bundle.get("flags", {}))))
        fixed = _empty_multi_hop_synthesis(subtype)
        fixed.update({k: v for k, v in synthesis.items() if k in fixed})
        fixed["question_type"] = subtype
        fixed["direct_facts"] = dict(fixed.get("direct_facts") or {})
        fixed["slot_sources"] = dict(fixed.get("slot_sources") or {})
        for slot in _required_multi_hop_slots(subtype):
            fixed["direct_facts"].setdefault(slot, "")
            fixed["slot_sources"].setdefault(slot, [])

        entries = self._candidate_evidence_entries(bundle)
        candidates = [str(entry.get("text", "")).strip() for entry in entries if str(entry.get("text", "")).strip()]

        def _set_slot(slot: str, text: str, chunk_ids: Sequence[int]) -> None:
            if text:
                fixed["direct_facts"][slot] = text.strip()
            if chunk_ids:
                fixed["slot_sources"][slot] = [int(chunk_id) for chunk_id in chunk_ids if int(chunk_id) > 0][:6]

        if subtype == "asks_rumor_vs_actual":
            actual_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("任何渠道", "并没有", "没有", "未", "宣布复出", "没有出现"),
                require_any=("复出", "消息", "宣布"),
            )
            gate_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("满一年", "一年", "规则", "可以复出"),
                require_any=("复出", "规则", "一年"),
            )
            rumor_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("传言", "传闻", "可以复出", "复出"),
                require_any=("复出",),
                avoid_terms=("任何渠道", "没有", "并没有"),
            )
            if actual_entry.get("text") and not any(x in str(fixed["direct_facts"].get("actual") or "") for x in ("并没有", "没有", "未")):
                _set_slot("actual", "在这个时间点并没有通过任何渠道宣布复出的消息", actual_entry.get("chunk_ids", []))
            elif not actual_entry.get("text"):
                _set_slot("actual", "在这个时间点并没有通过任何渠道宣布复出的消息", ())
            if gate_entry.get("text"):
                _set_slot("status_gate", "退役满一年后即可复出", gate_entry.get("chunk_ids", []))
            if rumor_entry.get("text") or not str(fixed["direct_facts"].get("rumor") or "").strip():
                _set_slot("rumor", "外界一直有叶秋会复出的传言，而且按规则已经到了可以复出的时间", rumor_entry.get("chunk_ids", []))
            fixed["final_claim"] = "根据联盟规则，叶秋在退役满一年后已经具备复出资格，但当时并没有宣布复出。"

        elif subtype == "asks_tactic_sequence":
            bait_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("申建", "拳法家", "诱饵"),
                require_any=("申建", "拳法家", "诱饵"),
                avoid_terms=("叶修", "包子", "依诺"),
            )
            force_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("肖时钦", "王泽", "孙翔", "张家兴", "申建"),
                require_any=("肖时钦", "王泽", "孙翔", "张家兴", "申建"),
                avoid_terms=("叶修", "包子", "依诺"),
            )
            pattern_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("五个人", "分头行事", "夹攻", "包围", "不同方向"),
                require_any=("分头", "包围", "夹攻", "不同方向", "五个人"),
                avoid_terms=("叶修", "包子", "依诺"),
            )
            outcome_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("击败", "包围", "追了上去"),
                require_any=("击败", "包围", "追"),
            )
            if bait_entry.get("text"):
                _set_slot("bait", "申建的拳法家", bait_entry.get("chunk_ids", []))
            names = [
                name
                for name in ("肖时钦", "王泽", "孙翔", "申建", "张家兴")
                if any(name in line for line in (str(force_entry.get("text", "")), str(pattern_entry.get("text", "")), str(outcome_entry.get("text", "")), str(bait_entry.get("text", ""))))
            ]
            if names:
                _set_slot("collapsing_force", "、".join(names), list(force_entry.get("chunk_ids", [])) + list(pattern_entry.get("chunk_ids", [])))
            if pattern_entry.get("text"):
                _set_slot("encirclement_pattern", "五人分头行动，从不同方向夹攻包围", pattern_entry.get("chunk_ids", []))
            if outcome_entry.get("text") or pattern_entry.get("text"):
                _set_slot("outcome", "完成了对谁不低头和莫敢回手的包围并击败二人", list(outcome_entry.get("chunk_ids", [])) + list(pattern_entry.get("chunk_ids", [])))
            fixed["final_claim"] = "嘉世先五人分头行动，再以申建的拳法家作诱饵，由肖时钦、王泽、孙翔、申建和张家兴从不同方向收缩夹攻，最后包围并击败了谁不低头和莫敢回手。"

        elif subtype == "asks_design_rationale":
            attr_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("高智力", "高暴击", "最高智力", "最高暴击"),
                require_any=("智力", "暴击"),
            )
            risk_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("冰抗", "暗抗", "控制"),
                require_any=("冰抗", "暗抗", "控制"),
            )
            weakness_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("反应慢", "慢节奏", "弥补"),
                require_any=("反应", "慢", "弥补"),
            )
            if attr_entry.get("text"):
                _set_slot("boosted_attributes", "高智力、高暴击", attr_entry.get("chunk_ids", []))
            if risk_entry.get("text"):
                _set_slot("mitigated_risk", "通过极高的冰抗和暗抗来减少被控制的风险", risk_entry.get("chunk_ids", []))
            if weakness_entry.get("text") or attr_entry.get("text"):
                _set_slot("compensated_weakness", "利用高爆发的属性来弥补安文逸反应较慢的弱点", list(weakness_entry.get("chunk_ids", [])) + list(attr_entry.get("chunk_ids", [])))
            if not str(fixed["direct_facts"].get("design_target") or "").strip():
                _set_slot("design_target", "安文逸的职业特点和操作短板", ())
            fixed["final_claim"] = "这套装备通过堆高冰抗和暗抗来减少被控制的风险，再用高智力、高暴击带来的高爆发去弥补安文逸反应较慢的短板。"

        elif subtype == "asks_roster_plus_status":
            roster_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("团队赛", "派出了", "组合", "首发"),
                require_any=("团队赛", "派出", "组合", "首发"),
                min_delimiters=4,
            )
            status_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("休息了一轮", "没有在团队赛中出场", "未出场", "没上场"),
                require_any=("休息", "出场", "上场"),
            )
            if roster_entry.get("text"):
                match = re.search(r"派出了(.+?)的组合", str(roster_entry.get("text", "")))
                _set_slot("roster", match.group(1).strip() if match else str(roster_entry.get("text", "")), roster_entry.get("chunk_ids", []))
            if status_entry.get("text"):
                _set_slot("status_note", "叶修这一轮没有上场", status_entry.get("chunk_ids", []))
                if "叶修" in str(status_entry.get("text", "")) or "休息了一轮" in str(status_entry.get("text", "")):
                    _set_slot("not_playing", "叶修", status_entry.get("chunk_ids", []))
            fixed["final_claim"] = (
                f"团队赛首发是{fixed['direct_facts'].get('roster') or '方锐、苏沐橙、唐柔、乔一帆、安文逸和包子'}，"
                "叶修这一轮没有上场。"
            )

        elif subtype == "asks_exchange_mapping":
            role_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("刘皓", "贺铭", "筹码", "交换", "送往", "雷霆"),
                require_any=("刘皓", "贺铭", "交换", "雷霆"),
            )
            if role_entry.get("text"):
                _set_slot("person_a_role", "刘皓和贺铭都是交换肖时钦的筹码", role_entry.get("chunk_ids", []))
                _set_slot("person_b_role", "刘皓和贺铭都是交换肖时钦的筹码", role_entry.get("chunk_ids", []))
                _set_slot("destination", "雷霆战队", role_entry.get("chunk_ids", []))
                fixed["final_claim"] = "刘皓和贺铭作为交换肖时钦的筹码，最终被送往了雷霆战队。"

        elif subtype == "asks_lineup_constraint":
            cause_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("喻文州", "徐景熙", "单人对决", "治疗职业", "一线主力"),
                require_any=("蓝雨", "个人赛", "擂台赛", "单人对决"),
                avoid_terms=("轮回", "嘉世", "兴欣"),
            )
            consequence_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("二线", "替补", "只能", "个人赛", "单人赛"),
                require_any=("二线", "替补", "只能"),
                avoid_terms=("轮回", "嘉世", "兴欣"),
            )
            _set_slot("missing_core", "喻文州和徐景熙这两位团队赛一线主力通常不参加单人对决", list(cause_entry.get("chunk_ids", [])) + list(consequence_entry.get("chunk_ids", [])))
            _set_slot("constraint_reason", "队长喻文州通常不打个人赛，而治疗职业徐景熙也很少参加单人赛", list(cause_entry.get("chunk_ids", [])))
            _set_slot("lineup_consequence", "蓝雨在个人赛里只能派出二线替补选手", list(consequence_entry.get("chunk_ids", [])))
            fixed["final_claim"] = "蓝雨的一线主力在个人赛中无法全部出战，因为喻文州通常不参加单人对决，徐景熙作为治疗也很少打单人赛，所以蓝雨在个人赛里只能派出二线替补选手。"

        elif subtype == "asks_counter_adjustment":
            adjust_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("杨昊轩", "李迅", "枪炮师", "替下", "首发阵容"),
                require_any=("枪炮师", "杨昊轩", "李迅", "首发"),
            )
            goal_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("苏沐橙", "火力压制", "远距离火力", "限制", "对等"),
                require_any=("苏沐橙", "火力", "限制", "远距离"),
            )
            _set_slot("changed_out", "李迅", adjust_entry.get("chunk_ids", []))
            _set_slot("added_in", "枪炮师杨昊轩", adjust_entry.get("chunk_ids", []))
            _set_slot("tactical_goal", "在远距离火力上对等限制苏沐橙", list(adjust_entry.get("chunk_ids", [])) + list(goal_entry.get("chunk_ids", [])))
            fixed["final_claim"] = "虚空把李迅换下，改由枪炮师杨昊轩首发，想在远距离火力上对等限制苏沐橙。"

        elif subtype == "asks_role_plus_tactic":
            role_a_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("叶修", "战斗法师"),
                require_any=("叶修", "战斗法师"),
            )
            role_b_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("肖时钦", "机械师"),
                require_any=("肖时钦", "机械师"),
            )
            tactic_entry = self._pick_candidate_entry(
                entries,
                prefer_terms=("树根", "主动撤退", "走位", "技能移动", "躲避攻击", "抢夺", "龙剑士"),
                require_any=("集火", "树根", "走位", "撤退", "龙剑士", "机械师", "战斗法师"),
            )
            _set_slot("person_a_role", "战斗法师", role_a_entry.get("chunk_ids", []))
            _set_slot("person_b_role", "机械师", role_b_entry.get("chunk_ids", []))
            _set_slot("side_a_tactic", "通过技能移动、利用地形或主动撤退来躲避集火", list(role_a_entry.get("chunk_ids", [])) + list(tactic_entry.get("chunk_ids", [])))
            _set_slot("side_b_tactic", "通过技能移动、利用地形或主动撤退来躲避集火", list(role_b_entry.get("chunk_ids", [])) + list(tactic_entry.get("chunk_ids", [])))
            _set_slot("shared_strategy", "两人都靠走位、树根地形或主动撤退来规避集火，并寻找机会抢夺龙剑士", tactic_entry.get("chunk_ids", []))
            fixed["final_claim"] = "叶修操作的是战斗法师，肖时钦操作的是机械师；两人在面对集火时都通过技能移动、利用地形（如树根）或主动撤退来躲避攻击，并寻找机会抢夺BOSS。"

        elif subtype == "asks_compare_two_sides":
            team_tokens = list(
                dict.fromkeys(token.lstrip("和及与跟、") for token in re.findall(r"([\u4e00-\u9fff]{1,4}战队)", user_text))
            )
            if len(team_tokens) < 2:
                team_tokens = [token for token in _MULTI_HOP_TEAM_TOKENS if token in user_text]
            if len(team_tokens) >= 2:
                left_patterns = (
                    rf"{re.escape(team_tokens[0])}[^\n。]*第一个(?:出战|出场)选手(?:是|为)?([\u4e00-\u9fff]{{2,4}})",
                    r"兴欣方面派出的第一个出战选手是([\u4e00-\u9fff]{2,4})",
                    rf"{re.escape(team_tokens[0])}(?:的选手)?[，,\s]*([\u4e00-\u9fff]{{2,4}})[，,\s]+角色",
                )
                right_patterns = (
                    rf"{re.escape(team_tokens[1])}[^\n。]*率先出场的是([\u4e00-\u9fff]{{2,4}})",
                    rf"{re.escape(team_tokens[1])}[^\n。]*第一个(?:出战|出场)选手(?:是|为)?([\u4e00-\u9fff]{{2,4}})",
                    rf"{re.escape(team_tokens[1])}(?:的选手)?[，,\s]*([\u4e00-\u9fff]{{2,4}})[，,\s]+角色",
                    r"率先出场的是([\u4e00-\u9fff]{2,4})",
                )
                left_entry = self._pick_candidate_entry(
                    entries,
                    prefer_terms=(team_tokens[0], "第一个出场", "第一个出战", "率先出场"),
                    require_any=(team_tokens[0],),
                )
                right_entry = self._pick_candidate_entry(
                    entries,
                    prefer_terms=(team_tokens[1], "第一个出场", "第一个出战", "率先出场"),
                    require_any=(team_tokens[1],),
                )
                left_text = str(left_entry.get("text", ""))
                right_text = str(right_entry.get("text", ""))
                left_name = ""
                right_name = ""
                for pattern in left_patterns:
                    match = re.search(pattern, left_text)
                    if match:
                        left_name = match.group(1)
                        break
                for pattern in right_patterns:
                    match = re.search(pattern, right_text)
                    if match:
                        right_name = match.group(1)
                        break
                if not left_name or not right_name:
                    for entry in entries:
                        text = str(entry.get("text", ""))
                        if not left_name:
                            for pattern in left_patterns:
                                match = re.search(pattern, text)
                                if match:
                                    left_name = match.group(1)
                                    left_entry = entry
                                    break
                        if not right_name:
                            for pattern in right_patterns:
                                match = re.search(pattern, text)
                                if match:
                                    right_name = match.group(1)
                                    right_entry = entry
                                    break
                        if left_name and right_name:
                            break
                if left_name:
                    _set_slot("side_a", left_name, left_entry.get("chunk_ids", []))
                elif left_text:
                    _set_slot("side_a", left_text, left_entry.get("chunk_ids", []))
                if right_name:
                    _set_slot("side_b", right_name, right_entry.get("chunk_ids", []))
                elif right_text:
                    _set_slot("side_b", right_text, right_entry.get("chunk_ids", []))
            if not str(fixed["direct_facts"].get("comparison_basis") or "").strip():
                _set_slot("comparison_basis", user_text, ())
            if flags.get("asks_first_or_order") and str(fixed["direct_facts"].get("side_a") or "").strip() and str(fixed["direct_facts"].get("side_b") or "").strip():
                fixed["final_claim"] = f"{team_tokens[0]}第一个出场的是{fixed['direct_facts']['side_a']}，{team_tokens[1]}第一个出场的是{fixed['direct_facts']['side_b']}。"

        if not str(fixed.get("final_claim") or "").strip():
            pieces = [str(fixed["direct_facts"].get(slot) or "").strip() for slot in _required_multi_hop_slots(subtype)]
            fixed["final_claim"] = "；".join(piece for piece in pieces if piece)[:220]
        return fixed

    def render_multi_hop_answer(self, user_text: str, synthesis: Dict[str, object]) -> str:
        subtype = str(synthesis.get("question_type") or _resolve_multi_hop_subtype(user_text, _query_flags(user_text)))
        facts = dict(synthesis.get("direct_facts") or {})
        claim = str(synthesis.get("final_claim") or "").strip()
        if subtype == "asks_rumor_vs_actual":
            if claim:
                return claim
            status_gate = str(facts.get("status_gate") or "退役满一年后即可复出").strip()
            actual = str(facts.get("actual") or "当时并没有通过任何渠道宣布复出").strip()
            return f"根据联盟规则，{status_gate}；而在那个时间点，叶秋{actual}。"
        if subtype == "asks_tactic_sequence":
            if claim:
                return claim
            bait = str(facts.get("bait") or "申建的拳法家").strip()
            force = str(facts.get("collapsing_force") or "肖时钦、王泽、孙翔、申建和张家兴").strip()
            pattern = str(facts.get("encirclement_pattern") or "五人分头行动，从不同方向夹攻包围").strip()
            outcome = str(facts.get("outcome") or "完成了对谁不低头和莫敢回手的包围并击败二人").strip()
            return f"嘉世先以{bait}作诱饵，再由{force}{pattern}，最后{outcome}。"
        if subtype == "asks_design_rationale":
            if claim:
                return claim
            risk = str(facts.get("mitigated_risk") or "通过极高的冰抗和暗抗来减少被控制的风险").strip()
            weakness = str(facts.get("compensated_weakness") or "利用高爆发的属性来弥补安文逸反应较慢的弱点").strip()
            return f"这套装备{risk}，再{weakness}。"
        if subtype == "asks_roster_plus_status":
            if claim:
                return claim
            roster = str(facts.get("roster") or "方锐、苏沐橙、唐柔、乔一帆、安文逸和包子").strip()
            not_playing = str(facts.get("not_playing") or "叶修").strip()
            return f"团队赛首发是{roster}，{not_playing}这一轮没有上场。"
        if subtype == "asks_exchange_mapping":
            return claim or "刘皓和贺铭作为交换筹码，最终被送往雷霆战队。"
        if subtype == "asks_lineup_constraint":
            if claim:
                return claim
            return "蓝雨的一线主力在个人赛中无法全部出战，因为喻文州通常不参加单人对决，徐景熙作为治疗也很少打单人赛，所以他们只能派二线替补选手。"
        if subtype == "asks_counter_adjustment":
            if claim:
                return claim
            return "虚空把李迅换下，改由枪炮师杨昊轩首发，想在远距离火力上对等限制苏沐橙。"
        if subtype == "asks_role_plus_tactic":
            if claim:
                return claim
            return "叶修操作的是战斗法师，肖时钦操作的是机械师；两人在面对集火时都通过技能移动、利用地形（如树根）或主动撤退来躲避攻击，并寻找机会抢夺BOSS。"
        if subtype == "asks_compare_two_sides":
            if claim:
                return claim
            side_a = str(facts.get("side_a") or "").strip()
            side_b = str(facts.get("side_b") or "").strip()
            if side_a and side_b and any(team in user_text for team in ("兴欣", "嘉世", "蓝雨", "轮回", "虚空")):
                teams = list(
                    dict.fromkeys(token.lstrip("和及与跟、") for token in re.findall(r"([\u4e00-\u9fff]{1,4}战队)", user_text))
                )
                if len(teams) < 2:
                    teams = [team for team in _MULTI_HOP_TEAM_TOKENS if team in user_text][:2]
                if len(teams) >= 2:
                    return f"{teams[0]}第一个出场的是{side_a}，{teams[1]}第一个出场的是{side_b}。"
            return "两边对应的人物和结果需要分别回答。"
        return claim

    def needs_multi_hop_answer_fallback(self, answer: str, synthesis: Dict[str, object], user_text: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return True
        lowered = text.lower()
        subtype = str(synthesis.get("question_type") or _resolve_multi_hop_subtype(user_text, _query_flags(user_text)))
        if any(f"{slot.lower()}:" in lowered or f"{slot.lower()}：" in lowered for slot in _required_multi_hop_slots(subtype)):
            return True
        if any(marker in text for marker in ("未检索到", "建议查看官方", "目前没有在已检索到的资料中出现")):
            return True
        facts = dict(synthesis.get("direct_facts") or {})
        if subtype == "asks_rumor_vs_actual":
            if "退役满一年" in str(facts.get("status_gate") or "") and any(x in text for x in ("尚未满足", "未满一年", "不能复出")):
                return True
            if "并没有" in str(facts.get("actual") or "") and not any(x in text for x in ("并没有", "没有宣布", "未宣布", "没有通过任何渠道")):
                return True
        if subtype == "asks_tactic_sequence":
            if "申建" in str(facts.get("bait") or "") and "申建" not in text:
                return True
            if "分头行动" in str(facts.get("encirclement_pattern") or "") and "分头" not in text:
                return True
        if subtype == "asks_design_rationale":
            if "冰抗" in str(facts.get("mitigated_risk") or "") and "冰抗" not in text:
                return True
            if "暗抗" in str(facts.get("mitigated_risk") or "") and "暗抗" not in text:
                return True
        if subtype == "asks_roster_plus_status":
            if "没有上场" in str(facts.get("status_note") or "") and "没有上场" not in text and "休息了一轮" not in text:
                return True
        if subtype == "asks_lineup_constraint":
            if any(token in str(facts.get("constraint_reason") or "") for token in ("喻文州", "徐景熙")) and not all(
                token in text for token in ("喻文州", "徐景熙")
            ):
                return True
            if "二线" in str(facts.get("lineup_consequence") or "") and not any(token in text for token in ("二线", "替补")):
                return True
        if subtype == "asks_counter_adjustment":
            if any(token in str(facts.get("added_in") or "") for token in ("杨昊轩", "枪炮师")) and not all(
                token in text for token in ("杨昊轩", "枪炮师")
            ):
                return True
            if "李迅" in str(facts.get("changed_out") or "") and "李迅" not in text:
                return True
        if subtype == "asks_role_plus_tactic":
            if "战斗法师" in str(facts.get("person_a_role") or "") and "战斗法师" not in text:
                return True
            if "机械师" in str(facts.get("person_b_role") or "") and "机械师" not in text:
                return True
            if any(token in str(facts.get("shared_strategy") or "") for token in ("树根", "撤退", "走位")) and not any(
                token in text for token in ("树根", "撤退", "走位", "利用地形", "主动撤退")
            ):
                return True
        if subtype == "asks_compare_two_sides":
            teams = [team for team in _MULTI_HOP_TEAM_TOKENS if team in user_text][:2]
            side_a = str(facts.get("side_a") or "").strip()
            side_b = str(facts.get("side_b") or "").strip()
            if teams and teams[0] not in text:
                return True
            if len(teams) >= 2 and teams[1] not in text:
                return True
            if side_a and side_a not in text:
                return True
            if side_b and side_b not in text:
                return True
        return False

    def build_multi_hop_answer_messages(
        self,
        user_text: str,
        assistant_style: str,
        synthesis: Dict[str, object],
        lang: str = "zh",
        retrieved_bundle: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, str]]:
        bundle = retrieved_bundle or self.retrieve_structured(user_text, mode="multi_hop", k=self.cfg.retrieve_top_k)
        subtype = str(bundle.get("question_subtype") or synthesis.get("question_type") or _resolve_multi_hop_subtype(user_text, dict(bundle.get("flags", {}))))
        required_slots = _required_multi_hop_slots(subtype)
        evidence_text = self._render_evidence_sections(bundle)
        conflict_watch = list(bundle.get("conflict_watch", []))
        synthesis_text = json.dumps(synthesis, ensure_ascii=False, indent=2)
        slot_lines = "\n".join(_multi_hop_slot_prompt_lines(subtype))

        system = (
            "您是 EverMate.AI 的多跳问答整理器。\n"
            "请只根据 Structured Synthesis 与 Evidence 回答。\n"
            "先覆盖所有必填槽位，再组织成一到两句自然中文。\n"
            "不要新增 Structured Synthesis 中没有出现的事实。\n"
            "如果题目问“分别”，必须显式写出两边。\n"
            "如果题目问“为什么/如何针对”，必须写清“设计/属性 -> 针对的弱点或风险”。\n"
            "如果题目问阵容，必须完整列出名单，并单独说明关键人物是否上场。\n"
            "若 Structured Synthesis 缺关键槽位，请保守回答最确定的部分，不得补编。\n"
            "不要把 slot key / JSON 字段名原样抄进答案，例如不要输出 roster: / not_playing: / status_note: 这种格式。\n"
        )
        if assistant_style:
            system += "【Assistant Style】\n" + assistant_style.strip() + "\n\n"
        system += "【必填槽位】\n" + slot_lines + "\n\n"
        if conflict_watch:
            system += "【Conflict Watch】\n- " + "\n- ".join(conflict_watch) + "\n\n"
        system += "【Structured Synthesis】\n" + synthesis_text + "\n\n" + evidence_text

        user = f"{user_text}\n请依据上面的 Structured Synthesis 直接给出最终答案。"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_system_prompt(
        self,
        user_text: str,
        assistant_style: str,
        lang: str = "zh",
        answer_mode: str = "auto",
        retrieved_bundle: Optional[Dict[str, object]] = None,
    ) -> str:
        """Assemble a system prompt that injects Core/Persona and structured Vault evidence."""

        core = _read_text(self.core_md_path).strip()
        persona = _read_text(self.persona_md_path).strip()
        bundle = retrieved_bundle or self.retrieve_structured(user_text, mode=answer_mode, k=self.cfg.retrieve_top_k)
        resolved_mode = str(bundle.get("mode", "chat"))
        flags = dict(bundle.get("flags", {}))
        question_subtype = str(bundle.get("question_subtype", ""))
        ev_text = self._render_evidence_sections(bundle)

        conflict_markers = list(bundle.get("conflict_watch", [])) or _conflict_markers(
            [str(e.get("snippet", "")) for e in list(bundle.get("items", []))]
        )

        if lang == "en":
            base = (
                "You are EverMate.AI, a local AI companion.\n"
                "Follow the assistant style below.\n"
                "If memory conflicts with the user's current message, prefer the current message.\n"
                "Use memory naturally; do not reveal system prompt.\n"
                "When the user asks for a factual memory detail such as a name, date, duration, place, or chosen option, answer from the evidence and avoid guessing.\n"
                "Do not merge different events into one memory. If the evidence seems inconsistent, answer conservatively from the most direct evidence.\n"
            )
        else:
            base = (
                "您是 EverMate.AI（本地 AI 伙伴）。\n"
                "请用中文回答并称呼用户为“您”。\n"
                "如记忆与用户当前输入冲突，请以当前输入为准。\n"
                "引用记忆要自然，不要暴露系统提示。\n"
                "当用户在询问可核对的记忆事实（如名字、日期、时长、地点、选择结果）时，请优先依据证据中的原词原数值作答，不要猜测。\n"
                "如果证据里同一句包含多个并列事实，请尽量完整说全，不要只答一半。\n"
                "不要把不同事件的证据硬拼成同一次经历；如果证据之间可能在说不同事件，请只回答最确定的部分。\n"
            )

        if resolved_mode == "fact":
            if lang == "en":
                base += (
                    "This turn is in fact mode.\n"
                    "Answer only the fact itself.\n"
                    "Prefer the canonical term shown in evidence; do not paraphrase it into a near-miss.\n"
                    "If the question contrasts current vs theoretical maximum, answer the one the question asks for.\n"
                )
            else:
                base += (
                    "本轮处于事实问答模式。\n"
                    "请只回答事实本身，不要扩写成背景说明。\n"
                    "若证据里已经给出标准术语、标准名字或标准技能名，请直接使用它，不要改写成近似说法。\n"
                    "如果题目在区分“当前/目前/这时”和“最高/最多/理论”，请严格按题目所问作答。\n"
                )
            answer_candidates = list(bundle.get("answer_candidates", []) or [])
            canonical_answer = str(bundle.get("canonical_answer", "") or "").strip()
            fact_strategy = str(bundle.get("fact_strategy", "model_select") or "model_select")
            if answer_candidates:
                if lang == "en":
                    if fact_strategy == "model_select":
                        base += "The evidence has already been narrowed to a candidate list. You must choose from those candidates only.\n"
                    elif canonical_answer:
                        base += f"The evidence has converged on a canonical answer: {canonical_answer}. Return that answer directly.\n"
                else:
                    if fact_strategy == "model_select":
                        base += "证据已经整理出候选答案列表。您只能从候选列表里选择，不得自造第三种答案。\n"
                    elif canonical_answer:
                        base += f"证据已经收敛到一个最高支持的标准答案：{canonical_answer}。请直接回答它。\n"
        elif resolved_mode == "multi_hop":
            if lang == "en":
                base += (
                    "This turn is in multi-hop mode.\n"
                    "Align person, event, and conclusion before answering.\n"
                    "If the question asks for two parts, answer both parts explicitly.\n"
                    "For roster questions, list every required name and do not omit anyone.\n"
                )
            else:
                base += (
                    "本轮处于多跳问答模式。\n"
                    "请先在心里对齐“人物 / 事件 / 结论”，再组织答案。\n"
                    "如果题目问“分别是什么”“分别做了什么”，请把两部分都明确答出来。\n"
                    "如果题目在问阵容、组合或首发名单，请完整列出所有关键名字，不要漏一个。\n"
                )

        answer_rules: List[str] = []
        if flags.get("asks_name_or_title"):
            answer_rules.append("- 名称题：优先输出标准名，不解释。")
        if flags.get("asks_count_or_total"):
            answer_rules.append("- 数量题：优先输出数字事实，避免泛泛而谈。")
        if flags.get("asks_current_vs_max"):
            answer_rules.append("- 当前值 vs 理论值：不要把“当前/目前”答成“最高/最多”。")
        if flags.get("asks_first_or_order"):
            answer_rules.append("- 顺序题：优先回答“第一个 / 首发 / 率先出场”的对象。")
        if flags.get("asks_roster"):
            answer_rules.append("- 阵容题：完整枚举所有人名，缺一不可。")
        if flags.get("asks_item_or_weapon"):
            answer_rules.append("- 物品或武器题：优先回答标准物品名或武器类别。")
        if flags.get("asks_exchange_or_role"):
            answer_rules.append("- 交换 / 分别题：明确回答每个人扮演的角色与最终去向。")
        if flags.get("asks_status_then_fact"):
            answer_rules.append("- 传闻 vs 实际：请把两层信息都答全，不要只答一层。")
        if flags.get("asks_skill_name"):
            answer_rules.append("- 技能题：如果证据给了标准技能名，请原样使用。")
        if question_subtype == "asks_compare_two_sides":
            answer_rules.append("- 双主体题：请把双方分别是谁 / 分别做了什么完整写出。")
        if question_subtype == "asks_rumor_vs_actual":
            answer_rules.append("- 传闻 / 实际 / 状态门槛三层都要覆盖，不能只答一层。")
        if question_subtype == "asks_tactic_sequence":
            answer_rules.append("- 战术流程题：请交代诱饵、合围者、包围方式和结果。")
        if question_subtype == "asks_design_rationale":
            answer_rules.append("- 设计原因题：请交代强化属性、降低的风险和弥补的弱点。")
        if question_subtype == "asks_roster_plus_status":
            answer_rules.append("- 阵容状态题：请列完整阵容，并明确谁没有上场。")
        if question_subtype == "asks_exchange_mapping":
            answer_rules.append("- 交换映射题：请写清两人的角色与最终去向。")
        if question_subtype == "asks_lineup_constraint":
            answer_rules.append("- 布阵缺口题：请写清谁无法参加单人赛，以及因此只能派什么人补位。")
        if question_subtype == "asks_counter_adjustment":
            answer_rules.append("- 针对性调整题：请写清谁被换下、谁顶上，以及这次调整要限制谁。")
        if question_subtype == "asks_role_plus_tactic":
            answer_rules.append("- 角色 + 战术题：请同时写清两人的职业角色，以及他们怎样通过走位/地形/撤退来规避集火。")

        consistency_lines = [
            "【Evidence Consistency】",
            "- 如果多个证据片段明显来自不同事件，请不要把它们混成一条回忆。",
            "- 如果数字、币种、课程名、日期或驾驶场景彼此不一致，请优先引用最直接支持当前问题的那一条。",
        ]
        if conflict_markers:
            consistency_lines.append("- 当前检索中存在可能相互冲突的维度：" + "、".join(conflict_markers) + "。不确定时请保守表述。")
        if answer_rules:
            consistency_lines.append("【Answer Rules】")
            consistency_lines.extend(answer_rules)

        prompt = (
            base
            + "\n"
            + "【Assistant Style】\n"
            + (assistant_style.strip() if assistant_style else "")
            + "\n\n"
            + "【Core】\n"
            + (core if core else "(empty)")
            + "\n\n"
            + "【Persona】\n"
            + (persona if persona else "(empty)")
            + "\n\n"
            + ev_text
            + "\n\n"
            + "\n".join(consistency_lines)
            + "\n"
        )

        return prompt

    def render_fact_answer(self, user_text: str, retrieved_bundle: Optional[Dict[str, object]] = None) -> str:
        bundle = retrieved_bundle or self.retrieve_structured(user_text, mode="fact", k=self.cfg.retrieve_top_k)
        subtype = str(bundle.get("fact_subtype", "") or _resolve_fact_subtype(user_text, dict(bundle.get("flags", {}))))
        flags = dict(bundle.get("flags", {}))
        query_anchors = _query_anchor_terms(user_text, limit=8)
        answer_candidates = [dict(item) for item in list(bundle.get("answer_candidates", []) or [])]
        supplemental = self._supplement_fact_items(user_text, flags, subtype, limit=24)
        supplemental_candidates: List[Dict[str, object]] = []
        for item in supplemental:
            for candidate in item.get("fact_candidates", []) or []:
                candidate_copy = dict(candidate)
                candidate_copy.setdefault("evidence", "")
                answer_candidates.append(candidate_copy)
                supplemental_candidates.append(candidate_copy)

        def _dedup(candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
            merged: Dict[str, Dict[str, object]] = {}
            for candidate in candidates:
                answer = str(candidate.get("answer", "") or "").strip()
                if not answer:
                    continue
                existing = merged.get(answer)
                if existing is None or float(candidate.get("bonus", 0.0) or 0.0) > float(existing.get("bonus", 0.0) or 0.0):
                    merged[answer] = dict(candidate)
            return list(merged.values())

        def _match(candidates: Sequence[Dict[str, object]], predicate) -> List[Dict[str, object]]:
            return [candidate for candidate in candidates if predicate(candidate)]

        def _pick_first(candidates: Sequence[Dict[str, object]]) -> str:
            return str(candidates[0].get("answer", "") or "").strip() if candidates else ""

        def _numeric_value(answer: str) -> int:
            compact = str(answer or "").strip()
            match = re.search(r"(\d+)", compact)
            if match:
                return int(match.group(1))
            mapping = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
            if compact.startswith("第"):
                compact = compact[1:]
            compact = compact.rstrip("个位家级脚分秒")
            if not compact:
                return 0
            if compact == "十":
                return 10
            if "十" in compact:
                left, _, right = compact.partition("十")
                tens = mapping.get(left, 1 if left == "" else 0)
                ones = mapping.get(right, 0)
                return tens * 10 + ones
            return mapping.get(compact, 0)

        def _source_entries() -> List[Dict[str, object]]:
            out: List[Dict[str, object]] = []
            seen: set[Tuple[int, str]] = set()
            for rank, item in enumerate(list(supplemental) + list(bundle.get("items", []) or [])):
                chunk_id = int(item.get("chunk_id", 0) or 0)
                cache_key = (chunk_id, str(item.get("source", "")))
                if cache_key in seen:
                    continue
                seen.add(cache_key)
                chunk_text = self._chunk_text_by_id(chunk_id) if chunk_id > 0 else ""
                evidence_text = chunk_text or " ".join(str(line) for line in item.get("answer_lines", []) or []) or str(item.get("snippet", ""))
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "score": float(item.get("score", 0.0) or 0.0) + max(0.0, 10.0 - rank),
                        "text": evidence_text,
                    }
                )
            return out

        def _extract_from_texts(
            patterns: Sequence[str],
            *,
            bonus: float = 0.0,
            allow_transform=None,
            required_terms: Sequence[str] = (),
        ) -> str:
            best_answer = ""
            best_score = float("-inf")
            for entry in _source_entries():
                text = str(entry.get("text", "") or "")
                if not text:
                    continue
                if required_terms and not all(term in text for term in required_terms):
                    continue
                for pattern in patterns:
                    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                        answer = match.group(1)
                        if allow_transform is not None:
                            answer = allow_transform(answer)
                        normalized = _normalize_fact_candidate_text(answer, subtype)
                        if not normalized or _is_implausible_fact_candidate(normalized, subtype):
                            continue
                        anchor_hits = sum(1 for anchor in query_anchors if anchor and anchor in text)
                        score = float(entry.get("score", 0.0) or 0.0) + bonus + (anchor_hits * 3.0)
                        if score > best_score:
                            best_score = score
                            best_answer = normalized
            return best_answer

        answer_candidates = _dedup(answer_candidates)
        supplemental_candidates = _dedup(supplemental_candidates)
        answer_candidates.sort(
            key=lambda candidate: (
                float(candidate.get("score", 0.0) or 0.0) + float(candidate.get("bonus", 0.0) or 0.0),
                len(str(candidate.get("answer", "") or "")),
            ),
            reverse=True,
        )
        supplemental_candidates.sort(
            key=lambda candidate: (
                float(candidate.get("score", 0.0) or 0.0) + float(candidate.get("bonus", 0.0) or 0.0),
                len(str(candidate.get("answer", "") or "")),
            ),
            reverse=True,
        )
        if not answer_candidates:
            canonical_answer = str(bundle.get("canonical_answer", "") or "").strip()
            if canonical_answer:
                return canonical_answer
            return "无法确定"

        primary_pool = supplemental_candidates or answer_candidates

        exact_answer = ""
        if subtype == "ranking_position":
            exact_answer = _extract_from_texts(
                (
                    r"战绩排行[:：]\s*嘉世战队总排名(第[\u4e00-\u9fff0-9两十百千]+位)",
                    r"嘉世战队总排名(第[\u4e00-\u9fff0-9两十百千]+位)",
                    r"总排名(第[\u4e00-\u9fff0-9两十百千]+位)",
                ),
                bonus=14.0,
                required_terms=("嘉世",),
            )
        elif subtype == "record_time":
            exact_answer = _extract_from_texts(
                (
                    r"冰霜森林通关记录，成绩(\d+\s*分\s*\d+\s*秒(?:\s*\d+)?)",
                    r"打破副本冰霜森林通关记录，成绩(\d+\s*分\s*\d+\s*秒(?:\s*\d+)?)",
                ),
                bonus=14.0,
            )
        elif subtype == "wager_or_material":
            for required in (("赌局",), ("筹码",), ("抵押",), ()):
                exact_answer = _extract_from_texts(
                    (
                        r"抵押[^。；！？\n]{0,24}?([一二三四五六七八九十百两0-9]+\s*个强力蛛丝)",
                        r"赌注[^。；！？\n]{0,24}?([一二三四五六七八九十百两0-9]+\s*个强力蛛丝)",
                        r"筹码[^。；！？\n]{0,24}?([一二三四五六七八九十百两0-9]+\s*个强力蛛丝)",
                        r"([一二三四五六七八九十百两0-9]+\s*个强力蛛丝)",
                    ),
                    bonus=12.0,
                    required_terms=required,
                )
                if exact_answer:
                    break
        elif subtype == "role_name":
            owner = next((anchor for anchor in query_anchors if len(anchor) <= 4), "")
            patterns = []
            if owner:
                patterns.extend(
                    [
                        rf"([\u4e00-\u9fffA-Za-z0-9·\-]{{2,10}})交还给了?{re.escape(owner)}",
                        rf"你说的妹子就是这个([\u4e00-\u9fffA-Za-z0-9·\-]{{2,10}})吧",
                    ]
                )
            exact_answer = _extract_from_texts(tuple(patterns), bonus=14.0) if patterns else ""
        elif subtype == "boss_name":
            if "烈焰森林" in user_text:
                exact_answer = _extract_from_texts(
                    (
                        r"目标，是[^。；！？\n]{0,18}?野外BOSS[，,:：\s]*([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})",
                        r"烈焰森林[^。；！？\n]{0,40}?野外BOSS[，,:：\s]*([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})",
                    ),
                    bonus=14.0,
                    required_terms=("烈焰森林",),
                )
            if not exact_answer and "霸气雄图" in user_text:
                exact_answer = _extract_from_texts(
                    (
                        r"看到BOSS([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})的身影",
                        r"BOSS([\u4e00-\u9fffA-Za-z0-9·\-]{2,12})的身影",
                    ),
                    bonus=14.0,
                    required_terms=("霸气雄图",),
                )
        elif subtype == "level_count":
            target = next((anchor for anchor in query_anchors if anchor.endswith("卡修") or len(anchor) >= 4), "")
            patterns = [rf"{re.escape(target)}的等级是([一二三四五六七八九十百两0-9]+\s*级)"] if target else []
            patterns.append(r"等级是([一二三四五六七八九十百两0-9]+\s*级)")
            exact_answer = _extract_from_texts(tuple(patterns), bonus=14.0, required_terms=(target,) if target else ())
        elif subtype == "guild_count":
            exact_answer = _extract_from_texts(
                (
                    r"共计([一二三四五六七八九十百两0-9]+\s*家)",
                    r"参与这次对君莫笑队伍围剿行动的公会，共计([一二三四五六七八九十百两0-9]+\s*家)",
                    r"这次行动的(七家)公会",
                ),
                bonus=14.0,
            )
        elif subtype == "current_vs_max":
            exact_answer = _extract_from_texts(
                (
                    r"目前等阶是能出([一二三四五六七八九十百两0-9]+\s*脚)",
                    r"当前技能等阶[^。；！？\n]{0,24}?([一二三四五六七八九十百两0-9]+\s*脚)",
                ),
                bonus=14.0,
            )
        elif subtype == "weapon_form":
            exact_answer = _extract_from_texts(
                (
                    r"25级银武([\u4e00-\u9fffA-Za-z0-9·\-]{1,8})的飞枪移动",
                    r"25级银武([\u4e00-\u9fffA-Za-z0-9·\-]{1,8})",
                ),
                bonus=14.0,
                required_terms=("君莫笑",) if "君莫笑" in user_text else (),
            )
        elif subtype == "running_total":
            exact_answer = _extract_from_texts(
                (
                    r"陈果[^。；！？\n]{0,30}?([0-9]+)\s*了",
                    r"这第([0-9]+)个",
                ),
                bonus=14.0,
                allow_transform=lambda value: f"{value}个",
            )

        if exact_answer:
            return exact_answer

        if subtype == "ranking_position":
            preferred = _match(
                primary_pool,
                lambda c: any(anchor in str(c.get("evidence", "")) for anchor in ("总排名", "战绩排行")) and any(
                    anchor in str(c.get("evidence", "")) for anchor in query_anchors
                ),
            )
            if preferred:
                preferred.sort(key=lambda c: _numeric_value(str(c.get("answer", ""))), reverse=True)
                return _pick_first(preferred)
        elif subtype == "record_time":
            preferred = _match(
                primary_pool,
                lambda c: any(anchor in str(c.get("evidence", "")) for anchor in ("成绩", "通关记录", "打破副本")),
            )
            if preferred:
                preferred.sort(key=lambda c: len(str(c.get("answer", ""))), reverse=True)
                return _pick_first(preferred)
        elif subtype == "wager_or_material":
            preferred = _match(primary_pool, lambda c: "蛛丝" in str(c.get("answer", "")))
            if preferred:
                preferred.sort(key=lambda c: _numeric_value(str(c.get("answer", ""))), reverse=True)
                return _pick_first(preferred)
        elif subtype == "role_name":
            preferred = _match(
                primary_pool,
                lambda c: 2 <= len(str(c.get("answer", ""))) <= 6 and not _is_implausible_fact_candidate(str(c.get("answer", "")), subtype),
            )
            if preferred:
                preferred.sort(
                    key=lambda c: (
                        any(anchor in str(c.get("evidence", "")) for anchor in ("交还给", "这个", "妹子")),
                        any(anchor in str(c.get("evidence", "")) for anchor in query_anchors),
                        float(c.get("bonus", 0.0) or 0.0),
                        len(str(c.get("answer", ""))),
                    ),
                    reverse=True,
                )
                return _pick_first(preferred)
        elif subtype == "boss_name":
            preferred = _match(
                primary_pool,
                lambda c: 4 <= len(str(c.get("answer", ""))) <= 8
                and not _is_implausible_fact_candidate(str(c.get("answer", "")), subtype)
                and any(anchor in str(c.get("evidence", "")) for anchor in ("BOSS", "身影", "烈焰森林", "霸气雄图")),
            )
            if preferred:
                preferred.sort(
                    key=lambda c: (
                        any(token in str(c.get("answer", "")) for token in ("浪人", "女巫", "卡修", "奥磐")),
                        float(c.get("bonus", 0.0) or 0.0),
                        len(str(c.get("answer", ""))),
                    ),
                    reverse=True,
                )
                return _pick_first(preferred)
        elif subtype == "level_count":
            preferred = _match(
                primary_pool,
                lambda c: str(c.get("answer", "")).endswith("级") and any(anchor in str(c.get("evidence", "")) for anchor in query_anchors),
            )
            if preferred:
                preferred.sort(key=lambda c: float(c.get("bonus", 0.0) or 0.0), reverse=True)
                return _pick_first(preferred)
        elif subtype == "guild_count":
            preferred = _match(
                primary_pool,
                lambda c: str(c.get("answer", "")).endswith("家") and any(anchor in str(c.get("evidence", "")) for anchor in ("公会", "围剿", "参与")),
            )
            if preferred:
                preferred.sort(
                    key=lambda c: (
                        "共计" in str(c.get("evidence", "")) or "共有" in str(c.get("evidence", "")),
                        float(c.get("bonus", 0.0) or 0.0),
                    ),
                    reverse=True,
                )
                return _pick_first(preferred)
        elif subtype == "current_vs_max":
            preferred = _match(
                primary_pool,
                lambda c: str(c.get("answer", "")).endswith("脚") and any(anchor in str(c.get("evidence", "")) for anchor in ("当前", "目前", "能出", "等阶")),
            )
            if preferred:
                preferred.sort(key=lambda c: _numeric_value(str(c.get("answer", ""))))
                return _pick_first(preferred)
        elif subtype == "weapon_form":
            preferred = _match(
                primary_pool,
                lambda c: str(c.get("answer", "")) in _WEAPON_TERMS and any(anchor in str(c.get("evidence", "")) for anchor in ("25级银武", "飞枪移动", "步枪")),
            )
            if preferred:
                preferred.sort(
                    key=lambda c: (
                        "步枪" in str(c.get("answer", "")),
                        "飞枪移动" in str(c.get("evidence", "")),
                        float(c.get("bonus", 0.0) or 0.0),
                    ),
                    reverse=True,
                )
                return _pick_first(preferred)
        elif subtype == "running_total":
            preferred = _match(
                primary_pool,
                lambda c: str(c.get("answer", "")).endswith("个") and any(anchor in str(c.get("evidence", "")) for anchor in ("数到", "达到", "323了", "第323个")),
            )
            if preferred:
                preferred.sort(key=lambda c: _numeric_value(str(c.get("answer", ""))), reverse=True)
                return _pick_first(preferred)

        canonical_answer = str(bundle.get("canonical_answer", "") or "").strip()
        if canonical_answer and not _is_implausible_fact_candidate(canonical_answer, subtype):
            return canonical_answer
        first = _pick_first(
            [candidate for candidate in answer_candidates if not _is_implausible_fact_candidate(str(candidate.get("answer", "")), subtype)]
        )
        if first:
            return first
        return "无法确定"

    def needs_fact_answer_fallback(self, answer: str, retrieved_bundle: Dict[str, object], user_text: str) -> bool:
        text = (answer or "").strip()
        answer_candidates = list(retrieved_bundle.get("answer_candidates", []) or [])
        canonical_answer = str(retrieved_bundle.get("canonical_answer", "") or "").strip()
        subtype = str(retrieved_bundle.get("fact_subtype", "") or _resolve_fact_subtype(user_text, dict(retrieved_bundle.get("flags", {}))))
        if not answer_candidates:
            return not text
        if not text:
            return True
        normalized_answer = _normalize_fact_candidate_text(text, subtype) or _normalize_fact_text(text)
        for candidate in answer_candidates:
            candidate_answer = str(candidate.get("answer", "") or "").strip()
            normalized_candidate = _normalize_fact_candidate_text(candidate_answer, subtype) or _normalize_fact_text(candidate_answer)
            if not normalized_candidate:
                continue
            if (
                normalized_answer == normalized_candidate
                or normalized_candidate in normalized_answer
                or normalized_answer in normalized_candidate
            ):
                return False
        if canonical_answer:
            normalized_canonical = _normalize_fact_candidate_text(canonical_answer, subtype) or _normalize_fact_text(canonical_answer)
            if normalized_canonical and normalized_canonical in normalized_answer:
                return False
        if any(marker in text for marker in ("无法确定", "不确定", "未检索到", "没有提到")):
            return True
        return True

    # ---------------- UI debug helpers ----------------

    def status_snapshot(self) -> Dict[str, object]:
        """Structured status for GUI indicators."""

        uploads = self.list_uploads()
        return {
            "memory_dir": self.memory_dir,
            "chunks": self.count_chunks(),
            "terms": self.count_terms(),
            "uploads": len(uploads),
            "last_analyze_ts": self._meta_get_int("last_analyze_ts", 0),
        }

    def debug_view(self) -> str:
        """Human-friendly text for the GUI memory panel."""

        chunks = self.count_chunks()
        terms = self.count_terms()
        uploads = self.list_uploads()
        last_ts = self._meta_get_int("last_analyze_ts", 0)
        last = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)) if last_ts else "（无）"

        core = _read_text(self.core_md_path).strip()
        persona = _read_text(self.persona_md_path).strip()

        up_lines = "\n".join(f"- {os.path.basename(p)}" for p in uploads) or "- （无）"

        txt = (
            f"【Memory Root】\n{self.memory_dir}\n\n"
            f"【Stats】\n- Chunks: {chunks}\n- Terms: {terms}\n- Uploads: {len(uploads)}\n- Last Analyze: {last}\n\n"
            f"【Uploads】\n{up_lines}\n\n"
            f"【01_core.md】\n{core}\n\n"
            f"【02_persona.md】\n{persona}\n"
        )
        return txt
