"""Core / Persona maintenance.

Core = stable style rules + high-frequency topics.
Persona = a handful of bullets about the user (local LLM preferred,
heuristics as fallback).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from engine.storage import MemoryStore, read_text, write_text
from engine.textutil import EN_STOP, ZH_STOP, tokenize

CORE_RULES = {
    "zh": [
        "称呼用户为“您”。",
        "回答尽量详细、结构清晰，必要时给例子。",
        "记忆与当前输入冲突时，以当前输入为准。",
    ],
    "en": [
        "Address the user respectfully.",
        "Prefer clear structure; include examples when helpful.",
        "When memory conflicts with the current input, prefer the current input.",
    ],
}


def _top_terms(store: MemoryStore, limit: int) -> List[str]:
    """High-frequency topics, preferring conversation over imported docs.

    Core is meant to reflect what the user talks about; a big imported novel
    would otherwise drown out the actual conversations.
    """

    cur = store.conn.cursor()

    def query(chat_only: bool) -> List[str]:
        sql = (
            "SELECT p.term AS term, SUM(p.tf) AS total_tf FROM postings p "
            + ("JOIN chunks c ON c.id = p.chunk_id WHERE c.source = 'chat' " if chat_only else "")
            + "GROUP BY p.term ORDER BY total_tf DESC LIMIT ?"
        )
        rows = cur.execute(sql, (limit * 2,)).fetchall()
        terms: List[str] = []
        for r in rows:
            t = str(r["term"])
            if len(t) <= 1 or t in EN_STOP or t in ZH_STOP:
                continue
            terms.append(t)
            if len(terms) >= limit:
                break
        return terms

    terms = query(chat_only=True)
    return terms if terms else query(chat_only=False)


def refresh_core(store: MemoryStore, core_md_path: str, top_terms: int, lang: str = "zh") -> None:
    terms = _top_terms(store, top_terms)
    rules = CORE_RULES.get(lang, CORE_RULES["zh"])
    topic_label = "高频关键词" if lang != "en" else "Frequent topics"
    topic_line = "、".join(terms) if terms else ("（暂无）" if lang != "en" else "(none yet)")
    content = (
        "# Core Memory\n\n"
        + "\n".join(f"- {r}" for r in rules)
        + f"\n\n- {topic_label}：{topic_line}\n"
    )
    write_text(core_md_path, content)


def refresh_persona(
    store: MemoryStore,
    persona_md_path: str,
    max_bullets: int,
    lang: str = "zh",
    model: Optional[str] = None,
) -> None:
    recent = store.recent_chunks_text(max_chunks=12, max_chars=12000)
    previous = _existing_bullets(persona_md_path)

    bullets = _persona_via_ollama(recent, previous, max_bullets, lang=lang, model=model)
    if not bullets:
        bullets = _persona_via_heuristics(recent, max_bullets)
    if not bullets:
        bullets = previous
    if not bullets:
        bullets = ["（尚未从对话中提取到稳定偏好）" if lang != "en" else "(no stable preferences extracted yet)"]

    bullets = bullets[: int(max_bullets)]
    md = f"# Persona (≤ {int(max_bullets)} bullets)\n\n" + "\n".join(f"- {b}" for b in bullets) + "\n"
    write_text(persona_md_path, md)


def _existing_bullets(persona_md_path: str) -> List[str]:
    out: List[str] = []
    for line in read_text(persona_md_path).splitlines():
        line = line.strip()
        if line.startswith("- "):
            bullet = line[2:].strip()
            if bullet and not bullet.startswith("（尚未") and not bullet.startswith("(no stable"):
                out.append(bullet)
    return out


def _persona_via_ollama(
    text: str,
    previous: List[str],
    max_bullets: int,
    lang: str = "zh",
    model: Optional[str] = None,
) -> List[str]:
    """Summarize the user persona with the local LLM. [] on any failure."""

    text = (text or "").strip()
    if not text:
        return []

    try:
        from ollama_client import chat as ollama_chat, default_model
    except Exception:
        return []

    resolved_model = model or os.getenv("OLLAMA_MODEL", "") or default_model()
    if not resolved_model:
        return []

    if lang == "en":
        system = "You are an assistant that distills stable user preferences."
        prompt = (
            "Update the user persona from the conversation below.\n"
            f"- At most {int(max_bullets)} bullets\n"
            "- One sentence each, concrete (communication style / info density / long-term interests / no-gos)\n"
            "- Keep still-valid bullets from the current persona; revise or drop stale ones\n"
            "- Output only the bullet list, one '- ' per line\n\n"
        )
    else:
        system = "您是一位擅长提炼用户偏好的助手。"
        prompt = (
            "请根据下面的对话内容，更新“用户画像 Persona”：\n"
            f"- 最多 {int(max_bullets)} 条要点\n"
            "- 每条 1 句话，尽量具体（沟通偏好/信息密度/长期兴趣/禁忌）\n"
            "- 当前画像中仍然成立的条目请保留，过时的请修订或删除\n"
            "- 只输出要点列表，每行以 '- ' 开头，不要输出其他文字\n\n"
        )
    if previous:
        current_label = "Current persona:\n" if lang == "en" else "当前画像：\n"
        prompt += current_label + "\n".join(f"- {b}" for b in previous) + "\n\n"
    prompt += ("Conversation:\n" if lang == "en" else "对话内容：\n") + text

    try:
        resp = ollama_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=resolved_model,
            timeout=120,
        )
    except Exception:
        return []

    bullets: List[str] = []
    for ln in (resp or "").splitlines():
        ln = ln.strip()
        if ln.startswith("-"):
            b = ln.lstrip("- ").strip()
            if b:
                bullets.append(b)
    return bullets[: int(max_bullets)]


def _persona_via_heuristics(text: str, max_bullets: int) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []

    bullets: List[str] = []
    if "详细" in t or "展开" in t or "多举例" in t:
        bullets.append("偏好回答详细、带例子")
    if "简洁" in t or "别太长" in t:
        bullets.append("偏好回答简洁、直达结论")

    freq: Dict[str, int] = {}
    for x in tokenize(t):
        if len(x) <= 1 or x in EN_STOP or x in ZH_STOP:
            continue
        freq[x] = freq.get(x, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]
    if top:
        bullets.append("近期高频话题关键词：" + "、".join(k for k, _ in top))

    seen = set()
    dedup: List[str] = []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            dedup.append(b)
    return dedup[: int(max_bullets)]
