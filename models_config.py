# models_config.py
from typing import List, Tuple

TARGET_MODELS = [
    {
        "key": "deepseek_qwen3_8b",
        "label": "DeepSeek-R1-0528-Qwen3-8B (Easy to use)",
        # Ollama re-tagged the 0528 Qwen3 distill as deepseek-r1:8b.
        "candidates": ["deepseek-r1:8b", "deepseek-r1:8b-0528-qwen3"],
    },
    {
        "key": "qwen3_30b_a3b",
        "label": "Qwen3-30B-A3B (Fast Respond)",
        "candidates": ["qwen3:30b-a3b", "qwen3:30b"],
    },
    {
        "key": "deepseek_r1_70b",
        "label": "Deepseek R1 70B (Better Stability)",
        "candidates": ["deepseek-r1:70b"],
    },
    {
        "key": "gpt_oss_120b",
        "label": "gpt-oss-120b (Best Performance)",
        "candidates": ["gpt-oss:120b"],
    },
]


def is_local_model(name: str) -> bool:
    """EverMate is local-first: never auto-select Ollama cloud models."""

    return "cloud" not in (name or "").lower()


def _matches(installed: str, candidate: str) -> bool:
    """True if `installed` is `candidate` or a more specific variant of it.

    Boundary-aware so that e.g. "deepseek-r1:8b" does not match a
    hypothetical "deepseek-r1:8bx", while still matching quantization
    suffixes like "deepseek-r1:8b-q4_K_M".
    """

    name = installed.strip().lower()
    cand = candidate.strip().lower()
    if name == cand:
        return True
    return name.startswith(cand) and name[len(cand)] in "-_.:"


def resolve_installed_model(installed: List[str], choice_key: str) -> Tuple[str, List[str]]:
    """
    Given installed model names and a target choice key, find a candidate that exists.
    Returns (name, candidates). name may be "" if not found.
    """
    cands: List[str] = []
    for m in TARGET_MODELS:
        if m["key"] == choice_key:
            cands = m["candidates"]
            break
    if not cands:
        return "", []
    local = [ins for ins in installed if is_local_model(ins)]
    for c in cands:
        for ins in local:
            if _matches(ins, c):
                return ins, cands
    return "", cands
