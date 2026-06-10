
# models_config.py
from typing import List, Tuple

TARGET_MODELS = [
    {
        "key": "deepseek_qwen3_8b",
        "label": "DeepSeek-R1-0528-Qwen3-8B (Easy to use)",
        "candidates": ["deepseek-r1:8b-qwen-0528","deepseek-r1-0528-qwen3:8b","deepseek-r1-qwen:8b","deepseek-r1:8b-qwen3"],
    },
    {
        "key": "qwen3_30b_a3b",
        "label": "Qwen3-30B-A3B (Fast Respond)",
        "candidates": ["qwen3:30b-a3b","qwen:30b-a3b","qwen3-30b-a3b"],
    },
    {
        "key": "deepseek_r1_70b",
        "label": "Deepseek R1 70B (Better Stability)",
        "candidates": ["deepseek-r1:70b","deepseek-r1-70b","deepseek:70b"],
    },
    {
        "key": "gpt_oss_120b",
        "label": "gpt-oss-120b (Best Performance)",
        "candidates": ["gpt-oss:120b","gpt-oss-120b","gpt-oss:120b-q4","gpt-oss:120b-q5"],
    },
]

def resolve_installed_model(installed: List[str], choice_key: str) -> Tuple[str, List[str]]:
    """
    Given installed model names and a target choice key, find a candidate that exists.
    Returns (name, candidates). name may be "" if not found.
    """
    cands = []
    for m in TARGET_MODELS:
        if m["key"] == choice_key:
            cands = m["candidates"]
            break
    if not cands:
        return "", []
    for c in cands:
        for ins in installed:
            if ins.strip().lower().startswith(c.lower()):
                return ins, cands
    return "", cands
