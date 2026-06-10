# ollama_client.py
import os
import re

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL","http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL","gpt-oss:20b")


def _clean_response_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"<\|channel>thought\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|channel\|>thought\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<channel\|>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def chat(messages, model=DEFAULT_MODEL, options=None, url=OLLAMA_URL, timeout=120):
    payload={"model":model,"messages":messages,"stream":False}
    if options: payload["options"]=options
    r=requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return _clean_response_text(r.json()["message"]["content"])
