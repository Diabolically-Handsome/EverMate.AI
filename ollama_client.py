# ollama_client.py
"""Thin client for a local Ollama server.

All EverMate model traffic goes through this module so that network
behavior (timeouts, context size, keep-alive, error reporting) stays
consistent and easy to audit.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterator, List, Optional, Sequence

import requests


def ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def default_model() -> str:
    return os.getenv("OLLAMA_MODEL", "gpt-oss:20b")


# Kept for backwards compatibility with older imports.
OLLAMA_URL = ollama_url()
DEFAULT_MODEL = default_model()

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# The Ollama default context window (4k) silently truncates long prompts from
# the top — which is exactly where EverMate injects memory evidence.
DEFAULT_NUM_CTX = _env_int("EVERMATE_NUM_CTX", 8192)

# Companion chats are intermittent; keep the model warm between turns instead
# of paying a cold load every few minutes.
DEFAULT_KEEP_ALIVE = os.getenv("EVERMATE_KEEP_ALIVE", "30m")

CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 300.0


class OllamaError(RuntimeError):
    """A failure talking to Ollama."""


class OllamaConnectionError(OllamaError):
    """The Ollama server is unreachable (not running, or wrong URL)."""


class OllamaModelNotFoundError(OllamaError):
    """The requested model is not available on the server."""

    def __init__(self, model: str, message: str = ""):
        super().__init__(message or f"model '{model}' not found")
        self.model = model


_CHANNEL_ARTIFACTS = (
    re.compile(r"<\|channel>thought\s*", re.IGNORECASE),
    re.compile(r"<\|channel\|>thought\s*", re.IGNORECASE),
    re.compile(r"<channel\|>\s*", re.IGNORECASE),
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def clean_response_text(text: str) -> str:
    """Strip reasoning-channel artifacts from a complete model reply.

    Closed ``<think>…</think>`` blocks are removed. An unclosed ``<think>``
    means the model was cut off while still reasoning — everything from that
    tag onward is dropped (text before it, if any, is kept). A reply that was
    entirely reasoning therefore comes back empty, and the caller decides how
    to recover (retry, fallback, or an honest error message).
    """

    cleaned = (text or "").strip()
    for pattern in _CHANNEL_ARTIFACTS:
        cleaned = pattern.sub("", cleaned)
    cleaned = _THINK_BLOCK.sub("", cleaned)
    open_match = _THINK_OPEN.search(cleaned)
    if open_match:
        cleaned = cleaned[: open_match.start()]
    return cleaned.strip()


def _partial_suffix_len(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""

    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if text[-length:].lower() == tag[:length]:
            return length
    return 0


class ThinkTagFilter:
    """Stateful stream sanitizer.

    Drops <think>…</think> spans AND gpt-oss reasoning-channel artifacts
    ("<|channel>thought", "<channel|>", …) from a token stream — the
    non-stream path cleans these in clean_response_text, and users must
    never see them in streamed replies either.
    """

    OPEN = "<think>"
    CLOSE = "</think>"
    # Regex search keeps indices aligned with the original buffer; lowercasing
    # the buffer for find() breaks alignment for characters whose lowercase
    # form has a different length (e.g. 'İ').
    _OPEN_RE = re.compile(re.escape(OPEN), re.IGNORECASE)
    _CLOSE_RE = re.compile(re.escape(CLOSE), re.IGNORECASE)
    _CHANNEL_RE = re.compile(r"<\|?channel\|?>(thought)?[ \t]*", re.IGNORECASE)
    # Longest channel artifact we might have to hold back at a chunk border.
    _HOLDBACK = len("<|channel|>thought")

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def _emit(self, text: str) -> str:
        return self._CHANNEL_RE.sub("", text)

    @staticmethod
    def _channel_partial_len(text: str) -> int:
        """Longest suffix of `text` that could still grow into an artifact."""

        max_len = min(len(text), ThinkTagFilter._HOLDBACK - 1)
        for length in range(max_len, 0, -1):
            tail = text[-length:].lower()
            for proto in ("<|channel|>thought", "<|channel>thought", "<channel|>"):
                if proto.startswith(tail):
                    return length
        return 0

    def feed(self, delta: str) -> str:
        self._buf += delta or ""
        out: List[str] = []
        while True:
            if self._in_think:
                m = self._CLOSE_RE.search(self._buf)
                if not m:
                    keep = _partial_suffix_len(self._buf, self.CLOSE)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    return self._emit("".join(out))
                self._buf = self._buf[m.end():]
                self._in_think = False
                continue
            m = self._OPEN_RE.search(self._buf)
            if not m:
                keep = max(
                    _partial_suffix_len(self._buf, self.OPEN),
                    self._channel_partial_len(self._buf),
                )
                cut = len(self._buf) - keep
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                return self._emit("".join(out))
            out.append(self._buf[: m.start()])
            self._buf = self._buf[m.end():]
            self._in_think = True

    def flush(self) -> str:
        """Emit whatever remains; an unclosed think span is discarded."""

        if self._in_think:
            self._buf = ""
            self._in_think = False
            return ""
        out, self._buf = self._buf, ""
        return self._emit(out)


def _error_from_response(r: requests.Response, model: str) -> OllamaError:
    detail = ""
    try:
        detail = str(r.json().get("error", "") or "")
    except Exception:
        detail = (r.text or "")[:300]
    lowered = detail.lower()
    # Require an explicit model-not-found message: a bare 404 can come from
    # any non-Ollama server behind a misconfigured OLLAMA_URL.
    if "not found" in lowered and "model" in lowered:
        return OllamaModelNotFoundError(model, detail)
    return OllamaError(detail or f"Ollama returned HTTP {r.status_code}")


def list_models(url: Optional[str] = None, timeout: float = 3.0) -> List[str]:
    """Names of locally installed models. Raises OllamaConnectionError if the
    server is unreachable — callers must distinguish that from "no models"."""

    base = (url or ollama_url()).rstrip("/")
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise OllamaConnectionError(str(e)) from e
    except requests.exceptions.RequestException as e:
        raise OllamaError(str(e)) from e
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def _build_payload(
    messages: Sequence[Dict[str, str]],
    model: Optional[str],
    options: Optional[Dict[str, object]],
    stream: bool,
) -> Dict[str, object]:
    merged_options: Dict[str, object] = {"num_ctx": DEFAULT_NUM_CTX}
    if options:
        merged_options.update(options)
    return {
        "model": model or default_model(),
        "messages": list(messages),
        "stream": stream,
        "options": merged_options,
        "keep_alive": DEFAULT_KEEP_ALIVE,
    }


def chat(
    messages: Sequence[Dict[str, str]],
    model: Optional[str] = None,
    options: Optional[Dict[str, object]] = None,
    url: Optional[str] = None,
    timeout: float = DEFAULT_READ_TIMEOUT,
) -> str:
    """Single-shot chat completion. Returns the cleaned reply text."""

    base = (url or ollama_url()).rstrip("/")
    resolved_model = model or default_model()
    payload = _build_payload(messages, resolved_model, options, stream=False)
    try:
        r = requests.post(
            f"{base}/api/chat", json=payload, timeout=(CONNECT_TIMEOUT, timeout)
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise OllamaConnectionError(str(e)) from e
    except requests.exceptions.RequestException as e:
        raise OllamaError(str(e)) from e
    if r.status_code != 200:
        raise _error_from_response(r, resolved_model)
    try:
        content = r.json()["message"]["content"]
    except (ValueError, KeyError) as e:
        raise OllamaError(f"unexpected Ollama response: {e}") from e
    return clean_response_text(content)


def chat_stream(
    messages: Sequence[Dict[str, str]],
    model: Optional[str] = None,
    options: Optional[Dict[str, object]] = None,
    url: Optional[str] = None,
    timeout: float = DEFAULT_READ_TIMEOUT,
) -> Iterator[str]:
    """Streaming chat completion yielding visible-text deltas.

    <think> spans are filtered out on the fly via ThinkTagFilter, so consumers
    can append every yielded delta directly to the UI.
    """

    base = (url or ollama_url()).rstrip("/")
    resolved_model = model or default_model()
    payload = _build_payload(messages, resolved_model, options, stream=True)
    think_filter = ThinkTagFilter()
    try:
        with requests.post(
            f"{base}/api/chat",
            json=payload,
            timeout=(CONNECT_TIMEOUT, timeout),
            stream=True,
        ) as r:
            if r.status_code != 200:
                raise _error_from_response(r, resolved_model)
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if data.get("error"):
                    raise OllamaError(str(data["error"]))
                delta = str(data.get("message", {}).get("content", "") or "")
                if delta:
                    visible = think_filter.feed(delta)
                    if visible:
                        yield visible
                if data.get("done"):
                    break
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise OllamaConnectionError(str(e)) from e
    except requests.exceptions.RequestException as e:
        raise OllamaError(str(e)) from e
    tail = think_filter.flush()
    if tail:
        yield tail
