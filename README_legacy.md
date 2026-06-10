
# EverMate – Integrated Build

This package includes the memory system, aligned topbar, combobox arrow fix, equal line spacing on welcome page, 
and a restricted model selection integrated with Ollama detection.
An "EM" vector icon has been added and set as the application window icon.

## Run

```bash
python app.py
```

Requires a local Ollama server (default http://localhost:11434).

## Models
Only these four choices are exposed:
- DeepSeek-R1-0528-Qwen3-8B (Easy to use)
- Qwen3-30B-A3B (Fast Respond)
- Deepseek R1 70B (Better Stability)
- gpt-oss-120b (Best Performance)

If a model is not installed, the UI will show a message with the `ollama pull ...` command.

## Memory
- Three layers: Core / Episodic / Short-term
- Episodic: BM25-like retrieval with time decay
- Short-term: last turns compressed by truncation
- Persistence: data/memory.json

## UI
- Topbar alignment (Language / Theme / Select Persona on same row)
- Language combo arrow no longer overlaps the border
- Equal spacing on the welcome page
- "EM" icon embedded (assets/icons/app_icon.svg)
