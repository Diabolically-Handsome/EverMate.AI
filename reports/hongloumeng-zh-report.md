# Public-corpus benchmark — hongloumeng (zh)

- Corpus: corpora/hongloumeng.txt (906,088 chars)
- Index: 326 chunks / 131,478 terms / 3.0s
- Answer model: gpt-oss:20b (temperature 0)
- Question generation: deterministic seeded cloze (seed=7), no LLM, no hand-written bank
- Scoring: exact string containment, no LLM judge

## Retrieval (top-6, 200 probes)
- Source-chunk hit rate: **99.50%**
- Evidence snippet contains answer: **100.00%**

## End-to-end (60 questions)
- Accuracy: **93.33%**

## Reproduce
```bash
python scripts/benchmark_public_corpus.py --txt corpora/hongloumeng.txt --lang zh \
    --model gpt-oss:20b --questions 60 --retrieval-probes 200 --seed 7
```
