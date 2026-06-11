# Public-corpus benchmark — moby_dick (en)

- Corpus: corpora/moby_dick.txt (1,218,938 chars)
- Index: 441 chunks / 17,115 terms / 0.4s
- Answer model: gpt-oss:20b (temperature 0)
- Question generation: deterministic seeded cloze (seed=7), no LLM, no hand-written bank
- Scoring: exact string containment, no LLM judge

## Retrieval (top-6, 200 probes)
- Source-chunk hit rate: **100.00%**
- Evidence snippet contains answer: **100.00%**

## End-to-end (60 questions)
- Accuracy: **98.33%**

## Reproduce
```bash
python scripts/benchmark_public_corpus.py --txt corpora/moby_dick.txt --lang en \
    --model gpt-oss:20b --questions 60 --retrieval-probes 200 --seed 7
```
