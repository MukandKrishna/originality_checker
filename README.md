# Originality Audit Toolkit v3

This toolkit combines a reusable ChatGPT/Claude-style skill with a local Python similarity checker.

## Files

- `Originality_Check_Skill.md` — orchestration and review instructions.
- `originality_checker.py` — local lexical + optional semantic similarity engine (v3).
- `requirements.txt` — Python dependencies.
- `example_draft.txt` / `example_source.txt` — quick smoke-test files.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> The first run with semantic mode enabled downloads the default embedding model
> (`sentence-transformers/all-mpnet-base-v2`, ~438 MB) from Hugging Face and caches it
> locally. Subsequent runs use the cached copy.

## Quick start

Minimal lexical mode (no model download):

```bash
python originality_checker.py --draft example_draft.txt --sources example_source.txt --no-semantic
```

Full mode (semantic + FAISS):

```bash
python originality_checker.py \
  --draft example_draft.txt \
  --sources example_source.txt \
  --json report.json \
  --text-report report.txt
```

With Cross-Encoder re-ranking (more precise, slower):

```bash
python originality_checker.py \
  --draft example_draft.txt \
  --sources example_source.txt \
  --rerank
```

> The default re-ranker is `cross-encoder/stsb-roberta-base` — trained on the
> Semantic Textual Similarity Benchmark (STS-B), which is exactly the task of
> judging how close two sentences are in meaning. It outputs calibrated 0-1
> similarity scores that match the `rr >= 0.75` thresholds.
>
> The older `cross-encoder/ms-marco-MiniLM-L-6-v2` is a **web-search relevance**
> model (MS MARCO = query→document matching). Its logits are unbounded, so
> thresholds like 0.75 have no meaningful interpretation for plagiarism scoring.

For a folder:

```bash
python originality_checker.py \
  --draft draft.docx \
  --sources-folder ./sources \
  --json originality_report.json
```

Control source window sizes (default indexes 1-, 2-, and 3-sentence windows):

```bash
python originality_checker.py --draft draft.txt --sources source.txt \
  --source-window-sizes 1,2,3
```

## What it checks (v3)

- **exact 2–12 word phrase windows** (now includes short 2–5 word phrases for entity/proper-name detection)
- **named entity overlap** (e.g. "Grid Code 2023", acronyms, capitalized sequences) — supporting signal only
- word 3-gram overlap
- character 5-gram overlap
- fuzzy similarity
- TF-IDF similarity
- optional SentenceTransformer semantic similarity (default: `all-mpnet-base-v2`)
- optional FAISS retrieval with **pre-built index** (corpus encoded once, not per draft sentence)
- optional **Cross-Encoder re-ranking** — **batched** (all candidates scored in one call, then re-ordered)
- **sliding source windows** — indexes 1-, 2-, and 3-sentence windows so multi-sentence paraphrasing is detected
- simple citation/quotation heuristics
- **phrase highlighting** in reports (shared material wrapped in `[brackets]`)
- **source-level aggregation** (summary per source file)
- **abbreviation-aware sentence splitting** (handles `e.g.`, `i.e.`, `et al.`, `U.S.`, etc.)

## Architecture (v3)

```
                  Draft
                    |
                    v
            Better sentence split
                    |
        +-----------+-----------+
        |                       |
        v                       v
  Lexical engine          Embedding engine
        |                 (pre-built FAISS)
  exact 2-12 phrases            |
  3-grams / char-grams          v
  fuzzy / TF-IDF         Semantic candidates
  entity overlap               |
        |                       |
        +-----------+-----------+
                    |
                    v
             Candidate pool
                    |
                    v
          Cross-Encoder re-rank (optional)
                    |
                    v
             Combined evidence
                    |
                    v
             Risk classification
                    |
                    v
        Human-readable report
        (phrase highlighting)
```

## Embedding model selection

| Model | Dimensions | Speed | Use case |
|---|---|---|---|
| `all-mpnet-base-v2` (default) | 768 | medium | Best quality of the all-* family — recommended |
| `all-MiniLM-L6-v2` | 384 | ~5x faster | Large corpora where speed matters |
| `BAAI/bge-m3` | 1024 | slower | Multilingual; **requires threshold recalibration** |
| `intfloat/multilingual-e5-large` | 1024 | medium/slow | Multilingual |

Switch with `--semantic-model`:

```bash
python originality_checker.py --draft draft.txt --sources source.txt \
  --semantic-model sentence-transformers/all-MiniLM-L6-v2
```

## Threshold calibration

Different embedding models produce different similarity distributions.
If you switch to a model other than `all-mpnet-base-v2`, calibrate the semantic
thresholds on your own data:

```bash
python originality_checker.py --draft draft.txt --sources source.txt \
  --sem-med-threshold 0.78 --sem-high-threshold 0.88
```

Defaults (calibrated for `all-mpnet-base-v2`):
- `sem_med` = 0.82 → "Semantically similar"
- `sem_high` = 0.90 → "Very high semantic similarity"

## Citation / quotation logic

| Case | Verdict |
|---|---|
| Proper paraphrase + citation | Usually acceptable |
| Near-verbatim wording + citation, no quotes | Potential quotation problem |
| Verbatim wording + quotes + citation | Usually acceptable |
| Verbatim wording + no citation | High risk |
| Paraphrased distinctive finding + no citation | Can still be plagiarism |

## Disclaimer

Similarity signals identify passages for review. They do not by themselves prove plagiarism.
This is NOT a replacement for Turnitin/iThenticate — it only compares against sources you provide locally.