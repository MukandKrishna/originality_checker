# Originality Checker

A local tool that checks a draft against source files you provide. It looks for wording that may be copied, lightly rewritten, or too close to the original.

This is not Turnitin. It cannot search the whole internet or private journal databases. What it does is compare your draft against the PDFs, Word files, and text files you give it, then flags passages that look suspicious.

## Why use this

When you write a review of a paper, you usually keep the source open. You rephrase as you go. Most of the time that's fine. Sometimes a sentence ends up closer to the source than you meant.

This tool catches that. It shows which sentences overlap with the source, how strong the overlap is, and what the risk looks like on a scale of 0 to 5. You decide what to do next.

## What it checks

- Exact phrase overlap, from 2 to 12 word windows
- Shared named entities, like "Grid Code 2023" or "MedScore"
- Word 3-gram overlap
- Character 5-gram overlap
- Fuzzy similarity
- TF-IDF similarity
- Semantic similarity with sentence embeddings (optional)
- Cross-encoder re-ranking (optional, more accurate)
- Citation and quotation presence

## Setup

You need Python 3.9 or newer. These commands are for Windows PowerShell.

Create a virtual environment and install the packages:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run with semantic mode downloads a model from Hugging Face. It's about 438 MB, so it takes a minute. After that it's cached.

## Quick start

Run it on the example files:

```powershell
python originality_checker.py --draft example_draft.txt --sources example_source.txt --no-semantic
```

`--no-semantic` skips the embedding model so it runs fast.

## Checking your own files

Give it a draft and one or more sources:

```powershell
python originality_checker.py --draft m_review.docx --sources paper1.pdf paper2.pdf
```

It reads .txt, .md, .rst, .csv, .json, .html, .pdf, and .docx files.

Add `--json report.json --text-report report.txt` to save both a human-readable report and a structured JSON file.

## Understanding the output

Each flagged passage gets a risk score:

| Score | Meaning |
|---|---|
| 0 | No material overlap |
| 1 | Low overlap, often just shared names |
| 2 | Semantically similar but wording looks independent |
| 3 | Close paraphrase or patchwriting |
| 4 | Near-verbatim overlap |
| 5 | Direct copying, especially without quotes or citation |

The report shows the draft sentence and the source sentence with overlapping phrases in brackets:

```
Draft : [Grid Code 2023] provides [technical requirements for] the [secure operation of Pakistan's transmission] network.
Source: [Grid Code 2023] establishes [technical requirements for] [secure operation of Pakistan's transmission] system.
```

## Sliding windows

Sources are indexed in three ways: single sentences, two-sentence windows, and three-sentence windows. If a draft merges two source sentences into one paragraph, the tool still notices.

## Semantic mode

Without `--no-semantic`, the tool uses an embedding model to find meaning-level matches. This helps when the wording is different but the meaning is close. The default model is `all-mpnet-base-v2`. It is slower than MiniLM but gives better quality.

If speed matters and your sources are large, switch to MiniLM:

```powershell
python originality_checker.py --draft my_review.docx --sources paper1.pdf --semantic-model sentence-transformers/all-MiniLM-L6-v2
```

## Cross-encoder re-ranking

Add `--rerank` to score candidates with a cross-encoder. It's more precise, but slower. The default model is `cross-encoder/stsb-roberta-base`, which outputs calibrated 0-1 similarity scores.

## Thresholds

Semantic thresholds are set for the default model. If you switch models, the score distribution changes, so recalibrate like this:

```powershell
python originality_checker.py --draft my_review.docx --sources paper1.pdf --sem-med-threshold 0.78 --sem-high-threshold 0.88
```

## What this tool does not do

It does not prove plagiarism. A high score means "look at this passage", not "this is copying". Shared entities like a paper title or a standard name are not plagiarism on their own, so the tool treats them as low-risk signals.

It also only compares against the sources you provide. It has no access to Turnitin or iThenticate databases, and it cannot search the web by itself.

## Files

- `originality_checker.py` — the checker itself
- `Originality_Check_Skill.md` — the skill guide describing the full audit workflow
- `requirements.txt` — Python dependencies
- `example_draft.txt` and `example_source.txt` — sample files to try