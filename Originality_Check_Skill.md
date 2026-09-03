---
name: originality-audit
description: |
  Audit writing for possible plagiarism, close paraphrasing, patchwriting,
  unattributed borrowing, weak quotation practice, and citation problems.
  Uses web phrase discovery plus a local Python checker for lexical and semantic
  comparison when source files are available.
version: 2.0.0
license: MIT
compatibility: claude-code opencode chatgpt
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - WebSearch
  - Bash
  - AskUserQuestion
---

# Originality Audit Skill v2

## Core goal

The goal is not to force a plagiarism detector to return 0%.

The goal is to:

1. find wording that is copied or too close to a source;
2. identify patchwriting;
3. find source-dependent ideas or claims that need citation;
4. distinguish legitimate similarity from problematic borrowing;
5. rewrite from meaning rather than through synonym substitution;
6. preserve facts, numbers, technical terminology, citations, and source meaning.

A similarity score is evidence for review, not proof of plagiarism.

---

# Available engine

If `originality_checker.py` is available, use it for local source comparison.

Example:

```bash
python originality_checker.py \
  --draft draft.txt \
  --sources source1.pdf source2.pdf \
  --json originality_report.json \
  --text-report originality_report.txt
```

For a folder:

```bash
python originality_checker.py \
  --draft draft.docx \
  --sources-folder ./sources \
  --json originality_report.json
```

If SentenceTransformers or FAISS are unavailable, do not fail the whole task.
The checker must still use its lexical engines.

If semantic packages are unavailable, report:

> Semantic similarity layer unavailable in this environment; lexical and source-aware checks completed.

---

# Layered audit

Use these layers in order.

## Layer 1 — Source and citation map

For each paragraph, identify:

- source-derived facts;
- author's interpretation;
- recommendations;
- quotations;
- uncited non-obvious claims.

Ask:

> Where did this claim come from?

Paraphrasing never removes the need to cite a source-derived finding.

---

## Layer 2 — Exact phrase discovery on the web

When the original source is unknown, search distinctive phrases.

Prefer 6–14 word phrases.

Do not search only complete sentences.

For a sentence such as:

> Pakistan's transmission system requires coordinated protection and secure operation during abnormal conditions.

Generate overlapping searches:

> "transmission system requires coordinated protection"

> "requires coordinated protection and secure operation"

> "secure operation during abnormal conditions"

Prioritise rare technical wording, numbers, dates, names, unusual noun combinations, and distinctive causal phrases.

Ignore generic wording such as:

> The results are shown below.

---

## Layer 3 — Local lexical comparison

Run `originality_checker.py` against supplied source documents.

The tool checks:

- exact 6–12 word phrase windows;
- word 3-gram containment;
- character 5-gram containment;
- fuzzy sentence similarity;
- TF-IDF cosine similarity.

Treat these as signals, not final judgements.

---

## Layer 4 — Semantic comparison

If SentenceTransformers is installed, use semantic similarity to retrieve and compare meaning-level matches.

Preferred default model:

`sentence-transformers/all-MiniLM-L6-v2`

For biomedical work, a domain-specific embedding model may be substituted when appropriate.

If FAISS is installed, use it for efficient top-k semantic retrieval.

High semantic similarity alone does not prove plagiarism.

Check whether:

- the claim is common knowledge;
- the finding is distinctive;
- a citation is present;
- the reasoning structure follows the source too closely.

---

## Layer 5 — Patchwriting check

Patchwriting often preserves:

- subject order;
- verb order;
- noun sequence;
- clause order;
- evidence order;
- causal sequence.

Example:

Source:

> Grid Code 2023 establishes technical requirements for secure operation of Pakistan's transmission system.

Draft:

> Grid Code 2023 provides technical requirements for the secure operation of Pakistan's transmission network.

This is not an independent paraphrase.

Classify as close paraphrase / patchwriting.

Do not fix it by swapping more synonyms.

---

# Risk scale

## 0 — No concern

Common wording, technical names, or no material overlap.

## 1 — Low overlap

Small generic overlap.

## 2 — Semantically similar

Meaning is close, but wording and structure appear independent. Citation may still be required.

## 3 — Close paraphrase / patchwriting

Sentence skeleton, phrase order, or wording remains too close.

## 4 — Near-verbatim overlap

Substantial phrase overlap.

## 5 — Direct or near-direct copying risk

Large exact overlap, especially without quotation or citation.

Do not present this as a legal or academic verdict.
It is a review priority score.

---

# Quotation and citation logic

### Proper paraphrase + citation
Usually acceptable.

### Near-verbatim wording + citation but no quotation marks
Potential quotation problem.

### Verbatim wording + quotation marks + citation
Usually acceptable.

### Verbatim wording + no citation
High risk.

### Fully paraphrased distinctive finding + no citation
Can still be plagiarism.

---

# Rewriting method

Never rewrite directly from the surface wording.

Use:

SOURCE
→ extract facts
→ identify what must remain exact
→ stop following source syntax
→ rebuild explanation from meaning
→ preserve citation
→ re-run originality checker

Do not:

- spin synonyms;
- change numbers;
- distort technical terms;
- insert errors;
- remove citations;
- use invisible characters;
- change punctuation solely to avoid matching;
- translate back and forth to evade detection.

---

# Working with local source files

When source files are available:

1. extract text;
2. split into sentence units;
3. compare draft sentences with source sentences;
4. inspect top lexical and semantic candidates;
5. manually verify the strongest matches;
6. classify risk;
7. rewrite only genuinely problematic passages;
8. re-run the checker.

For PDFs and DOCX, install optional parsers if needed:

```bash
pip install pypdf python-docx
```

---

# Web discovery fallback

If no source files are supplied:

1. select high-distinctiveness phrases;
2. search them in quotation marks;
3. try overlapping fragments;
4. search numbers + topic;
5. search rare phrase + author/topic;
6. open likely sources;
7. compare locally or manually.

Do not claim full-database plagiarism coverage.

Public web search does not include:

- all licensed journal databases;
- private student-paper repositories;
- institutional submissions;
- every archived webpage.

---

# Coverage reporting

Always report what was actually checked.

Examples:

> Coverage: 100% of the supplied draft against 8 supplied source files.

> Web discovery coverage: selective phrase search across high-distinctiveness passages.

> Private institutional databases were not available.

Never treat "500 of 957 words scanned" as a full-document plagiarism score.

---

# Recommended output

## Overall assessment

State:

- originality risk: low / moderate / high;
- coverage;
- number of sources checked;
- whether semantic checking was available.

## Passage findings

Use:

| Passage | Source | Match type | Risk | Citation | Action |
|---|---|---|---:|---|---|

## Why flagged

Use precise reasons:

- exact multi-word overlap;
- same sentence skeleton;
- 3-gram containment;
- strong character overlap;
- fuzzy near-match;
- high semantic similarity;
- uncited source-dependent claim.

## Rewrite

Only rewrite passages that genuinely need it.

Preserve:

- facts;
- figures;
- dates;
- technical terms;
- citation meaning.

## Residual uncertainty

Example:

> No additional public or supplied-source matches were found. This does not rule out matches in Turnitin/iThenticate private databases.

---

# Recommended workflow with Humanizer

For AI-assisted academic writing:

1. Originality Audit
2. Fix citation and quotation problems
3. Rewrite patchwriting from meaning
4. Humanizer
5. Grammar / technical accuracy check
6. Final Originality Audit

Do not humanize first when the draft may already be too close to a source.

Humanizing surface wording can hide lexical overlap without fixing attribution.

---

# Final benchmark

The target is not:

> 0% similarity

The target is:

> independently written work that uses sources honestly, cites borrowed evidence correctly, preserves technical accuracy, and contains no avoidable close copying or patchwriting.
