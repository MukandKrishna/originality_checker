#!/usr/bin/env python3
"""
Originality Audit Toolkit v3

Purpose
-------
A lightweight originality / plagiarism-risk pre-checker for:
- exact overlap (including short 2-5 word phrases)
- named entity overlap
- word n-gram overlap
- character n-gram overlap
- fuzzy sentence similarity
- TF-IDF similarity
- optional semantic similarity with SentenceTransformers
- optional FAISS candidate retrieval (pre-built index)
- optional Cross-Encoder re-ranking
- citation / quotation heuristics

This is NOT a replacement for Turnitin/iThenticate. It only compares against
the sources you provide locally, and optionally whatever web search workflow
your surrounding agent/skill performs.

Architecture (v3)
-----------------
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

Usage
-----
python originality_checker.py --draft draft.txt --sources source1.txt source2.txt
python originality_checker.py --draft draft.txt --sources-folder ./sources --json report.json

Supported source text inputs:
.txt .md .rst .csv .json .html .htm
Optional extraction:
.pdf (pypdf)
.docx (python-docx)

Dependencies:
Core: rapidfuzz, scikit-learn, numpy
Optional semantic: sentence-transformers, faiss-cpu
Optional re-rank: sentence-transformers (cross-encoder)
Optional document parsing: pypdf, python-docx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional, Set

try:
    import numpy as np
except Exception as e:
    raise SystemExit("numpy is required. Install with: pip install numpy") from e

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn") from e


# -----------------------------
# Data models
# -----------------------------

@dataclass
class SharedPhrase:
    text: str
    start_draft: int
    end_draft: int
    start_source: int
    end_source: int

@dataclass
class Match:
    draft_unit: str
    source_unit: str
    source_name: str
    exact_phrase_overlap: float
    word_ngram_overlap: float
    char_ngram_overlap: float
    fuzzy_ratio: float
    tfidf_similarity: float
    entity_overlap: float
    shared_phrases: List[SharedPhrase]
    semantic_similarity: Optional[float]
    rerank_score: Optional[float]
    risk_score: int
    risk_label: str
    reasons: List[str]

@dataclass
class SourceSummary:
    source_name: str
    unit_count: int
    max_risk: int
    max_risk_label: str
    top_matches: int

@dataclass
class Report:
    draft_name: str
    source_count: int
    draft_units: int
    semantic_enabled: bool
    faiss_enabled: bool
    rerank_enabled: bool
    matches: List[Match]
    source_summaries: List[SourceSummary]
    coverage_note: str
    disclaimer: str


# -----------------------------
# Text loading
# -----------------------------

TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".json", ".html", ".htm"}

def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError("PDF support needs pypdf: pip install pypdf") from e
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except Exception as e:
            raise RuntimeError("DOCX support needs python-docx: pip install python-docx") from e
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise RuntimeError(f"Unsupported file type: {path.suffix}")


# -----------------------------
# Normalisation / segmentation
# -----------------------------

def normalize_ws(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s%-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Abbreviations that should NOT be treated as sentence boundaries.
# Followed by either a period or are common academic shorthand.
_ABBREVIATIONS = {
    "e.g", "i.e", "et al", "etc", "vs", "viz", "cf", "al",
    "fig", "figs", "eq", "eqs", "ref", "refs", "ch", "chap",
    "sec", "sect", "no", "nos", "vol", "pp", "p", "ed",
    "eds", "trans", "ca", "approx", "dept", "univ", "inc",
    "corp", "ltd", "co", "dr", "prof", "sr", "jr", "mr",
    "mrs", "ms", "st", "mt", "ave", "blvd", "rd", "ln",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "u.n", "e.u",
}

# Known entity abbreviations often followed by digits (e.g. "Grid Code 2023")
_ENTITY_ABBREV = {
    "grid code", "iso", "iec", "ieee", "who", "un", "unicef",
    "oecd", "nasa", "fda", "epa", "eu", "uk", "us", "usa",
    "gdp", "cpu", "gpu", "ram", "ai", "ml", "nlp", "llm",
    "gpt", "bert", "medscore",
}

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_CAPWORD_RE = re.compile(r"\b[A-Z][a-z]{1,}\b")
_DIGIT_WORD_RE = re.compile(r"\b[A-Za-z]+\s+\d{2,4}\b")
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def _is_abbrev(text: str) -> bool:
    """Return True if text (e.g. 'e.g' or 'U.S') is a known abbreviation."""
    lowered = text.strip().rstrip(".").lower()
    return lowered in _ABBREVIATIONS


def split_sentences(text: str, min_words: int = 5) -> List[str]:
    """Split text into sentences, handling common abbreviations correctly."""
    text = normalize_ws(text)
    if not text:
        return []

    # Step 1: protect known abbreviations by inserting placeholder tokens.
    # Replace "e.g." -> "e_g_" so they don't get split.
    protected = []
    for m in re.finditer(r"\b[\w\.]+\.", text):
        token = m.group(0).rstrip(".")
        if _is_abbrev(token):
            # Replace with a placeholder that won't match sentence boundary
            text = text[:m.start()] + token.replace(".", "_") + text[m.end():]

    # Step 2: split on sentence boundaries (handles the protected abbrevs now)
    rough = _SENT_BOUNDARY.split(text)

    # Step 3: restore placeholders and finalize
    out = []
    for s in rough:
        s = s.strip().replace("_", ".")
        if len(s.split()) >= min_words:
            out.append(s)
    return out


def sliding_word_windows(text: str, size: int = 10, stride: int = 5) -> List[str]:
    words = normalize_for_match(text).split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i+size]) for i in range(0, len(words)-size+1, stride)]


def ngrams(tokens: List[str], n: int) -> set:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}


def char_ngrams(text: str, n: int = 5) -> set:
    text = normalize_for_match(text).replace(" ", "_")
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text)-n+1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set, b: set) -> float:
    """How much of a is contained in b."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


# -----------------------------
# Entity detection
# -----------------------------

def extract_entities(text: str) -> Set[str]:
    """Extract candidate named entities from text.

    Heuristics:
    - Capitalized multi-word sequences (e.g. "Grid Code", "Pakistan's")
    - Acronyms (e.g. "GPT", "WHO", "IEEE")
    - Capitalized word + year (e.g. "Grid Code 2023", "ISO 27001")
    - Known entity abbreviations (grid code, iso, etc.)
    """
    entities: Set[str] = set()

    # Capitalized word sequences (2-4 words)
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}\b", text):
        entities.add(m.group(0).lower())

    # Capitalized word + digits (e.g. "Grid Code 2023", "ISO 27001")
    for m in re.finditer(r"\b[A-Za-z]+\s+\d{2,4}\b", text):
        entities.add(m.group(0).lower())

    # Acronyms
    for m in _ACRONYM_RE.finditer(text):
        entities.add(m.group(0).lower())

    # Known entity abbreviations (even if lowercase)
    for tok in re.findall(r"\b[a-z][a-z-]{1,}\b", text.lower()):
        if tok in _ENTITY_ABBREV:
            entities.add(tok)

    return entities


def entity_overlap(draft: str, source: str) -> float:
    """Fraction of draft entities that appear in source. 0.0-1.0."""
    de = extract_entities(draft)
    se = extract_entities(source)
    if not de:
        return 0.0
    return len(de & se) / len(de)


# -----------------------------
# Exact phrase signal
# -----------------------------

def exact_phrase_overlap(
    draft: str,
    source: str,
    min_words: int = 2,
    max_words: int = 12,
) -> Tuple[float, List[SharedPhrase]]:
    """
    Returns (fraction of draft word windows that appear exactly in source,
             list of shared phrases with positions).

    Now supports short 2-5 word phrases for entity/proper-name detection.
    """
    d = normalize_for_match(draft)
    s = normalize_for_match(source)
    dwords = d.split()
    swords = s.split()
    if len(dwords) < min_words:
        return 0.0, []

    # Build source phrase index for fast lookup
    source_grams: Set[str] = set()
    for n in range(min_words, max_words + 1):
        if len(swords) >= n:
            source_grams.update(
                " ".join(swords[i:i+n]) for i in range(len(swords)-n+1)
            )

    best = 0.0
    shared_phrases: List[SharedPhrase] = []

    # Check largest to smallest for best match first
    found_phrases: Set[str] = set()
    for n in range(min(max_words, len(dwords)), min_words - 1, -1):
        if n < min_words or n > max_words:
            continue
        grams = [" ".join(dwords[i:i+n]) for i in range(len(dwords)-n+1)]
        if not grams:
            continue

        for gi, g in enumerate(grams):
            if g in source_grams and g not in found_phrases:
                found_phrases.add(g)
                # Find position in source
                start_src = s.find(g)
                if start_src >= 0:
                    start_draft = d.find(g)
                    shared_phrases.append(SharedPhrase(
                        text=g,
                        start_draft=start_draft,
                        end_draft=start_draft + len(g),
                        start_source=start_src,
                        end_source=start_src + len(g),
                    ))

        # Compute overlap for this window size
        hits = sum(1 for g in grams if g in source_grams)
        best = max(best, hits / len(grams))

    return best, shared_phrases


# -----------------------------
# Similarity engines
# -----------------------------

def lexical_scores(draft: str, source: str) -> Dict[str, float]:
    dnorm = normalize_for_match(draft)
    snorm = normalize_for_match(source)
    dtok = dnorm.split()
    stok = snorm.split()

    w3 = containment(ngrams(dtok, 3), ngrams(stok, 3))
    c5 = containment(char_ngrams(dnorm, 5), char_ngrams(snorm, 5))
    exact, _ = exact_phrase_overlap(draft, source, min_words=6, max_words=12)
    ent = entity_overlap(draft, source)

    if fuzz is not None:
        fuzzy = fuzz.token_set_ratio(dnorm, snorm) / 100.0
    else:
        fuzzy = 0.0

    try:
        tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1).fit_transform([dnorm, snorm])
        tfidf_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0,0])
    except Exception:
        tfidf_sim = 0.0

    return {
        "exact": exact,
        "word3": w3,
        "char5": c5,
        "fuzzy": fuzzy,
        "tfidf": tfidf_sim,
        "entity": ent,
    }


def make_fingerprint(text: str) -> Dict[str, object]:
    """Cheap lexical features for fast prefiltering across large corpora.

    Used to rank all source units with lightweight set operations BEFORE
    running the expensive full lexical scores (TF-IDF, fuzzy, exact phrases)
    on only the top candidates.
    """
    dnorm = normalize_for_match(text)
    tokens = dnorm.split()
    return {
        "norm": dnorm,
        "w3": ngrams(tokens, 3),
        "c5": char_ngrams(dnorm, 5),
        "entities": extract_entities(text),
    }


def cheap_lexical_rank(draft_fp: Dict[str, object], source_fp: Dict[str, object]) -> float:
    """Fast set-based similarity used to prefilter candidates.

    This is O(set operations) only — no TF-IDF, no fuzzy, no phrase scanning —
    so it can be run across thousands of source units per draft sentence.
    """
    w3s = containment(draft_fp["w3"], source_fp["w3"])
    c5s = containment(draft_fp["c5"], source_fp["c5"])
    de = draft_fp["entities"]
    se = source_fp["entities"]
    ents = (len(de & se) / len(de)) if de else 0.0
    return max(w3s, c5s * 0.9, ents * 0.6)


class SemanticEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
        self.faiss = None
        self._corpus_cache: Optional[np.ndarray] = None
        self._corpus_texts: Optional[List[str]] = None
        self._corpus_index = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

        try:
            import faiss
            self.faiss = faiss
        except Exception:
            self.faiss = None

    @property
    def enabled(self):
        return self.model is not None

    def encode(self, texts: List[str], show_progress: bool = False) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype="float32")

    def build_index(self, corpus: List[str]) -> bool:
        """Encode ALL source sentences once and build the FAISS index."""
        if not self.enabled or not corpus:
            return False

        cemb = self.encode(corpus, show_progress=True)
        if cemb is None:
            return False

        self._corpus_cache = cemb
        self._corpus_texts = list(corpus)

        if self.faiss is not None:
            index = self.faiss.IndexFlatIP(cemb.shape[1])
            index.add(cemb)
            self._corpus_index = index

        return True

    def pair_similarity(self, a: str, b: str) -> Optional[float]:
        emb = self.encode([a, b])
        if emb is None:
            return None
        return float(np.dot(emb[0], emb[1]))

    def topk(self, query: str, k: int = 8) -> List[Tuple[int, float]]:
        """Query the pre-built FAISS index. Returns (index, score) pairs."""
        if not self.enabled or self._corpus_cache is None or not self._corpus_texts:
            return []

        qemb = self.encode([query])
        if qemb is None:
            return []

        k = min(k, len(self._corpus_texts))

        if self._corpus_index is not None:
            scores, ids = self._corpus_index.search(qemb, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

        # Fallback: numpy dot product
        sims = self._corpus_cache @ qemb[0]
        ids = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in ids]


class CrossEncoderReranker:
    """Optional cross-encoder reranker for more precise pairwise scoring.

    Default: cross-encoder/stsb-roberta-base
      - Trained on STS-B (Semantic Textual Similarity Benchmark) — the exact
        task of judging how close two sentences are in meaning.
      - Outputs calibrated 0-1 similarity scores (sigmoid), which matches the
        rr >= 0.75 thresholds used in classify_risk().
      - The old default (ms-marco-MiniLM-L-6-v2) is a web-search relevance
        model whose logits are unbounded, so thresholds like 0.75 have no
        meaningful interpretation for plagiarism scoring.
    """

    def __init__(self, model_name: str = "cross-encoder/stsb-roberta-base"):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    @property
    def enabled(self):
        return self.model is not None

    def score(self, pairs: List[Tuple[str, str]]) -> Optional[List[float]]:
        if not self.enabled or not pairs:
            return None
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            return [float(s) for s in scores]
        except Exception:
            return None


# -----------------------------
# Risk scoring (calibrated)
# -----------------------------

def classify_risk(
    scores: Dict[str, float],
    semantic: Optional[float],
    rerank: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[int, str, List[str]]:
    """Risk classification with configurable thresholds.

    Thresholds can be overridden for calibration with different embedding models.
    """
    if thresholds is None:
        thresholds = {
            "exact_high": 0.60,
            "exact_med": 0.30,
            "exact_low": 0.05,
            "w3_high": 0.50,
            "w3_med": 0.28,
            "w3_low": 0.10,
            "c5_high": 0.65,
            "c5_med": 0.50,
            "c5_low": 0.25,
            "fuzzy_high": 0.88,
            "fuzzy_med": 0.78,
            "fuzzy_low": 0.60,
            "tfidf_high": 0.70,
            "tfidf_med": 0.55,
            "sem_high": 0.90,
            "sem_med": 0.82,
            "entity_flag": 0.50,
        }

    reasons = []
    exact = scores["exact"]
    w3 = scores["word3"]
    c5 = scores["char5"]
    fuzzy = scores["fuzzy"]
    tfidf = scores["tfidf"]
    ent = scores["entity"]
    sem = semantic if semantic is not None else 0.0
    rr = rerank if rerank is not None else 0.0

    # Conservative thresholds: flags passages for review, not guilt.
    if exact >= thresholds["exact_high"] or (w3 >= thresholds["w3_high"] and fuzzy >= thresholds["fuzzy_high"]):
        score, label = 5, "Direct/near-direct copying risk"
        reasons.append("large exact or near-exact phrase overlap")
    elif exact >= thresholds["exact_med"] or w3 >= thresholds["w3_high"] or (c5 >= thresholds["c5_high"] and fuzzy >= thresholds["fuzzy_high"]):
        score, label = 4, "Near-verbatim overlap"
        reasons.append("substantial lexical overlap")
    elif w3 >= thresholds["w3_med"] or c5 >= thresholds["c5_med"] or fuzzy >= thresholds["fuzzy_med"] or tfidf >= thresholds["tfidf_high"]:
        score, label = 3, "Close paraphrase / patchwriting risk"
        reasons.append("sentence structure or wording remains close")
    elif sem >= thresholds["sem_med"] or tfidf >= thresholds["tfidf_med"] or rr >= 0.75:
        score, label = 2, "Semantically similar"
        reasons.append("meaning is close even though wording is less similar")
    elif exact >= thresholds["exact_low"] or w3 >= thresholds["w3_low"] or c5 >= thresholds["c5_low"]:
        score, label = 1, "Low overlap"
        reasons.append("small lexical overlap")
    else:
        score, label = 0, "No material overlap detected"

    # Entity overlap is a supporting signal only
    if ent >= thresholds["entity_flag"] and score == 0:
        score, label = 1, "Low overlap"
        reasons.append("named entity overlap detected (shared terms do not prove copying)")

    if sem >= thresholds["sem_high"]:
        reasons.append("very high semantic similarity")
    elif sem >= thresholds["sem_med"]:
        reasons.append("high semantic similarity")

    if rr >= 0.85:
        reasons.append("cross-encoder confirms strong pairwise similarity")
    elif rr >= 0.75:
        reasons.append("cross-encoder indicates meaningful pairwise similarity")

    if fuzzy >= 0.90:
        reasons.append("very high fuzzy sentence similarity")
    if exact >= thresholds["exact_med"]:
        reasons.append("multiple exact multi-word windows match")
    if w3 >= thresholds["w3_high"]:
        reasons.append("large share of draft 3-grams found in source")
    if c5 >= thresholds["c5_high"]:
        reasons.append("strong character-level overlap")
    if ent >= thresholds["entity_flag"]:
        reasons.append(f"shared named entities (e.g., {ent:.0%} of draft entities found)")

    return score, label, list(dict.fromkeys(reasons))


# -----------------------------
# Citation / quotation heuristics
# -----------------------------

CITATION_RE = re.compile(
    r"(\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,?\s+\d{4}[a-z]?\)|\[[0-9,\-\s]+\])"
)
QUOTE_RE = re.compile(r'["“”][^"“”]{8,}["“”]')

def has_citation(text: str) -> bool:
    return bool(CITATION_RE.search(text))

def has_quote(text: str) -> bool:
    return bool(QUOTE_RE.search(text))


# -----------------------------
# Phrase highlighting
# -----------------------------

def highlight_phrases(text: str, phrases: List[SharedPhrase], use_source: bool = False) -> str:
    """Return text with shared phrases wrapped in [brackets] for visibility.

    Overlapping/adjacent phrases are merged into a single bracketed span to avoid
    nested-bracket noise.  Use use_source=True to highlight positions in the
    source-text units instead of draft-text units."""
    if not phrases:
        return text

    # Collect spans (start, end) and merge overlapping/adjacent ones
    spans = []
    for p in phrases:
        start = p.start_source if use_source else p.start_draft
        end = p.end_source if use_source else p.end_draft
        if start >= 0 and end <= len(text):
            spans.append((start, end))
    if not spans:
        return text

    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Insert brackets from the end so earlier offsets stay valid
    result = text
    for start, end in reversed(merged):
        result = result[:start] + "[" + result[start:end] + "]" + result[end:]
    return result


# -----------------------------
# Main comparison
# -----------------------------

def build_source_units(
    sources: List[Tuple[str, str]],
    window_sizes: Tuple[int, ...] = (1, 2, 3),
) -> List[Tuple[str, str]]:
    """Build source units including multi-sentence windows.

    Indexes single sentences plus 2- and 3-sentence windows so that
    multi-sentence paraphrasing can be detected.
    """
    units: List[Tuple[str, str]] = []
    for name, text in sources:
        sentences = split_sentences(text)
        for size in window_sizes:
            for i in range(len(sentences) - size + 1):
                unit = " ".join(sentences[i:i + size])
                units.append((name, unit))
    return units


def compare(
    draft_text: str,
    sources: List[Tuple[str, str]],
    top_k_lexical: int = 5,
    top_k_semantic: int = 8,
    semantic: bool = True,
    semantic_model: str = "sentence-transformers/all-mpnet-base-v2",
    rerank: bool = False,
    rerank_model: str = "cross-encoder/stsb-roberta-base",
    thresholds: Optional[Dict[str, float]] = None,
    source_window_sizes: Tuple[int, ...] = (1, 2, 3),
) -> Report:
    draft_units = split_sentences(draft_text)
    source_units = build_source_units(sources, source_window_sizes)

    sem_engine = SemanticEngine(semantic_model) if semantic else None
    reranker = CrossEncoderReranker(rerank_model) if rerank else None

    matches: List[Match] = []

    corpus_only = [u for _, u in source_units]

    # Pre-build FAISS index ONCE (encode all source sentences once)
    if sem_engine and sem_engine.enabled and corpus_only:
        print(
            f"Encoding {len(corpus_only)} source units with {semantic_model}... "
            "(this is a one-time per-run cost; progress bar below)",
            file=sys.stderr,
        )
        sem_engine.build_index(corpus_only)

    # Precompute cheap lexical fingerprints for ALL source units once.
    # This lets us prefilter candidates with fast set operations before running
    # expensive full lexical scores (TF-IDF, fuzzy, exact phrases) on top-K only.
    source_features = [make_fingerprint(u) for _, u in source_units]

    for dunit in draft_units:
        dfeat = make_fingerprint(dunit)

        # 1) Fast lexical prefilter over all source units (cheap set ops only)
        cheap_ranked = sorted(
            (
                (cheap_lexical_rank(dfeat, sf), i)
                for i, sf in enumerate(source_features)
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        # Keep a buffer (3x) so promising candidates are not pruned before
        # the expensive full scoring pass.
        prefilter_k = max(top_k_lexical * 3, top_k_lexical)
        candidate_ids = {i for _, i in cheap_ranked[:prefilter_k]}

        # 2) Semantic retrieval from pre-built FAISS
        sem_scores: Dict[int, float] = {}
        if sem_engine and sem_engine.enabled and corpus_only:
            for idx, sim in sem_engine.topk(dunit, k=min(top_k_semantic, len(corpus_only))):
                candidate_ids.add(idx)
                sem_scores[idx] = sim

        # 3) Full lexical scores ONLY for candidate units (cheap prefilter done above)
        full_score_cache: Dict[int, Dict[str, float]] = {}
        for idx in candidate_ids:
            _, sunit = source_units[idx]
            full_score_cache[idx] = lexical_scores(dunit, sunit)

        # 4) Batched Cross-Encoder re-ranking: score ALL candidates at once
        rerank_scores: Dict[int, float] = {}
        if reranker and reranker.enabled and candidate_ids:
            pairs = [(dunit, source_units[idx][1]) for idx in candidate_ids]
            scores_list = reranker.score(pairs)
            if scores_list:
                rerank_scores = dict(zip(candidate_ids, scores_list))

        # Order candidates: rerank score first (if available), then semantic, then lexical
        ordered_candidates = sorted(
            candidate_ids,
            key=lambda idx: (
                rerank_scores.get(idx, -1.0),
                sem_scores.get(idx, 0.0),
                full_score_cache[idx].get("word3", 0.0),
            ),
            reverse=True,
        )

        for idx in ordered_candidates:
            sname, sunit = source_units[idx]
            score_map = full_score_cache[idx]

            # Also compute short-phrase overlap for highlighting
            _, shared_phrases = exact_phrase_overlap(dunit, sunit, min_words=2, max_words=12)

            sem = sem_scores.get(idx)
            if sem is None and sem_engine and sem_engine.enabled:
                sem = sem_engine.pair_similarity(dunit, sunit)

            rr_score = rerank_scores.get(idx)

            risk, label, reasons = classify_risk(score_map, sem, rr_score, thresholds)

            # only retain non-trivial candidates
            if risk >= 1:
                if risk >= 4 and has_citation(dunit) and not has_quote(dunit):
                    reasons.append("citation present, but near-verbatim wording may still need quotation marks")
                elif risk >= 4 and not has_citation(dunit):
                    reasons.append("high lexical overlap and no obvious citation detected")

                matches.append(Match(
                    draft_unit=dunit,
                    source_unit=sunit,
                    source_name=sname,
                    exact_phrase_overlap=round(score_map["exact"], 4),
                    word_ngram_overlap=round(score_map["word3"], 4),
                    char_ngram_overlap=round(score_map["char5"], 4),
                    fuzzy_ratio=round(score_map["fuzzy"], 4),
                    tfidf_similarity=round(score_map["tfidf"], 4),
                    entity_overlap=round(score_map["entity"], 4),
                    shared_phrases=shared_phrases,
                    semantic_similarity=None if sem is None else round(float(sem), 4),
                    rerank_score=None if rr_score is None else round(float(rr_score), 4),
                    risk_score=risk,
                    risk_label=label,
                    reasons=reasons,
                ))

    # keep strongest match per draft unit/source pair, then sort by risk
    dedup = {}
    for m in matches:
        key = (m.draft_unit, m.source_name)
        old = dedup.get(key)
        quality = (
            m.risk_score,
            m.exact_phrase_overlap,
            m.word_ngram_overlap,
            m.semantic_similarity or 0.0,
        )
        if old is None:
            dedup[key] = m
        else:
            oldq = (
                old.risk_score,
                old.exact_phrase_overlap,
                old.word_ngram_overlap,
                old.semantic_similarity or 0.0,
            )
            if quality > oldq:
                dedup[key] = m

    final_matches = sorted(
        dedup.values(),
        key=lambda m: (
            m.risk_score,
            m.exact_phrase_overlap,
            m.word_ngram_overlap,
            m.semantic_similarity or 0.0
        ),
        reverse=True
    )

    # Source-level aggregation
    source_summaries = []
    src_match_counts: Dict[str, List[Match]] = {}
    for m in final_matches:
        src_match_counts.setdefault(m.source_name, []).append(m)
    for sname, ms in src_match_counts.items():
        max_risk = max(m.risk_score for m in ms)
        source_summaries.append(SourceSummary(
            source_name=sname,
            unit_count=len(ms),
            max_risk=max_risk,
            max_risk_label=next(m.risk_label for m in ms if m.risk_score == max_risk),
            top_matches=len(ms),
        ))
    source_summaries.sort(key=lambda s: s.max_risk, reverse=True)

    return Report(
        draft_name="draft",
        source_count=len(sources),
        draft_units=len(draft_units),
        semantic_enabled=bool(sem_engine and sem_engine.enabled),
        faiss_enabled=bool(sem_engine and sem_engine.enabled and sem_engine.faiss is not None),
        rerank_enabled=bool(reranker and reranker.enabled),
        matches=final_matches,
        source_summaries=source_summaries,
        coverage_note="Local comparison coverage is 100% against the source files supplied to this tool. It does not cover private institutional databases or sources you did not provide.",
        disclaimer="Similarity signals identify passages for review. They do not by themselves prove plagiarism.",
    )


# -----------------------------
# Rendering
# -----------------------------

def report_to_dict(report: Report) -> dict:
    d = asdict(report)
    return d


def render_text(report: Report, max_matches: int = 30) -> str:
    lines = []
    lines.append("ORIGINALITY AUDIT REPORT")
    lines.append("=" * 78)
    lines.append(f"Sources compared: {report.source_count}")
    lines.append(f"Draft sentence units: {report.draft_units}")
    lines.append(f"Semantic model enabled: {report.semantic_enabled}")
    lines.append(f"FAISS enabled: {report.faiss_enabled}")
    lines.append(f"Cross-Encoder re-rank enabled: {report.rerank_enabled}")
    lines.append("")
    lines.append(report.coverage_note)
    lines.append(report.disclaimer)
    lines.append("")

    if not report.matches:
        lines.append("No material overlap was detected against the supplied sources.")
        return "\n".join(lines)

    # Source summary
    lines.append("SOURCE SUMMARY (by max risk)")
    lines.append("-" * 78)
    for ss in report.source_summaries:
        lines.append(
            f"  {ss.source_name}: {ss.unit_count} flagged unit(s), "
            f"max risk {ss.max_risk}/5 ({ss.max_risk_label})"
        )
    lines.append("")

    # Detail
    lines.append("FLAGGED PASSAGES")
    lines.append("=" * 78)
    for i, m in enumerate(report.matches[:max_matches], 1):
        lines.append(f"[{i}] Risk {m.risk_score}/5 — {m.risk_label}")
        lines.append(f"Source: {m.source_name}")

        # Highlight shared phrases in draft
        draft_hl = highlight_phrases(m.draft_unit, m.shared_phrases)
        source_hl = highlight_phrases(m.source_unit, m.shared_phrases, use_source=True)

        lines.append(f"Draft : {draft_hl}")
        lines.append(f"Source: {source_hl}")

        # Show shared phrases explicitly
        if m.shared_phrases:
            unique_phrases = list(dict.fromkeys(p.text for p in m.shared_phrases))
            lines.append("Shared phrases: " + ", ".join(f'"{p}"' for p in unique_phrases[:8]))

        lines.append(
            "Scores: "
            f"exact={m.exact_phrase_overlap:.2f}, "
            f"word3={m.word_ngram_overlap:.2f}, "
            f"char5={m.char_ngram_overlap:.2f}, "
            f"fuzzy={m.fuzzy_ratio:.2f}, "
            f"tfidf={m.tfidf_similarity:.2f}, "
            f"entity={m.entity_overlap:.2f}, "
            f"semantic={'n/a' if m.semantic_similarity is None else f'{m.semantic_similarity:.2f}'}"
        )
        if m.rerank_score is not None:
            lines.append(f"Rerank: {m.rerank_score:.2f}")
        if m.reasons:
            lines.append("Why: " + "; ".join(m.reasons))
        lines.append("-" * 78)

    return "\n".join(lines)


# -----------------------------
# CLI
# -----------------------------

def gather_sources(paths: List[str], folder: Optional[str]) -> List[Path]:
    out = [Path(p) for p in paths]
    if folder:
        f = Path(folder)
        for p in sorted(f.rglob("*")):
            if p.is_file() and p.suffix.lower() in (TEXT_EXTS | {".pdf", ".docx"}):
                out.append(p)
    # unique preserving order
    seen = set()
    uniq = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main():
    ap = argparse.ArgumentParser(description="Local originality / plagiarism-risk pre-checker v3")
    ap.add_argument("--draft", required=True, help="Draft file to audit")
    ap.add_argument("--sources", nargs="*", default=[], help="Source files to compare against")
    ap.add_argument("--sources-folder", help="Folder containing source files")
    ap.add_argument("--json", dest="json_path", help="Write JSON report to this path")
    ap.add_argument("--text-report", help="Write human-readable report to this path")
    ap.add_argument("--no-semantic", action="store_true", help="Disable SentenceTransformers semantic checking")
    ap.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model (default: all-mpnet-base-v2, better quality than MiniLM)",
    )
    ap.add_argument("--rerank", action="store_true", help="Enable Cross-Encoder re-ranking (optional, more precise)")
    ap.add_argument("--rerank-model", default="cross-encoder/stsb-roberta-base",
                    help="Cross-encoder model for re-ranking (default: stsb-roberta-base, outputs calibrated 0-1 similarity)")
    ap.add_argument("--top-k-lexical", type=int, default=5)
    ap.add_argument("--top-k-semantic", type=int, default=8)
    ap.add_argument("--max-matches", type=int, default=30)
    ap.add_argument("--source-window-sizes", default="1,2,3",
                    help="Comma-separated sentence window sizes to index from sources (default: 1,2,3)")

    # Threshold calibration flags (optional)
    ap.add_argument("--sem-med-threshold", type=float, default=None,
                    help="Override semantic threshold for 'semantically similar' (default 0.82)")
    ap.add_argument("--sem-high-threshold", type=float, default=None,
                    help="Override semantic threshold for 'very high semantic similarity' (default 0.90)")

    args = ap.parse_args()

    draft_path = Path(args.draft)
    draft_text = read_text_file(draft_path)

    source_paths = gather_sources(args.sources, args.sources_folder)
    if not source_paths:
        raise SystemExit("Provide --sources and/or --sources-folder")

    sources = []
    for p in source_paths:
        try:
            sources.append((p.name, read_text_file(p)))
        except Exception as e:
            print(f"Warning: skipped {p}: {e}", file=sys.stderr)

    # Build custom thresholds if overrides provided
    thresholds = None
    if args.sem_med_threshold is not None or args.sem_high_threshold is not None:
        thresholds = {
            "exact_high": 0.60,
            "exact_med": 0.30,
            "exact_low": 0.05,
            "w3_high": 0.50,
            "w3_med": 0.28,
            "w3_low": 0.10,
            "c5_high": 0.65,
            "c5_med": 0.50,
            "c5_low": 0.25,
            "fuzzy_high": 0.88,
            "fuzzy_med": 0.78,
            "fuzzy_low": 0.60,
            "tfidf_high": 0.70,
            "tfidf_med": 0.55,
            "sem_high": args.sem_high_threshold if args.sem_high_threshold is not None else 0.90,
            "sem_med": args.sem_med_threshold if args.sem_med_threshold is not None else 0.82,
            "entity_flag": 0.50,
        }

    window_sizes = tuple(
        int(x) for x in args.source_window_sizes.split(",") if x.strip()
    ) or (1,)

    report = compare(
        draft_text=draft_text,
        sources=sources,
        top_k_lexical=args.top_k_lexical,
        top_k_semantic=args.top_k_semantic,
        semantic=not args.no_semantic,
        semantic_model=args.semantic_model,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        thresholds=thresholds,
        source_window_sizes=window_sizes,
    )
    report.draft_name = draft_path.name

    txt = render_text(report, max_matches=args.max_matches)
    print(txt)

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.text_report:
        Path(args.text_report).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()