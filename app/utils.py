import math
import os
import re
from collections import Counter
from typing import Iterable, Iterator, List


WORD_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_\-]+\b")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_whitespace_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    raw_parts = re.split(r"\n\s*\n", text)
    parts = [normalize_whitespace_inline(p) for p in raw_parts]
    return [p for p in parts if p]


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace_inline(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def looks_like_heading(text: str) -> bool:
    t = normalize_whitespace_inline(text)
    if not t:
        return False
    if len(t) > 120:
        return False
    if re.fullmatch(r"([A-Z][A-Z0-9/&,\- ]{3,}|[0-9]+(\.[0-9]+)*\s+.+)", t):
        return True
    if t.endswith(":") and len(t) < 100:
        return True
    return False


def split_large_paragraph(paragraph: str, max_chars: int, overlap: int) -> List[str]:
    paragraph = normalize_whitespace_inline(paragraph)
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = split_sentences(paragraph)
    if not sentences:
        return sliding_window_chunk(paragraph, max_chars, overlap)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + (1 if current else 0)
        if current and current_len + sentence_len > max_chars:
            chunks.append(" ".join(current).strip())

            carry = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1

            current = carry[:] if carry else []
            current_len = len(" ".join(current)) if current else 0

        current.append(sentence)
        current_len = len(" ".join(current))

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def chunk_by_paragraphs(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_chars: int,
    max_paragraph_chars: int,
) -> List[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    normalized_paragraphs: List[str] = []
    for p in paragraphs:
        normalized_paragraphs.extend(split_large_paragraph(p, max_paragraph_chars, overlap))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in normalized_paragraphs:
        addition_len = len(paragraph) + (2 if current else 0)

        if current and current_len + addition_len > chunk_size:
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append(chunk)

            carry: List[str] = []
            carry_chars = 0
            for prev in reversed(current):
                prev_len = len(prev) + (2 if carry else 0)
                if carry_chars + prev_len > overlap:
                    break
                carry.insert(0, prev)
                carry_chars += prev_len

            current = carry[:]
            current_len = len("\n\n".join(current)) if current else 0

        current.append(paragraph)
        current_len = len("\n\n".join(current))

    if current:
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

    # Merge tiny trailing chunks when possible
    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chunk_chars:
            candidate = merged[-1] + "\n\n" + chunk
            if len(candidate) <= chunk_size + overlap:
                merged[-1] = candidate
                continue
        merged.append(chunk)

    return merged


def sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = normalize_whitespace_inline(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def term_frequency(tokens: Iterable[str]) -> Counter:
    return Counter(tokens)


def sparse_overlap_score(query: str, document: str) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(document)

    if not q_tokens or not d_tokens:
        return 0.0

    q_counts = term_frequency(q_tokens)
    d_counts = term_frequency(d_tokens)

    overlap = 0.0
    for token, q_count in q_counts.items():
        overlap += min(q_count, d_counts.get(token, 0))

    length_penalty = math.sqrt(max(len(d_tokens), 1))
    return float(overlap / length_penalty)


def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) < 1e-9:
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]


def iter_files(root_dir: str) -> Iterator[str]:
    supported = {".pdf", ".txt", ".md", ".html", ".htm"}
    for root, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported:
                yield os.path.join(root, file)