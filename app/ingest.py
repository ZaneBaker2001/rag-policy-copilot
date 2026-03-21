import os
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import settings
from app.models import ChunkRecord
from app.utils import (
    chunk_by_paragraphs,
    clean_text,
    looks_like_heading,
    normalize_whitespace_inline,
    split_paragraphs,
)


def _extract_sectioned_blocks(text: str) -> List[dict]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = split_paragraphs(text)
    blocks: List[dict] = []

    current_section = None
    current_buffer: List[str] = []

    def flush() -> None:
        if current_buffer:
            blocks.append(
                {
                    "page": None,
                    "section": current_section,
                    "text": "\n\n".join(current_buffer).strip(),
                }
            )
            current_buffer.clear()

    for para in paragraphs:
        if looks_like_heading(para):
            flush()
            current_section = normalize_whitespace_inline(para.rstrip(":"))
        else:
            current_buffer.append(para)

    flush()
    if not blocks:
        blocks.append({"page": None, "section": None, "text": text})
    return blocks


def read_pdf(path: str) -> List[dict]:
    reader = PdfReader(path)
    units: List[dict] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            text = page.extract_text() or ""
        except Exception:
            text = page.extract_text() or ""

        text = clean_text(text)
        if not text:
            continue

        blocks = _extract_sectioned_blocks(text)
        for block in blocks:
            block["page"] = i + 1
            units.append(block)

    return units


def read_text_file(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _extract_sectioned_blocks(f.read())


def read_html(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    blocks: List[dict] = []
    current_section = None

    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = clean_text(element.get_text(separator=" "))
        if not text:
            continue

        if element.name in {"h1", "h2", "h3", "h4"}:
            current_section = normalize_whitespace_inline(text)
        else:
            blocks.append(
                {
                    "page": None,
                    "section": current_section,
                    "text": text,
                }
            )

    if not blocks:
        fallback = clean_text(soup.get_text(separator="\n"))
        if fallback:
            return _extract_sectioned_blocks(fallback)

    return blocks


def parse_file(path: str) -> List[dict]:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix in {".html", ".htm"}:
        return read_html(path)
    return []


def build_chunks_from_file(path: str) -> List[ChunkRecord]:
    file_name = os.path.basename(path)
    title = os.path.splitext(file_name)[0]
    parsed_units = parse_file(path)

    chunk_records: List[ChunkRecord] = []
    idx = 0

    inferred_visibility = "public" if "public" in file_name.lower() else "internal"

    for unit in parsed_units:
        text = unit["text"]
        page = unit.get("page")
        section = unit.get("section")

        chunks = chunk_by_paragraphs(
            text=text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            min_chunk_chars=settings.min_chunk_chars,
            max_paragraph_chars=settings.max_paragraph_chars,
        )

        for chunk in chunks:
            chunk_id = f"{file_name}::chunk_{idx}"
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source=file_name,
                    title=title,
                    text=chunk,
                    page=page,
                    section=section,
                    extra_metadata={
                        "filepath": path,
                        "doc_type": Path(path).suffix.lower().replace(".", ""),
                        "visibility": inferred_visibility,
                        "allowed_users": [],
                        "allowed_roles": [],
                    },
                )
            )
            idx += 1

    return chunk_records