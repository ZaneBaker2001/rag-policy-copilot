from typing import List

from openai import OpenAI

from app.config import settings
from app.models import RetrievedChunk, SearchDiagnostics


SYSTEM_PROMPT = """You are a careful enterprise RAG assistant.
Answer only from the provided context.
If the answer is not clearly supported by the context, say that you do not have enough information in the documents.
Be precise and concise.
Always include inline citations in the form [source | chunk_id | page X] when available.
Do not invent policy terms, dates, rules, or obligations.
"""


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks[: settings.max_context_chunks]:
        page_part = f"page {c.page}" if c.page is not None else "page n/a"
        section_part = f" | section {c.section}" if c.section else ""
        parts.append(
            f"[SOURCE: {c.source} | CHUNK: {c.chunk_id} | {page_part}{section_part}]\n{c.text}"
        )
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: List[RetrievedChunk],
    diagnostics: SearchDiagnostics,
) -> str:
    if not chunks or diagnostics.abstained:
        reason = diagnostics.reason or "insufficient_evidence"
        return (
            "I do not have enough reliable information in the indexed documents to answer "
            f"that confidently. Reason: {reason}."
        )

    context = build_context(chunks)

    if not settings.openai_api_key:
        fallback = [
            "OpenAI API key not configured, so this is an extractive fallback answer.",
            "",
            "Most relevant evidence:",
        ]
        for c in chunks[:3]:
            page_part = f"page {c.page}" if c.page is not None else "page n/a"
            fallback.append(
                f"- {c.text[:350]}... [{c.source} | {c.chunk_id} | {page_part}]"
            )
        return "\n".join(fallback)

    client = OpenAI(api_key=settings.openai_api_key)
    user_prompt = f"""Question:
{question}

Context:
{context}

Instructions:
- Answer only using the context.
- If the context is insufficient, explicitly say so.
- Include inline citations for each substantive claim.
- Prefer the shortest accurate answer.
"""

    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content or "No answer generated."