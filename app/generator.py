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


def build_fallback_answer(
    question: str,
    chunks: List[RetrievedChunk],
    diagnostics: SearchDiagnostics,
    reason: str,
) -> str:
    if not chunks or diagnostics.abstained:
        abstain_reason = diagnostics.reason or reason or "insufficient_evidence"
        return (
            "I do not have enough reliable information in the indexed documents to answer "
            f"that confidently. Reason: {abstain_reason}."
        )

    top = chunks[0]
    page_part = f"page {top.page}" if top.page is not None else "page n/a"

    return (
        "Using extractive fallback because answer generation was unavailable.\n\n"
        f"Best matching evidence for: {question}\n"
        f"- {top.text[:700].strip()} [{'{} | {} | {}'.format(top.source, top.chunk_id, page_part)}]"
    )


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
        return build_fallback_answer(
            question=question,
            chunks=chunks,
            diagnostics=diagnostics,
            reason="openai_api_key_not_configured",
        )

    try:
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

        return response.choices[0].message.content or build_fallback_answer(
            question=question,
            chunks=chunks,
            diagnostics=diagnostics,
            reason="empty_model_response",
        )

    except Exception as exc:
        return build_fallback_answer(
            question=question,
            chunks=chunks,
            diagnostics=diagnostics,
            reason=f"openai_error: {exc.__class__.__name__}",
        )