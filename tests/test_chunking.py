from app.utils import chunk_by_paragraphs


def test_chunk_by_paragraphs_preserves_paragraph_boundaries():
    text = """

    Section A

    This is the first paragraph. It has a few sentences. It should stay together.

    This is the second paragraph. It should not be merged strangely in the middle
    of a sentence if the chunking is paragraph-aware.

    This is the third paragraph.
    """

    chunks = chunk_by_paragraphs(
        text=text,
        chunk_size=180,
        overlap=40,
        min_chunk_chars=40,
        max_paragraph_chars=300,
    )

    assert len(chunks) >= 2
    assert "first paragraph" in chunks[0]
    assert all("\n\n" in c or len(c) < 180 for c in chunks)