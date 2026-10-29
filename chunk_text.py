import re
from read_pdf import extract_pages, PDF_PATH


def split_into_chunks(
    text,
    chunk_size=1000,
    overlap_sentences=2
):
    """
    Split text into sentence-aware chunks.

    chunk_size:
        Approximate maximum chunk size in characters.

    overlap_sentences:
        Number of sentences shared between adjacent chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Basic sentence splitting
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_length = len(sentence)

        # Flush the current chunk before adding a sentence
        if (
            current_sentences
            and current_length + sentence_length + 1 > chunk_size
        ):
            chunks.append(" ".join(current_sentences))

            # Preserve the last few sentences for context
            if overlap_sentences:
                current_sentences = current_sentences[
                    -overlap_sentences:
                ]
            else:
                current_sentences = []

            current_length = sum(
                len(item) for item in current_sentences
            ) + max(0, len(current_sentences) - 1)

        # Handle an unusually long sentence
        if sentence_length > chunk_size and not current_sentences:
            for start in range(0, sentence_length, chunk_size):
                chunks.append(
                    sentence[start:start + chunk_size]
                )
            continue

        current_sentences.append(sentence)

        current_length = sum(
            len(item) for item in current_sentences
        ) + max(0, len(current_sentences) - 1)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def create_document_chunks(
    pages,
    chunk_size=1000,
    overlap_sentences=2
):
    """
    Split each PDF page into chunks and preserve page numbers.
    """

    document_chunks = []

    for item in pages:
        page_chunks = split_into_chunks(
            item["text"],
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences
        )

        for chunk in page_chunks:
            document_chunks.append({
                "page": item["page"],
                "text": chunk
            })

    return document_chunks


if __name__ == "__main__":
    pages = extract_pages(PDF_PATH)

    chunks = create_document_chunks(pages)

    print("Pages with text:", len(pages))
    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks[:3], start=1):
        print(
            f"\n--- Chunk {index}, "
            f"Page {chunk['page']} ---"
        )
        print(chunk["text"])