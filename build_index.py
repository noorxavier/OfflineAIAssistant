import faiss
import numpy as np
import pickle

from pathlib import Path
from sentence_transformers import SentenceTransformer

from read_pdf import extract_pages, PDF_PATH
from chunk_text import create_document_chunks


BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "faiss_index.bin"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(pdf_path=PDF_PATH):
    print("Extracting PDF pages...")

    pages = extract_pages(pdf_path)

    if not pages:
        raise ValueError(
            "No text was extracted. The PDF may contain scanned pages."
        )

    print("Creating chunks...")

    chunks = create_document_chunks(pages)

    if not chunks:
        raise ValueError("No chunks were created.")

    print("Total pages with text:", len(pages))
    print("Total chunks:", len(chunks))

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        local_files_only=True
    )

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print("Creating FAISS index...")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("Saving index...")

    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print("\nIndex built successfully!")
    print("Vectors:", index.ntotal)
    print("Saved:", INDEX_PATH)
    print("Saved:", CHUNKS_PATH)


if __name__ == "__main__":
    build_index()