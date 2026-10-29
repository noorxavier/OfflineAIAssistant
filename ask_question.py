import pickle
import requests
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# Project configuration
BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "faiss_index.bin"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:1.5b"


def load_resources():
    """Load the FAISS index, text chunks, and embedding model."""

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Index files are missing. Run build_index.py first."
        )

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading document chunks...")
    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            "The FAISS index and chunks do not match. "
            "Run build_index.py again."
        )

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True
    )

    return index, chunks, embedding_model


def retrieve_chunks(question, index, chunks, embedding_model, top_k=3):
    """Retrieve the most relevant chunks for a question."""

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )

    k = min(top_k, index.ntotal)

    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "text": chunks[int(idx)]
        })

    return results


def generate_answer(question, retrieved_chunks):
    """Generate an answer using the local Ollama model."""

    if not retrieved_chunks:
        return "No relevant passages were found in the document."

    context = "\n\n".join(
        result["text"] for result in retrieved_chunks
    )

    prompt = f"""
You are an assistant that answers questions using the provided document.

Instructions:
1. Answer using the document context below.
2. Do not invent facts that are not supported by the context.
3. If the answer is not available in the context, say:
   "I could not find this information in the provided document."
4. Explain the answer clearly and concisely.

Document context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]


def main():
    try:
        index, chunks, embedding_model = load_resources()

        print("\nOffline PDF Assistant is ready!")
        print("Ask questions about your PDF.")
        print("Type 'exit' to quit.\n")

        while True:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            retrieved = retrieve_chunks(
                question,
                index,
                chunks,
                embedding_model,
                top_k=3
            )

            print("\nRelevant passages retrieved:")

            for number, result in enumerate(retrieved, start=1):
                print(
                    f"\nPassage {number} "
                    f"(similarity: {result['score']:.3f})"
                )
                print(result["text"][:300])

            print("\nGenerating answer...")

            answer = generate_answer(question, retrieved)

            print("\nAssistant:", answer)
            print("\n" + "-" * 60 + "\n")

    except FileNotFoundError as error:
        print("File error:", error)

    except requests.RequestException as error:
        print("Could not connect to Ollama:", error)

    except Exception as error:
        print("An unexpected error occurred:", error)


if __name__ == "__main__":
    main()