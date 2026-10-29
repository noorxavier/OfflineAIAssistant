import pymupdf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "documents" / "software_engineering.pdf"


def extract_pages(pdf_path):
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        A list of dictionaries containing page numbers and text.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


def extract_text(pdf_path):
    """
    Compatibility function for scripts that need the
    complete extracted text.
    """

    pages = extract_pages(pdf_path)

    return "\n\n".join(
        f"--- Page {item['page']} ---\n{item['text']}"
        for item in pages
    )


if __name__ == "__main__":
    pages = extract_pages(PDF_PATH)

    print("PDF extraction completed.")
    print("Pages containing text:", len(pages))

    for item in pages[:2]:
        print(f"\n--- Page {item['page']} ---")
        print(item["text"][:500])