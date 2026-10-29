# Offline AI Document Assistant

A privacy-focused PDF question-answering application built with
**Python, Streamlit, FAISS, Sentence Transformers, and Ollama**. It uses
Retrieval-Augmented Generation (RAG) to retrieve relevant passages from
a PDF and ask a locally running language model to generate an answer.

The application is designed to run on your own computer without sending
PDF content to a cloud-based LLM API.

## Features

-   **Local LLM inference:** Generates answers through Ollama using
    `qwen2.5:1.5b`.
-   **PDF upload:** Upload a text-based PDF through the Streamlit
    sidebar.
-   **PDF text extraction:** Extracts page text using PyMuPDF.
-   **Chunking:** Splits document text into smaller, overlapping,
    sentence-aware chunks.
-   **Semantic retrieval:** Uses
    `sentence-transformers/all-MiniLM-L6-v2` embeddings and a FAISS
    index to find relevant passages.
-   **Page references:** Retains PDF page numbers for retrieved chunks
    and displays the source passages.
-   **Similarity filtering:** Supports a configurable cosine-similarity
    threshold in the semantic retrieval version of the application.
-   **Local browser interface:** Ask questions and inspect retrieved
    passages through Streamlit.
-   **Offline use:** After dependencies and model files have been
    downloaded, the application can run without an internet connection,
    provided Ollama and the local embedding model are available.

> **Note:** Retrieval and generated answers are not guaranteed to be
> correct. Always check the displayed source passages.
> Scanned/image-only PDFs may require OCR before their text can be
> searched.

## Technology Stack

  ------------------------------------------------------------------------------------------
  Component               Technology                                 Purpose
  ----------------------- ------------------------------------------ -----------------------
  Language                Python                                     Application logic

  UI                      Streamlit                                  Browser-based chat
                                                                     interface

  Local model runtime     Ollama                                     Runs the LLM locally

  LLM                     Qwen2.5 1.5B (`qwen2.5:1.5b`)              Generates answers

  Embedding model         `sentence-transformers/all-MiniLM-L6-v2`   Converts text into
                                                                     vectors

  Vector search           FAISS                                      Retrieves semantically
                                                                     similar chunks

  PDF extraction          PyMuPDF                                    Extracts text and page
                                                                     numbers

  Numerical processing    NumPy                                      Handles embedding
                                                                     arrays

  HTTP client             Requests                                   Calls the local Ollama
                                                                     API
  ------------------------------------------------------------------------------------------

## How It Works

1.  The user selects the default PDF or uploads a PDF.
2.  PyMuPDF extracts text from each page.
3.  The text is split into chunks, with page metadata preserved.
4.  Sentence Transformers converts each chunk into an embedding.
5.  FAISS searches for passages similar to the user's question.
6.  The application sends the question and retrieved context to Ollama.
7.  Qwen generates an answer, and the interface displays the answer and
    retrieved source pages.

``` text
PDF
 |
 v
PyMuPDF (extract text and page numbers)
 |
 v
Sentence-aware chunking
 |
 v
Sentence Transformers (embeddings)
 |
 v
FAISS (semantic retrieval)
 |
 v
Retrieved passages + question
 |
 v
Ollama / Qwen2.5 1.5B
 |
 v
Answer + source passages in Streamlit
```

## Project Structure

The project may contain the following files and folders:

``` text
OfflineAIAssistant/
├── documents/
│   └── software_engineering.pdf
├── venv/                 # Local virtual environment; do not upload
├── app.py                # Streamlit web application
├── read_pdf.py           # PDF text extraction and page metadata
├── chunk_text.py         # Sentence-aware chunking
├── build_index.py        # Builds the default PDF's FAISS index
├── ask_question.py       # Terminal-based RAG interface
├── test_ollama.py        # Tests Python-to-Ollama communication
├── faiss_index.bin       # Saved FAISS index
└── chunks.pkl            # Saved text chunks and page metadata
```

Your local folder may differ as the project evolves. Uploaded PDFs are
processed by the app; they do not necessarily need to be stored
permanently in `documents/`.

## Prerequisites

-   Windows, Linux, or macOS
-   Python 3.11 or 3.12 recommended
-   Ollama installed and running
-   The `qwen2.5:1.5b` model downloaded in Ollama
-   The embedding model downloaded and available in the Sentence
    Transformers cache
-   Internet access for the initial installation and model downloads;
    internet is not required for normal operation after setup

## Setup

### 1. Open the project directory

On Windows Command Prompt:

``` cmd
cd /d C:\Users\User\OfflineAIAssistant
```

### 2. Create and activate a virtual environment

If you already have a working `venv`, activate it instead of creating
another one.

``` cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install dependencies

``` cmd
python -m pip install --upgrade pip
pip install streamlit requests pymupdf sentence-transformers faiss-cpu numpy rank-bm25
```

`rank-bm25` is needed only if you have added the optional BM25
hybrid-retrieval code. The base semantic-search application uses FAISS.

### 4. Download the Ollama model

``` cmd
ollama pull qwen2.5:1.5b
```

If the model is already installed, this command should not require you
to download it again.

Check available models:

``` cmd
ollama list
```

### 5. Download the embedding model

Run this once while online:

``` cmd
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Embedding model ready')"
```

The application uses `local_files_only=True`, so the embedding model
must be available locally when running offline.

### 6. Prepare the default PDF index

Place a text-based PDF at:

``` text
documents/software_engineering.pdf
```

Then build the default index:

``` cmd
python build_index.py
```

This creates or updates `faiss_index.bin` and `chunks.pkl`.

If the default index and chunks have already been generated with the
current code and metadata format, you do not need to rebuild them unless
the PDF or chunking strategy changes.

### 7. Start Ollama and run the app

Make sure the Ollama service is running and that `qwen2.5:1.5b` is
available. Then run:

``` cmd
streamlit run app.py
```

Open the local address printed by Streamlit, usually:

``` text
http://localhost:8501
```

## Usage

1.  Open the application in your browser.
2.  Keep the default indexed document selected, or upload a PDF in the
    sidebar.
3.  Wait for a newly uploaded document to be processed and indexed.
4.  Enter a question about the document.
5.  Read the generated answer.
6.  Expand **View retrieved source passages** to inspect the evidence
    and PDF page numbers.
7.  If the answer looks incorrect, check whether the retrieved passages
    actually support it.

Example questions for a software engineering document:

-   What is cohesion?
-   Explain the Waterfall Model.
-   What are the disadvantages of the classical Waterfall Model?
-   What is the difference between function-oriented and object-oriented
    design?

The answer depends on the content of the selected PDF.

## Running the Terminal Version

The project also includes a terminal-based question-answering script:

``` cmd
python ask_question.py
```

Type a question when prompted. Enter `exit` to quit.

## Offline and Privacy Notes

-   Ollama serves the language model locally through
    `http://localhost:11434`.
-   The application uses local embedding-model files and a local FAISS
    index.
-   After all required packages and models are installed, normal
    document processing and answer generation can run without internet
    access.
-   This does not mean every dependency installation or first-time model
    download works offline; prepare the environment before
    disconnecting.
-   Avoid uploading confidential documents to public repositories. The
    `documents/` folder may contain personal or copyrighted files.

## Limitations

-   The default retrieval approach uses vector similarity; relevant
    passages may still be missed.
-   Similarity scores are ranking signals, not probabilities of
    correctness.
-   Displayed page numbers identify the source of retrieved text; they
    do not independently prove every statement in the generated answer.
-   Scanned PDFs may require OCR.
-   A small local model may produce incomplete or inaccurate answers.
-   The current application processes one selected PDF at a time.
-   Follow-up questions may not use earlier conversation turns as
    context unless conversation-memory logic is added.

## Troubleshooting

### Ollama connection error

Check that Ollama is installed and the model is available:

``` cmd
ollama list
```

If needed, start Ollama from your installed application or launch the
Ollama service, then retry.

### FAISS index or chunks are missing

Run:

``` cmd
python build_index.py
```

### Index and chunks do not match

Rebuild the index using the same `chunk_text.py`, embedding model, and
PDF-processing code that the app uses:

``` cmd
python build_index.py
```

### Embedding model is unavailable offline

Reconnect to the internet and run the embedding-model download command
in the setup section. Then test offline again.

### No text extracted from a PDF

The PDF may be scanned or image-only. OCR may be required before the
document can be searched.

## Future Improvements

Possible next steps include:

-   Evaluate and improve retrieval quality with a repeatable question
    set.
-   Add hybrid BM25 + FAISS retrieval and compare it with vector-only
    retrieval.
-   Add reranking for retrieved passages.
-   Improve handling of scanned PDFs with OCR.
-   Cache indexes for previously uploaded documents.
-   Add conversation-aware follow-up questions.
-   Add tests and a reproducible dependency file.
-   Add answer evaluation and source-grounding checks.

## License

Choose and add a license before publishing this project publicly. Do not
claim a license until you have selected one.
