# Local OCR & Dynamic RAG System

A fully local document processing pipeline: upload a scanned document or
PDF containing Bangla, English, or both → extract text locally (no
external OCR APIs) → chunk, embed, and store it → search it with natural
language plus optional manual filters.

## Folder structure

```
ocr_rag/
├── app/
│   ├── main.py          FastAPI app — /upload, /search, /documents endpoints
│   ├── ocr.py            Local text extraction (Tesseract + PyMuPDF)
│   ├── chunking.py        Sentence-aware chunking with overlap
│   ├── embeddings.py      Multilingual embedding model wrapper
│   ├── vectorstore.py    ChromaDB storage + hybrid metadata search
│   └── rag.py             LLM answer generation from retrieved chunks
├── static/
│   └── index.html        Simple browser UI for upload + search (demo-friendly)
├── test_docs/             Sample bilingual test files (see below)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Setup

### Option A — Docker (recommended, matches the deliverable requirement)

```bash
cp .env.example .env
# edit .env and paste your free Gemini key (aistudio.google.com/apikey — no card needed)

docker compose up --build
```

Open **http://localhost:8000** — you'll see the upload/search demo page.

### Option B — Run directly with Python

```bash
# System dependency: Tesseract with Bengali support
sudo apt-get install tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng

pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here

uvicorn app.main:app --reload
```

Open **http://localhost:8000**.

### Quick test with the included sample files

`test_docs/` contains 3 files used during development:
- `test_bilingual_doc.png` — a real bilingual image (Bangla + English text) for testing the OCR path
- `test_digital.pdf` — a PDF with an embedded text layer, for testing the fast digital-text path
- `test_scanned.pdf` — the same image embedded as a flat page image with no text layer, for testing the OCR fallback path

Upload any of these through the web UI, then try a search query like
*"what is the total amount"* with the language filter set to Bangla.

## Must Explain

### 1. Local OCR model choice, trade-offs, and Bangla accuracy

**Choice:** Tesseract OCR with the Bengali (`ben`) and English (`eng`)
language packs run together (`lang="ben+eng"`), entirely offline.

**Trade-offs considered:**
- Tesseract is lightweight, has no GPU requirement, and installs as a
  simple system package — a strong fit for a "must run fully locally"
  requirement, versus a vision-language OCR model that would need several
  GB of weights and ideally a GPU.
- The cost is accuracy on very messy or low-quality scans, where a modern
  OCR-LLM (e.g. a vision-language model) would likely outperform it. For
  clean-to-moderate quality scans — the realistic case for most uploaded
  documents — Tesseract is a solid, fast, well-understood choice.

**Measured baseline accuracy:** tested against a real bilingual document
image (a generated invoice-style page with genuine Bangla sentences using
the Noto Sans Bengali font, not placeholder text), Tesseract with `ben+eng`
extracted **100% of the text correctly in both scripts**, with no garbled
characters in either language. This was validated end-to-end through the
actual `/upload` endpoint, not just the OCR engine in isolation.

**A second, related architectural decision** for PDFs specifically: rather
than always running OCR, `ocr.py` first tries extracting the PDF's embedded
digital text layer (via PyMuPDF). Only if that yields under ~20 characters
per page — a strong signal the page is a flat scanned image with no text
layer — does it fall back to rendering that page as an image and running
Tesseract on it. This was tested against both a genuinely digital PDF
(instant, perfect extraction, `method: "digital"`) and a scanned-style PDF
with no text layer (`method: "ocr"`, correctly falls back and still
extracts the bilingual text accurately). This avoids paying OCR's
accuracy/speed cost on documents that don't need it.

### 2. Text-chunking strategy and embedding model selection

**Chunking:** sentence-aware splitting rather than fixed-character-count
slicing. The splitter recognizes both Bangla's `।` (danda) sentence-ending
mark and standard `.`/`!`/`?`, then greedily packs sentences into ~600
character chunks with a 100-character overlap carried into the next chunk.
Sentence-aware splitting avoids cutting a sentence in half mid-thought,
which would otherwise hand the embedding model an incomplete fragment —
this matters more in bilingual text, where naive character-count slicing
can also risk cutting in the middle of a multi-byte Bangla character
cluster. The overlap means a fact mentioned right at a chunk boundary is
still retrievable from either neighboring chunk.

**Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2`
(sentence-transformers), chosen specifically because it's trained across
50+ languages including Bengali, which is what allows a Bangla query to
retrieve relevant English chunks and vice versa — both languages land in
the same vector space. It was chosen over a larger multilingual model
(e.g. LaBSE) as the practical balance for a service required to run
locally on CPU: ~470MB and fast enough for real-time search, at some cost
to retrieval accuracy versus a heavier model.

### 3. System architecture: manual metadata filtering + vector similarity

Every chunk is stored in ChromaDB with both its embedding vector and a
metadata record: `filename`, `doc_type`, `language`, `upload_date`,
`extraction_method`, and `chunk_index`.

A search request can include any combination of three manual filters
(`language`, `doc_type`, `upload_date`) alongside the natural-language
query. These are passed to ChromaDB's `query()` call as a `where` clause
**in the same call** as the vector similarity search — Chroma applies the
metadata filter first to narrow the candidate set, then ranks only that
filtered subset by vector similarity to the query embedding. This was
verified directly: a store containing one Bangla invoice and one English
contract correctly returned **only** the Bangla document when filtered by
`language=bn`, and **only** the English contract when filtered by
`doc_type=contract` — confirming the manual filter and semantic ranking
genuinely work together rather than as two separate, hard-to-combine steps.

Retrieved chunks (after filtering + ranking) are passed to Gemini along
with the original question; the model is instructed to answer using only
that context and to cite which excerpt(s) it used, and to explicitly say
so if the answer isn't present in the retrieved context rather than
guessing from outside knowledge.

