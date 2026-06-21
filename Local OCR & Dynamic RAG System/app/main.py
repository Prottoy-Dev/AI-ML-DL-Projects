"""
FastAPI backend for the local OCR + Dynamic RAG system.

Endpoints:
  POST /upload  — accepts a document (PDF or image) + metadata, runs OCR
                  locally, chunks and embeds the text, stores it.
  POST /search  — accepts a natural-language query + optional manual
                  filters (language, doc_type, date), runs hybrid search,
                  and returns an LLM-generated answer with sources.
  GET  /documents — lists what's currently stored.
"""

from dotenv import load_dotenv
load_dotenv()

import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.ocr import extract_text
from app.chunking import chunk_text
from app.embeddings import embed_texts, embed_query
from app.vectorstore import add_document_chunks, hybrid_search, list_documents
from app.rag import generate_answer

app = FastAPI(title="Local OCR & Dynamic RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("general"),
    language: str = Form("auto"),
    upload_date: str = Form(...),
):
    file_bytes = await file.read()

    try:
        extraction = extract_text(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    text = extraction["text"].strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from this document.",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Extracted text produced no chunks.")

    embeddings = embed_texts(chunks)

    doc_id = str(uuid.uuid4())[:8]
    base_metadata = {
        "filename": file.filename,
        "doc_type": doc_type,
        "language": language,
        "upload_date": upload_date,
        "extraction_method": extraction["method"],
    }

    add_document_chunks(chunks, embeddings, base_metadata, doc_id=doc_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": extraction["pages"],
        "extraction_method": extraction["method"],
        "chunks_stored": len(chunks),
        "preview": text[:300],
    }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    language: str | None = None
    doc_type: str | None = None
    upload_date: str | None = None


@app.post("/search")
async def search(req: SearchRequest):
    query_vec = embed_query(req.query)

    filters = {
        "language": req.language,
        "doc_type": req.doc_type,
        "upload_date": req.upload_date,
    }

    results = hybrid_search(query_vec, top_k=req.top_k, filters=filters)

    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    answer = await generate_answer(req.query, chunks)

    return {
        "answer": answer,
        "sources": [
            {"text": c[:200], "metadata": m}
            for c, m in zip(chunks, metadatas)
        ],
    }


@app.get("/documents")
async def get_documents():
    return {"documents": list_documents()}


@app.get("/")
async def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
