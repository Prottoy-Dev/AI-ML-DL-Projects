"""
Vector store module — wraps ChromaDB for storage and hybrid search.

Hybrid search architecture: ChromaDB's query() accepts both a vector
(for semantic similarity ranking) and a `where` clause (for exact metadata
matching) in the same call. Chroma applies the metadata filter FIRST to
narrow the candidate set, then runs vector similarity ranking only within
that filtered subset. This means a manual filter like "language=bn" and
a semantic query work together rather than as two separate, hard-to-combine
steps — exactly the "dynamic hybrid search" the brief asks for.
"""

import uuid
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_document_chunks(chunks: list[str], embeddings: list[list[float]],
                         base_metadata: dict, doc_id: str):
    """Stores each chunk with shared document-level metadata plus its own
    chunk index, so search results can be traced back to source + position."""
    collection = get_collection()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {**base_metadata, "chunk_index": i, "doc_id": doc_id}
        for i in range(len(chunks))
    ]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return ids


def hybrid_search(query_embedding: list[float], top_k: int = 5,
                   filters: dict | None = None) -> dict:
    """
    filters example: {"language": "bn", "doc_type": "invoice"}
    Combined with AND semantics automatically when more than one key is given.
    """
    collection = get_collection()

    where_clause = None
    if filters:
        active = {k: v for k, v in filters.items() if v}
        if len(active) == 1:
            where_clause = active
        elif len(active) > 1:
            where_clause = {"$and": [{k: v} for k, v in active.items()]}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
    )
    return results


def list_documents() -> list[dict]:
    """Returns one row per unique uploaded document (not per chunk), for
    a quick overview of what's in the store."""
    collection = get_collection()
    all_items = collection.get()
    seen = {}
    for meta in all_items["metadatas"]:
        doc_id = meta.get("doc_id")
        if doc_id not in seen:
            seen[doc_id] = {k: v for k, v in meta.items() if k != "chunk_index"}
    return list(seen.values())
