"""
Embeddings module.

Uses a multilingual sentence-embedding model so that Bangla and English
chunks land in the same vector space — this is what lets a query in
either language retrieve relevant chunks regardless of which language
the source document was written in.

Model: paraphrase-multilingual-MiniLM-L12-v2
- Trained on 50+ languages including Bengali.
- Small (~470MB) and fast enough to run locally on CPU, which matters
  since the brief requires this to run without external API calls.
- Trade-off: a larger multilingual model (e.g. LaBSE) would likely improve
  cross-lingual retrieval accuracy further, at a meaningfully higher
  memory/latency cost — MiniLM was chosen as the practical balance for a
  locally-run service.
"""

from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
