"""
RAG answer generation — takes retrieved chunks + the user's question and
asks Gemini to answer using only that context. Reuses the same free
Gemini API key/pattern from Assessment 1.
"""
import os
import httpx

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer the user's question using ONLY the provided context excerpts below.
The context may be in Bangla, English, or both — answer in the same
language as the question where possible.
If the answer isn't contained in the context, say so explicitly rather
than guessing or using outside knowledge.
Cite which excerpt number(s) you used, like [1], [2]."""


async def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    if not retrieved_chunks:
        return "No relevant documents were found for this query (try removing filters or rephrasing)."

    context = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)
    )

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{
                "text": f"Context excerpts:\n{context}\n\nQuestion: {question}"
            }]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
