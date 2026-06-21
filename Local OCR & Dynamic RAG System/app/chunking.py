"""
Chunking module.

Strategy: split on sentence boundaries (both Bangla's '।' danda and
standard '.', '!', '?'), then greedily pack sentences into chunks up to
a target size, with a small overlap between consecutive chunks so that
context isn't lost at a chunk boundary mid-thought.

Sentence-aware chunking (rather than fixed-character-count slicing) avoids
cutting a sentence in half, which would otherwise hand the embedding model
a fragment that doesn't represent a complete idea — this matters more in
bilingual text where naive splitting can also accidentally cut a Bangla
sentence in the middle of a multi-byte character cluster.
"""

import re

SENTENCE_BOUNDARY = re.compile(r"(?<=[।.!?])\s+")

TARGET_CHUNK_CHARS = 600
OVERLAP_CHARS = 100


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, target_chars: int = TARGET_CHUNK_CHARS,
               overlap_chars: int = OVERLAP_CHARS) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence

        if len(candidate) > target_chars and current:
            chunks.append(current)
            # carry the tail of the previous chunk forward for overlap/context
            overlap = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = (overlap + " " + sentence).strip()
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
