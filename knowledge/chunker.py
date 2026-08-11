"""
knowledge/chunker.py
─────────────────────
Splits a document into retrievable chunks that keep their heading context.

Retrieval quality is mostly decided here rather than in the vector store: a
chunk that has lost the heading it sat under ("5 days for immediate family")
is far less useful than one that kept it ("Section 1 – Leave Policy › 1.5
Bereavement Leave: 5 days for immediate family"). So headings are detected,
carried onto every chunk as metadata, and prefixed onto the embedded text.

Falls back to paragraph packing for documents with no discernible structure,
which is what most ad-hoc HR uploads look like.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

# "SECTION 3 – EXPENSES" / "SECTION 3 - EXPENSES"
_MAJOR = re.compile(r"^\s*(SECTION\s+\d+\s*[–—-]?\s*.*)$")
# "2.4 Equipment and Security"
_MINOR = re.compile(r"^\s*(\d+\.\d+\s+\S.*)$")
# Markdown "## Heading"
_MARKDOWN = re.compile(r"^\s*#{1,6}\s+(\S.*)$")
# "OVERTIME POLICY" — a short all-caps line used as a heading
_UPPER = re.compile(r"^\s*([A-Z][A-Z0-9 &/(),'.-]{3,60})\s*$")
# Rules and underlines used as separators
_DIVIDER = re.compile(r"^\s*([=\-─━_*]{3,}|-{3,})\s*$")


@dataclass
class Chunk:
    """One retrievable unit of a document."""
    text: str            # body text, without the heading prefix
    section: str         # heading trail, e.g. "SECTION 1 – LEAVE POLICY › 1.1 Annual Leave"

    def embedding_text(self) -> str:
        """Text handed to the embedder and to BM25 — heading included."""
        return f"{self.section}\n{self.text}".strip() if self.section else self.text


def _detect_heading(line: str) -> Optional[tuple]:
    """Return (level, heading_text) when the line looks like a heading."""
    if _DIVIDER.match(line):
        return None
    m = _MARKDOWN.match(line)
    if m:
        return (1 if line.lstrip().startswith("##") is False else 2, m.group(1).strip())
    m = _MAJOR.match(line)
    if m:
        return (1, m.group(1).strip())
    m = _MINOR.match(line)
    if m:
        return (2, m.group(1).strip())
    m = _UPPER.match(line)
    if m and len(line.strip()) <= 60 and not line.strip().endswith((".", ":", "•")):
        return (1, m.group(1).strip())
    return None


def _trail(major: str, minor: str) -> str:
    parts = [p for p in (major, minor) if p]
    return " › ".join(parts)


def _split_long(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Break an oversized block on paragraph, then sentence, then hard boundaries.

    Overlap carries the tail of one piece into the next so a fact spanning a
    split is still fully present in at least one chunk.
    """
    if len(text) <= max_chars:
        return [text]

    # Prefer paragraph boundaries.
    units = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(units) == 1:
        # Single paragraph — fall back to sentence boundaries.
        units = re.split(r"(?<=[.!?])\s+", text)

    pieces, current = [], ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{unit}".strip() if tail else unit
        else:
            current = unit
        # A single unit larger than the budget still has to be cut somewhere.
        while len(current) > max_chars:
            pieces.append(current[:max_chars])
            current = current[max_chars - overlap:] if overlap else current[max_chars:]
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_document(
    text: str,
    max_chars: int = 1200,
    overlap: int = 150,
    min_chars: int = 40,
) -> List[Chunk]:
    """
    Split `text` into chunks, preserving the heading each one sat under.

    Args:
        max_chars: soft ceiling on a chunk's body length.
        overlap:   characters carried across a forced split.
        min_chars: chunks shorter than this are merged into the previous one,
                   so a stray heading line does not become its own chunk.
    """
    if not text or not text.strip():
        return []

    major = minor = ""
    blocks: List[tuple] = []          # (section_trail, body)
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            blocks.append((_trail(major, minor), body))

    for line in text.splitlines():
        heading = _detect_heading(line)
        if heading:
            level, title = heading
            flush()
            if level == 1:
                major, minor = title, ""
            else:
                minor = title
            continue
        if _DIVIDER.match(line):
            continue
        buffer.append(line)
    flush()

    if not blocks:                    # no structure at all — treat as one block
        blocks = [("", text.strip())]

    chunks: List[Chunk] = []
    for section, body in blocks:
        for piece in _split_long(body, max_chars, overlap):
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) < min_chars and chunks:
                # Too small to retrieve on its own; fold into the previous chunk.
                chunks[-1] = Chunk(text=f"{chunks[-1].text}\n{piece}".strip(),
                                   section=chunks[-1].section)
            else:
                chunks.append(Chunk(text=piece, section=section))

    return chunks
