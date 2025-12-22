from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    source: str

def chunk_text(
    *,
    text: str,
    doc_id: str,
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk,
                    source=source,
                )
            )
            chunk_id += 1

        if end == length:
            break

        start = end - overlap

    return chunks

