from dataclasses import dataclass
from typing import TextIO, Iterable

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    source: str

def chunk_text(
    *,
    file_object: TextIO,
    doc_id: str,
    source: str,
    chunk_size: int,
    overlap: int,
) -> Iterable[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    buffer = ""
    chunk_id = 0

    for line in file_object:
        buffer += line

        while len(buffer) >= chunk_size:
            text = buffer[:chunk_size].strip()
            if text:
                yield Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                )
                chunk_id += 1

            buffer = buffer[chunk_size - overlap:]

    # remain text
    if buffer.strip():
        yield Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=buffer,
            source=source,
        )
