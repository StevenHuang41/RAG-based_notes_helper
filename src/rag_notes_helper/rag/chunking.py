import re
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

    buffer = []
    buffer_count = 0
    chunk_id = 0

    for line in file_object:
        line = line.strip()
        if line:
            buffer.append(line)
            buffer_count += len(line) + 2

        # buffer exceed chunk size
        while buffer_count - 2 >= chunk_size:
            text = ", ".join(buffer[:-1])
            buffer = [line]
            buffer_count = len(line)

            if text:
                yield Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                )
                chunk_id += 1

    # remain buffer
    if buffer:
        text = ", ".join(buffer)
        yield Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=text,
            source=source,
        )
