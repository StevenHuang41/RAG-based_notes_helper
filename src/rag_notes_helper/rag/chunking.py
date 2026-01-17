from collections import deque
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

    buffer = deque()
    buffer_len = 0
    chunk_id = 0

    for line in file_object:
        line = line.strip()
        if not line:
            continue

        line_len = len(line)

        if buffer_len + line_len >= chunk_size and buffer:
            text = ", ".join(buffer)
            if text:
                yield Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                )
                chunk_id += 1

            tmp_buffer = deque()
            tmp_len = 0

            while tmp_len < overlap and buffer:
                last_line = buffer.pop()
                tmp_buffer.appendleft(last_line)
                tmp_len += len(last_line)

            buffer = tmp_buffer
            buffer_len = tmp_len


        buffer.append(line)
        buffer_len += line_len

    # remain buffer
    if buffer:
        text = ", ".join(buffer)
        yield Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=text,
            source=source,
        )
