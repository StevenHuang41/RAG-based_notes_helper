from collections import deque
from dataclasses import dataclass
from collections.abc import Iterator

from rag_notes_helper.core.config import get_settings

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    source: str

def chunk_lines(
    *,
    lines: Iterator[str],
    doc_id: str,
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> Iterator[Chunk]:

    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    buffer = deque()
    buffer_len = 0
    chunk_id = 0

    for line in lines:
        line_len = len(line)

        if buffer_len + line_len >= chunk_size and buffer:
            yield Chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=", ".join(buffer),
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
        yield Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=", ".join(buffer),
            source=source,
        )
