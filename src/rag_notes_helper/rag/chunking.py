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
    start_line: int | None = None
    end_line: int | None = None

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

    buffer: deque[tuple[int, str]] = deque()
    buffer_len = 0
    chunk_id = 0

    def make_chunk() -> Chunk:
        return Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text="\n".join(text for _, text in buffer),
            source=source,
            start_line=buffer[0][0],
            end_line=buffer[-1][0],
        )

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")

        # Leading blank lines are not part of a chunk, but still count toward
        # the physical line number of the next non-blank line.
        if not line.strip() and not buffer:
            continue

        line_len = len(line)

        if buffer_len + line_len + 1 >= chunk_size and buffer:
            yield make_chunk()
            chunk_id += 1

            tmp_buffer = deque()
            tmp_len = 0

            while tmp_len < overlap and buffer:
                last_line = buffer.pop()
                tmp_buffer.appendleft(last_line)
                tmp_len += len(last_line[1]) + 1

            buffer = tmp_buffer
            buffer_len = tmp_len

        buffer.append((line_number, line))
        buffer_len += line_len + (1 if len(buffer) > 0 else 0)

    # remain buffer
    if buffer and any(text.strip() for _, text in buffer):
        yield make_chunk()
