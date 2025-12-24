from pathlib import Path
import tempfile

from rag_notes_helper.rag.loaders import is_text_file


def test_text_file_detected():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("This is text")
        path = Path(f.name)

    assert is_text_file(path) is True
    path.unlink()


def test_binary_file_detected():
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"\x00\x01\x02")
        path = Path(f.name)

    assert is_text_file(path) is False
    path.unlink()

