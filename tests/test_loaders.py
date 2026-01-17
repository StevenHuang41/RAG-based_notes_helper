from pathlib import Path

from rag_notes_helper.rag.loaders import is_text_file


def test_text_file_detected(tmp_path):
    test_path = tmp_path / "test"
    test_path.write_text("This is text", encoding="utf-8")

    assert is_text_file(test_path) is True


def test_binary_file_detected(tmp_path):
    test_path = tmp_path / "test.bin"
    test_path.write_bytes(b"\x00")

    assert is_text_file(test_path) is False


def test_empty_file(tmp_path):
    test_path = tmp_path / "empty"
    test_path.touch()

    assert is_text_file(test_path) is False
