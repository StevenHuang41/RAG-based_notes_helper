from pathlib import Path

# TEXT_EXTENSIONS = {
#     ".md",
#     ".txt",
# }

def _is_text_file(data: bytes) -> bool:
    return b"\x00" not in data

def load_text_file(path: Path) -> str | None:
    # Load a file if it is likely text.

    # filter out symlinks or directories
    if not path.is_file():
        return None

    # check first 4KB characters
    try :
        with path.open("rb") as f:
            head = f.read(4 * 1024) # read first 4KB
    except Exception:
        return None

    if not head or not _is_text_file(head):
        return None

    try :
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

