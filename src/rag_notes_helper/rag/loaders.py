from pathlib import Path

# TEXT_EXTENSIONS = {
#     ".md",
#     ".txt",
# }


def _has_null(data: bytes) -> bool:
    return b"\x00" in data

def is_text_file(path: Path) -> bool:
    try :
        with path.open("rb") as f:
            # check first 4KB characters
            head = f.read(4 * 1024)
    except Exception:
        # print(f"Error when checking _is_text_file: {e}")
        return False

    if not head or _has_null(head):
        return  False

    return True
