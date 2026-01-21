import time
from contextlib import contextmanager
from rag_notes_helper.utils.logger import get_logger


def _get_logger():
    return get_logger("latency")

@contextmanager
def time_block(name: str):
    start = time.perf_counter()

    try :
        yield
    finally :
        end = time.perf_counter()
        _get_logger().info(f"{name} latency={(end - start) * 1000:.2f} ms")

def deco_time_block(func):
    def wrapper(*args, **kws):
        with time_block(func.__name__):
            return func(*args, **kws)

    return wrapper


