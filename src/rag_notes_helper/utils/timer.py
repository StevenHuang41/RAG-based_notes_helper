import time
from contextlib import contextmanager
from .logger import get_logger

logger = get_logger("latency")

@contextmanager
def time_block(name: str):
    start = time.perf_counter()

    try :
        yield
    finally :
        end = time.perf_counter()
        logger.info(f"{name} latency={(end - start) * 1000:.2f} ms")

def deco_time_block(func):
    def wrapper(*args, **kws):
        with time_block(func.__name__):
            return func(*args, **kws)

    return wrapper


