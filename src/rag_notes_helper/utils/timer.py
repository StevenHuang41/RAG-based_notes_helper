import time
from .logger import get_logger

logger = get_logger("timer")

class LapTimer:
    def __init__(self) -> None:
        self._last = time.perf_counter()

    def start(self):
        self._last  = time.perf_counter()
        return ""


    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self._last) * 1000
        self._last = now
        return elapsed


