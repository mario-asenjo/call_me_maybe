"""Performance utilities for lightweight timing"""
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def measure_time(label: str) -> Iterator[None]:
    """Measure and print the execution time of a code block"""
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000.0
        print(f"[timing] {label}: {elapsed_ms} ms")
