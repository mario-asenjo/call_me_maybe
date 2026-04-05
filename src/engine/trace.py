"""Trace utilities for constrained generation"""

from __future__ import annotations

from src.domain import GenerationTraceStep


class GenerationTrace:
    """In-memory trace collecto for decoding steps"""

    def __init__(self) -> None:
        """Initialize an empty trace"""
        self._steps: list[GenerationTraceStep] = []

    def add_step(self, step: GenerationTraceStep) -> None:
        """Append one trace step"""
        self._steps.append(step)

    def get_steps(self) -> list[GenerationTraceStep]:
        """Return a copy of the recorded steps"""
        return list(self._steps)

    def clear(self) -> None:
        """Remove all recorded steps"""
        self._steps.clear()
