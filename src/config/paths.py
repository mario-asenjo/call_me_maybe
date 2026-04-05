"""Default filesystem paths used by the project"""
from __future__ import annotations

from pathlib import Path


DEFAULT_INPUT_DIR: Path = Path("data/input")
DEFAULT_OUTPUT_DIR: Path = Path("data/output")
DEFAULT_FUNCTION_DEFINITIONS: Path = (
        DEFAULT_INPUT_DIR / "function_definitions.json"
)
DEFAULT_PROMPT_TESTS: Path = (
        DEFAULT_INPUT_DIR / "function_calling_tests.json"
)
DEFAULT_OUTPUT_FILE: Path = (
        DEFAULT_OUTPUT_DIR / "function_calling_results.json"
)
