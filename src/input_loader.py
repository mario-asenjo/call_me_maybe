"""
Utilities for loading and validating input JSON files
"""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.errors import InputFileError, InputJsonError, InputValidationError
from src.models import FunctionDefinition, PromptItem


def load_json_file(path: Path) -> Any:
    """
    Load a JSON file and return the parsed Python object
    :param path: Path to the JSON file
    :return: The parsed JSON content
    :raises: InputFileError: If the file does not exist or cannot be read
             InputJsonError: If the file content is not a valid JSON
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise InputFileError(f"Input file not found: {path}") from exc
    except OSError as exc:
        raise InputFileError(f"Could not read input file: {path}") from exc
    except JSONDecodeError as exc:
        raise InputJsonError(
            f"Invalid JSON in file: '{path}': line {exc.lineno},"
            f" column {exc.colno}"
        ) from exc


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """
    Load and validate function definitions from a JSON file

    :param path: Path to the function definitions file
    :return: A validated list of FunctionDefinition
    :raises: InputValidationError: If the data does not match
             the expected schema
    """
    data = load_json_file(path)
    adapter: TypeAdapter[
        list[FunctionDefinition]
    ] = TypeAdapter(
        list[FunctionDefinition]
    )

    try:
        definitions = adapter.validate_python(data)
    except ValidationError as exc:
        raise InputValidationError(
            f"Invalid function definitions in '{path}': {exc}"
        ) from exc

    names = [item.name for item in definitions]
    duplicated_names = sorted(
        {name for name in names if names.count(name) > 1}
    )
    if duplicated_names:
        duplicates = ", ".join(duplicated_names)
        raise InputValidationError(
            f"Duplicated function names found in '{path}': {duplicates}"
        )

    return definitions


def load_prompt_items(path: Path) -> list[PromptItem]:
    """
    Load and validate prompt items from a JSON file
    :param path: Path to the prompt item files
    :return: A validated list of PromptItem
    :raises: InputValidationError: If the data does not match the expected
             schema
    """
    data = load_json_file(path)
    adapter: TypeAdapter[list[PromptItem]] = TypeAdapter(list[PromptItem])

    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        raise InputValidationError(
            f"Invalid prompt items in: '{path}': {exc}"
        ) from exc
