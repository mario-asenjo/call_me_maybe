"""Validation and serialization helpers for final function-calling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.domain import (
    FunctionCallCore,
    FunctionCallResult,
    FunctionDefinition,
    OutputFileError,
    OutputValidationError,
)


def validate_function_call_core(
    function_call_core: FunctionCallCore,
    function_definitions: list[FunctionDefinition],
) -> FunctionCallCore:
    """Validate and normalize a generated function call core."""
    definition_map = {definition.name: definition for definition in function_definitions}
    function_definition = definition_map.get(function_call_core.fn_name)
    if function_definition is None:
        raise OutputValidationError(
            f"Unknown function name generated: {function_call_core.fn_name!r}"
        )

    expected_parameter_names = list(function_definition.parameters.keys())
    actual_parameter_names = list(function_call_core.args.keys())

    missing_parameter_names = [
        name for name in expected_parameter_names if name not in function_call_core.args
    ]
    if missing_parameter_names:
        raise OutputValidationError(
            f"Missing required parameters for {function_definition.name!r}: "
            f"{missing_parameter_names}"
        )

    extra_parameter_names = [
        name for name in actual_parameter_names if name not in function_definition.parameters
    ]
    if extra_parameter_names:
        raise OutputValidationError(
            f"Unexpected parameters for {function_definition.name!r}: "
            f"{extra_parameter_names}"
        )

    normalized_arguments: dict[str, Any] = {}
    for parameter_name in expected_parameter_names:
        parameter_value = function_call_core.args[parameter_name]
        parameter_type = function_definition.parameters[parameter_name].type
        normalized_arguments[parameter_name] = _normalize_argument_value(
            value=parameter_value,
            expected_type=parameter_type,
            function_name=function_definition.name,
            parameter_name=parameter_name,
        )

    return FunctionCallCore(fn_name=function_call_core.fn_name, args=normalized_arguments)


def build_function_call_result(
    prompt: str,
    function_call_core: FunctionCallCore,
    function_definitions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Build the validated final result item for one prompt."""
    normalized_core = validate_function_call_core(function_call_core, function_definitions)
    return FunctionCallResult(
        prompt=prompt,
        fn_name=normalized_core.fn_name,
        args=normalized_core.args,
    )


def write_function_call_results(
    output_path: Path,
    results: list[FunctionCallResult],
) -> None:
    """Write the final JSON output file to disk."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump([result.model_dump() for result in results], handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise OutputFileError(f"Could not write output file: {output_path}") from exc


def _normalize_argument_value(
    value: Any,
    expected_type: str,
    function_name: str,
    parameter_name: str,
) -> Any:
    """Normalize and validate one argument value against the schema type."""
    if expected_type == "string":
        if not isinstance(value, str):
            raise OutputValidationError(
                _build_type_error_message(function_name, parameter_name, expected_type, value)
            )
        return value

    if expected_type == "boolean":
        if not isinstance(value, bool):
            raise OutputValidationError(
                _build_type_error_message(function_name, parameter_name, expected_type, value)
            )
        return value

    if expected_type == "integer":
        if isinstance(value, bool):
            raise OutputValidationError(
                _build_type_error_message(function_name, parameter_name, expected_type, value)
            )
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise OutputValidationError(
            _build_type_error_message(function_name, parameter_name, expected_type, value)
        )

    if expected_type == "number":
        if isinstance(value, bool):
            raise OutputValidationError(
                _build_type_error_message(function_name, parameter_name, expected_type, value)
            )
        if isinstance(value, (int, float)):
            return float(value)
        raise OutputValidationError(
            _build_type_error_message(function_name, parameter_name, expected_type, value)
        )

    raise OutputValidationError(
        f"Unsupported schema type {expected_type!r} for {function_name!r}.{parameter_name!r}"
    )


def _build_type_error_message(
    function_name: str,
    parameter_name: str,
    expected_type: str,
    value: Any,
) -> str:
    """Build a stable and readable type error message."""
    actual_type_name = type(value).__name__
    return (
        f"Invalid type for {function_name!r}.{parameter_name!r}: expected "
        f"{expected_type}, got {actual_type_name}: {value!r}"
    )
