"""Command-line entry point for the function-calling project."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.application import build_function_call_result, load_function_definitions, load_prompt_items, write_function_call_results
from src.config import DEFAULT_FUNCTION_DEFINITIONS, DEFAULT_OUTPUT_FILE, DEFAULT_PROMPT_TESTS
from src.domain import ProjectError
from src.engine import GenerationEngine
from src.infrastructure import LlmClient
from src.utils import measure_time


def build_argument_parser() -> ArgumentParser:
    """Build the CLI argument parser."""
    parser = ArgumentParser(
        description="Generate structured function calls from natural-language prompts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROMPT_TESTS,
        help="Path to the JSON file containing prompt tests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path.",
    )
    return parser


def main() -> int:
    """Run the end-to-end function-calling command-line program."""
    parser = build_argument_parser()
    args: Namespace = parser.parse_args()

    try:
        function_definitions = load_function_definitions(DEFAULT_FUNCTION_DEFINITIONS)
        prompt_items = load_prompt_items(args.input)
        llm_client = LlmClient()
        generation_engine = GenerationEngine(function_definitions, llm_client)

        print("\n" + "=" * 15 + " INFO - FILE LOADING " + "=" * 15)
        print(
            "Inputs loaded successfully: "
            f"{len(function_definitions)} function definitions, "
            f"{len(prompt_items)} prompts."
        )
        print(f"Prompt input file: {args.input}")
        print(f"Planned output path: {args.output}\n")

        results = []
        print("\n" + "=" * 15 + " INFO - CONSTRAINED GENERATION " + "=" * 15)
        with measure_time("full_generation_batch"):
            for prompt_item in prompt_items:
                with measure_time("part_prompt_each"):
                    generated_core = generation_engine.generate_function_call_core(prompt_item.prompt)
                    result = build_function_call_result(
                        prompt=prompt_item.prompt,
                        function_call_core=generated_core,
                        function_definitions=function_definitions,
                    )
                    results.append(result)
                    print(f"Prompt: {prompt_item.prompt}")
                    print(f"Generated core: {generated_core.model_dump()}")
                    print("-" * 78 + "\n")

        write_function_call_results(args.output, results)
        print(f"[SUCCESS] Results written to {args.output}")
        return 0
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI wrapper
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
