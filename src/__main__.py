"""
Command-line entry point for the function-calling project: call-me-maybe
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.domain import ProjectError
from src.application import load_function_definitions, load_prompt_items
from src.infrastructure import LlmClient
from src.config import (
    DEFAULT_FUNCTION_DEFINITIONS,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_PROMPT_TESTS
)


def build_argument_parser() -> ArgumentParser:
    """Build the CLI argument parser"""
    parser = ArgumentParser(
        description="Generate structured function calls "
                    "from natural-language prompts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROMPT_TESTS,
        help=(
            "Input path. It can be either a directory containing the"
            " default input files or a JSON file with prompt tests."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path."
    )

    return parser


def main() -> int:
    """Run the command-line interface"""
    parser = build_argument_parser()
    args: Namespace = parser.parse_args()

    try:
        function_definitions = load_function_definitions(
            DEFAULT_FUNCTION_DEFINITIONS
        )
        prompt_items = load_prompt_items(args.input)
        llm_client = LlmClient()

        print("\n" + "=" * 15 + " INFO - FILE LOADING " + "=" * 15)
        print(
            "Inputs loaded successfully: "
            f"{len(function_definitions)} function definitions, "
            f"{len(prompt_items)} prompts."
        )

        print(f"Prompt input file: {args.input}")
        print(f"Planned output path: {args.output}\n")

        from src.engine import GenerationEngine

        generation_engine = GenerationEngine(function_definitions, llm_client)

        print("\n" + "=" * 15 + " INFO - REAL GENERATION TEST " + "=" * 15)
        for prompt_item in prompt_items:
            generated_core = generation_engine.generate_function_call_core(
                prompt_item.prompt
            )
            print(f"Prompt: {prompt_item.prompt}")
            print(f"Generated core: {generated_core.model_dump()}")
            print("-" * 78 + "\n")

        return 0
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
