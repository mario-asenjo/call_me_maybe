"""
Command-line entry point for the function-calling project: call-me-maybe
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.domain import ProjectError
from src.application import load_function_definitions, load_prompt_items
from src.infrastructure import load_json_object, invert_vocab_mapping, LlmClient
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
        vocab_path = llm_client.get_vocab_file_path()
        vocab_data = load_json_object(vocab_path)
        inverted_vocab = invert_vocab_mapping(vocab_data)

        print(
            "Inputs loaded successfully: "
            f"{len(function_definitions)} function definitions, "
            f"{len(prompt_items)} prompts."
        )
        print(f"Vocabulary file: {vocab_path}")
        print(f"Vocabulary size: {len(inverted_vocab)}")
        print(f"Prompt input file: {args.input}")
        print(f"Planned output path: {args.output}\n")
        sample_texts = [
            '{"fn_name":"fn_add_numbers","args":{"a":2.0,"b":3.0})',
            '"hello"',
            '"\\\\d+"',
            '"C:\\\\Users\\\\john\\\\config.ini"'
        ]
        for sample in sample_texts:
            token_ids = llm_client.encode(sample)
            decoded = llm_client.decode(token_ids)
            print("-" * 60)
            print(f"Sample: {sample}")
            print(f"Token count: {len(token_ids)}")
            print(f"Token IDs: {token_ids}")
            print(f"Decoded: {decoded}")
        return 0
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
