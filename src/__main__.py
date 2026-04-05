"""
Command-line entry point for the function-calling project: call-me-maybe
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.domain import ProjectError
from src.application import load_function_definitions, load_prompt_items
from src.infrastructure import (
    load_json_object,
    invert_vocab_mapping,
    LlmClient
)
from src.config import (
    DEFAULT_FUNCTION_DEFINITIONS,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_PROMPT_TESTS
)
from src.engine import ConstraintEngine


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
            '{"fn_name":"fn_add_numbers","args":{"a":2.0,"b":3.0}}',
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
        print("-" * 40)
        print("TEST CONSTRAINT ENGINE - 1")
        constraint_engine = ConstraintEngine(function_definitions, llm_client)
        state = constraint_engine.initial_state()
        print("-" * 60)
        print("Constraint engine dry-run for function header generation")
        test_context_text = (
            "You must output a function call as JSON with keys "
            "'fn_name' and 'args'."
        )
        test_context_token_ids = llm_client.encode(test_context_text)
        for step_index in range(64):
            decision = constraint_engine.compute_valid_tokens(state)
            print(
                f"Step: {step_index}: phase={decision.phase}, "
                f"valid_token_count={len(decision.valid_token_ids)}"
            )
            if decision.error is not None:
                print(f"Constraint error: {decision.error.message}")
                break

            if not decision.valid_token_ids:
                print("Header generation completed.")
                print(f"Selected function: {state.selected_function_name}")
                print(f"Pending parameters: {state.pending_parameter_names}")
                print(f"Generated text: {state.partial_output_text}")
                break

            model_input_token_ids = (
                    test_context_token_ids + state.partial_output_token_ids
            )
            logits = llm_client.get_next_token_logits(model_input_token_ids)
            chosen_token_id = max(
                decision.valid_token_ids,
                key=lambda token_id: logits[token_id]
            )
            state = constraint_engine.advance_state_with_token(
                state,
                chosen_token_id
            )
        return 0
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
