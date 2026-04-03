"""
Utilities for loading and inspecting tokenizer vocabulary files
"""
import json
from json import JSONDecodeError
from pathlib import Path


def load_json_object(path: Path) -> dict[str, Any]:
    """
    Load a JSON object from a file
    :param path: Path to the JSON file
    :return: Parsed JSON object as a dictionary
    :raises: InputFileError: If the file cannot be read
             InputJsonError: If the file content is invalid JSON
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise InputFileError(f"Vocabulary file not found: {path}") from exc
    except JSONDecodeError as exc:
        raise InputJsonError(
            f"Invalid JSON in vocabulary file '{path}': "
            f"line {exc.lineno}, column {exc.colno}."
        )

    if not isinstance(data, dict):
        raise InputJsonError(f"Expected a JSON object in vocabulary file: {path}")

    return data


def invert_vocab_mapping(vocab: dict[str, Any]) -> dict[int, str]:
    """
    Invert a token->id vocabulary mapping into id->token
    :param vocab: Original vocabulary
    :return: A mapping from tokenID to token string
    """
    inverted: dict[int, str] = {}

    for token_text, token_id in vocab.items():
        if not isinstance(token_text, str):
            raise InputJsonError("Vocabulary token text must be a string")
        if not isinstance(token_id, int):
            raise InputJsonError("Vocabulary token ID must be an integer")
        if token_id in inverted:
            raise InputJsonError(f"Duplicated token ID found in vocabulary: {token_id}")
        inverted[token_id] = token_text

    return inverted
