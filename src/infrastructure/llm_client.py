"""
Minimal adapter for real SDK.
"""

from __future__ import annotations

from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.config import DEFAULT_MODEL_NAME


class LlmClient:
    """Public adapter for interacting with the small LLM model"""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Initialize the LLM client.
        :param model_name: Default model identifier to load.
        """
        self._model = Small_LLM_Model(model_name=model_name)

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs
        :param text: Input text
        :return: A list of token IDs
        """
        encoded = self._model.encode(text)
        if hasattr(encoded, "tolist"):
            nested = encoded.tolist()
            if (
                    isinstance(nested, list)
                    and nested
                    and isinstance(nested[0], list)
            ):
                return [int(token_id) for token_id in nested[0]]
            return [int(token_id) for token_id in nested]
        raise TypeError("Unexpected encode() return type from llm_sdk")

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs into text
        :param token_ids: Token IDs to decode
        :return: Decoded text
        """
        return self._model.decode(token_ids)

    def get_next_token_logits(self, input_ids: list[int]) -> list[float]:
        """
        Get next-token logits for the provided input IDs
        :param input_ids: Full current token sequence
        :return: Raw logits for the next token
        """
        return self._model.get_logits_from_input_ids(input_ids)

    def get_vocab_file_path(self) -> Path:
        """
        Return the public vocabulary file path exposed by de SDK
        """
        return Path(self._model.get_path_to_vocab_file())

    def get_tokenizer_file_path(self) -> Path:
        """
        Return the public tokenizer file path exposed by the SDK
        """
        return Path(self._model.get_path_to_tokenizer_file())

    def get_merges_file_path(self) -> Path:
        """
        Return the public merges file path exposed by the SDK
        """
        return Path(self._model.get_path_to_merges_file())
