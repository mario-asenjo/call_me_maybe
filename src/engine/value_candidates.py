"""Prompt-driven candidate extraction for constrained value generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from src.domain import SchemaPrimitiveType


@dataclass(frozen=True)
class RawCandidate:
    """Single raw candidate before JSON serialization."""

    value: str
    score: int
    category: str


class ValueCandidateBuilder:
    """Build ordered JSON literal candidates for one parameter value."""

    _NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
    _QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'')
    _WINDOWS_PATH_RE = re.compile(
        r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"
    )
    _UNIX_PATH_RE = re.compile(r"/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")
    _ENCODING_RE = re.compile(
        r"\b(?:utf-8|utf8|latin-1|latin1|ascii|utf-16|cp1252)\b",
        re.IGNORECASE,
    )
    _WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\:-]+")
    _STOPWORDS = frozenset(
        {
            "a",
            "all",
            "an",
            "and",
            "another",
            "are",
            "at",
            "by",
            "calculate",
            "execute",
            "format",
            "for",
            "from",
            "file",
            "greet",
            "in",
            "is",
            "of",
            "on",
            "read",
            "replace",
            "reverse",
            "root",
            "run",
            "square",
            "string",
            "substitute",
            "sum",
            "the",
            "what",
            "with",
            "word",
        }
    )

    def build_serialized_candidates(
        self,
        prompt: str,
        function_description: str,
        parameter_name: str,
        parameter_type: SchemaPrimitiveType,
    ) -> list[str]:
        """Return ordered JSON literal candidates for a parameter."""
        return self._build_cached(
            prompt=prompt,
            function_description=function_description,
            parameter_name=parameter_name,
            parameter_type=parameter_type,
        )

    @lru_cache(maxsize=4096)
    def _build_cached(
        self,
        prompt: str,
        function_description: str,
        parameter_name: str,
        parameter_type: SchemaPrimitiveType,
    ) -> list[str]:
        if parameter_type == "string":
            return self._build_string_candidates(
                prompt=prompt,
                function_description=function_description,
                parameter_name=parameter_name,
            )
        if parameter_type == "number":
            return self._build_number_candidates(prompt)
        if parameter_type == "integer":
            return self._build_integer_candidates(prompt)
        if parameter_type == "boolean":
            return self._build_boolean_candidates(prompt)
        return []

    def _build_number_candidates(self, prompt: str) -> list[str]:
        """Build JSON float literals in order of appearance."""
        return [
            self._serialize_number(float(raw_value))
            for raw_value in self._extract_numeric_slots(prompt)
        ]

    def _build_integer_candidates(self, prompt: str) -> list[str]:
        """Build JSON integer literals in order of appearance."""
        candidates: list[str] = []
        for raw_value in self._extract_numeric_slots(prompt):
            if "." in raw_value:
                continue
            candidates.append(str(int(raw_value)))
        return self._dedupe_serialized(candidates)

    def build_numeric_slot_candidate(
        self,
        prompt: str,
        parameter_type: SchemaPrimitiveType,
        slot_index: int,
    ) -> list[str]:
        """Build the candidate for one numeric slot preserving prompt order."""
        numeric_slots = self._extract_numeric_slots(prompt)
        if slot_index >= len(numeric_slots):
            return []

        raw_value = numeric_slots[slot_index]
        if parameter_type == "number":
            return [self._serialize_number(float(raw_value))]
        if parameter_type == "integer":
            if "." in raw_value:
                return []
            return [str(int(raw_value))]
        return []

    @lru_cache(maxsize=2048)
    def _extract_numeric_slots(self, prompt: str) -> list[str]:
        """Extract numeric slots from the prompt in original order."""
        return [match.group(0) for match in self._NUMBER_RE.finditer(prompt)]

    def _build_boolean_candidates(self, prompt: str) -> list[str]:
        """Build plausible JSON boolean literals from prompt wording."""
        prompt_lower = prompt.lower()
        candidates: list[str] = []

        truthy_markers = (" true", " yes", " enable", " enabled", " on ")
        falsy_markers = (" false", " no", " disable", " disabled", " off ")

        if any(marker in f" {prompt_lower} " for marker in truthy_markers):
            candidates.append("true")
        if any(marker in f" {prompt_lower} " for marker in falsy_markers):
            candidates.append("false")

        if prompt_lower.startswith(("is ", "are ", "has ", "can ", "should ")):
            candidates.extend(["true", "false"])

        if not candidates:
            candidates.extend(["true", "false"])

        return self._dedupe_serialized(candidates)

    def _build_string_candidates(
        self,
        prompt: str,
        function_description: str,
        parameter_name: str,
    ) -> list[str]:
        """Build ranked JSON string literals for a parameter."""
        raw_candidates: list[RawCandidate] = []
        raw_candidates.extend(self._extract_quoted_candidates(prompt))
        raw_candidates.extend(self._extract_path_candidates(prompt))
        raw_candidates.extend(self._extract_encoding_candidates(prompt))
        raw_candidates.extend(self._extract_template_candidates(prompt))
        raw_candidates.extend(
            self._extract_regex_projection_candidates(prompt)
        )
        raw_candidates.extend(self._extract_safe_word_candidates(prompt))

        preferred_categories = self._infer_preferred_categories(
            parameter_name=parameter_name,
            function_description=function_description,
        )

        return self._rank_and_serialize(
            raw_candidates=raw_candidates,
            preferred_categories=preferred_categories,
        )

    def _infer_preferred_categories(
        self,
        parameter_name: str,
        function_description: str,
    ) -> tuple[str, ...]:
        """Infer which candidate categories are most appropriate."""
        parameter_keywords = self._split_keywords(parameter_name)
        description_keywords = self._split_keywords(function_description)
        combined_keywords = parameter_keywords | description_keywords

        if {"regex", "pattern"} & combined_keywords:
            return ("regex",)

        if {"replacement", "replace"} & combined_keywords:
            return ("replacement", "quoted_atom", "single_word", "full_text")

        if {"query", "template", "source", "text", "string", "message"} & parameter_keywords:
            return ("full_text", "template", "path")

        if {"path", "file", "filepath", "location"} & combined_keywords:
            return ("path",)

        if {"encoding", "charset"} & combined_keywords:
            return ("encoding", "quoted_atom", "single_word")

        if {"database", "db", "name"} & combined_keywords:
            return ("quoted_atom", "single_word")

        return (
            "full_text",
            "quoted_atom",
            "single_word",
            "template",
            "path",
            "encoding",
        )

    def _extract_quoted_candidates(self, prompt: str) -> list[RawCandidate]:
        """Extract exact quoted string literals from the prompt."""
        candidates: list[RawCandidate] = []

        for match in self._QUOTED_RE.finditer(prompt):
            double_quoted, single_quoted = match.groups()
            value = double_quoted if double_quoted is not None else single_quoted
            normalized = self._normalize_text(value)
            if not normalized:
                continue

            if self._looks_like_full_text_block(normalized):
                candidates.append(
                    RawCandidate(
                        value=normalized,
                        score=180,
                        category="full_text",
                    )
                )
            else:
                candidates.append(
                    RawCandidate(
                        value=normalized,
                        score=150,
                        category="quoted_atom",
                    )
                )

        return candidates

    def _extract_path_candidates(self, prompt: str) -> list[RawCandidate]:
        """Extract Windows and Unix style file paths."""
        candidates: list[RawCandidate] = []

        for match in self._WINDOWS_PATH_RE.finditer(prompt):
            candidates.append(
                RawCandidate(
                    value=match.group(0),
                    score=190,
                    category="path",
                )
            )

        for match in self._UNIX_PATH_RE.finditer(prompt):
            candidates.append(
                RawCandidate(
                    value=match.group(0),
                    score=190,
                    category="path",
                )
            )

        return candidates

    def _extract_encoding_candidates(self, prompt: str) -> list[RawCandidate]:
        """Extract common encoding names."""
        return [
            RawCandidate(
                value=match.group(0),
                score=175,
                category="encoding",
            )
            for match in self._ENCODING_RE.finditer(prompt)
        ]

    def _extract_template_candidates(self, prompt: str) -> list[RawCandidate]:
        """Extract template-like text after a template marker."""
        match = re.search(r"\btemplate\s*:\s*(.+)$", prompt, re.IGNORECASE)
        if not match:
            return []

        value = self._normalize_text(match.group(1))
        if not value:
            return []

        return [
            RawCandidate(
                value=value,
                score=185,
                category="template",
            )
        ]

    def _extract_regex_projection_candidates(
        self,
        prompt: str,
    ) -> list[RawCandidate]:
        """Project generic regex and replacement literals."""
        prompt_lower = prompt.lower()
        candidates: list[RawCandidate] = []

        projection_map: tuple[tuple[tuple[str, ...], str], ...] = (
            (("replace all numbers", "replace all digits"), r"\d+"),
            (("replace all vowels",), r"[aeiouAEIOU]"),
            (
                ("replace all consonants",),
                r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]",
            ),
            (("replace all spaces", "replace all whitespace"), r"\s+"),
            (("replace all letters",), r"[A-Za-z]"),
            (("replace all uppercase letters",), r"[A-Z]"),
            (("replace all lowercase letters",), r"[a-z]"),
        )

        for triggers, regex_literal in projection_map:
            if any(trigger in prompt_lower for trigger in triggers):
                candidates.append(
                    RawCandidate(
                        value=regex_literal,
                        score=210,
                        category="regex",
                    )
                )

        word_match = re.search(
            r"\bword\s+[\"']([^\"']+)[\"']",
            prompt,
            re.IGNORECASE,
        )
        if word_match:
            escaped_word = re.escape(word_match.group(1))
            candidates.append(
                RawCandidate(
                    value=rf"\b{escaped_word}\b",
                    score=220,
                    category="regex",
                )
            )

        replacement_match = re.search(
            r"\bwith\s+[\"']([^\"']+)[\"'](?:\s+\bin\b|\s*$)",
            prompt,
            re.IGNORECASE,
        )
        if replacement_match:
            replacement_value = self._normalize_text(replacement_match.group(1))
            if replacement_value:
                candidates.append(
                    RawCandidate(
                        value=replacement_value,
                        score=215,
                        category="replacement",
                    )
                )

        if "asterisk" in prompt_lower or "asterisks" in prompt_lower:
            candidates.append(
                RawCandidate(
                    value="*",
                    score=215,
                    category="replacement",
                )
            )

        tail_match = re.search(
            r"\bwith\s+([A-Za-z0-9_*+-]+)\s*$",
            prompt.strip(),
            re.IGNORECASE,
        )
        if tail_match:
            tail_value = self._normalize_text(tail_match.group(1))
            if tail_value:
                candidates.append(
                    RawCandidate(
                        value=tail_value,
                        score=170,
                        category="replacement",
                    )
                )

        return candidates

    def _extract_safe_word_candidates(self, prompt: str) -> list[RawCandidate]:
        """Extract low-noise single-word fallback candidates."""
        prompt_without_quotes = self._QUOTED_RE.sub(" ", prompt)
        tokens = self._WORD_TOKEN_RE.findall(prompt_without_quotes)

        candidates: list[RawCandidate] = []
        for token in tokens:
            normalized = self._normalize_text(token)
            if not normalized:
                continue
            if self._NUMBER_RE.fullmatch(normalized):
                continue
            if normalized.lower() in self._STOPWORDS:
                continue

            candidates.append(
                RawCandidate(
                    value=normalized,
                    score=70,
                    category="single_word",
                )
            )

        return candidates

    def _rank_and_serialize(
        self,
        raw_candidates: Iterable[RawCandidate],
        preferred_categories: tuple[str, ...],
    ) -> list[str]:
        """Rank, filter and serialize candidates."""
        raw_candidates_list = list(raw_candidates)
        if not raw_candidates_list:
            return []

        preferred_set = set(preferred_categories)
        if any(candidate.category in preferred_set for candidate in raw_candidates_list):
            filtered_candidates = [
                candidate
                for candidate in raw_candidates_list
                if candidate.category in preferred_set
            ]
        else:
            filtered_candidates = raw_candidates_list

        best_by_value: dict[str, int] = {}
        for candidate in filtered_candidates:
            score = candidate.score
            if candidate.category in preferred_set:
                score += 120

            previous_score = best_by_value.get(candidate.value)
            if previous_score is None or score > previous_score:
                best_by_value[candidate.value] = score

        sorted_values = sorted(
            best_by_value.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )

        return [
            json.dumps(value, ensure_ascii=False)
            for value, _score in sorted_values
        ]

    def _serialize_number(self, value: float) -> str:
        """Serialize a JSON number while preserving float-ness."""
        if value.is_integer():
            return f"{int(value)}.0"
        return repr(value)

    def _dedupe_serialized(self, values: Iterable[str]) -> list[str]:
        """Preserve order while removing duplicates."""
        seen: set[str] = set()
        deduped: list[str] = []

        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)

        return deduped

    def _normalize_text(self, value: str) -> str:
        """Normalize extracted prompt text."""
        return value.strip().strip(",.;:!? ")

    def _looks_like_full_text_block(self, value: str) -> bool:
        """Return True when a string candidate looks like a full text block."""
        return any(character.isspace() for character in value) or len(value) > 18

    def _split_keywords(self, value: str) -> set[str]:
        """Split a schema or description field into lowercase lexical keywords."""
        return {
            token.lower()
            for token in re.split(r"[^A-Za-z0-9]+", value)
            if token
        }