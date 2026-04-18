"""State model and token constraints for structured function-call generation."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.domain import (
    ConstraintDecision,
    ConstraintPhase,
    FunctionDefinition,
    GenerationErrorInfo,
    SchemaPrimitiveType,
)
from src.engine.value_candidates import ValueCandidateBuilder
from src.infrastructure import LlmClient


class ConstraintState(BaseModel):
    """Mutable logical state for one constrained generation run."""

    model_config = ConfigDict(extra="forbid")

    phase: ConstraintPhase = ConstraintPhase.START_OBJECT
    selected_function_name: str | None = None
    pending_parameter_names: list[str] = Field(default_factory=list)
    emitted_parameter_names: list[str] = Field(default_factory=list)
    completed_arguments: dict[str, Any] = Field(default_factory=dict)

    current_parameter_name: str | None = None
    current_parameter_type: SchemaPrimitiveType | None = None
    current_key_text: str = ""
    current_key_token_ids: list[int] = Field(default_factory=list)
    current_value_text: str = ""
    current_value_token_ids: list[int] = Field(default_factory=list)

    partial_output_text: str = ""
    partial_output_token_ids: list[int] = Field(default_factory=list)
    source_prompt: str = ""

    consumed_numeric_slots: int = 0
    used_string_literals: list[str] = Field(default_factory=list)


class ConstraintEngine:
    """Compute valid next-token sets for the structured JSON output."""

    def __init__(
        self,
        function_definitions: list[FunctionDefinition],
        llm_client: LlmClient,
    ) -> None:
        self._function_definitions = function_definitions
        self._function_map = {definition.name: definition for definition in function_definitions}
        self._llm_client = llm_client
        self._candidate_builder = ValueCandidateBuilder()

        self._function_header_token_ids = {
            definition.name: self._llm_client.encode(
                self._build_function_header_text(definition.name)
            )
            for definition in function_definitions
        }
        self._arg_key_token_ids = {
            parameter_name: self._llm_client.encode(
                self._build_next_arg_key_text(parameter_name)
            )
            for definition in function_definitions
            for parameter_name in definition.parameters.keys()
        }
        self._comma_token_id = self._llm_client.encode(",")[0]
        self._closing_brace_token_id = self._llm_client.encode("}")[0]

    def initial_state(self, source_prompt: str = "") -> ConstraintState:
        """Create a new initial constraint state."""
        return ConstraintState(source_prompt=source_prompt)

    def get_function_definition(self, function_name: str) -> FunctionDefinition:
        """Return the function definition associated with one function name."""
        return self._function_map[function_name]

    def get_current_value_candidates(self, state: ConstraintState) -> list[str]:
        """Return active candidate JSON literals for the current parameter."""
        if state.selected_function_name is None:
            return []
        if state.current_parameter_name is None or state.current_parameter_type is None:
            return []

        function_definition = self.get_function_definition(state.selected_function_name)
        base_candidates = self._candidate_builder.build_serialized_candidates(
            prompt=state.source_prompt,
            function_description=function_definition.description,
            parameter_name=state.current_parameter_name,
            parameter_type=state.current_parameter_type,
        )

        if state.current_parameter_type in {"number", "integer"}:
            return self._candidate_builder.build_numeric_slot_candidate(
                prompt=state.source_prompt,
                parameter_type=state.current_parameter_type,
                slot_index=state.consumed_numeric_slots,
            )

        if state.current_parameter_type == "string":
            return self._filter_used_string_literals(
                candidates=base_candidates,
                used_literals=state.used_string_literals,
            )

        return base_candidates

    def compute_valid_tokens(self, state: ConstraintState) -> ConstraintDecision:
        """Compute the valid next-token IDs for the current state."""
        if state.phase in {
            ConstraintPhase.START_OBJECT,
            ConstraintPhase.EXPECT_FN_NAME_KEY,
            ConstraintPhase.EXPECT_FN_NAME_VALUE,
            ConstraintPhase.EXPECT_ARGS_KEY,
            ConstraintPhase.EXPECT_ARGS_OBJECT_START,
        }:
            return self._compute_function_header_tokens(state)

        if state.phase == ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END:
            return self._compute_next_arg_key_or_end_tokens(state)

        if state.phase == ConstraintPhase.EXPECT_ARG_VALUE:
            return self._compute_value_tokens(state)

        if state.phase == ConstraintPhase.EXPECT_ARG_SEPARATOR_OR_END:
            return self._compute_separator_or_end_tokens(state)

        if state.phase == ConstraintPhase.EXPECT_FINAL_OBJECT_END:
            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=[self._closing_brace_token_id],
                note="Closing the root JSON object.",
            )

        return ConstraintDecision(
            phase=ConstraintPhase.ERROR,
            valid_token_ids=[],
            error=GenerationErrorInfo(
                phase=state.phase,
                message="Unsupported constraint phase.",
                partial_text=state.partial_output_text,
                partial_token_ids=state.partial_output_token_ids,
            ),
        )

    def advance_state_with_token(self, state: ConstraintState, token_id: int) -> ConstraintState:
        """Advance the logical state by appending one generated token."""
        previous_selected_function_name = state.selected_function_name
        new_token_ids = [*state.partial_output_token_ids, token_id]
        new_text = self._llm_client.decode(new_token_ids)

        new_state = state.model_copy(
            update={
                "partial_output_token_ids": new_token_ids,
                "partial_output_text": new_text,
            }
        )
        new_state = self._try_finalize_function_selection(new_state)

        if previous_selected_function_name is None and new_state.selected_function_name is not None:
            return new_state

        if new_state.phase == ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END:
            return self._advance_key_state(new_state, token_id)

        if new_state.phase == ConstraintPhase.EXPECT_ARG_VALUE:
            return self._advance_value_state(new_state, token_id)

        if new_state.phase == ConstraintPhase.EXPECT_ARG_SEPARATOR_OR_END:
            return self._advance_separator_state(new_state, token_id)

        if new_state.phase == ConstraintPhase.EXPECT_FINAL_OBJECT_END:
            if token_id == self._closing_brace_token_id:
                new_state.phase = ConstraintPhase.DONE
            return new_state

        return new_state

    def _compute_function_header_tokens(self, state: ConstraintState) -> ConstraintDecision:
        """Restrict generation to valid function header continuations."""
        prefix_token_ids = state.partial_output_token_ids
        matching_headers = self._get_matching_function_headers(prefix_token_ids)
        if not matching_headers:
            return ConstraintDecision(
                phase=ConstraintPhase.ERROR,
                valid_token_ids=[],
                error=GenerationErrorInfo(
                    phase=state.phase,
                    message="No valid function header matches the current token prefix.",
                    partial_text=state.partial_output_text,
                    partial_token_ids=state.partial_output_token_ids,
                ),
            )

        next_token_ids: set[int] = set()
        for header_token_ids in matching_headers.values():
            if len(prefix_token_ids) < len(header_token_ids):
                next_token_ids.add(header_token_ids[len(prefix_token_ids)])

        return ConstraintDecision(
            phase=state.phase,
            valid_token_ids=sorted(next_token_ids),
            note="Restricting generation to valid function headers.",
        )

    def _compute_next_arg_key_or_end_tokens(self, state: ConstraintState) -> ConstraintDecision:
        """Restrict generation to the next argument key or the args object end."""
        if state.selected_function_name is None:
            return self._unexpected_state_error(
                state,
                "Function must be selected before argument-key generation.",
            )

        if not state.pending_parameter_names:
            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=[self._closing_brace_token_id],
                note="All arguments were emitted; the args object can be closed.",
            )

        next_parameter_name = state.pending_parameter_names[0]
        option_token_ids = self._arg_key_token_ids[next_parameter_name]
        relative_prefix = state.current_key_token_ids

        if len(relative_prefix) > len(option_token_ids):
            return self._unexpected_state_error(
                state,
                "Current argument-key prefix is longer than the valid option.",
            )

        if option_token_ids[: len(relative_prefix)] != relative_prefix:
            return self._unexpected_state_error(
                state,
                "Current argument-key prefix does not match the expected parameter.",
            )

        if len(relative_prefix) == len(option_token_ids):
            return self._unexpected_state_error(
                state,
                "Argument-key state is complete but was not advanced.",
            )

        next_token_id = option_token_ids[len(relative_prefix)]
        return ConstraintDecision(
            phase=state.phase,
            valid_token_ids=[next_token_id],
            note="Restricting generation to the next argument key.",
        )

    def _compute_value_tokens(self, state: ConstraintState) -> ConstraintDecision:
        """Restrict generation to continuations of valid value candidates."""
        candidates = self.get_current_value_candidates(state)
        if not candidates:
            return ConstraintDecision(
                phase=ConstraintPhase.ERROR,
                valid_token_ids=[],
                error=GenerationErrorInfo(
                    phase=state.phase,
                    message=(
                        "No value candidates are available for the current parameter. "
                        f"selected_function={state.selected_function_name!r}, "
                        f"current_parameter={state.current_parameter_name!r}, "
                        f"current_type={state.current_parameter_type!r}, "
                        f"consumed_numeric_slots={state.consumed_numeric_slots}"
                    ),
                    partial_text=state.partial_output_text,
                    partial_token_ids=state.partial_output_token_ids,
                ),
            )

        current_prefix = state.current_value_token_ids
        matching_value_options: list[list[int]] = []
        for candidate in candidates:
            option_token_ids = self._llm_client.encode(candidate)
            if len(current_prefix) > len(option_token_ids):
                continue
            if option_token_ids[: len(current_prefix)] == current_prefix:
                matching_value_options.append(option_token_ids)

        if not matching_value_options:
            return ConstraintDecision(
                phase=ConstraintPhase.ERROR,
                valid_token_ids=[],
                error=GenerationErrorInfo(
                    phase=state.phase,
                    message="No valid value candidate matches the current value prefix.",
                    partial_text=state.partial_output_text,
                    partial_token_ids=state.partial_output_token_ids,
                ),
            )

        next_token_ids: set[int] = set()
        for option_token_ids in matching_value_options:
            if len(current_prefix) < len(option_token_ids):
                next_token_ids.add(option_token_ids[len(current_prefix)])

        return ConstraintDecision(
            phase=state.phase,
            valid_token_ids=sorted(next_token_ids),
            note="Restricting generation to valid value candidate continuations.",
        )

    def _compute_separator_or_end_tokens(self, state: ConstraintState) -> ConstraintDecision:
        """Restrict generation to a comma or the args-object end."""
        if state.pending_parameter_names:
            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=[self._comma_token_id],
                note="More parameters remain, so a comma must be generated.",
            )
        return ConstraintDecision(
            phase=state.phase,
            valid_token_ids=[self._closing_brace_token_id],
            note="All parameters were emitted, so the args object can be closed.",
        )

    def _advance_key_state(self, state: ConstraintState, token_id: int) -> ConstraintState:
        """Advance the state while building the next argument key."""
        if not state.pending_parameter_names or state.selected_function_name is None:
            return state

        next_parameter_name = state.pending_parameter_names[0]
        option_token_ids = self._arg_key_token_ids[next_parameter_name]
        new_key_token_ids = [*state.current_key_token_ids, token_id]
        new_key_text = self._llm_client.decode(new_key_token_ids)

        state.current_key_token_ids = new_key_token_ids
        state.current_key_text = new_key_text

        if new_key_token_ids == option_token_ids:
            function_definition = self.get_function_definition(state.selected_function_name)
            parameter_spec = function_definition.parameters[next_parameter_name]
            state.current_parameter_name = next_parameter_name
            state.current_parameter_type = parameter_spec.type
            self._reset_current_value_state(state)
            state.phase = ConstraintPhase.EXPECT_ARG_VALUE
        return state

    def _advance_value_state(self, state: ConstraintState, token_id: int) -> ConstraintState:
        """Advance the state while building the current argument value."""
        state.current_value_token_ids = [*state.current_value_token_ids, token_id]
        state.current_value_text = self._llm_client.decode(state.current_value_token_ids)

        candidate_token_ids = [
            self._llm_client.encode(candidate)
            for candidate in self.get_current_value_candidates(state)
        ]
        if any(state.current_value_token_ids == candidate_ids for candidate_ids in candidate_token_ids):
            if state.current_parameter_name is None or state.current_parameter_type is None:
                return state

            parsed_value = json.loads(state.current_value_text)
            state.completed_arguments[state.current_parameter_name] = parsed_value
            state.emitted_parameter_names = [*state.emitted_parameter_names, state.current_parameter_name]
            state.pending_parameter_names = [
                name for name in state.pending_parameter_names if name != state.current_parameter_name
            ]

            if state.current_parameter_type in {"number", "integer"}:
                state.consumed_numeric_slots += 1
            elif state.current_parameter_type == "string":
                state.used_string_literals = [*state.used_string_literals, state.current_value_text]

            state.phase = ConstraintPhase.EXPECT_ARG_SEPARATOR_OR_END
        return state

    def _advance_separator_state(self, state: ConstraintState, token_id: int) -> ConstraintState:
        """Advance the state after a value, consuming either comma or closing brace."""
        if token_id == self._comma_token_id:
            state.current_parameter_name = None
            state.current_parameter_type = None
            self._reset_current_key_state(state)
            self._reset_current_value_state(state)
            state.phase = ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END
            return state

        if token_id == self._closing_brace_token_id:
            state.current_parameter_name = None
            state.current_parameter_type = None
            self._reset_current_key_state(state)
            self._reset_current_value_state(state)
            state.phase = ConstraintPhase.EXPECT_FINAL_OBJECT_END
        return state

    def _try_finalize_function_selection(self, state: ConstraintState) -> ConstraintState:
        """Finalize function selection when one header has been fully emitted."""
        if state.selected_function_name is not None:
            return state

        for function_name, header_token_ids in self._function_header_token_ids.items():
            if state.partial_output_token_ids == header_token_ids:
                function_definition = self.get_function_definition(function_name)
                state.selected_function_name = function_name
                state.pending_parameter_names = list(function_definition.parameters.keys())
                self._reset_current_key_state(state)
                self._reset_current_value_state(state)
                state.phase = ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END
                return state
        return state

    def _build_function_header_text(self, function_name: str) -> str:
        """Build the fixed JSON prefix for a function-call header."""
        return f'{{"fn_name":"{function_name}","args":{{'

    def _build_next_arg_key_text(self, parameter_name: str) -> str:
        """Build the next JSON argument-key fragment."""
        return f'"{parameter_name}":'

    def _get_matching_function_headers(self, prefix_token_ids: list[int]) -> dict[str, list[int]]:
        """Return all function headers compatible with the current token prefix."""
        return {
            function_name: header_token_ids
            for function_name, header_token_ids in self._function_header_token_ids.items()
            if len(prefix_token_ids) <= len(header_token_ids)
            and header_token_ids[: len(prefix_token_ids)] == prefix_token_ids
        }

    def _filter_used_string_literals(self, candidates: list[str], used_literals: list[str]) -> list[str]:
        """Filter already-used string literals while preserving duplicate accounting."""
        remaining_counts = Counter(used_literals)
        filtered: list[str] = []
        for candidate in candidates:
            if remaining_counts[candidate] > 0:
                remaining_counts[candidate] -= 1
                continue
            filtered.append(candidate)
        return filtered

    def _reset_current_key_state(self, state: ConstraintState) -> None:
        """Reset the state associated with the current argument key."""
        state.current_key_text = ""
        state.current_key_token_ids = []

    def _reset_current_value_state(self, state: ConstraintState) -> None:
        """Reset the state associated with the current argument value."""
        state.current_value_text = ""
        state.current_value_token_ids = []

    def _unexpected_state_error(self, state: ConstraintState, message: str) -> ConstraintDecision:
        """Build a stable structured error when the state machine desynchronizes."""
        return ConstraintDecision(
            phase=ConstraintPhase.ERROR,
            valid_token_ids=[],
            error=GenerationErrorInfo(
                phase=state.phase,
                message=message,
                partial_text=state.partial_output_text,
                partial_token_ids=state.partial_output_token_ids,
            ),
        )
