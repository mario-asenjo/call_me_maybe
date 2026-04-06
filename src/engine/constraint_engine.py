"""State model for constrained JSON generation"""
from pydantic import BaseModel, ConfigDict, Field

from src import GenerationErrorInfo
from src.domain import (
    ConstraintPhase,
    SchemaPrimitiveType,
    ConstraintDecision,
    FunctionDefinition
)
from src.infrastructure import LlmClient


class ConstraintState(BaseModel):
    """Mutable logical state of one constrained generation"""

    model_config = ConfigDict(extra="forbid")

    phase: ConstraintPhase = ConstraintPhase.START_OBJECT
    selected_function_name: str | None = None
    pending_parameter_names: list[str] = Field(default_factory=list)
    emitted_parameter_names: list[str] = Field(default_factory=list)
    current_parameter_name: str | None = None
    current_parameter_type: SchemaPrimitiveType | None = None

    partial_output_text: str = ""
    partial_output_token_ids: list[int] = Field(default_factory=list)

    source_prompt: str = ""

    current_value_text: str = ""
    current_value_token_ids: list[int] = Field(default_factory=list)

    current_key_text: str = ""
    current_key_token_ids: list[int] = Field(default_factory=list)


class ConstraintEngine:
    """Compute valid next-token sets for the structured JSON output"""

    def __init__(
            self,
            function_definitions: list[FunctionDefinition],
            llm_client: LlmClient
    ) -> None:
        """
        Initialize the engine with available function definitions
        :param function_definitions: list of valid FunctionDefinition items
        """
        self._function_definitions = function_definitions
        self._function_map = {item.name: item for item in function_definitions}
        self._llm_client = llm_client

        self._function_header_texts = {
            item.name: self._build_function_header_text(item.name)
            for item in function_definitions
        }
        self._function_header_token_ids = {
            function_name: self._llm_client.encode(header_text)
            for function_name, header_text
            in self._function_header_texts.items()
        }

    def initial_state(self, source_prompt: str = "") -> ConstraintState:
        """
        Create the initial constraint state
        :return: A newly created ConstraintState object
        """
        return ConstraintState(source_prompt=source_prompt)

    def get_function_definition(
            self,
            function_name: str
    ) -> FunctionDefinition:
        """
        Return a function definition by name
        :param function_name: The function definition name
        :return: The function definition whose name matches parameter
        """
        return self._function_map[function_name]

    def _build_function_header_text(self, function_name: str) -> str:
        """Build the fixed JSON prefix for a function call header"""
        return f'{{"fn_name":"{function_name}","args":{{'

    def _build_next_arg_key_text(self, parameter_name: str) -> str:
        """Build the JSON fragment for the next argument key"""
        return f'"{parameter_name}":'

    def _get_matching_function_headers(
            self,
            prefix_token_ids: list[int]
    ) -> dict[str, list[int]]:
        """Return all function headers compatible with the current prefix"""
        matches: dict[str, list[int]] = {}
        for function_name, header_token_ids \
                in self._function_header_token_ids.items():
            if len(prefix_token_ids) > len(header_token_ids):
                continue
            if header_token_ids[: len(prefix_token_ids)] == prefix_token_ids:
                matches[function_name] = header_token_ids
        return matches

    def _get_matching_next_arg_key_options(
            self,
            state: ConstraintState
    ) -> dict[str, list[int]]:
        """Return compatible tokenized options for the next argument key"""
        if not state.pending_parameter_names:
            return {}

        next_parameter_name = state.pending_parameter_names[0]
        option_text = self._build_next_arg_key_text(next_parameter_name)
        option_token_ids = self._llm_client.encode(option_text)

        return {next_parameter_name: option_token_ids}

    def _get_comma_token_id(self) -> int:
        """Return the token ID used for a JSON comma"""
        return self._llm_client.encode(",")[0]

    def _get_closing_brace_token_id(self) -> int:
        """Return the token ID used for a closing JSON brace"""
        return self._llm_client.encode("}")[0]

    def _extract_number_candidates_from_prompt(
            self,
            prompt: str
    ) -> list[str]:
        """Extract numeric candidates from the source prompt"""
        import re

        raw_numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
        candidates: list[str] = []

        for raw_number in raw_numbers:
            if raw_number not in candidates:
                candidates.append(raw_number)

            if "." not in raw_number:
                float_variant = f"{raw_number}.0"
                if float_variant not in candidates:
                    candidates.append(float_variant)

        return candidates

    def _get_number_value_options(self, prompt: str) -> list[list[int]]:
        """Build tokenized numeric value options from the prompt"""
        candidates = self._extract_number_candidates_from_prompt(prompt)
        return [self._llm_client.encode(candidate) for candidate in candidates]

    def _reset_current_key_state(self, state: ConstraintState) -> None:
        """Reset the current argument-key buffer state"""
        state.current_key_text = ""
        state.current_key_token_ids = []

    def _reset_current_value_state(self, state: ConstraintState) -> None:
        """Reset the current argument-value buffer state"""
        state.current_value_text = ""
        state.current_value_token_ids = []

    def _try_finalize_function_selection(
            self,
            state: ConstraintState
    ) -> ConstraintState:
        """Set the selected function once one full header has been generated"""
        if state.selected_function_name is not None:
            return state

        for function_name, header_token_ids \
                in self._function_header_token_ids.items():
            if state.partial_output_token_ids == header_token_ids:
                function_definition = self.get_function_definition(
                    function_name
                )
                state.selected_function_name = function_name
                state.pending_parameter_names = list(
                    function_definition.parameters
                )
                self._reset_current_key_state(state)
                self._reset_current_value_state(state)
                state.phase = ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END
                return state

        return state

    def compute_valid_tokens(
            self,
            state: ConstraintState
    ) -> ConstraintDecision:
        """Compute valid token IDs for the current state."""
        if state.phase in (
                ConstraintPhase.START_OBJECT,
                ConstraintPhase.EXPECT_FN_NAME_KEY,
                ConstraintPhase.EXPECT_FN_NAME_VALUE,
                ConstraintPhase.EXPECT_ARGS_KEY,
                ConstraintPhase.EXPECT_ARGS_OBJECT_START,
        ):
            prefix_token_ids = state.partial_output_token_ids
            matching_headers = self._get_matching_function_headers(
                prefix_token_ids
            )

            if not matching_headers:
                return ConstraintDecision(
                    phase=ConstraintPhase.ERROR,
                    valid_token_ids=[],
                    note=None,
                    error=GenerationErrorInfo(
                        phase=state.phase,
                        message="No valid function header "
                                "matches the current token prefix",
                        partial_text=state.partial_output_text,
                        partial_token_ids=state.partial_output_token_ids
                    )
                )

            next_token_ids: set[int] = set()
            for header_token_ids in matching_headers.values():
                if len(prefix_token_ids) < len(header_token_ids):
                    next_token_ids.add(header_token_ids[len(prefix_token_ids)])

            if not next_token_ids:
                return ConstraintDecision(
                    phase=ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END,
                    valid_token_ids=[],
                    note="Function header fully generated",
                    error=None
                )

            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=sorted(next_token_ids),
                note="Valid next tokens computed from matching"
                     " function headers",
                error=None
            )

        if state.phase == ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END:
            if not state.pending_parameter_names:
                closing_token_ids = self._llm_client.encode("}")
                return ConstraintDecision(
                    phase=state.phase,
                    valid_token_ids=[closing_token_ids[0]],
                    note="No pending parameters left, args object"
                         " can be closed.",
                    error=None
                )

            next_parameter_name = state.pending_parameter_names[0]
            option_token_ids = self._llm_client.encode(
                self._build_next_arg_key_text(next_parameter_name)
            )

            relative_prefix = state.current_key_token_ids

            if len(relative_prefix) > len(option_token_ids):
                return ConstraintDecision(
                    phase=ConstraintPhase.ERROR,
                    valid_token_ids=[],
                    note=None,
                    error=GenerationErrorInfo(
                        phase=state.phase,
                        message="Current argument-key prefix is longer"
                                " than the valid option.",
                        partial_text=state.partial_output_text,
                        partial_token_ids=state.partial_output_token_ids
                    )
                )

            if option_token_ids[: len(relative_prefix)] != relative_prefix:
                expected_key_text = self._build_next_arg_key_text(
                    next_parameter_name
                )
                return ConstraintDecision(
                    phase=ConstraintPhase.ERROR,
                    valid_token_ids=[],
                    note=None,
                    error=GenerationErrorInfo(
                        phase=state.phase,
                        message=(
                            "Current argument-key prefix does not "
                            "match the expected parameter. Expected key "
                            f"fragment: {expected_key_text!r}, current "
                            f"key text: {state.current_key_text!r}"
                        ),
                        partial_text=state.partial_output_text,
                        partial_token_ids=state.partial_output_token_ids
                    )
                )

            if len(relative_prefix) == len(option_token_ids):
                return ConstraintDecision(
                    phase=ConstraintPhase.EXPECT_ARG_VALUE,
                    valid_token_ids=[],
                    note="Argument key fully generated, next step is the"
                         " value.",
                    error=None
                )

            next_token_id = option_token_ids[len(relative_prefix)]
            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=[next_token_id],
                note="Restricting generation to the next argument key.",
                error=None
            )

        if state.phase == ConstraintPhase.EXPECT_ARG_VALUE:
            if state.current_parameter_type != "number":
                return ConstraintDecision(
                    phase=state.phase,
                    valid_token_ids=[],
                    note="Current implementation only supports numeric "
                         "argument values.",
                    error=None
                )

            value_options = self._get_number_value_options(
                state.source_prompt
            )
            if not value_options:
                return ConstraintDecision(
                    phase=ConstraintPhase.ERROR,
                    valid_token_ids=[],
                    note=None,
                    error=GenerationErrorInfo(
                        phase=state.phase,
                        message="No numeric candidates could be extracted "
                                "from the prompt.",
                        partial_text=state.partial_output_text,
                        partial_token_ids=state.partial_output_token_ids
                    )
                )

            relative_value_prefix = state.current_value_token_ids
            matching_value_options: list[list[int]] = []

            for option_token_ids in value_options:
                if len(relative_value_prefix) > len(option_token_ids):
                    continue
                if option_token_ids[
                    : len(relative_value_prefix)
                ] == relative_value_prefix:
                    matching_value_options.append(option_token_ids)

            if not matching_value_options:
                return ConstraintDecision(
                    phase=ConstraintPhase.ERROR,
                    valid_token_ids=[],
                    note=None,
                    error=GenerationErrorInfo(
                        phase=state.phase,
                        message="No numeric value option matches the current"
                                " value prefix.",
                        partial_text=state.partial_output_text,
                        partial_token_ids=state.partial_output_token_ids
                    )
                )

            next_token_ids = set()
            fully_matched_option_exists = False

            for option_token_ids in matching_value_options:
                if len(relative_value_prefix) == len(option_token_ids):
                    fully_matched_option_exists = True
                else:
                    next_token_ids.add(
                        option_token_ids[len(relative_value_prefix)]
                    )

            if fully_matched_option_exists and not next_token_ids:
                return ConstraintDecision(
                    phase=ConstraintPhase.EXPECT_ARG_SEPARATOR_OR_END,
                    valid_token_ids=[],
                    note="Numeric argument value fully generated.",
                    error=None
                )

            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=sorted(next_token_ids),
                note="Restricting generation to numeric candidates extracted"
                     " from the prompt.",
                error=None
            )

        if state.phase == ConstraintPhase.EXPECT_ARG_SEPARATOR_OR_END:
            if state.pending_parameter_names:
                return ConstraintDecision(
                    phase=state.phase,
                    valid_token_ids=[self._get_comma_token_id()],
                    note="More parameters remain, so a comma must be"
                         " generated.",
                    error=None
                )

            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=[self._get_closing_brace_token_id()],
                note="All parameters emitted, args object can be closed.",
                error=None
            )

        if state.phase == ConstraintPhase.EXPECT_FINAL_OBJECT_END:
            return ConstraintDecision(
                phase=state.phase,
                valid_token_ids=[self._get_closing_brace_token_id()],
                note="Closing the root JSON object.",
                error=None
            )

        return ConstraintDecision(
            phase=ConstraintPhase.ERROR,
            valid_token_ids=[],
            note=None,
            error=GenerationErrorInfo(
                phase=state.phase,
                message="Unsupported constraint phase in current"
                        " implementation.",
                partial_text=state.partial_output_text,
                partial_token_ids=state.partial_output_token_ids
            )
        )

    def advance_state_with_token(
            self,
            state: ConstraintState,
            token_id: int
    ) -> ConstraintState:
        """Advance the logical state by appending one generated token."""
        previous_selected_function_name = state.selected_function_name

        new_token_ids = [*state.partial_output_token_ids, token_id]
        new_text = self._llm_client.decode(new_token_ids)

        new_state = state.model_copy(
            update={
                "partial_output_token_ids": new_token_ids,
                "partial_output_text": new_text
            }
        )
        new_state = self._try_finalize_function_selection(new_state)

        if (
            previous_selected_function_name is None
            and new_state.selected_function_name is not None
        ):
            return new_state
        if (
                new_state.selected_function_name is not None
                and (new_state.phase ==
                     ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END)
                and new_state.pending_parameter_names
        ):
            next_parameter_name = new_state.pending_parameter_names[0]
            option_token_ids = self._llm_client.encode(
                self._build_next_arg_key_text(next_parameter_name)
            )

            candidate_key_token_ids = [
                *new_state.current_key_token_ids, token_id
            ]

            if option_token_ids[
                : len(candidate_key_token_ids)
            ] == candidate_key_token_ids:
                new_key_token_ids = candidate_key_token_ids
            elif option_token_ids[:1] == [token_id]:
                new_key_token_ids = [token_id]
            else:
                new_key_token_ids = candidate_key_token_ids

            new_key_text = self._llm_client.decode(new_key_token_ids)

            new_state.current_key_token_ids = new_key_token_ids
            new_state.current_key_text = new_key_text

            if new_key_token_ids == option_token_ids:
                function_definition = self.get_function_definition(
                    new_state.selected_function_name
                )
                parameter_spec = function_definition.parameters[
                    next_parameter_name
                ]

                new_state.current_parameter_name = next_parameter_name
                new_state.current_parameter_type = parameter_spec.type
                self._reset_current_value_state(new_state)
                new_state.phase = ConstraintPhase.EXPECT_ARG_VALUE
                return new_state

            return new_state

        if new_state.phase == ConstraintPhase.EXPECT_ARG_VALUE:
            new_value_token_ids = [
                *new_state.current_value_token_ids, token_id
            ]
            new_value_text = self._llm_client.decode(new_value_token_ids)

            new_state.current_value_token_ids = new_value_token_ids
            new_state.current_value_text = new_value_text

            if new_state.current_parameter_type == "number":
                value_options = self._get_number_value_options(
                    new_state.source_prompt
                )
                if any(
                        new_value_token_ids == option
                        for option in value_options
                ):
                    if new_state.current_parameter_name is not None:
                        new_state.emitted_parameter_names = [
                            *new_state.emitted_parameter_names,
                            new_state.current_parameter_name
                        ]
                        new_state.pending_parameter_names = [
                            name
                            for name in new_state.pending_parameter_names
                            if name != new_state.current_parameter_name
                        ]
                        new_state.phase = (
                            ConstraintPhase.EXPECT_ARG_SEPARATOR_OR_END
                        )
            return new_state

        if new_state.phase == ConstraintPhase.EXPECT_ARG_SEPARATOR_OR_END:
            comma_token_id = self._get_comma_token_id()
            closing_brace_token_id = self._get_closing_brace_token_id()

            if token_id == comma_token_id:
                new_state.current_parameter_name = None
                new_state.current_parameter_type = None
                self._reset_current_key_state(new_state)
                self._reset_current_value_state(new_state)
                new_state.phase = ConstraintPhase.EXPECT_NEXT_ARG_KEY_OR_END
                return new_state

            if token_id == closing_brace_token_id:
                new_state.current_parameter_name = None
                new_state.current_parameter_type = None
                self._reset_current_key_state(new_state)
                self._reset_current_value_state(new_state)
                new_state.phase = ConstraintPhase.EXPECT_FINAL_OBJECT_END
                return new_state

        if new_state.phase == ConstraintPhase.EXPECT_FINAL_OBJECT_END:
            if token_id == self._get_closing_brace_token_id():
                new_state.phase = ConstraintPhase.DONE
                return new_state

        return new_state
