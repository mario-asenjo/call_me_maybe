"""Real constrained generation engine for structured function calling."""

from __future__ import annotations

import json
from functools import lru_cache

from src.config import DEFAULT_MAX_NEW_TOKENS
from src.domain import ConstraintPhase, FunctionCallCore, FunctionDefinition, GenerationFailureError
from src.engine.constraint_engine import ConstraintEngine, ConstraintState
from src.infrastructure import LlmClient


class GenerationEngine:
    """Run constrained token-by-token generation for one prompt."""

    def __init__(
        self,
        function_definitions: list[FunctionDefinition],
        llm_client: LlmClient,
    ) -> None:
        self._function_definitions = function_definitions
        self._llm_client = llm_client
        self._constraint_engine = ConstraintEngine(function_definitions, llm_client)
        self._function_catalog = self._build_function_catalog()

    def generate_function_call_core(
        self,
        prompt: str,
        max_steps: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> FunctionCallCore:
        """Generate one constrained function-call core from a natural-language prompt."""
        state = self._constraint_engine.initial_state(prompt)

        for _ in range(max_steps):
            if state.phase == ConstraintPhase.DONE:
                return self._parse_final_core(state)

            decision = self._constraint_engine.compute_valid_tokens(state)
            if decision.error is not None:
                raise GenerationFailureError(
                    f"Constraint error: {decision.error.message} | "
                    f"phase={state.phase.value} | prompt={prompt!r} | "
                    f"partial_output={state.partial_output_text!r}"
                )
            if not decision.valid_token_ids:
                raise GenerationFailureError(
                    f"No valid next tokens available before reaching DONE | "
                    f"phase={state.phase.value} | prompt={prompt!r} | "
                    f"partial_output={state.partial_output_text!r}"
                )

            candidate_values = tuple(self._constraint_engine.get_current_value_candidates(state)[:24])
            completed_arguments = tuple(
                (name, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for name, value in state.completed_arguments.items()
            )
            context_token_ids = list(
                self._get_context_token_ids(
                    prompt=prompt,
                    selected_function_name=state.selected_function_name,
                    current_parameter_name=state.current_parameter_name,
                    current_parameter_type=state.current_parameter_type,
                    completed_arguments=completed_arguments,
                    candidate_values=candidate_values,
                )
            )
            model_input_token_ids = context_token_ids + state.partial_output_token_ids
            logits = self._llm_client.get_next_token_logits(model_input_token_ids)
            chosen_token_id = max(decision.valid_token_ids, key=lambda token_id: logits[token_id])
            state = self._constraint_engine.advance_state_with_token(state, chosen_token_id)

        raise GenerationFailureError(
            f"Generation exceeded the maximum number of steps ({max_steps}) for prompt {prompt!r}"
        )

    @lru_cache(maxsize=4096)
    def _get_context_token_ids(
        self,
        prompt: str,
        selected_function_name: str | None,
        current_parameter_name: str | None,
        current_parameter_type: str | None,
        completed_arguments: tuple[tuple[str, str], ...],
        candidate_values: tuple[str, ...],
    ) -> tuple[int, ...]:
        """Build and encode one reusable generation context."""
        context_lines: list[str] = [
            "You are a function-calling system.",
            "Choose exactly one function from the available catalog.",
            "Generate only one JSON object with keys \"fn_name\" and \"args\".",
            "Do not output natural language.",
            "Available functions:",
            self._function_catalog,
            "User prompt:",
            prompt,
        ]

        if selected_function_name is not None:
            context_lines.append(f"Selected function: {selected_function_name}")
        if completed_arguments:
            context_lines.append("Arguments already fixed:")
            for argument_name, serialized_value in completed_arguments:
                context_lines.append(f"- {argument_name} = {serialized_value}")
        if current_parameter_name is not None and current_parameter_type is not None:
            context_lines.append(f"Current argument name: {current_parameter_name}")
            context_lines.append(f"Current argument type: {current_parameter_type}")
        if candidate_values:
            context_lines.append("Valid candidate literals for the current argument:")
            for candidate_value in candidate_values:
                context_lines.append(f"- {candidate_value}")

        context_text = "\n".join(context_lines)
        return tuple(self._llm_client.encode(context_text))

    def _build_function_catalog(self) -> str:
        """Build the static function catalog included in every model context."""
        function_lines: list[str] = []
        for function_definition in self._function_definitions:
            parameters_text = ", ".join(
                f"{parameter_name}: {parameter_spec.type}"
                for parameter_name, parameter_spec in function_definition.parameters.items()
            )
            function_lines.append(
                f"- {function_definition.name}: {function_definition.description} | "
                f"params=({parameters_text}) | returns={function_definition.returns.type}"
            )
        return "\n".join(function_lines)

    def _parse_final_core(self, state: ConstraintState) -> FunctionCallCore:
        """Parse the final generated JSON core and validate its shape."""
        try:
            parsed = json.loads(state.partial_output_text)
        except json.JSONDecodeError as exc:
            raise GenerationFailureError(
                f"Generated output is not valid JSON: {state.partial_output_text!r}"
            ) from exc

        try:
            return FunctionCallCore.model_validate(parsed)
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise GenerationFailureError(
                f"Generated JSON does not match FunctionCallCore: {parsed!r}"
            ) from exc
