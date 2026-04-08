"""
Real constrained generation engine
"""

from __future__ import annotations

import json

from functools import lru_cache

from src.domain import FunctionDefinition, FunctionCallCore
from src.engine.constraint_engine import ConstraintEngine, ConstraintState
from src.infrastructure import LlmClient


class GenerationEngine:
    """Run constrained token-by-token generation for one prompt"""

    def __init__(
            self,
            function_definitions: list[FunctionDefinition],
            llm_client: LlmClient
    ):
        """Initialize the generation engine"""
        self._function_definitions = function_definitions
        self._llm_client = llm_client
        self._constraint_engine = ConstraintEngine(
            function_definitions, llm_client
        )
        self._function_catalog = self._build_function_catalog()

    def _build_function_catalog(self) -> str:
        """Build the static function catalog text once"""
        function_lines: list[str] = []

        for function_definition in self._function_definitions:
            parameters_text = ', '.join(
                f"{name}: {spec.type}"
                for name, spec in function_definition.parameters.items()
            )
            function_lines.append(
                f"- {function_definition.name}: "
                f"{function_definition.description} | "
                f"params=({parameters_text}) | "
                f"returns={function_definition.returns.type}"
            )

        return "\n".join(function_lines)

    @lru_cache(maxsize=2048)
    def _get_context_token_ids(
            self,
            prompt: str,
            selected_function_name: str | None,
            current_parameter_name: str | None,
            current_parameter_type: str | None,
            candidate_values: tuple[str, ...]
    ) -> tuple[int, ...]:
        """Build and encode a cached generation context"""
        state_like = ConstraintState(
            source_prompt=prompt,
            selected_function_name=selected_function_name,
            current_parameter_name=current_parameter_name,
            current_parameter_type=current_parameter_type
        )

        context_parts: list[str] = [
            "You are a function-calling system.",
            "Choose exactly one function from the available catalog.",
            "Available functions:",
            self._function_catalog,
            "User prompt:",
            prompt,
            'Return exactly one JSON object with keys "fn_name" and "args".',
            "Do not answer in natural language."
        ]

        if selected_function_name is not None:
            context_parts.append(f"Selected function so far: {selected_function_name}")

        if current_parameter_name is not None and current_parameter_type is not None:
            context_parts.append(f"Current argument to fill: {current_parameter_name}")
            context_parts.append(f"Current argument type: {current_parameter_type}")

            if candidate_values:
                context_parts.append("Valid candidate literals for the current argument:")
                for candidate in candidate_values[:20]:
                    context_parts.append(f"- {candidate}")

        context_text = "\n".join(context_parts)
        return tuple(self._llm_client.encode(context_text))

    def _build_generation_context(
            self,
            prompt: str,
            state: ConstraintState
    ) -> str:
        """
        Build a semantic context for function selection and argument generation
        The context is self-aware to help the model choose the right value among valid candidates
        """
        function_catalog = self._function_catalog

        context_parts: list = [
            "You are a function-calling system.",
            "Choose exactly one function from the available catalog.",
            "Available functions:",
            function_catalog,
            "User prompt:",
            prompt,
            'Return exactly one JSON object with keys "fn_name" and "args".',
            "Do not answer in natural language."
        ]

        if state.selected_function_name is not None:
            context_parts.extend([
                f"Selected function so far: {state.selected_function_name}"
            ])

        if (
            state.current_parameter_name is not None
            and state.current_parameter_type is not None
        ):
            context_parts.extend([
                f"Current argument to fill: {state.current_parameter_name}",
                f"Current argument type: {state.current_parameter_type}"
            ])

            candidate_values = self._constraint_engine.get_current_value_candidates(
                state
            )
            if candidate_values:
                context_parts.append("Valid candidate literals for the current argument:")
                for candidate in candidate_values[:20]:
                    context_parts.append(f"- {candidate}")

        return "\n".join(context_parts)


    def generate_function_call_core(
            self,
            prompt: str,
            max_steps: int = 256
    ) -> FunctionCallCore:
        """Generate a constrained function call core for one prompt"""
        state = self._constraint_engine.initial_state(prompt)

        for _ in range(max_steps):
            if state.phase.name == "DONE":
                parsed = json.loads(state.partial_output_text)
                return FunctionCallCore.model_validate(parsed)

            decision = self._constraint_engine.compute_valid_tokens(state)
            if decision.error is not None:
                raise ValueError(
                    f"Constraint error during generation: "
                    f"{decision.error.message} | "
                    f"phase={state.phase} | "
                    f"selected_function={state.selected_function_name} | "
                    f"pending={state.pending_parameter_names} | "
                    f"current_key={state.current_key_text!r} | "
                    f"current_value={state.current_value_text!r} |"
                    f"partial_output={state.partial_output_text!r}"
                )

            if not decision.valid_token_ids:
                raise ValueError(
                    f"No valid tokens available in phase {state.phase} "
                    "before reaching DONE"
                )

            candidate_values = tuple(
                self._constraint_engine.get_current_value_candidates(state)[:20]
            )
            context_token_ids = list(
                self._get_context_token_ids(
                    prompt,
                    state.selected_function_name,
                    state.current_parameter_name,
                    state.current_parameter_type,
                    candidate_values
                )
            )

            model_input_token_ids = (
                context_token_ids + state.partial_output_token_ids
            )

            logits = self._llm_client.get_next_token_logits(
                model_input_token_ids
            )
            chosen_token_id = max(
                decision.valid_token_ids,
                key=lambda token_id: logits[token_id]
            )

            state = self._constraint_engine.advance_state_with_token(
                state,
                chosen_token_id
            )

        raise ValueError("Generation exceeded the maximum number of steps")
