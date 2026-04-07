"""
Real constrained generation engine
"""

from __future__ import annotations

import json

from src.domain import FunctionDefinition, FunctionCallCore
from src.engine.constraint_engine import ConstraintEngine
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

    def _build_generation_context(self, prompt: str) -> str:
        """
        Build a semantic context for function selection and argument generation
        """
        function_lines: list[str] = []

        for function_definition in self._function_definitions:
            parameters_text = ", ".join(
                f"{name}: {spec.type}"
                for name, spec in function_definition.parameters.items()
            )
            function_lines.append(
                f"- {function_definition.name}({parameters_text}): "
                f"{function_definition.description}"
            )

        function_catalog = "\n".join(function_lines)

        return (
            "You are a function-calling system.\n"
            "Choose exactly one function from the available catalog:\n\n"
            "Available functions:\n"
            f"{function_catalog}\n\n"
            "User prompt:\n"
            f"{prompt}\n\n"
            'Return exactly one JSON object with keys "fn_name" and "args".\n'
            "Do not answer in natural language."
        )

    def generate_function_call_core(
            self,
            prompt: str,
            max_steps: int = 256
    ) -> FunctionCallCore:
        """Generate a constrained function call core for one prompt"""
        state = self._constraint_engine.initial_state(prompt)
        context_text = self._build_generation_context(prompt)
        context_token_ids = self._llm_client.encode(context_text)

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
