"""State model for constrained JSON generation"""
from pydantic import BaseModel, ConfigDict, Field

from src.domain import (
    ConstraintPhase,
    SchemaPrimitiveType,
    ConstraintDecision,
    FunctionDefinition
)


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


class ConstraintEngine:
    """Compute valid next-token sets for the structured JSON output"""

    def __init__(self, function_definitions: list[FunctionDefinition]) -> None:
        """
        Initialize the engine with available function definitions
        :param function_definitions: list of valid FunctionDefinition items
        """
        self._function_definitions = function_definitions
        self._function_map = {item.name: item for item in function_definitions}

    def initial_state(self) -> ConstraintState:
        """
        Create the initial constraint state
        :return: A newly created ConstraintState object
        """
        return ConstraintState()

    def get_function_definition(self, function_name: str) -> FunctionDefinition:
        """
        Return a function definition by name
        :param function_name: The function definition name
        :return: The function definition whose name matches parameter
        """
        return self._function_map[function_name]

    def compute_valid_tokens(
            self,
            state: ConstraintState
    ) -> ConstraintDecision:
        """
        Compute valid token IDs for the current state
        (Skeleton for now)
        :param state: Current state
        :return: Valid toke IDs for the current state
        """
        return ConstraintDecision(
            phase=state.phase,
            valid_token_ids=[],
            note="Constraint engine skeleton initialized.",
            error=None
        )
