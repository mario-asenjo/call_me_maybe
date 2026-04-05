"""Shared types for constrained generation"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import (
    DEFAULT_ENABLE_TRACE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME
)


SchemaPrimitiveType = Literal["string", "number", "boolean", "integer"]


class ConstraintPhase(str, Enum):
    """High-level phases of constrained JSON generation"""

    START_OBJECT = "start_object"
    EXPECT_FN_NAME_KEY = "expect_fn_name_key"
    EXPECT_FN_NAME_VALUE = "expect_fn_name_value"
    EXPECT_ARGS_KEY = "expect_args_key"
    EXPECT_ARGS_OBJECT_START = "expect_args_object_start"
    EXPECT_NEXT_ARG_KEY_OR_END = "expect_next_arg_key_or_end"
    EXPECT_ARG_VALUE = "expect_arg_value"
    EXPECT_ARG_SEPARATOR_OR_END = "expect_arg_separator_or_end"
    EXPECT_FINAL_OBJECT_END = "expect_final_object_end"
    DONE = "done"
    ERROR = "error"


class GenerationConfig(BaseModel):
    """Configuration for constrained generation"""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default=DEFAULT_MODEL_NAME, min_length=1)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1)
    enable_trace: bool = DEFAULT_ENABLE_TRACE


class GenerationTraceStep(BaseModel):
    """Single trace entry for one decoding step"""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=0)
    phase: ConstraintPhase
    generated_text: str
    generated_token_ids: list[int]
    valid_token_count: int = Field(ge=0)
    chosen_token_id: int | None = None   # ¿Por que no Optional[int]?
    note: str | None = None  # Lo mismo


class GenerationErrorInfo(BaseModel):
    """Structured error information for controlled generation failures"""

    model_config = ConfigDict(extra="forbid")

    phase: ConstraintPhase
    message: str = Field(min_length=1)
    partial_text: str
    partial_token_ids: list[int]


class ConstraintDecision(BaseModel):
    """Result of one constraint computation step"""

    model_config = ConfigDict(extra="forbid")

    phase: ConstraintPhase
    valid_token_ids: list[int]
    note: str | None = None
    error: GenerationErrorInfo | None = None
