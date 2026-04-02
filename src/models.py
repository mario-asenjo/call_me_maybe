"""Pydantic models for project inputs and outputs"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator
)


SupportedParameterTypes = Literal["string", "number", "boolean"]


class FunctionParameterSpec(BaseModel):
    """Specification for a single instruction parameter"""

    model_config = ConfigDict(extra="forbid")
    type: SupportedParameterTypes


class FunctionReturnSpec(BaseModel):
    """Specification for a function return type"""

    model_config = ConfigDict(extra="forbid")
    type: SupportedParameterTypes


class FunctionDefinition(BaseModel):
    """Definition of a callable function exposed to the LLM"""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, FunctionParameterSpec]
    returns: FunctionReturnSpec

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
            cls, value: dict[str, FunctionParameterSpec]
    ) -> dict[str, FunctionParameterSpec]:
        """Ensure parameter names are not empty"""

        for key in value:
            if not key.strip():
                raise ValueError("Parameter names must not be empty.")
        return value


class PromptItem(BaseModel):
    """Single natural-language prompt to process"""

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Ensure the prompt is not blank"""
        if not value.strip():
            raise ValueError("Prompt must not be blank.")
        return value


class FunctionCallCore(BaseModel):
    """Restricted LLM output before final prompt assembly"""
    model_config = ConfigDict(extra="forbid")
    fn_name: str = Field(min_length=1)
    args: dict[str, Any]


class FunctionCallResult(BaseModel):
    """Final output item written to the results JSON file"""

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    fn_name: str = Field(min_length=1)
    args: dict[str, Any]

    @model_validator(mode="after")
    def validate_non_blank_strings(self) -> "FunctionCallResult":
        """Ensure string fields are not blank"""
        if not self.prompt.strip():
            raise ValueError("Prompt must not be blank.")
        if not self.fn_name.strip():
            raise ValueError("fn_name must not be blank.")
        return self
