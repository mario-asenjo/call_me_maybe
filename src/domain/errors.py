"""
Custom exceptions for the project
"""

from __future__ import annotations


class ProjectError(Exception):
    """Base exception for project-specific errors"""


class InputFileError(ProjectError):
    """Raised when an input file cannot be read"""


class InputJsonError(ProjectError):
    """Raised when an input file contains invalid JSON"""


class InputValidationError(ProjectError):
    """Raised when parsed input data does not match the spected schema"""


class GenerationFailureError(ProjectError):
    """Raised when constrained generation cannot produce a valid call"""


class OutputValidationError(ProjectError):
    """Raised when a generated call does not match the function schema"""


class OutputFileError(ProjectError):
    """Raised when the output file cannot be written"""
