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
