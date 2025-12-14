"""Custom exceptions for ZKAEDI PRIME Engine.

Provides granular exception types for better error handling and debugging.
"""


class ZKAEDIError(Exception):
    """Base exception for all ZKAEDI PRIME errors."""
    pass


class HamiltonianError(ZKAEDIError):
    """Raised when Hamiltonian operations fail."""
    pass


class EngineInitializationError(ZKAEDIError):
    """Raised when engine initialization fails."""
    pass


class StateError(ZKAEDIError):
    """Raised when state operations fail."""
    pass


class QECError(ZKAEDIError):
    """Raised when quantum error correction fails."""
    pass


class BackendError(ZKAEDIError):
    """Raised when backend operations fail."""
    pass

