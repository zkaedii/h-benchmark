"""Custom exceptions for ZKAEDI PRIME Engine.

Provides granular exception types for better error handling and debugging.

This module maintains backward compatibility. For advanced exception features,
see zkaedi_prime_engine.exceptions module.
"""

# Import from the new exceptions module for backward compatibility
from .exceptions.base import (
    ZKAEDIError,
    HamiltonianError,
    EngineInitializationError,
    StateError,
    QECError,
    BackendError,
)

# Re-export for backward compatibility
__all__ = [
    "ZKAEDIError",
    "HamiltonianError",
    "EngineInitializationError",
    "StateError",
    "QECError",
    "BackendError",
]

