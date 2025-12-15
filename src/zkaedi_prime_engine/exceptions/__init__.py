"""Advanced Exception System for ZKAEDI PRIME Engine.

This module provides a comprehensive, novel exception handling system with:
- Self-healing exceptions
- Evolutionary/aging exceptions
- Swarm intelligence
- FSM-based stateful exceptions
- Explainable exceptions with XAI integration
- Rich tracebacks and metadata
- Observability integrations
- Exception analytics and taxonomy
"""

from .base import (
    ZKAEDIError,
    HamiltonianError,
    EngineInitializationError,
    StateError,
    QECError,
    BackendError,
    MetadataZKAEDIError,
    SelfDiagnosingZKAEDIError,
)

from .healing import SelfHealingError
from .evolution import AgeingError
from .swarm import SwarmError
from .fsm import FSMError
from .explainable import ExplainableZKAEDIError
from .diagnostics import (
    ObservabilityException,
    ExceptionTracker,
    generate_exception_hierarchy,
)
from .rich_traceback import install_rich_traceback

__all__ = [
    # Base exceptions
    "ZKAEDIError",
    "HamiltonianError",
    "EngineInitializationError",
    "StateError",
    "QECError",
    "BackendError",
    "MetadataZKAEDIError",
    "SelfDiagnosingZKAEDIError",
    # Advanced exceptions
    "SelfHealingError",
    "AgeingError",
    "SwarmError",
    "FSMError",
    "ExplainableZKAEDIError",
    "ObservabilityException",
    "ExceptionTracker",
    "generate_exception_hierarchy",
    "install_rich_traceback",
]

# Auto-install rich traceback if available
try:
    install_rich_traceback()
except ImportError:
    pass  # rich is optional

