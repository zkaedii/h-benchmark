"""ZKAEDI PRIME Engine - Unified Quantum Computing Engine

One Engine. All Seals. End-to-End.
"""

from .engine import (
    ZKAEDIPrimeHamiltonian,
    SparseState,
    SurfaceCode,
    LDPCDecoder,
    MPSEngine,
    ZKAEDIEngine,
    ZKAEDIDiagnostics,
    BackendType,
    create_example_hamiltonian,
    demonstrate_zkaedi_prime,
    sigmoid,
    hermitian_projection,
)

from .exceptions import (
    ZKAEDIError,
    HamiltonianError,
    EngineInitializationError,
    StateError,
    QECError,
    BackendError,
)

__version__ = "1.0.0"
__author__ = "ZKAEDI PRIME Contributors"

__all__ = [
    "ZKAEDIPrimeHamiltonian",
    "SparseState",
    "SurfaceCode",
    "LDPCDecoder",
    "MPSEngine",
    "ZKAEDIEngine",
    "ZKAEDIDiagnostics",
    "BackendType",
    "create_example_hamiltonian",
    "demonstrate_zkaedi_prime",
    "sigmoid",
    "hermitian_projection",
    "ZKAEDIError",
    "HamiltonianError",
    "EngineInitializationError",
    "StateError",
    "QECError",
    "BackendError",
]

# Leaderboard module (optional)
try:
    from .leaderboard import BenchmarkLeaderboard
    __all__.append("BenchmarkLeaderboard")
except ImportError:
    pass
