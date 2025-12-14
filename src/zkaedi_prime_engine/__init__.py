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
]
