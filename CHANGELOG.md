# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-13

### Added
- Initial release of ZKAEDI PRIME Engine
- Recursive Hamiltonian evolution (ZKAEDIPrimeHamiltonian)
- Sparse state management (SparseState) with gate operations
- Quantum error correction (SurfaceCode, LDPCDecoder)
- MPS engine for high-entanglement systems
- Unified ZKAEDIEngine with automatic backend selection
- Comprehensive test suite (118 tests)
- MVP benchmark suite
- Full documentation and examples

### Features
- Automatic backend selection (Sparse/MPS/Dense)
- Quantum error correction with adaptive selection
- Guaranteed invariants (norm preservation, Hermiticity)
- Scalable from 2 to 20+ qubits
- Production-ready code with full test coverage

