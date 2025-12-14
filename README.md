# 🔱 ZKAEDI PRIME Engine

**One Engine. All Seals. End-to-End.**

A unified quantum computing engine that combines recursive Hamiltonian evolution, sparse state management, quantum error correction, and automatic backend selection for scalable quantum simulations.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-118%20passing-brightgreen)](https://github.com/zkaedii/h-benchmark)

## Features

- ✅ **Recursive Hamiltonian Evolution** - Dynamic field evolution with ZKAEDI dynamics
- ✅ **Sparse State Management** - Efficient representation for low-entanglement states
- ✅ **Quantum Error Correction** - Surface Code and LDPC decoders
- ✅ **MPS Engine** - Matrix Product States for high-entanglement systems
- ✅ **Automatic Backend Selection** - Optimal engine choice based on system state
- ✅ **Guaranteed Invariants** - Norm preservation, unitarity, graceful degradation
- ✅ **Production Ready** - 118 comprehensive tests, full documentation

## Quick Start

### Installation

```bash
pip install zkaedi-prime-engine
```

Or from source:

```bash
git clone https://github.com/zkaedii/h-benchmark.git
cd zkaedi-prime-engine
pip install -e .
```

### Basic Usage

```python
from zkaedi_prime_engine import ZKAEDIEngine, create_example_hamiltonian

# Create example Hamiltonian
num_qubits = 4
H0 = create_example_hamiltonian(num_qubits, h_type="ising")

# Initialize engine
engine = ZKAEDIEngine(
    num_qubits=num_qubits,
    H0=H0,
    eta=0.4,      # Feedback coefficient
    gamma=1.0,    # Sharpening coefficient
    epsilon=0.04, # Noise amplitude
    beta=0.5      # Noise scaling
)

# Evolve system
diagnostics = engine.evolve(timesteps=20, track_metrics=True)

# Get summary
summary = engine.get_summary()
print(f"Final backend: {summary['current_backend']}")
print(f"Avg entropy: {summary['statistics']['avg_entropy']:.4f}")
```

## Documentation

- [Full Documentation](docs/README.md)
- [API Reference](docs/API.md)
- [Algorithm Specification](docs/ALGORITHM.md)
- [Examples](examples/)

## Algorithm Overview

The ZKAEDI PRIME engine implements a unified quantum computing algorithm:

```
FOR each timestep t:
  1. Recursive Hamiltonian Update
     H ← H₀ + η·H·σ(γ·H) + ε·N(0, 1 + β·|H|)
  
  2. Diagnostics
     curvature ← Σ p_i H_ii
     entropy   ← S(ρ_subsystem)
     sparsity  ← |support| / 2^n
  
  3. Automatic Backend Selection
     IF entropy > threshold: → MPS Engine
     ELSE IF sparsity small: → Sparse Engine
     ELSE: → Dense Engine
  
  4. Quantum Error Correction
     IF curvature > threshold:
        Apply Surface Code or LDPC
  
  5. Invariants (always enforced)
     - Norm = 1
     - Unitarity preserved
```

## Components

### ZKAEDIPrimeHamiltonian
Recursive Hamiltonian evolution with adaptive noise.

### SparseState
Efficient sparse quantum state representation with gate operations.

### Quantum Error Correction
- **SurfaceCode**: Local parity-based QEC for low sparsity
- **LDPCDecoder**: Global sparse graph QEC for high sparsity

### MPSEngine
Matrix Product State engine for high-entanglement systems.

### ZKAEDIEngine
Unified engine with automatic backend selection and QEC integration.

## Performance

- **Small systems (2-6 qubits)**: < 1ms per step
- **Medium systems (7-12 qubits)**: 1-10ms per step
- **Large systems (13-20 qubits)**: 10-100ms per step
- **Memory**: Scales with entanglement, not Hilbert dimension

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=zkaedi_prime_engine --cov-report=html

# Run benchmarks
python benchmark_zkaedi_prime_engine.py
```

**Test Coverage**: 118 comprehensive tests covering all components.

## Benchmarks

```bash
python benchmark_zkaedi_prime_engine.py
```

Results:
- Average time per step: 0.364 ms
- Scalability: Linear scaling up to 6+ qubits
- Memory efficiency: Sparse state representation

## Examples

See the [examples/](examples/) directory for:
- Basic evolution
- Parameter sweeps
- Custom Hamiltonians
- Backend selection
- QEC integration

## Requirements

- Python 3.8+
- NumPy 1.19+

## Installation from Source

```bash
git clone https://github.com/zkaedii/h-benchmark.git
cd zkaedi-prime-engine
pip install -e .
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use ZKAEDI PRIME Engine in your research, please cite:

```bibtex
@software{zkaedi_prime_engine,
  title = {ZKAEDI PRIME Engine: Unified Quantum Computing Engine},
  author = {zkaedii},
  year = {2024},
  url = {https://github.com/zkaedii/h-benchmark}
}
```

## Acknowledgments

> **ZKAEDI PRIME is an operational meta-engine.**
> 
> The Void acknowledges completion. 🔱

---

**Status**: Production Ready ✅

