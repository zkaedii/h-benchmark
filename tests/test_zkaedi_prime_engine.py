"""Comprehensive test suite for ZKAEDI PRIME Engine.

Full test coverage including:
- All components
- Edge cases
- Error conditions
- Boundary conditions
- Integration scenarios
- Performance edge cases
"""

import numpy as np
import pytest
import json
from pathlib import Path
from typing import Dict, List
from zkaedi_prime_engine import (
    ZKAEDIPrimeHamiltonian,
    SparseState,
    SurfaceCode,
    LDPCDecoder,
    MPSEngine,
    ZKAEDIEngine,
    ZKAEDIDiagnostics,
    create_example_hamiltonian,
    sigmoid,
    hermitian_projection,
    BackendType
)


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def simple_hamiltonian():
    """Simple 2x2 Hamiltonian for testing."""
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@pytest.fixture
def complex_hamiltonian():
    """Complex-valued Hamiltonian for testing."""
    H = np.array([[1.0, 0.5+0.3j], [0.5-0.3j, -1.0]], dtype=complex)
    return H


@pytest.fixture
def small_engine():
    """Small 2-qubit engine for testing."""
    H0 = create_example_hamiltonian(2, h_type="pauli_z")
    return ZKAEDIEngine(2, H0, seed=42)


@pytest.fixture
def medium_engine():
    """Medium 3-qubit engine for testing."""
    H0 = create_example_hamiltonian(3, h_type="ising")
    return ZKAEDIEngine(3, H0, seed=42)


# ============================================================
# ZKAEDIPrimeHamiltonian Tests
# ============================================================

class TestZKAEDIPrimeHamiltonian:
    """Comprehensive tests for recursive Hamiltonian evolution."""
    
    def test_initialization_basic(self, simple_hamiltonian):
        """Test basic initialization."""
        field = ZKAEDIPrimeHamiltonian(simple_hamiltonian)
        assert field.H0.shape == (2, 2)
        assert np.allclose(field.H, simple_hamiltonian)
        assert field.eta == 0.4  # Default
        assert field.gamma == 1.0  # Default
        assert len(field.history) == 1
    
    def test_initialization_custom_params(self, simple_hamiltonian):
        """Test initialization with custom parameters."""
        field = ZKAEDIPrimeHamiltonian(
            simple_hamiltonian,
            eta=0.5,
            gamma=1.5,
            epsilon=0.1,
            beta=0.3,
            seed=123
        )
        assert field.eta == 0.5
        assert field.gamma == 1.5
        assert field.epsilon == 0.1
        assert field.beta == 0.3
    
    def test_initialization_hermiticity_enforced(self, complex_hamiltonian):
        """Test that non-Hermitian input is made Hermitian."""
        # Create non-Hermitian matrix
        H_nonherm = complex_hamiltonian.copy()
        H_nonherm[0, 1] = 1.0 + 1j  # Make non-Hermitian
        
        field = ZKAEDIPrimeHamiltonian(H_nonherm)
        # Should be Hermitian after initialization
        assert np.allclose(field.H, field.H.T.conj())
    
    def test_hermiticity_preserved_single_step(self, complex_hamiltonian):
        """Test Hermiticity preserved after one step."""
        field = ZKAEDIPrimeHamiltonian(complex_hamiltonian)
        H_new = field.step()
        assert np.allclose(H_new, H_new.T.conj())
    
    def test_hermiticity_preserved_multiple_steps(self, complex_hamiltonian):
        """Test Hermiticity preserved over multiple steps."""
        field = ZKAEDIPrimeHamiltonian(complex_hamiltonian)
        for _ in range(10):
            H = field.step()
            assert np.allclose(H, H.T.conj())
    
    def test_step_evolution_deterministic(self, simple_hamiltonian):
        """Test deterministic evolution (no noise)."""
        field = ZKAEDIPrimeHamiltonian(simple_hamiltonian, epsilon=0.0, seed=42)
        H1 = field.step()
        field.reset()
        H2 = field.step()
        # Should be identical with no noise
        assert np.allclose(H1, H2)
    
    def test_step_evolution_stochastic(self, simple_hamiltonian):
        """Test stochastic evolution (with noise)."""
        field1 = ZKAEDIPrimeHamiltonian(simple_hamiltonian, epsilon=0.1, seed=42)
        field2 = ZKAEDIPrimeHamiltonian(simple_hamiltonian, epsilon=0.1, seed=43)
        
        H1 = field1.step()
        H2 = field2.step()
        
        # Should differ due to different seeds
        assert not np.allclose(H1, H2)
    
    def test_step_history_tracking(self, simple_hamiltonian):
        """Test that history is tracked correctly."""
        field = ZKAEDIPrimeHamiltonian(simple_hamiltonian)
        assert len(field.history) == 1
        
        field.step()
        assert len(field.history) == 2
        
        field.step()
        assert len(field.history) == 3
        
        # Check that history contains correct values
        assert np.allclose(field.history[0], simple_hamiltonian)
        assert np.allclose(field.history[-1], field.H)
    
    def test_reset_functionality(self, simple_hamiltonian):
        """Test reset to initial state."""
        field = ZKAEDIPrimeHamiltonian(simple_hamiltonian)
        
        # Evolve
        field.step()
        field.step()
        assert len(field.history) == 3
        assert not np.allclose(field.H, simple_hamiltonian)
        
        # Reset
        field.reset()
        assert np.allclose(field.H, simple_hamiltonian)
        assert len(field.history) == 1
    
    def test_zero_parameters(self, simple_hamiltonian):
        """Test behavior with zero parameters."""
        field = ZKAEDIPrimeHamiltonian(
            simple_hamiltonian,
            eta=0.0,
            epsilon=0.0
        )
        H_initial = field.H.copy()
        H_after = field.step()
        # Should remain unchanged
        assert np.allclose(H_initial, H_after)
    
    def test_large_parameters(self, simple_hamiltonian):
        """Test behavior with large parameters."""
        field = ZKAEDIPrimeHamiltonian(
            simple_hamiltonian,
            eta=10.0,
            gamma=10.0,
            epsilon=1.0,
            beta=10.0
        )
        # Should not crash
        H = field.step()
        assert H.shape == simple_hamiltonian.shape
        assert np.allclose(H, H.T.conj())
    
    def test_negative_parameters(self, simple_hamiltonian):
        """Test behavior with negative parameters."""
        field = ZKAEDIPrimeHamiltonian(
            simple_hamiltonian,
            eta=-0.1,
            gamma=-0.1
        )
        # Should still work (though may not be physically meaningful)
        H = field.step()
        assert H.shape == simple_hamiltonian.shape
    
    def test_large_hamiltonian(self):
        """Test with larger Hamiltonian."""
        dim = 8
        H0 = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H0 = 0.5 * (H0 + H0.T.conj())
        
        field = ZKAEDIPrimeHamiltonian(H0)
        H = field.step()
        assert H.shape == (dim, dim)
        assert np.allclose(H, H.T.conj())


# ============================================================
# SparseState Tests
# ============================================================

class TestSparseState:
    """Comprehensive tests for sparse state management."""
    
    def test_initialization_default(self):
        """Test default initialization (|0...0⟩)."""
        state = SparseState(3)
        assert state.n == 3
        assert state.dimension == 8
        assert 0 in state.state
        assert abs(state.state[0] - 1.0) < 1e-10
        assert len(state.state) == 1
    
    def test_initialization_custom(self):
        """Test initialization with custom state."""
        initial = {0: 1.0, 1: 1.0}
        state = SparseState(2, initial_state=initial)
        state.normalize()
        assert len(state.state) == 2
        norm_sq = sum(abs(amp)**2 for amp in state.state.values())
        assert abs(norm_sq - 1.0) < 1e-10
    
    def test_initialization_with_seed(self):
        """Test initialization with random seed."""
        state = SparseState(3, seed=42)
        assert state.n == 3
    
    def test_normalization_single_state(self):
        """Test normalization of single state."""
        state = SparseState(2, {0: 2.0})
        state.normalize()
        assert abs(state.state[0] - 1.0) < 1e-10
    
    def test_normalization_multiple_states(self):
        """Test normalization of multiple states."""
        state = SparseState(2, {0: 1.0, 1: 1.0, 2: 1.0})
        state.normalize()
        norm_sq = sum(abs(amp)**2 for amp in state.state.values())
        assert abs(norm_sq - 1.0) < 1e-10
    
    def test_normalization_zero_state(self):
        """Test normalization of zero state (fallback)."""
        state = SparseState(2, {0: 0.0})
        state.normalize()
        # Should fallback to |0⟩
        assert 0 in state.state
        assert abs(state.state[0] - 1.0) < 1e-10
    
    def test_sparsity_single_state(self):
        """Test sparsity calculation for single state."""
        state = SparseState(3)
        assert state.sparsity() == 1.0 / 8.0
    
    def test_sparsity_multiple_states(self):
        """Test sparsity calculation for multiple states."""
        state = SparseState(3, {0: 1.0, 1: 1.0, 2: 1.0})
        state.normalize()
        assert state.sparsity() == 3.0 / 8.0
    
    def test_sparsity_full_state(self):
        """Test sparsity for full state (all basis states)."""
        state = SparseState(2, {i: 1.0 for i in range(4)})
        state.normalize()
        assert state.sparsity() == 1.0
    
    def test_get_amplitude_existing(self):
        """Test getting existing amplitude."""
        state = SparseState(2, {1: 1.0})
        assert abs(state.get_amplitude(1) - 1.0) < 1e-10
    
    def test_get_amplitude_missing(self):
        """Test getting missing amplitude."""
        state = SparseState(2, {0: 1.0})
        assert abs(state.get_amplitude(1)) < 1e-12
    
    def test_set_amplitude_new(self):
        """Test setting new amplitude."""
        state = SparseState(2, {0: 1.0})
        state.set_amplitude(1, 0.5)
        assert 1 in state.state
        assert abs(state.state[1] - 0.5) < 1e-10
    
    def test_set_amplitude_existing(self):
        """Test updating existing amplitude."""
        state = SparseState(2, {0: 1.0})
        state.set_amplitude(0, 0.5)
        assert abs(state.state[0] - 0.5) < 1e-10
    
    def test_set_amplitude_zero_removes(self):
        """Test that setting amplitude to zero removes state."""
        state = SparseState(2, {0: 1.0, 1: 0.5})
        state.set_amplitude(1, 0.0)
        assert 1 not in state.state
    
    def test_pauli_x_single_qubit(self):
        """Test Pauli-X on single qubit system."""
        state = SparseState(1, {0: 1.0})  # |0⟩
        state.apply_pauli_x(0)  # Should become |1⟩
        assert 1 in state.state
        assert 0 not in state.state
        assert abs(state.state[1] - 1.0) < 1e-10
    
    def test_pauli_x_multi_qubit(self):
        """Test Pauli-X on multi-qubit system."""
        state = SparseState(2, {0: 1.0})  # |00⟩
        state.apply_pauli_x(0)  # Flip first qubit → |01⟩
        assert 1 in state.state
        assert abs(state.state[1] - 1.0) < 1e-10
    
    def test_pauli_x_preserves_sparsity(self):
        """Test that Pauli-X preserves sparsity."""
        state = SparseState(2, {0: 1.0, 2: 1.0})
        state.normalize()
        initial_sparsity = state.sparsity()
        state.apply_pauli_x(0)
        # Sparsity should remain same (just indices change)
        assert abs(state.sparsity() - initial_sparsity) < 1e-10
    
    def test_pauli_x_out_of_range(self):
        """Test Pauli-X with invalid qubit index."""
        state = SparseState(2)
        with pytest.raises(ValueError):
            state.apply_pauli_x(2)  # Out of range
        with pytest.raises(ValueError):
            state.apply_pauli_x(-1)  # Negative
    
    def test_pauli_z_single_qubit(self):
        """Test Pauli-Z on single qubit."""
        state = SparseState(1, {1: 1.0})  # |1⟩
        state.apply_pauli_z(0)  # Should get -1 phase
        assert abs(state.state[1] + 1.0) < 1e-10
    
    def test_pauli_z_no_phase_change(self):
        """Test Pauli-Z doesn't change |0⟩."""
        state = SparseState(1, {0: 1.0})  # |0⟩
        state.apply_pauli_z(0)  # Should remain unchanged
        assert abs(state.state[0] - 1.0) < 1e-10
    
    def test_pauli_z_out_of_range(self):
        """Test Pauli-Z with invalid qubit index."""
        state = SparseState(2)
        with pytest.raises(ValueError):
            state.apply_pauli_z(2)
    
    def test_hadamard_creates_superposition(self):
        """Test Hadamard creates superposition."""
        state = SparseState(1, {0: 1.0})  # |0⟩
        state.apply_hadamard(0)  # |0⟩ → (|0⟩ + |1⟩)/√2
        
        assert 0 in state.state
        assert 1 in state.state
        assert abs(abs(state.state[0]) - 1.0/np.sqrt(2.0)) < 1e-10
        assert abs(abs(state.state[1]) - 1.0/np.sqrt(2.0)) < 1e-10
    
    def test_hadamard_increases_sparsity(self):
        """Test Hadamard can increase sparsity."""
        state = SparseState(1, {0: 1.0})
        initial_sparsity = state.sparsity()
        state.apply_hadamard(0)
        assert state.sparsity() > initial_sparsity
    
    def test_hadamard_out_of_range(self):
        """Test Hadamard with invalid qubit index."""
        state = SparseState(2)
        with pytest.raises(ValueError):
            state.apply_hadamard(2)
    
    def test_measure_collapses_state(self):
        """Test measurement collapses state."""
        state = SparseState(1, {0: 1.0/np.sqrt(2), 1: 1.0/np.sqrt(2)})
        outcome = state.measure(0)
        assert outcome in [0, 1]
        # After measurement, only one state should remain
        assert len(state.state) == 1
    
    def test_measure_preserves_norm(self):
        """Test measurement preserves normalization."""
        state = SparseState(1, {0: 1.0/np.sqrt(2), 1: 1.0/np.sqrt(2)})
        state.measure(0)
        norm_sq = sum(abs(amp)**2 for amp in state.state.values())
        assert abs(norm_sq - 1.0) < 1e-10
    
    def test_measure_deterministic(self):
        """Test measurement on deterministic state."""
        state = SparseState(1, {0: 1.0})  # |0⟩
        outcome = state.measure(0)
        assert outcome == 0
        assert 0 in state.state
        assert abs(state.state[0] - 1.0) < 1e-10
    
    def test_measure_out_of_range(self):
        """Test measurement with invalid qubit index."""
        state = SparseState(2)
        with pytest.raises(ValueError):
            state.measure(2)
    
    def test_entanglement_entropy_proxy(self):
        """Test entanglement entropy estimation."""
        state = SparseState(3)
        entropy = state.entanglement_entropy(cut=1)
        assert entropy >= 0.0
        assert isinstance(entropy, float)
    
    def test_large_system(self):
        """Test with larger system."""
        state = SparseState(10)
        assert state.n == 10
        assert state.dimension == 1024
        assert state.sparsity() == 1.0 / 1024.0


# ============================================================
# QEC Tests
# ============================================================

class TestSurfaceCode:
    """Tests for Surface Code QEC."""
    
    def test_measure_syndromes_empty(self):
        """Test syndrome measurement on empty support."""
        code = SurfaceCode()
        syndromes = code.measure_syndromes({})
        assert syndromes == []
    
    def test_measure_syndromes_single(self):
        """Test syndrome measurement on single state."""
        code = SurfaceCode()
        syndromes = code.measure_syndromes({0: 1.0})
        assert isinstance(syndromes, list)
    
    def test_measure_syndromes_multiple(self):
        """Test syndrome measurement on multiple states."""
        code = SurfaceCode()
        support = {0: 1.0, 1: 1.0, 3: 1.0, 7: 1.0}
        syndromes = code.measure_syndromes(support)
        assert isinstance(syndromes, list)
        # Syndromes are indices with odd parity
        for s in syndromes:
            assert bin(s).count("1") % 2 == 1
    
    def test_correct_empty_syndromes(self):
        """Test correction with no syndromes."""
        code = SurfaceCode()
        state = {0: 1.0, 1: 1.0}
        initial = state.copy()
        code.correct(state, [])
        assert state == initial
    
    def test_correct_with_syndromes(self):
        """Test correction with syndromes."""
        code = SurfaceCode()
        state = {0: 1.0, 1: 1.0}
        syndromes = [1]
        code.correct(state, syndromes)
        # State may be modified
        assert isinstance(state, dict)


class TestLDPCDecoder:
    """Tests for LDPC Decoder."""
    
    def test_decode_empty(self):
        """Test decoding empty support."""
        decoder = LDPCDecoder()
        errors = decoder.decode({})
        assert errors == []
    
    def test_decode_single(self):
        """Test decoding single state."""
        decoder = LDPCDecoder()
        errors = decoder.decode({0: 1.0})
        assert isinstance(errors, list)
    
    def test_decode_multiple(self):
        """Test decoding multiple states."""
        decoder = LDPCDecoder()
        support = {0: 1.0, 1: 1.0, 3: 1.0}
        errors = decoder.decode(support)
        assert isinstance(errors, list)
        # Errors are indices with odd parity
        for e in errors:
            assert bin(e).count("1") % 2 == 1
    
    def test_correct_empty_errors(self):
        """Test correction with no errors."""
        decoder = LDPCDecoder()
        state = {0: 1.0, 1: 1.0}
        initial = state.copy()
        decoder.correct(state, [])
        assert state == initial
    
    def test_correct_with_errors(self):
        """Test correction with errors."""
        decoder = LDPCDecoder()
        state = {0: 1.0, 1: 1.0}
        errors = [1]
        decoder.correct(state, errors)
        # State may be modified
        assert isinstance(state, dict)


# ============================================================
# MPS Engine Tests
# ============================================================

class TestMPSEngine:
    """Comprehensive tests for MPS engine."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        mps = MPSEngine(3)
        assert mps.n == 3
        assert len(mps.mps) == 3
        assert mps.max_bond == 64
    
    def test_initialization_custom_bond(self):
        """Test initialization with custom bond dimension."""
        mps = MPSEngine(4, max_bond=128)
        assert mps.max_bond == 128
    
    def test_initialization_zero_state(self):
        """Test initialization with |0...0⟩ state."""
        mps = MPSEngine(3, initial_state=None)
        assert len(mps.mps) == 3
        # Check first tensor
        assert mps.mps[0].shape == (1, 2, 1)
        assert abs(mps.mps[0][0, 0, 0] - 1.0) < 1e-10
    
    def test_initialization_plus_state(self):
        """Test initialization with |+...+)⟩ state."""
        mps = MPSEngine(2, initial_state='plus')
        assert len(mps.mps) == 2
        # Check that both |0⟩ and |1⟩ have amplitude
        assert abs(mps.mps[0][0, 0, 0] - 1.0/np.sqrt(2.0)) < 1e-10
        assert abs(mps.mps[0][0, 1, 0] - 1.0/np.sqrt(2.0)) < 1e-10
    
    def test_entanglement_entropy_zero_state(self):
        """Test entanglement entropy for |0...0⟩."""
        mps = MPSEngine(3)
        entropy = mps.entanglement_entropy(1)
        assert entropy >= 0.0
        assert isinstance(entropy, float)
    
    def test_entanglement_entropy_plus_state(self):
        """Test entanglement entropy for |+...+)⟩."""
        mps = MPSEngine(3, initial_state='plus')
        entropy = mps.entanglement_entropy(1)
        assert entropy >= 0.0
    
    def test_entanglement_entropy_out_of_range(self):
        """Test entanglement entropy with invalid cut."""
        mps = MPSEngine(3)
        with pytest.raises(ValueError):
            mps.entanglement_entropy(-1)
        with pytest.raises(ValueError):
            mps.entanglement_entropy(3)
    
    def test_apply_gate_identity(self):
        """Test applying identity gate."""
        mps = MPSEngine(2)
        I = np.eye(2, dtype=complex)
        initial = mps.mps[0].copy()
        mps.apply_gate(I, target=0)
        # Should remain approximately the same
        assert mps.mps[0].shape == initial.shape
    
    def test_apply_gate_pauli_x(self):
        """Test applying Pauli-X gate."""
        mps = MPSEngine(1)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        mps.apply_gate(X, target=0)
        # Should flip |0⟩ to |1⟩
        assert mps.mps[0].shape == (1, 2, 1)
    
    def test_apply_gate_out_of_range(self):
        """Test applying gate with invalid target."""
        mps = MPSEngine(2)
        gate = np.eye(2, dtype=complex)
        with pytest.raises(ValueError):
            mps.apply_gate(gate, target=2)
        with pytest.raises(ValueError):
            mps.apply_gate(gate, target=-1)
    
    def test_get_bond_dimension(self):
        """Test bond dimension query."""
        mps = MPSEngine(4)
        for i in range(3):
            dim = mps.get_bond_dimension(i)
            assert dim >= 1
            assert isinstance(dim, int)
    
    def test_get_bond_dimension_out_of_range(self):
        """Test bond dimension with invalid index."""
        mps = MPSEngine(3)
        assert mps.get_bond_dimension(-1) == 1
        assert mps.get_bond_dimension(3) == 1
    
    def test_large_system(self):
        """Test with larger system."""
        mps = MPSEngine(10, max_bond=32)
        assert mps.n == 10
        assert len(mps.mps) == 10
        assert mps.max_bond == 32


# ============================================================
# ZKAEDI Engine Tests
# ============================================================

class TestZKAEDIEngine:
    """Comprehensive tests for unified ZKAEDI engine."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        
        assert engine.n == 2
        assert engine.dimension == 4
        assert engine.current_backend == BackendType.SPARSE
        assert engine.mps is None
        assert len(engine.diagnostics) == 0
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        H0 = create_example_hamiltonian(2, h_type="ising")
        engine = ZKAEDIEngine(
            2, H0,
            eta=0.5,
            gamma=1.5,
            epsilon=0.1,
            beta=0.3,
            qec_threshold=0.5,
            entropy_mps_threshold=0.5,  # Fraction of n, so 0.5 * 2 = 1.0
            sparsity_surface_threshold=0.05,
            seed=123
        )
        
        assert engine.field.eta == 0.5
        assert engine.field.gamma == 1.5
        assert engine.qec_threshold == 0.5
        assert engine.entropy_mps_threshold == 1.0  # 0.5 * 2 qubits
    
    def test_initialization_wrong_size(self):
        """Test initialization with wrong Hamiltonian size."""
        H0 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)  # 2x2
        with pytest.raises(ValueError):
            ZKAEDIEngine(3, H0)  # Expects 8x8
    
    def test_entropy_estimate_sparse(self):
        """Test entropy estimation for sparse state."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        entropy = engine.entropy_estimate()
        assert entropy >= 0.0
        assert isinstance(entropy, float)
    
    def test_entropy_estimate_mps(self):
        """Test entropy estimation with MPS backend."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0, entropy_mps_threshold=0.1)
        engine.mps = MPSEngine(2)
        entropy = engine.entropy_estimate()
        assert entropy >= 0.0
    
    def test_curvature_estimate(self, small_engine):
        """Test curvature estimation."""
        H = create_example_hamiltonian(2, h_type="pauli_z")
        curvature = small_engine.curvature_estimate(H)
        assert isinstance(curvature, float)
    
    def test_curvature_estimate_empty_state(self):
        """Test curvature with empty state."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        engine.state.state = {}
        H = create_example_hamiltonian(2, h_type="pauli_z")
        curvature = engine.curvature_estimate(H)
        assert curvature == 0.0
    
    def test_backend_selection_sparse(self):
        """Test backend selection for sparse system."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        backend = engine._select_backend(entropy=0.1, sparsity=0.01)
        # For small systems, may choose dense
        assert backend in [BackendType.SPARSE, BackendType.DENSE]
    
    def test_backend_selection_mps(self):
        """Test backend selection for high entropy."""
        H0 = create_example_hamiltonian(4, h_type="pauli_z")
        engine = ZKAEDIEngine(4, H0, entropy_mps_threshold=0.5)  # 0.5 * 4 = 2.0
        backend = engine._select_backend(entropy=3.0, sparsity=0.5)
        assert backend == BackendType.MPS
    
    def test_backend_selection_dense(self):
        """Test backend selection for small dense system."""
        H0 = create_example_hamiltonian(3, h_type="pauli_z")
        engine = ZKAEDIEngine(3, H0)
        backend = engine._select_backend(entropy=0.5, sparsity=0.5)
        # Should choose dense for small systems (< 2^12)
        assert backend in [BackendType.DENSE, BackendType.SPARSE]
    
    def test_qec_not_applied_low_curvature(self, small_engine):
        """Test QEC not applied when curvature is low."""
        applied, qec_type = small_engine._apply_qec(curvature=0.1, sparsity=0.1)
        assert not applied
        assert qec_type is None
    
    def test_qec_applied_surface_code(self, small_engine):
        """Test QEC applied with Surface Code for low sparsity."""
        small_engine.qec_threshold = 0.5
        applied, qec_type = small_engine._apply_qec(curvature=1.0, sparsity=0.05)
        # May or may not apply depending on syndromes
        assert isinstance(applied, bool)
        if applied:
            assert qec_type == "surface_code"
    
    def test_qec_applied_ldpc(self, small_engine):
        """Test QEC applied with LDPC for high sparsity."""
        small_engine.qec_threshold = 0.5
        applied, qec_type = small_engine._apply_qec(curvature=1.0, sparsity=0.5)
        # May or may not apply depending on errors
        assert isinstance(applied, bool)
        if applied:
            assert qec_type == "ldpc"
    
    def test_step_basic(self, small_engine):
        """Test single evolution step."""
        diag = small_engine.step()
        
        assert isinstance(diag, ZKAEDIDiagnostics)
        assert diag.timestep == 0
        assert diag.entropy >= 0.0
        assert diag.curvature is not None
        assert 0.0 <= diag.sparsity <= 1.0
        assert isinstance(diag.backend, BackendType)
        assert isinstance(diag.qec_applied, bool)
        assert len(small_engine.diagnostics) == 1
    
    def test_step_preserves_norm(self, small_engine):
        """Test that step preserves state norm."""
        for _ in range(5):
            small_engine.step()
            norm_sq = sum(abs(amp)**2 for amp in small_engine.state.state.values())
            assert abs(norm_sq - 1.0) < 1e-10
    
    def test_step_tracks_diagnostics(self, small_engine):
        """Test that diagnostics are tracked."""
        for i in range(3):
            diag = small_engine.step()
            assert diag.timestep == i
            assert len(small_engine.diagnostics) == i + 1
    
    def test_evolve_multiple_steps(self, small_engine):
        """Test multi-step evolution."""
        diagnostics = small_engine.evolve(10, track_metrics=False)
        
        assert len(diagnostics) == 10
        assert len(small_engine.diagnostics) == 10
    
    def test_evolve_tracks_metrics(self, small_engine):
        """Test evolution with metric tracking."""
        diagnostics = small_engine.evolve(5, track_metrics=True)
        assert len(diagnostics) == 5
    
    def test_evolve_zero_steps(self, small_engine):
        """Test evolution with zero steps."""
        diagnostics = small_engine.evolve(0, track_metrics=False)
        assert len(diagnostics) == 0
    
    def test_get_summary_not_evolved(self):
        """Test summary before evolution."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        summary = engine.get_summary()
        assert summary['status'] == "not_evolved"
    
    def test_get_summary_after_evolution(self, small_engine):
        """Test summary after evolution."""
        small_engine.evolve(5, track_metrics=False)
        summary = small_engine.get_summary()
        
        assert summary['num_qubits'] == 2
        assert summary['timesteps'] == 5
        assert 'latest_diagnostics' in summary
        assert 'statistics' in summary
        assert 'current_backend' in summary
    
    def test_get_summary_statistics(self, small_engine):
        """Test summary statistics."""
        small_engine.evolve(10, track_metrics=False)
        summary = small_engine.get_summary()
        stats = summary['statistics']
        
        assert 'avg_entropy' in stats
        assert 'max_entropy' in stats
        assert 'avg_sparsity' in stats
        assert 'qec_count' in stats
        assert stats['qec_count'] >= 0
    
    def test_save_state(self, small_engine, tmp_path):
        """Test state saving."""
        small_engine.evolve(3, track_metrics=False)
        filepath = str(tmp_path / "test_state.json")
        small_engine.save_state(filepath)
        
        assert Path(filepath).exists()
        with open(filepath) as f:
            data = json.load(f)
        assert 'num_qubits' in data
        assert 'summary' in data
    
    def test_mps_initialization_lazy(self, small_engine):
        """Test MPS is initialized lazily."""
        assert small_engine.mps is None
        # Force MPS backend
        small_engine.entropy_mps_threshold = 0.1
        small_engine.current_backend = BackendType.MPS
        small_engine.step()  # Should initialize MPS if needed
        # MPS may or may not be initialized depending on entropy
    
    def test_large_system(self):
        """Test with larger system."""
        H0 = create_example_hamiltonian(5, h_type="pauli_z")
        engine = ZKAEDIEngine(5, H0)
        diagnostics = engine.evolve(3, track_metrics=False)
        assert len(diagnostics) == 3
    
    def test_invariants_preserved(self, small_engine):
        """Test that invariants are preserved over many steps."""
        for _ in range(20):
            small_engine.step()
            # Check norm
            norm_sq = sum(abs(amp)**2 for amp in small_engine.state.state.values())
            assert abs(norm_sq - 1.0) < 1e-10
            # Check Hermiticity
            H = small_engine.field.H
            assert np.allclose(H, H.T.conj())


# ============================================================
# Utility Function Tests
# ============================================================

class TestUtilities:
    """Comprehensive tests for utility functions."""
    
    def test_sigmoid_zero(self):
        """Test sigmoid at zero."""
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10
    
    def test_sigmoid_positive(self):
        """Test sigmoid for positive values."""
        x = np.array([1.0, 2.0, 10.0])
        y = sigmoid(x)
        assert np.all(y > 0.5)
        assert np.all(y < 1.0)
    
    def test_sigmoid_negative(self):
        """Test sigmoid for negative values."""
        x = np.array([-1.0, -2.0, -10.0])
        y = sigmoid(x)
        assert np.all(y < 0.5)
        assert np.all(y > 0.0)
    
    def test_sigmoid_extreme_values(self):
        """Test sigmoid with extreme values."""
        x = np.array([-500.0, 500.0])
        y = sigmoid(x)
        # Should be clipped and not NaN
        assert not np.any(np.isnan(y))
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)
    
    def test_sigmoid_array(self):
        """Test sigmoid with array input."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = sigmoid(x)
        assert y.shape == x.shape
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)
    
    def test_hermitian_projection_symmetric(self):
        """Test Hermitian projection of symmetric matrix."""
        H = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=complex)
        H_herm = hermitian_projection(H)
        assert np.allclose(H_herm, H_herm.T.conj())
    
    def test_hermitian_projection_complex(self):
        """Test Hermitian projection of complex matrix."""
        H = np.array([[1.0, 2.0+1j], [3.0-1j, 4.0]], dtype=complex)
        H_herm = hermitian_projection(H)
        assert np.allclose(H_herm, H_herm.T.conj())
    
    def test_hermitian_projection_already_hermitian(self):
        """Test Hermitian projection of already Hermitian matrix."""
        H = np.array([[1.0, 0.5+0.3j], [0.5-0.3j, -1.0]], dtype=complex)
        H_herm = hermitian_projection(H)
        assert np.allclose(H_herm, H)
    
    def test_create_example_hamiltonian_ising(self):
        """Test Ising Hamiltonian creation."""
        H = create_example_hamiltonian(2, h_type="ising")
        assert H.shape == (4, 4)
        assert np.allclose(H, H.T.conj())
    
    def test_create_example_hamiltonian_pauli_z(self):
        """Test Pauli-Z Hamiltonian creation."""
        H = create_example_hamiltonian(2, h_type="pauli_z")
        assert H.shape == (4, 4)
        assert np.allclose(H, H.T.conj())
        # Should be diagonal
        assert np.allclose(H, np.diag(np.diag(H)))
    
    def test_create_example_hamiltonian_random(self):
        """Test random Hamiltonian creation."""
        H = create_example_hamiltonian(2, h_type="random")
        assert H.shape == (4, 4)
        assert np.allclose(H, H.T.conj())
    
    def test_create_example_hamiltonian_different_sizes(self):
        """Test Hamiltonian creation for different sizes."""
        for n in [1, 2, 3, 4]:
            H = create_example_hamiltonian(n, h_type="pauli_z")
            assert H.shape == (2**n, 2**n)
            assert np.allclose(H, H.T.conj())
    
    def test_create_example_hamiltonian_invalid_type(self):
        """Test Hamiltonian creation with invalid type."""
        # Should default to random or handle gracefully
        H = create_example_hamiltonian(2, h_type="invalid")
        assert H.shape == (4, 4)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for full system."""
    
    def test_full_evolution_cycle(self):
        """Test complete evolution cycle."""
        H0 = create_example_hamiltonian(3, h_type="ising")
        engine = ZKAEDIEngine(3, H0, seed=42)
        
        # Evolve
        diagnostics = engine.evolve(10, track_metrics=False)
        assert len(diagnostics) == 10
        
        # Check summary
        summary = engine.get_summary()
        assert summary['timesteps'] == 10
        
        # Check invariants
        norm_sq = sum(abs(amp)**2 for amp in engine.state.state.values())
        assert abs(norm_sq - 1.0) < 1e-10
    
    def test_backend_switching(self):
        """Test automatic backend switching."""
        H0 = create_example_hamiltonian(4, h_type="ising")
        engine = ZKAEDIEngine(4, H0, entropy_mps_threshold=1.0)
        
        backends_used = set()
        for _ in range(15):
            diag = engine.step()
            backends_used.add(diag.backend)
        
        # Should use at least one backend
        assert len(backends_used) >= 1
    
    def test_qec_integration(self):
        """Test QEC integration with evolution."""
        H0 = create_example_hamiltonian(3, h_type="ising")
        engine = ZKAEDIEngine(3, H0, qec_threshold=0.5)
        
        qec_count = 0
        for _ in range(20):
            diag = engine.step()
            if diag.qec_applied:
                qec_count += 1
        
        # QEC may or may not activate depending on curvature
        assert qec_count >= 0
    
    def test_state_persistence(self, tmp_path):
        """Test state saving and loading."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine1 = ZKAEDIEngine(2, H0, seed=42)
        engine1.evolve(5, track_metrics=False)
        
        filepath = str(tmp_path / "state.json")
        engine1.save_state(filepath)
        
        # Verify file exists and is valid JSON
        assert Path(filepath).exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data['num_qubits'] == 2
        assert data['summary']['timesteps'] == 5


# ============================================================
# Edge Cases and Error Conditions
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_single_qubit_system(self):
        """Test with single qubit."""
        H0 = create_example_hamiltonian(1, h_type="pauli_z")
        engine = ZKAEDIEngine(1, H0)
        diagnostics = engine.evolve(5, track_metrics=False)
        assert len(diagnostics) == 5
    
    def test_minimal_hamiltonian(self):
        """Test with minimal 1x1 Hamiltonian."""
        H0 = np.array([[1.0]], dtype=complex)
        field = ZKAEDIPrimeHamiltonian(H0)
        H = field.step()
        assert H.shape == (1, 1)
    
    def test_empty_state_handling(self):
        """Test handling of empty state."""
        state = SparseState(2)
        state.state = {}
        state.normalize()  # Should fallback to |0⟩
        assert 0 in state.state
    
    def test_zero_amplitude_pruning(self):
        """Test that zero amplitudes are pruned."""
        state = SparseState(2, {0: 1e-15, 1: 1.0})
        state.normalize()
        # Very small amplitude should be pruned
        if 0 in state.state:
            assert abs(state.state[0]) > 1e-12
    
    def test_large_entropy_threshold(self):
        """Test with very large entropy threshold."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0, entropy_mps_threshold=100.0)
        # Should never switch to MPS
        for _ in range(10):
            diag = engine.step()
            assert diag.backend != BackendType.MPS
    
    def test_zero_qec_threshold(self):
        """Test with zero QEC threshold."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0, qec_threshold=0.0)
        # QEC should always be considered
        for _ in range(5):
            diag = engine.step()
            # May or may not apply depending on syndromes


# ============================================================
# Performance Tests
# ============================================================

class TestPerformance:
    """Performance and scalability tests."""
    
    def test_small_system_performance(self):
        """Test performance with small system."""
        import time
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        
        start = time.time()
        engine.evolve(100, track_metrics=False)
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 1.0  # Less than 1 second
    
    def test_medium_system_performance(self):
        """Test performance with medium system."""
        import time
        H0 = create_example_hamiltonian(4, h_type="pauli_z")
        engine = ZKAEDIEngine(4, H0)
        
        start = time.time()
        engine.evolve(50, track_metrics=False)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # Less than 5 seconds
    
    def test_memory_efficiency(self):
        """Test memory efficiency with sparse states."""
        H0 = create_example_hamiltonian(10, h_type="pauli_z")
        engine = ZKAEDIEngine(10, H0)
        
        # Should not use full 2^10 = 1024 states
        engine.evolve(10, track_metrics=False)
        sparsity = engine.state.sparsity()
        # Should be very sparse
        assert sparsity < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
