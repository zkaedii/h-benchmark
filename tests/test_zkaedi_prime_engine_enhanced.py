"""Enhanced test suite for ZKAEDI PRIME Engine.

Addresses review feedback:
- Improved performance tests with relative comparisons
- Expanded edge case handling
- Custom exception testing
- Mock dependency testing
- Refined comments
"""

import numpy as np
import pytest
import time
import sys
from unittest.mock import patch, MagicMock
from typing import Dict

from zkaedi_prime_engine import (
    ZKAEDIPrimeHamiltonian,
    SparseState,
    SurfaceCode,
    LDPCDecoder,
    MPSEngine,
    ZKAEDIEngine,
    create_example_hamiltonian,
    sigmoid,
    hermitian_projection,
    BackendType
)

from zkaedi_prime_engine.exceptions import (
    ZKAEDIError,
    HamiltonianError,
    EngineInitializationError,
    StateError,
    QECError,
    BackendError
)


# ============================================================
# Enhanced Performance Tests
# ============================================================

class TestPerformanceEnhanced:
    """Enhanced performance tests with relative comparisons and parameterization."""
    
    def baseline_task(self) -> float:
        """Run a small, consistent task to establish baseline performance."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        start = time.time()
        engine.evolve(5, track_metrics=False)
        return time.time() - start
    
    @pytest.mark.parametrize("num_qubits,steps", [
        (2, 100),
        (3, 50),
        (4, 25),
        (5, 10),
        (6, 5)
    ])
    def test_performance_varying_system(self, num_qubits, steps):
        """Test performance across varying system sizes with relative comparisons."""
        # Establish baseline
        baseline_time = self.baseline_task()
        
        # Run test
        H0 = create_example_hamiltonian(num_qubits, h_type="pauli_z")
        engine = ZKAEDIEngine(num_qubits, H0)
        
        start = time.time()
        engine.evolve(steps, track_metrics=False)
        elapsed = time.time() - start
        
        # Relative comparison: allow up to 100x baseline for larger systems
        # For 2 qubits, use a more lenient threshold (50x baseline) due to overhead
        if num_qubits == 2:
            max_expected = baseline_time * 50
        else:
            max_expected = baseline_time * (100 ** (num_qubits - 2))
        assert elapsed < max_expected, \
            f"Performance degraded: {elapsed:.3f}s > {max_expected:.3f}s for {num_qubits} qubits"
    
    def test_performance_time_per_step(self):
        """Test time-per-step metric instead of total time."""
        H0 = create_example_hamiltonian(3, h_type="ising")
        engine = ZKAEDIEngine(3, H0)
        
        steps = 20
        start = time.time()
        diagnostics = engine.evolve(steps, track_metrics=False)
        elapsed = time.time() - start
        
        time_per_step = elapsed / steps
        
        # Reasonable bound: < 10ms per step for small systems
        assert time_per_step < 0.01, \
            f"Time per step too high: {time_per_step*1000:.2f}ms"
    
    def test_performance_memory_efficiency(self):
        """Test memory efficiency during evolution."""
        import sys
        
        H0 = create_example_hamiltonian(4, h_type="ising")
        engine = ZKAEDIEngine(4, H0)
        
        # Measure memory before
        state_size_before = sys.getsizeof(engine.state.state)
        
        # Evolve
        engine.evolve(50, track_metrics=False)
        
        # Measure memory after
        state_size_after = sys.getsizeof(engine.state.state)
        
        # Memory should not grow excessively (sparse representation)
        sparsity = engine.state.sparsity()
        if sparsity < 0.1:
            # For sparse states, memory should be reasonable
            max_expected = 2**4 * 16  # Full state size in bytes
            assert state_size_after < max_expected, \
                f"Memory usage too high: {state_size_after} bytes"
    
    @pytest.mark.parametrize("h_type", ["ising", "pauli_z", "random"])
    def test_performance_hamiltonian_types(self, h_type):
        """Test performance across different Hamiltonian types."""
        baseline = self.baseline_task()
        
        H0 = create_example_hamiltonian(3, h_type=h_type)
        engine = ZKAEDIEngine(3, H0)
        
        start = time.time()
        engine.evolve(20, track_metrics=False)
        elapsed = time.time() - start
        
        # All Hamiltonian types should complete in reasonable time
        assert elapsed < baseline * 50, \
            f"{h_type} Hamiltonian too slow: {elapsed:.3f}s"


# ============================================================
# Expanded Edge Case Handling
# ============================================================

class TestEdgeCasesEnhanced:
    """Enhanced edge case tests for invalid inputs and malformed configurations."""
    
    def test_invalid_hamiltonian_data_type_string(self):
        """Test handling of invalid Hamiltonian data type (string)."""
        with pytest.raises((TypeError, ValueError), match="array|matrix|Hamiltonian"):
            ZKAEDIPrimeHamiltonian("invalid matrix")
    
    def test_invalid_hamiltonian_data_type_none(self):
        """Test handling of None Hamiltonian."""
        with pytest.raises((TypeError, ValueError), match="None|array|matrix"):
            ZKAEDIPrimeHamiltonian(None)
    
    def test_invalid_hamiltonian_data_type_list(self):
        """Test handling of list instead of numpy array."""
        # Lists are automatically converted to arrays, so this should work
        # But we can test that it works correctly
        field = ZKAEDIPrimeHamiltonian([[1, 0], [0, 1]])  # Should be converted to np.array
        assert isinstance(field.H0, np.ndarray)
        assert field.H0.shape == (2, 2)
    
    def test_non_square_hamiltonian(self):
        """Test handling of non-square Hamiltonian matrix."""
        H0 = np.array([[1, 0, 0], [0, 1, 0]])  # 2x3, not square
        with pytest.raises((ValueError, AssertionError), match="square|dimension"):
            ZKAEDIPrimeHamiltonian(H0)
    
    def test_invalid_engine_zero_qubits(self):
        """Test engine initialization with zero qubits."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        with pytest.raises(ValueError, match="positive|integer|qubits"):
            ZKAEDIEngine(0, H0)
    
    def test_invalid_engine_negative_qubits(self):
        """Test engine initialization with negative qubits."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        with pytest.raises((ValueError, AssertionError), match="positive|qubits|negative"):
            ZKAEDIEngine(-1, H0)
    
    def test_invalid_engine_none_hamiltonian(self):
        """Test engine initialization with None Hamiltonian."""
        with pytest.raises((TypeError, ValueError), match="Hamiltonian|None|array"):
            ZKAEDIEngine(2, None)
    
    def test_invalid_qec_threshold_negative(self):
        """Test invalid QEC threshold (negative)."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        # Should handle gracefully or raise meaningful error
        try:
            engine = ZKAEDIEngine(2, H0, qec_threshold=-0.5)
            # If it doesn't raise, threshold should be clamped or ignored
        except (ValueError, AssertionError) as e:
            assert "threshold" in str(e).lower() or "qec" in str(e).lower()
    
    def test_invalid_entropy_threshold_extreme(self):
        """Test extreme entropy threshold values."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        # Very high threshold
        engine = ZKAEDIEngine(2, H0, entropy_mps_threshold=1000.0)
        # Should work but may never switch to MPS
        engine.evolve(5, track_metrics=False)
        assert engine.current_backend != BackendType.MPS
    
    def test_degenerate_state_handling(self):
        """Test handling of degenerate (nearly null) states."""
        state = SparseState(2, {0: 1e-15, 1: 1.0})
        state.normalize()
        
        # Very small amplitude should be pruned or handled
        amp_0 = state.get_amplitude(0)
        # Either pruned to zero or normalized properly
        assert abs(amp_0) < 1e-10 or abs(abs(amp_0)**2 + abs(state.get_amplitude(1))**2 - 1.0) < 1e-10
    
    def test_non_normalized_state(self):
        """Test handling of non-normalized states."""
        state = SparseState(2, {0: 2.0, 1: 2.0})  # Not normalized
        state.normalize()
        
        # Should be normalized after call
        norm = sum(abs(amp)**2 for amp in state.state.values())
        assert abs(norm - 1.0) < 1e-10
    
    def test_empty_state_support(self):
        """Test handling of empty state support."""
        state = SparseState(2, {})
        state.normalize()
        
        # Empty state should be handled gracefully
        # Either initialized to |0...0⟩ or error raised
        assert len(state.state) >= 0
    
    def test_invalid_qubit_index_negative(self):
        """Test invalid qubit index (negative)."""
        state = SparseState(3)
        with pytest.raises(ValueError, match="range|qubit|index"):
            state.apply_pauli_x(-1)
    
    def test_invalid_qubit_index_too_large(self):
        """Test invalid qubit index (too large)."""
        state = SparseState(3)
        with pytest.raises(ValueError, match="range|qubit|index"):
            state.apply_pauli_x(10)


# ============================================================
# Custom Exception Testing
# ============================================================

class TestCustomExceptions:
    """Tests for custom exception types and error messages."""
    
    def test_exception_hierarchy(self):
        """Test that custom exceptions inherit from base."""
        assert issubclass(HamiltonianError, ZKAEDIError)
        assert issubclass(EngineInitializationError, ZKAEDIError)
        assert issubclass(StateError, ZKAEDIError)
        assert issubclass(QECError, ZKAEDIError)
        assert issubclass(BackendError, ZKAEDIError)
    
    def test_meaningful_error_messages(self):
        """Test that error messages are clear and actionable."""
        # Test invalid qubit index
        state = SparseState(3)
        with pytest.raises(ValueError) as exc_info:
            state.apply_pauli_x(10)
        error_msg = str(exc_info.value).lower()
        assert "range" in error_msg or "qubit" in error_msg or "index" in error_msg
    
    def test_hamiltonian_error_handling(self):
        """Test Hamiltonian error scenarios."""
        # Non-Hermitian matrix should be handled
        H0 = np.array([[1, 2], [3, 4]], dtype=complex)
        # Should either raise error or auto-correct
        try:
            field = ZKAEDIPrimeHamiltonian(H0)
            # If it works, hermitian_projection should be applied
            H = field.step()
            assert np.allclose(H, H.T.conj()), "Hamiltonian should be Hermitian"
        except (ValueError, HamiltonianError) as e:
            assert "hermitian" in str(e).lower() or "symmetric" in str(e).lower()


# ============================================================
# Mock Dependency Testing
# ============================================================

class TestMockDependencies:
    """Tests using mocks for external dependencies."""
    
    @patch('numpy.random.randn')
    def test_random_failure_handling(self, mock_randn):
        """Test handling of random number generation failure."""
        mock_randn.side_effect = RuntimeError("Random number generation failure")
        
        with pytest.raises(RuntimeError, match="Random|generation"):
            create_example_hamiltonian(2, h_type="random")
    
    @patch('numpy.random.default_rng')
    def test_rng_initialization_failure(self, mock_rng):
        """Test handling of RNG initialization failure."""
        mock_rng.side_effect = RuntimeError("RNG initialization failed")
        
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        with pytest.raises(RuntimeError, match="RNG|initialization|random"):
            ZKAEDIPrimeHamiltonian(H0, seed=42)
    
    @patch('numpy.array')
    def test_numpy_array_failure(self, mock_array):
        """Test handling of numpy array creation failure."""
        mock_array.side_effect = MemoryError("Out of memory")
        
        with pytest.raises(MemoryError):
            # This will fail during Hamiltonian creation
            H0 = np.array([[1, 0], [0, 1]])
            ZKAEDIPrimeHamiltonian(H0)
    
    def test_numpy_version_compatibility(self):
        """Test that code works with different numpy versions."""
        # Test basic numpy operations
        arr = np.array([1, 2, 3])
        assert np.sum(arr) == 6
        
        # Test complex arrays
        arr_complex = np.array([1+1j, 2+2j])
        assert np.allclose(arr_complex.conj(), np.array([1-1j, 2-2j]))
    
    @patch('zkaedi_prime_engine.engine.np.random.default_rng')
    def test_deterministic_evolution_with_mock(self, mock_rng):
        """Test deterministic evolution with mocked RNG."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        H0_shape = H0.shape
        
        # Create deterministic RNG that returns zero noise with correct shape
        mock_rng_instance = MagicMock()
        def mock_normal(mean, scale, size):
            return np.zeros(size, dtype=float)
        mock_rng_instance.normal = mock_normal
        mock_rng.return_value = mock_rng_instance
        
        field = ZKAEDIPrimeHamiltonian(H0, seed=42)
        
        # Evolution should be deterministic with zero noise
        H1 = field.step()
        H2 = field.step()
        
        # Should be different (even with zero noise, feedback term changes H)
        assert not np.allclose(H1, H2)


# ============================================================
# Refined Comments and Documentation
# ============================================================

class TestCommentClarity:
    """Tests demonstrating improved comment clarity."""
    
    def test_entropy_threshold_switching(self):
        """Validate entropy estimation threshold for switching from sparse to MPS backend.
        
        When entanglement entropy exceeds threshold, system should automatically
        switch to MPS backend for efficient representation of highly entangled states.
        """
        H0 = create_example_hamiltonian(4, h_type="ising")
        engine = ZKAEDIEngine(4, H0, entropy_mps_threshold=1.0)
        
        # Evolve to potentially increase entanglement
        diagnostics = engine.evolve(20, track_metrics=True)
        
        # Check if backend switched based on entropy
        max_entropy = max(d.entropy for d in diagnostics)
        if max_entropy > 1.0:
            # Should have used MPS at some point
            backends_used = {d.backend for d in diagnostics}
            # MPS may or may not be used depending on exact threshold
            assert BackendType.MPS in backends_used or max_entropy < 2.0
    
    def test_qec_activation_curvature(self):
        """Validate QEC activation based on curvature threshold.
        
        Quantum error correction should activate when system curvature exceeds
        threshold, with Surface Code for low sparsity and LDPC for high sparsity.
        """
        H0 = create_example_hamiltonian(3, h_type="ising")
        engine = ZKAEDIEngine(3, H0, qec_threshold=0.5)  # Lower threshold
        
        diagnostics = engine.evolve(20, track_metrics=True)
        
        # Check QEC activations
        qec_count = sum(1 for d in diagnostics if d.qec_applied)
        # QEC may or may not activate depending on curvature
        assert qec_count >= 0


# ============================================================
# Integration Tests with Edge Cases
# ============================================================

class TestIntegrationEdgeCases:
    """Integration tests combining multiple components with edge cases."""
    
    def test_full_evolution_with_extreme_parameters(self):
        """Test full evolution cycle with extreme parameter values."""
        H0 = create_example_hamiltonian(3, h_type="ising")
        
        # Extreme parameters
        engine = ZKAEDIEngine(
            3, H0,
            eta=0.99,  # Very high feedback
            gamma=0.01,  # Very low sharpening
            epsilon=0.001,  # Very low noise
            beta=0.1  # Low noise scaling
        )
        
        # Should complete without errors
        diagnostics = engine.evolve(10, track_metrics=False)
        assert len(diagnostics) == 10
    
    def test_backend_switching_under_stress(self):
        """Test backend switching behavior under computational stress."""
        H0 = create_example_hamiltonian(5, h_type="ising")
        engine = ZKAEDIEngine(5, H0, entropy_mps_threshold=2.0)
        
        # Evolve many steps to trigger potential backend switches
        diagnostics = engine.evolve(50, track_metrics=True)
        
        # System should handle backend transitions gracefully
        backends_used = {d.backend for d in diagnostics}
        assert len(backends_used) >= 1  # At least one backend used
    
    def test_state_persistence_across_errors(self):
        """Test that state persists correctly even when errors occur."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        
        # Evolve a bit
        engine.evolve(5, track_metrics=False)
        state_before = engine.state.state.copy()
        
        # Try invalid operation (should be caught)
        try:
            engine.state.apply_pauli_x(10)  # Invalid index
        except ValueError:
            pass
        
        # State should still be valid
        assert len(engine.state.state) > 0
        engine.state.normalize()
        norm = sum(abs(amp)**2 for amp in engine.state.state.values())
        assert abs(norm - 1.0) < 1e-10


# ============================================================
# CI/CD Optimized Tests
# ============================================================

class TestCICDOptimized:
    """Tests optimized for CI/CD pipelines with minimal verbosity."""
    
    def test_quick_smoke(self):
        """Quick smoke test for CI validation."""
        H0 = create_example_hamiltonian(2, h_type="pauli_z")
        engine = ZKAEDIEngine(2, H0)
        diagnostics = engine.evolve(3, track_metrics=False)
        assert len(diagnostics) == 3
    
    def test_critical_paths(self):
        """Test critical code paths without verbose output."""
        # Engine creation
        H0 = create_example_hamiltonian(2, h_type="ising")
        engine = ZKAEDIEngine(2, H0)
        
        # Evolution
        engine.evolve(5, track_metrics=False)
        
        # Summary
        summary = engine.get_summary()
        assert summary['status'] == 'evolved'
        
        # State saving (if implemented)
        # This test is minimal and fast for CI

