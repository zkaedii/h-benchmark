"""Unit tests for benchmark suite."""

import pytest
import numpy as np
from pathlib import Path
from zkaedi_prime_engine.benchmark import (
    ZKAEDIPrimeBenchmark,
    BenchmarkResult,
    DEFAULT_QBIT_COUNTS,
    DEFAULT_TIMESTEPS
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            config_name="test_config",
            num_qubits=2,
            timesteps=10,
            total_time=0.1,
            time_per_step=0.01,
            final_entropy=0.5,
            final_sparsity=0.25,
            final_backend="dense",
            qec_activations=2,
            memory_efficient=True
        )
        
        assert result.config_name == "test_config"
        assert result.num_qubits == 2
        assert result.success is True
        assert result.time_per_step == 0.01
    
    def test_benchmark_result_failure(self):
        """Test failed benchmark result."""
        result = BenchmarkResult(
            config_name="failed_config",
            num_qubits=2,
            timesteps=10,
            total_time=0.0,
            time_per_step=0.0,
            final_entropy=0.0,
            final_sparsity=0.0,
            final_backend="error",
            qec_activations=0,
            memory_efficient=False,
            success=False,
            error="Test error"
        )
        
        assert result.success is False
        assert result.error == "Test error"


class TestZKAEDIPrimeBenchmark:
    """Tests for ZKAEDIPrimeBenchmark class."""
    
    def test_initialization(self, tmp_path):
        """Test benchmark initialization."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        
        assert benchmark.output_dir == tmp_path
        assert len(benchmark.results) == 0
        assert benchmark.max_workers >= 1
    
    def test_create_engine(self, tmp_path):
        """Test engine creation."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        engine = benchmark.create_engine(num_qubits=2, h_type="pauli_z")
        
        assert engine.n == 2
        assert engine.field is not None
    
    def test_run_evolution(self, tmp_path):
        """Test evolution execution."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        engine = benchmark.create_engine(num_qubits=2)
        
        diagnostics, elapsed = benchmark.run_evolution(engine, timesteps=5)
        
        assert len(diagnostics) == 5
        assert elapsed > 0
    
    def test_benchmark_config_success(self, tmp_path):
        """Test successful benchmark configuration."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        result = benchmark.benchmark_config(
            config_name="test_success",
            num_qubits=2,
            timesteps=5,
            track_memory=False
        )
        
        assert result.success is True
        assert result.config_name == "test_success"
        assert result.num_qubits == 2
        assert result.timesteps == 5
        assert result.time_per_step > 0
    
    def test_benchmark_config_failure_handling(self, tmp_path):
        """Test benchmark handles failures gracefully."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        # This should handle errors gracefully
        result = benchmark.benchmark_config(
            config_name="test_invalid",
            num_qubits=100,  # Too large, will fail
            timesteps=1,
            max_retries=0
        )
        
        # Should return a result, even if failed
        assert isinstance(result, BenchmarkResult)
    
    def test_generate_summary(self, tmp_path):
        """Test summary generation."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        
        # Add some test results
        benchmark.results = [
            BenchmarkResult(
                config_name="test1",
                num_qubits=2,
                timesteps=10,
                total_time=0.1,
                time_per_step=0.01,
                final_entropy=0.5,
                final_sparsity=0.25,
                final_backend="dense",
                qec_activations=1,
                memory_efficient=True,
                success=True
            ),
            BenchmarkResult(
                config_name="test2",
                num_qubits=3,
                timesteps=10,
                total_time=0.2,
                time_per_step=0.02,
                final_entropy=0.6,
                final_sparsity=0.15,
                final_backend="sparse",
                qec_activations=2,
                memory_efficient=True,
                success=True
            )
        ]
        
        summary = benchmark.generate_summary()
        
        assert summary["total_benchmarks"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert "statistics" in summary
        assert "backend_distribution" in summary
        assert summary["statistics"]["avg_time_per_step_ms"] > 0
    
    def test_generate_summary_with_failures(self, tmp_path):
        """Test summary with failed benchmarks."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        
        benchmark.results = [
            BenchmarkResult(
                config_name="success",
                num_qubits=2,
                timesteps=10,
                total_time=0.1,
                time_per_step=0.01,
                final_entropy=0.5,
                final_sparsity=0.25,
                final_backend="dense",
                qec_activations=1,
                memory_efficient=True,
                success=True
            ),
            BenchmarkResult(
                config_name="failed",
                num_qubits=2,
                timesteps=10,
                total_time=0.0,
                time_per_step=0.0,
                final_entropy=0.0,
                final_sparsity=0.0,
                final_backend="error",
                qec_activations=0,
                memory_efficient=False,
                success=False,
                error="Test error"
            )
        ]
        
        summary = benchmark.generate_summary()
        
        assert summary["total_benchmarks"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert len(summary["failed_benchmarks"]) == 1
    
    def test_save_results(self, tmp_path):
        """Test saving results to file."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        
        benchmark.results = [
            BenchmarkResult(
                config_name="test",
                num_qubits=2,
                timesteps=10,
                total_time=0.1,
                time_per_step=0.01,
                final_entropy=0.5,
                final_sparsity=0.25,
                final_backend="dense",
                qec_activations=1,
                memory_efficient=True,
                success=True
            )
        ]
        
        filename = "test_results.json"
        benchmark.save_results(filename=filename)
        
        filepath = tmp_path / filename
        assert filepath.exists()
        
        # Verify file contents
        import json
        with open(filepath) as f:
            data = json.load(f)
        
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 1


class TestBenchmarkSuites:
    """Tests for individual benchmark suites."""
    
    def test_benchmark_scalability(self, tmp_path):
        """Test scalability benchmark."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        results = benchmark.benchmark_scalability(qubit_counts=[2, 3], timesteps=5)
        
        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.num_qubits in [2, 3] for r in results)
    
    def test_benchmark_parameters(self, tmp_path):
        """Test parameter sensitivity benchmark."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        results = benchmark.benchmark_parameters(
            num_qubits=2,
            timesteps=5,
            eta_values=[0.1, 0.5],
            gamma_values=[1.0],
            epsilon_values=[0.0]
        )
        
        assert len(results) > 0
        # Should have results for eta, gamma, epsilon sweeps
    
    def test_benchmark_hamiltonian_types(self, tmp_path):
        """Test Hamiltonian type benchmark."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        results = benchmark.benchmark_hamiltonian_types()
        
        assert len(results) == 3
        assert all(r.success for r in results)
        h_types = {r.config_name.split("_")[-1] for r in results}
        assert "ising" in h_types or any("ising" in r.config_name for r in results)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_results_summary(self, tmp_path):
        """Test summary with no results."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        summary = benchmark.generate_summary()
        
        assert "error" in summary
    
    def test_minimal_timesteps(self, tmp_path):
        """Test with minimal timesteps."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        result = benchmark.benchmark_config(
            config_name="minimal",
            num_qubits=2,
            timesteps=1,
            track_memory=False
        )
        
        assert result.success is True
        assert result.timesteps == 1
    
    def test_small_qubit_count(self, tmp_path):
        """Test with minimal qubit count."""
        benchmark = ZKAEDIPrimeBenchmark(output_dir=str(tmp_path))
        result = benchmark.benchmark_config(
            config_name="small",
            num_qubits=1,
            timesteps=5,
            track_memory=False
        )
        
        # Should handle gracefully (may need adjustment for 1 qubit)
        assert isinstance(result, BenchmarkResult)


@pytest.fixture
def tmp_path(tmp_path):
    """Fixture for temporary directory."""
    return tmp_path

