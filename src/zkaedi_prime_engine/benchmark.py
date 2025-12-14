"""MVP Benchmark for ZKAEDI PRIME Engine

Simple, focused benchmark that measures:
- Evolution speed
- Memory efficiency
- Scalability
- Parameter sensitivity
- Backend selection
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

from .engine import (
    ZKAEDIEngine,
    create_example_hamiltonian,
    BackendType
)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    config_name: str
    num_qubits: int
    timesteps: int
    total_time: float
    time_per_step: float
    final_entropy: float
    final_sparsity: float
    final_backend: str
    qec_activations: int
    memory_efficient: bool
    success: bool
    error: str = ""


class ZKAEDIPrimeBenchmark:
    """MVP Benchmark for ZKAEDI PRIME Engine."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def benchmark_config(self,
                       config_name: str,
                       num_qubits: int,
                       timesteps: int,
                       h_type: str = "ising",
                       eta: float = 0.4,
                       gamma: float = 1.0,
                       epsilon: float = 0.04,
                       beta: float = 0.5,
                       seed: int = 42) -> BenchmarkResult:
        """Benchmark a single configuration.
        
        Args:
            config_name: Name for this configuration
            num_qubits: Number of qubits
            timesteps: Number of evolution steps
            h_type: Hamiltonian type ('ising', 'pauli_z', 'random')
            eta: Feedback coefficient
            gamma: Sharpening coefficient
            epsilon: Noise amplitude
            beta: Noise scaling
            seed: Random seed
            
        Returns:
            BenchmarkResult
        """
        print(f"  Benchmarking: {config_name} ({num_qubits} qubits, {timesteps} steps)...", end=" ")
        
        try:
            # Create Hamiltonian
            H0 = create_example_hamiltonian(num_qubits, h_type=h_type)
            
            # Create engine
            engine = ZKAEDIEngine(
                num_qubits=num_qubits,
                H0=H0,
                eta=eta,
                gamma=gamma,
                epsilon=epsilon,
                beta=beta,
                seed=seed
            )
            
            # Measure memory before
            import sys
            state_size_before = sys.getsizeof(engine.state.state)
            
            # Run evolution and measure time
            start_time = time.time()
            diagnostics = engine.evolve(timesteps, track_metrics=False)
            elapsed_time = time.time() - start_time
            
            # Get final state
            summary = engine.get_summary()
            final_diag = diagnostics[-1] if diagnostics else None
            
            # Count QEC activations
            qec_count = sum(1 for d in diagnostics if d.qec_applied) if diagnostics else 0
            
            # Check memory efficiency (sparse state should be small)
            state_size_after = sys.getsizeof(engine.state.state)
            memory_efficient = engine.state.sparsity() < 0.1
            
            result = BenchmarkResult(
                config_name=config_name,
                num_qubits=num_qubits,
                timesteps=timesteps,
                total_time=elapsed_time,
                time_per_step=elapsed_time / timesteps if timesteps > 0 else 0.0,
                final_entropy=final_diag.entropy if final_diag else 0.0,
                final_sparsity=final_diag.sparsity if final_diag else 0.0,
                final_backend=final_diag.backend.value if final_diag else "unknown",
                qec_activations=qec_count,
                memory_efficient=memory_efficient,
                success=True
            )
            
            print(f"OK ({elapsed_time:.3f}s)")
            return result
            
        except Exception as e:
            print(f"FAILED: {e}")
            return BenchmarkResult(
                config_name=config_name,
                num_qubits=num_qubits,
                timesteps=timesteps,
                total_time=0.0,
                time_per_step=0.0,
                final_entropy=0.0,
                final_sparsity=0.0,
                final_backend="error",
                qec_activations=0,
                memory_efficient=False,
                success=False,
                error=str(e)
            )
    
    def benchmark_scalability(self) -> List[BenchmarkResult]:
        """Benchmark scalability across different system sizes."""
        print("\n" + "=" * 70)
        print("SCALABILITY BENCHMARK")
        print("=" * 70)
        
        results = []
        
        # Test different qubit counts
        qubit_counts = [2, 3, 4, 5, 6]
        timesteps = 20
        
        for n in qubit_counts:
            result = self.benchmark_config(
                config_name=f"scalability_{n}q",
                num_qubits=n,
                timesteps=timesteps,
                h_type="pauli_z"
            )
            results.append(result)
        
        return results
    
    def benchmark_parameters(self) -> List[BenchmarkResult]:
        """Benchmark different parameter configurations."""
        print("\n" + "=" * 70)
        print("PARAMETER SENSITIVITY BENCHMARK")
        print("=" * 70)
        
        results = []
        num_qubits = 3
        timesteps = 15
        
        # Test different eta values
        for eta in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = self.benchmark_config(
                config_name=f"eta_{eta}",
                num_qubits=num_qubits,
                timesteps=timesteps,
                eta=eta,
                epsilon=0.0  # No noise for cleaner comparison
            )
            results.append(result)
        
        # Test different gamma values
        for gamma in [0.5, 1.0, 1.5, 2.0]:
            result = self.benchmark_config(
                config_name=f"gamma_{gamma}",
                num_qubits=num_qubits,
                timesteps=timesteps,
                gamma=gamma,
                epsilon=0.0
            )
            results.append(result)
        
        return results
    
    def benchmark_hamiltonian_types(self) -> List[BenchmarkResult]:
        """Benchmark different Hamiltonian types."""
        print("\n" + "=" * 70)
        print("HAMILTONIAN TYPE BENCHMARK")
        print("=" * 70)
        
        results = []
        num_qubits = 4
        timesteps = 20
        
        for h_type in ["ising", "pauli_z", "random"]:
            result = self.benchmark_config(
                config_name=f"hamiltonian_{h_type}",
                num_qubits=num_qubits,
                timesteps=timesteps,
                h_type=h_type
            )
            results.append(result)
        
        return results
    
    def benchmark_backend_selection(self) -> List[BenchmarkResult]:
        """Benchmark backend selection behavior."""
        print("\n" + "=" * 70)
        print("BACKEND SELECTION BENCHMARK")
        print("=" * 70)
        
        results = []
        
        # Test with different entropy thresholds to trigger different backends
        configs = [
            ("low_entropy", 3, 0.1, 10),  # Should use sparse
            ("medium_entropy", 4, 1.0, 15),  # May use dense
            ("high_entropy", 5, 2.0, 20),  # May use MPS
        ]
        
        for name, n, entropy_thresh, steps in configs:
            result = self.benchmark_config(
                config_name=f"backend_{name}",
                num_qubits=n,
                timesteps=steps,
                h_type="ising"
            )
            results.append(result)
        
        return results
    
    def run_all(self) -> Dict[str, Any]:
        """Run all benchmark suites."""
        print("=" * 70)
        print("ZKAEDI PRIME ENGINE - MVP BENCHMARK")
        print("=" * 70)
        print()
        
        all_results = []
        
        # Run all benchmark suites
        all_results.extend(self.benchmark_scalability())
        all_results.extend(self.benchmark_parameters())
        all_results.extend(self.benchmark_hamiltonian_types())
        all_results.extend(self.benchmark_backend_selection())
        
        self.results = all_results
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary(summary)
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        if not successful:
            return {"error": "No successful benchmarks"}
        
        # Calculate statistics
        times = [r.time_per_step for r in successful]
        sparsities = [r.final_sparsity for r in successful]
        entropies = [r.final_entropy for r in successful]
        
        # Backend distribution
        backends = defaultdict(int)
        for r in successful:
            backends[r.final_backend] += 1
        
        summary = {
            "total_benchmarks": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0.0,
            "performance": {
                "avg_time_per_step_ms": np.mean(times) * 1000 if times else 0.0,
                "min_time_per_step_ms": np.min(times) * 1000 if times else 0.0,
                "max_time_per_step_ms": np.max(times) * 1000 if times else 0.0,
            },
            "sparsity": {
                "avg": np.mean(sparsities) if sparsities else 0.0,
                "min": np.min(sparsities) if sparsities else 0.0,
                "max": np.max(sparsities) if sparsities else 0.0,
            },
            "entropy": {
                "avg": np.mean(entropies) if entropies else 0.0,
                "min": np.min(entropies) if entropies else 0.0,
                "max": np.max(entropies) if entropies else 0.0,
            },
            "backend_distribution": dict(backends),
            "memory_efficient_count": sum(1 for r in successful if r.memory_efficient),
            "total_qec_activations": sum(r.qec_activations for r in successful),
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal Benchmarks: {summary['total_benchmarks']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']*100:.1f}%")
        
        print("\nPerformance:")
        perf = summary['performance']
        print(f"  Avg Time/Step: {perf['avg_time_per_step_ms']:.3f} ms")
        print(f"  Min Time/Step: {perf['min_time_per_step_ms']:.3f} ms")
        print(f"  Max Time/Step: {perf['max_time_per_step_ms']:.3f} ms")
        
        print("\nSparsity:")
        sp = summary['sparsity']
        print(f"  Average: {sp['avg']:.4f}")
        print(f"  Range: [{sp['min']:.4f}, {sp['max']:.4f}]")
        
        print("\nEntropy:")
        ent = summary['entropy']
        print(f"  Average: {ent['avg']:.3f}")
        print(f"  Range: [{ent['min']:.3f}, {ent['max']:.3f}]")
        
        print("\nBackend Distribution:")
        for backend, count in summary['backend_distribution'].items():
            print(f"  {backend}: {count}")
        
        print(f"\nMemory Efficient: {summary['memory_efficient_count']}/{summary['successful']}")
        print(f"Total QEC Activations: {summary['total_qec_activations']}")
        
        print("\n" + "=" * 70)
    
    def save_results(self):
        """Save benchmark results to JSON."""
        output_file = self.output_dir / "zkaedi_prime_engine_benchmark.json"
        
        data = {
            "benchmark_version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in self.results],
            "summary": self.generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
    
    def print_detailed_results(self):
        """Print detailed results table."""
        print("\n" + "=" * 70)
        print("DETAILED RESULTS")
        print("=" * 70)
        print(f"{'Config':<25} {'Qubits':<8} {'Steps':<8} {'Time/Step (ms)':<15} {'Sparsity':<10} {'Backend':<10}")
        print("-" * 70)
        
        for r in self.results:
            if r.success:
                print(f"{r.config_name:<25} {r.num_qubits:<8} {r.timesteps:<8} "
                      f"{r.time_per_step*1000:<15.3f} {r.final_sparsity:<10.4f} {r.final_backend:<10}")
            else:
                print(f"{r.config_name:<25} {'FAILED':<8} {'':<8} {'':<15} {'':<10} {r.error[:20]:<10}")


def main():
    """Run MVP benchmark."""
    benchmark = ZKAEDIPrimeBenchmark()
    
    # Run all benchmarks
    summary = benchmark.run_all()
    
    # Print detailed results
    benchmark.print_detailed_results()
    
    print("\n[OK] Benchmark complete!")
    return summary


if __name__ == "__main__":
    main()

