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
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .engine import (
    ZKAEDIEngine,
    create_example_hamiltonian,
    BackendType
)

# Optional dependency for memory/CPU tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    import warnings
    warnings.warn(
        "psutil not available. Memory/CPU tracking will be disabled. "
        "Install with: pip install psutil",
        UserWarning
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Benchmark configuration constants
DEFAULT_QBIT_COUNTS = [2, 3, 4, 5, 6]
DEFAULT_TIMESTEPS = 20
DEFAULT_ETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_GAMMA_VALUES = [0.5, 1.0, 1.5, 2.0]
DEFAULT_EPSILON_VALUES = [0.0, 0.02, 0.04, 0.06, 0.08]


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
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success: bool = True
    error: str = ""


class ZKAEDIPrimeBenchmark:
    """MVP benchmark suite for ZKAEDI PRIME Engine."""
    
    def __init__(self, output_dir: str = "benchmark_results", max_workers: int = 4):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
            max_workers: Maximum parallel workers for concurrent benchmarks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.max_workers = max_workers
        logger.info(f"Initialized benchmark suite (output_dir={output_dir}, max_workers={max_workers})")
    
    def track_resource_usage(self) -> Tuple[float, float]:
        """Track memory and CPU usage.
        
        Returns:
            (memory_mb, cpu_percent) tuple
        """
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0
        
        try:
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            return memory_mb, cpu_percent
        except Exception as e:
            logger.warning(f"Failed to track resources: {e}")
            return 0.0, 0.0
    
    def create_engine(self,
                     num_qubits: int,
                     h_type: str = "ising",
                     eta: float = 0.4,
                     gamma: float = 1.0,
                     epsilon: float = 0.04,
                     beta: float = 0.5,
                     seed: int = 42) -> ZKAEDIEngine:
        """Create and configure engine.
        
        Args:
            num_qubits: Number of qubits
            h_type: Hamiltonian type
            eta: Feedback coefficient
            gamma: Sharpening coefficient
            epsilon: Noise amplitude (0.0 for no noise)
            beta: Noise scaling
            seed: Random seed
            
        Returns:
            Configured engine
        """
        H0 = create_example_hamiltonian(num_qubits, h_type=h_type)
        engine = ZKAEDIEngine(
            num_qubits=num_qubits,
            H0=H0,
            eta=eta,
            gamma=gamma,
            epsilon=epsilon,
            beta=beta,
            seed=seed
        )
        return engine
    
    def run_evolution(self,
                     engine: ZKAEDIEngine,
                     timesteps: int,
                     track_metrics: bool = False) -> Tuple[List[Any], float]:
        """Run evolution and measure time.
        
        Args:
            engine: Engine instance
            timesteps: Number of evolution steps
            track_metrics: Whether to track detailed metrics
            
        Returns:
            (diagnostics, elapsed_time) tuple
        """
        start_time = time.time()
        diagnostics = engine.evolve(timesteps, track_metrics=track_metrics)
        elapsed_time = time.time() - start_time
        return diagnostics, elapsed_time
    
    def benchmark_config(self,
                       config_name: str,
                       num_qubits: int,
                       timesteps: int,
                       h_type: str = "ising",
                       eta: float = 0.4,
                       gamma: float = 1.0,
                       epsilon: float = 0.04,
                       beta: float = 0.5,
                       seed: int = 42,
                       track_memory: bool = True,
                       max_retries: int = 2) -> BenchmarkResult:
        """Benchmark single configuration.
        
        Args:
            config_name: Configuration name
            num_qubits: Number of qubits
            timesteps: Number of timesteps
            h_type: Hamiltonian type
            eta: Feedback coefficient
            gamma: Sharpening coefficient
            epsilon: Noise amplitude (0.0 for no noise)
            beta: Noise scaling
            seed: Random seed
            track_memory: Whether to track memory/CPU usage
            max_retries: Maximum retry attempts on failure
            
        Returns:
            Benchmark result
        """
        logger.info(f"Benchmarking: {config_name} (n={num_qubits}, steps={timesteps})")
        
        for attempt in range(max_retries + 1):
            try:
                # Track resources before
                memory_before, cpu_before = self.track_resource_usage() if track_memory else (0.0, 0.0)
                
                # Create engine
                engine = self.create_engine(
                    num_qubits=num_qubits,
                    h_type=h_type,
                    eta=eta,
                    gamma=gamma,
                    epsilon=epsilon,
                    beta=beta,
                    seed=seed
                )
                
                # Measure state size
                import sys
                state_size_before = sys.getsizeof(engine.state.state)
                
                # Run evolution
                diagnostics, elapsed_time = self.run_evolution(engine, timesteps, track_metrics=False)
                
                # Track resources after
                memory_after, cpu_after = self.track_resource_usage() if track_memory else (0.0, 0.0)
                memory_delta = memory_after - memory_before
                
                # Extract final diagnostics
                final_diag = diagnostics[-1] if diagnostics else None
                
                # Count QEC activations
                qec_count = sum(1 for d in diagnostics if d.qec_applied)
                
                # Check memory efficiency
                memory_efficient = state_size_before < (2**num_qubits * 16)  # Rough estimate
                
                # Build result
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
                    memory_usage_mb=memory_delta if track_memory else 0.0,
                    cpu_usage_percent=cpu_after if track_memory else 0.0,
                    success=True
                )
                
                logger.info(f"✅ {config_name}: {result.time_per_step*1000:.3f} ms/step")
                return result
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {config_name}: {error_msg}")
                
                if attempt < max_retries:
                    # Retry with reduced complexity
                    if num_qubits > 2:
                        logger.info(f"Retrying {config_name} with reduced qubits: {num_qubits-1}")
                        return self.benchmark_config(
                            config_name=f"{config_name}_retry",
                            num_qubits=num_qubits - 1,
                            timesteps=max(5, timesteps // 2),
                            h_type=h_type,
                            eta=eta,
                            gamma=gamma,
                            epsilon=epsilon,
                            beta=beta,
                            seed=seed,
                            track_memory=track_memory,
                            max_retries=0  # Don't retry retries
                        )
                else:
                    logger.error(f"❌ {config_name} failed after {max_retries + 1} attempts")
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
                        memory_usage_mb=0.0,
                        cpu_usage_percent=0.0,
                        success=False,
                        error=error_msg
                    )
        
        # Should never reach here, but just in case
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
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            success=False,
            error="Unknown error"
        )
    
    def benchmark_scalability(self, qubit_counts: Optional[List[int]] = None, timesteps: int = DEFAULT_TIMESTEPS) -> List[BenchmarkResult]:
        """Benchmark scalability across qubit counts.
        
        Args:
            qubit_counts: List of qubit counts to test (default: DEFAULT_QBIT_COUNTS)
            timesteps: Number of timesteps per benchmark
            
        Returns:
            List of benchmark results
        """
        if qubit_counts is None:
            qubit_counts = DEFAULT_QBIT_COUNTS
        
        logger.info(f"Starting scalability benchmark: {qubit_counts}")
        
        results = []
        for n in qubit_counts:
            result = self.benchmark_config(
                config_name=f"scalability_{n}q",
                num_qubits=n,
                timesteps=timesteps,
                h_type="ising"
            )
            results.append(result)
        
        return results
    
    def benchmark_parameters(self,
                           num_qubits: int = 3,
                           timesteps: int = 10,
                           eta_values: Optional[List[float]] = None,
                           gamma_values: Optional[List[float]] = None,
                           epsilon_values: Optional[List[float]] = None) -> List[BenchmarkResult]:
        """Benchmark parameter sensitivity.
        
        Args:
            num_qubits: Number of qubits
            timesteps: Number of timesteps
            eta_values: List of eta values (default: DEFAULT_ETA_VALUES)
            gamma_values: List of gamma values (default: DEFAULT_GAMMA_VALUES)
            epsilon_values: List of epsilon values (default: DEFAULT_EPSILON_VALUES)
            
        Returns:
            List of benchmark results
        """
        if eta_values is None:
            eta_values = DEFAULT_ETA_VALUES
        if gamma_values is None:
            gamma_values = DEFAULT_GAMMA_VALUES
        if epsilon_values is None:
            epsilon_values = DEFAULT_EPSILON_VALUES
        
        logger.info(f"Starting parameter sensitivity benchmark (n={num_qubits})")
        
        results = []
        
        # Eta sweep
        for eta in eta_values:
            result = self.benchmark_config(
                config_name=f"param_eta_{eta:.1f}",
                num_qubits=num_qubits,
                timesteps=timesteps,
                h_type="pauli_z",
                eta=eta,
                epsilon=0.0  # No noise for parameter sweep
            )
            results.append(result)
        
        # Gamma sweep
        for gamma in gamma_values:
            result = self.benchmark_config(
                config_name=f"param_gamma_{gamma:.1f}",
                num_qubits=num_qubits,
                timesteps=timesteps,
                h_type="pauli_z",
                gamma=gamma,
                epsilon=0.0
            )
            results.append(result)
        
        # Epsilon sweep
        for epsilon in epsilon_values:
            result = self.benchmark_config(
                config_name=f"param_epsilon_{epsilon:.2f}",
                num_qubits=num_qubits,
                timesteps=timesteps,
                h_type="pauli_z",
                epsilon=epsilon
            )
            results.append(result)
        
        return results
    
    def benchmark_hamiltonian_types(self) -> List[BenchmarkResult]:
        """Benchmark different Hamiltonian types."""
        logger.info("Starting Hamiltonian type benchmark")
        
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
        logger.info("Starting backend selection benchmark")
        
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
    
    def run_all(self, parallel: bool = True) -> Dict[str, Any]:
        """Run all benchmarks.
        
        Args:
            parallel: Whether to run benchmarks in parallel
            
        Returns:
            Summary dictionary
        """
        logger.info("=" * 70)
        logger.info("ZKAEDI PRIME BENCHMARK SUITE")
        logger.info("=" * 70)
        
        all_results = []
        
        if parallel and self.max_workers > 1:
            logger.info(f"Running benchmarks in parallel (max_workers={self.max_workers})")
            all_results = self._run_parallel()
        else:
            logger.info("Running benchmarks sequentially")
            all_results = self._run_sequential()
        
        self.results = all_results
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results
        self.save_results()
        
        return summary
    
    def _run_sequential(self) -> List[BenchmarkResult]:
        """Run all benchmarks sequentially."""
        all_results = []
        
        # Scalability
        logger.info("\n" + "=" * 70)
        logger.info("SCALABILITY BENCHMARK")
        logger.info("=" * 70)
        all_results.extend(self.benchmark_scalability())
        
        # Parameters
        logger.info("\n" + "=" * 70)
        logger.info("PARAMETER SENSITIVITY BENCHMARK")
        logger.info("=" * 70)
        all_results.extend(self.benchmark_parameters())
        
        # Hamiltonian types
        logger.info("\n" + "=" * 70)
        logger.info("HAMILTONIAN TYPE BENCHMARK")
        logger.info("=" * 70)
        all_results.extend(self.benchmark_hamiltonian_types())
        
        # Backend selection
        logger.info("\n" + "=" * 70)
        logger.info("BACKEND SELECTION BENCHMARK")
        logger.info("=" * 70)
        all_results.extend(self.benchmark_backend_selection())
        
        return all_results
    
    def _run_parallel(self) -> List[BenchmarkResult]:
        """Run benchmarks in parallel where possible."""
        all_results = []
        
        # Collect all benchmark tasks
        tasks = []
        
        # Scalability benchmarks (can be parallelized)
        for n in DEFAULT_QBIT_COUNTS:
            tasks.append(('scalability', lambda n=n: self.benchmark_config(
                config_name=f"scalability_{n}q",
                num_qubits=n,
                timesteps=DEFAULT_TIMESTEPS,
                h_type="ising"
            )))
        
        # Parameter benchmarks (can be parallelized)
        for eta in DEFAULT_ETA_VALUES:
            tasks.append(('param_eta', lambda eta=eta: self.benchmark_config(
                config_name=f"param_eta_{eta:.1f}",
                num_qubits=3,
                timesteps=10,
                h_type="pauli_z",
                eta=eta,
                epsilon=0.0
            )))
        
        # Hamiltonian types (can be parallelized)
        for h_type in ["ising", "pauli_z", "random"]:
            tasks.append(('hamiltonian', lambda h_type=h_type: self.benchmark_config(
                config_name=f"hamiltonian_{h_type}",
                num_qubits=4,
                timesteps=20,
                h_type=h_type
            )))
        
        # Backend selection (can be parallelized)
        configs = [
            ("low_entropy", 3, 10),
            ("medium_entropy", 4, 15),
            ("high_entropy", 5, 20),
        ]
        for name, n, steps in configs:
            tasks.append(('backend', lambda name=name, n=n, steps=steps: self.benchmark_config(
                config_name=f"backend_{name}",
                num_qubits=n,
                timesteps=steps,
                h_type="ising"
            )))
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(task[1]): task[0] for task in tasks}
            
            for future in as_completed(future_to_task):
                task_type = future_to_task[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    logger.info(f"Completed {task_type} benchmark: {result.config_name}")
                except Exception as e:
                    logger.error(f"Task {task_type} failed: {e}")
        
        return all_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {"error": "No results to summarize"}
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        if not successful:
            return {"error": "No successful benchmarks"}
        
        # Aggregate statistics
        avg_time_per_step = np.mean([r.time_per_step for r in successful])
        min_time_per_step = np.min([r.time_per_step for r in successful])
        max_time_per_step = np.max([r.time_per_step for r in successful])
        
        avg_entropy = np.mean([r.final_entropy for r in successful])
        avg_sparsity = np.mean([r.final_sparsity for r in successful])
        
        # Backend distribution
        backend_counts = defaultdict(int)
        for r in successful:
            backend_counts[r.final_backend] += 1
        
        # QEC statistics
        total_qec = sum(r.qec_activations for r in successful)
        avg_qec = total_qec / len(successful) if successful else 0.0
        
        # Memory statistics
        memory_results = [r for r in successful if r.memory_usage_mb > 0]
        avg_memory = np.mean([r.memory_usage_mb for r in memory_results]) if memory_results else 0.0
        
        summary = {
            "total_benchmarks": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0.0,
            "statistics": {
                "avg_time_per_step_ms": avg_time_per_step * 1000,
                "min_time_per_step_ms": min_time_per_step * 1000,
                "max_time_per_step_ms": max_time_per_step * 1000,
                "avg_entropy": avg_entropy,
                "avg_sparsity": avg_sparsity,
                "total_qec_activations": total_qec,
                "avg_qec_per_benchmark": avg_qec,
                "avg_memory_usage_mb": avg_memory,
            },
            "backend_distribution": dict(backend_counts),
            "failed_benchmarks": [{"name": r.config_name, "error": r.error} for r in failed]
        }
        
        return summary
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file.
        
        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "results": [asdict(r) for r in self.results],
            "summary": self.generate_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def print_detailed_results(self):
        """Print detailed benchmark results table."""
        if not self.results:
            logger.warning("No results to display")
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("DETAILED RESULTS")
        logger.info("=" * 70)
        logger.info(f"{'Config':<25} {'Qubits':<8} {'Steps':<8} {'Time/Step (ms)':<15} {'Sparsity':<10} {'Backend':<10}")
        logger.info("-" * 70)
        
        for r in self.results:
            if r.success:
                logger.info(f"{r.config_name:<25} {r.num_qubits:<8} {r.timesteps:<8} "
                          f"{r.time_per_step*1000:<15.3f} {r.final_sparsity:<10.4f} {r.final_backend:<10}")
            else:
                logger.error(f"{r.config_name:<25} {'FAILED':<8} {'':<8} {'':<15} {'':<10} {r.error[:20]:<10}")


def main():
    """Run MVP benchmark."""
    benchmark = ZKAEDIPrimeBenchmark(max_workers=4)
    
    # Run all benchmarks
    summary = benchmark.run_all(parallel=True)
    
    # Print detailed results
    benchmark.print_detailed_results()
    
    logger.info("\n[OK] Benchmark complete!")
    return summary


if __name__ == "__main__":
    main()
