"""
Parameter Sweep Example - Production-Grade Implementation
==========================================================
Perform comprehensive sensitivity analysis on the ZKAEDI PRIME Engine by varying
key parameters (eta, epsilon, gamma, beta) and Hamiltonian types.

Features:
- High-resolution parameter sweeps
- Parallel execution for speedup
- Multiple noise models
- Various Hamiltonian types
- Results saved to CSV/JSON
- Visualization support
- Modular, testable design

Usage:
    python parameter_sweep.py [--eta-steps 50] [--parallel] [--visualize]
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
import numpy as np

from zkaedi_prime_engine import ZKAEDIEngine, create_example_hamiltonian


@dataclass
class SweepResult:
    """Single parameter sweep result."""
    eta: float
    epsilon: float
    gamma: float
    beta: float
    h_type: str
    num_qubits: int
    timesteps: int
    final_entropy: float
    final_sparsity: float
    final_backend: str
    qec_activations: int
    total_time: float
    time_per_step: float


def generate_diagnostics(
    num_qubits: int,
    H0: np.ndarray,
    eta: float,
    epsilon: float = 0.0,
    gamma: float = 1.0,
    beta: float = 0.5,
    timesteps: int = 10,
    seed: int = 42
) -> Tuple[float, float, str, int, float]:
    """Generate diagnostics for a single parameter configuration.
    
    Args:
        num_qubits: Number of qubits
        H0: Base Hamiltonian
        eta: Feedback coefficient
        epsilon: Noise amplitude
        gamma: Sharpening coefficient
        beta: Noise scaling
        timesteps: Number of evolution steps
        seed: Random seed
        
    Returns:
        (final_entropy, final_sparsity, final_backend, qec_count, elapsed_time)
    """
    start_time = time.time()
    
    engine = ZKAEDIEngine(
        num_qubits=num_qubits,
        H0=H0,
        eta=eta,
        gamma=gamma,
        epsilon=epsilon,
        beta=beta,
        seed=seed
    )
    
    diagnostics = engine.evolve(timesteps, track_metrics=False)
    elapsed_time = time.time() - start_time
    
    final = diagnostics[-1] if diagnostics else None
    if final is None:
        return 0.0, 0.0, "unknown", 0, elapsed_time
    
    qec_count = sum(1 for d in diagnostics if d.qec_applied)
    
    return (
        final.entropy,
        final.sparsity,
        final.backend.value,
        qec_count,
        elapsed_time
    )


def evaluate_configuration(
    config: Dict
) -> SweepResult:
    """Evaluate a single parameter configuration.
    
    Args:
        config: Configuration dictionary with all parameters
        
    Returns:
        SweepResult with diagnostics
    """
    num_qubits = config['num_qubits']
    h_type = config['h_type']
    
    # Create Hamiltonian
    H0 = create_example_hamiltonian(num_qubits, h_type=h_type)
    
    # Generate diagnostics
    entropy, sparsity, backend, qec_count, elapsed = generate_diagnostics(
        num_qubits=num_qubits,
        H0=H0,
        eta=config['eta'],
        epsilon=config['epsilon'],
        gamma=config['gamma'],
        beta=config['beta'],
        timesteps=config['timesteps'],
        seed=config.get('seed', 42)
    )
    
    return SweepResult(
        eta=config['eta'],
        epsilon=config['epsilon'],
        gamma=config['gamma'],
        beta=config['beta'],
        h_type=h_type,
        num_qubits=num_qubits,
        timesteps=config['timesteps'],
        final_entropy=entropy,
        final_sparsity=sparsity,
        final_backend=backend,
        qec_activations=qec_count,
        total_time=elapsed,
        time_per_step=elapsed / config['timesteps'] if config['timesteps'] > 0 else 0.0
    )


def generate_configurations(
    num_qubits: int = 3,
    eta_range: Tuple[float, float] = (0.1, 0.9),
    eta_steps: int = 50,
    epsilon_values: Optional[List[float]] = None,
    gamma_values: Optional[List[float]] = None,
    beta_values: Optional[List[float]] = None,
    h_types: Optional[List[str]] = None,
    timesteps: int = 10
) -> List[Dict]:
    """Generate all parameter configurations to sweep.
    
    Args:
        num_qubits: Number of qubits
        eta_range: (min, max) range for eta
        eta_steps: Number of steps in eta sweep
        epsilon_values: List of epsilon values (default: [0.0, 0.05, 0.1])
        gamma_values: List of gamma values (default: [1.0])
        beta_values: List of beta values (default: [0.5])
        h_types: List of Hamiltonian types (default: ["pauli_z"])
        timesteps: Number of evolution steps
        
    Returns:
        List of configuration dictionaries
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.1]
    if gamma_values is None:
        gamma_values = [1.0]
    if beta_values is None:
        beta_values = [0.5]
    if h_types is None:
        h_types = ["pauli_z"]
    
    configurations = []
    etas = np.linspace(eta_range[0], eta_range[1], eta_steps)
    
    for h_type in h_types:
        for eta in etas:
            for epsilon in epsilon_values:
                for gamma in gamma_values:
                    for beta in beta_values:
                        configurations.append({
                            'num_qubits': num_qubits,
                            'h_type': h_type,
                            'eta': float(eta),
                            'epsilon': epsilon,
                            'gamma': gamma,
                            'beta': beta,
                            'timesteps': timesteps,
                            'seed': 42
                        })
    
    return configurations


def run_sequential(configurations: List[Dict]) -> List[SweepResult]:
    """Run parameter sweep sequentially.
    
    Args:
        configurations: List of parameter configurations
        
    Returns:
        List of sweep results
    """
    results = []
    total = len(configurations)
    
    print(f"Running {total} configurations sequentially...")
    for i, config in enumerate(configurations, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{total} ({100*i/total:.1f}%)")
        
        try:
            result = evaluate_configuration(config)
            results.append(result)
        except Exception as e:
            print(f"  Error in configuration {i}: {e}")
            continue
    
    return results


def run_parallel(configurations: List[Dict], num_workers: Optional[int] = None) -> List[SweepResult]:
    """Run parameter sweep in parallel.
    
    Args:
        configurations: List of parameter configurations
        num_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        List of sweep results
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Running {len(configurations)} configurations in parallel ({num_workers} workers)...")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(evaluate_configuration, configurations)
    
    return results


def display_results(results: List[SweepResult]):
    """Display results in a formatted table.
    
    Args:
        results: List of sweep results
    """
    print("\n" + "=" * 100)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 100)
    print(f"{'Eta':<10} {'Epsilon':<10} {'Gamma':<10} {'H-Type':<12} {'Entropy':<12} {'Sparsity':<12} {'Backend':<10} {'QEC':<6} {'Time/Step (ms)':<15}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.eta:<10.3f} {r.epsilon:<10.3f} {r.gamma:<10.3f} {r.h_type:<12} "
              f"{r.final_entropy:<12.4f} {r.final_sparsity:<12.4f} {r.final_backend:<10} "
              f"{r.qec_activations:<6} {r.time_per_step*1000:<15.3f}")
    
    print("=" * 100)
    print(f"Total configurations: {len(results)}")
    
    # Summary statistics
    if results:
        avg_entropy = np.mean([r.final_entropy for r in results])
        avg_sparsity = np.mean([r.final_sparsity for r in results])
        avg_time = np.mean([r.time_per_step for r in results])
        
        print(f"\nSummary Statistics:")
        print(f"  Average Entropy: {avg_entropy:.4f}")
        print(f"  Average Sparsity: {avg_sparsity:.4f}")
        print(f"  Average Time/Step: {avg_time*1000:.3f} ms")


def save_results_csv(results: List[SweepResult], filepath: str):
    """Save results to CSV file.
    
    Args:
        results: List of sweep results
        filepath: Output file path
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Eta', 'Epsilon', 'Gamma', 'Beta', 'H_Type', 'Num_Qubits',
            'Timesteps', 'Final_Entropy', 'Final_Sparsity', 'Final_Backend',
            'QEC_Activations', 'Total_Time', 'Time_Per_Step'
        ])
        
        for r in results:
            writer.writerow([
                r.eta, r.epsilon, r.gamma, r.beta, r.h_type, r.num_qubits,
                r.timesteps, r.final_entropy, r.final_sparsity, r.final_backend,
                r.qec_activations, r.total_time, r.time_per_step
            ])
    
    print(f"\nResults saved to CSV: {filepath}")


def save_results_json(results: List[SweepResult], filepath: str):
    """Save results to JSON file.
    
    Args:
        results: List of sweep results
        filepath: Output file path
    """
    data = {
        'metadata': {
            'total_configurations': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': [asdict(r) for r in results]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to JSON: {filepath}")


def visualize_results(results: List[SweepResult], output_dir: str = "outputs"):
    """Visualize parameter sweep results.
    
    Args:
        results: List of sweep results
        output_dir: Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping visualization.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Filter results for single epsilon/gamma/beta for cleaner plots
    base_results = [r for r in results if r.epsilon == 0.0 and r.gamma == 1.0 and r.beta == 0.5]
    
    if not base_results:
        base_results = results
    
    # Plot 1: Entropy vs Eta
    plt.figure(figsize=(10, 6))
    etas = [r.eta for r in base_results]
    entropies = [r.final_entropy for r in base_results]
    plt.plot(etas, entropies, 'b-', marker='o', markersize=3, label='Entropy')
    plt.xlabel('Eta (Feedback Coefficient)')
    plt.ylabel('Final Entropy')
    plt.title('Entropy vs Eta Parameter')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'entropy_vs_eta.png', dpi=150)
    print(f"Saved plot: {output_path / 'entropy_vs_eta.png'}")
    plt.close()
    
    # Plot 2: Sparsity vs Eta
    plt.figure(figsize=(10, 6))
    sparsities = [r.final_sparsity for r in base_results]
    plt.plot(etas, sparsities, 'r-', marker='s', markersize=3, label='Sparsity')
    plt.xlabel('Eta (Feedback Coefficient)')
    plt.ylabel('Final Sparsity')
    plt.title('Sparsity vs Eta Parameter')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'sparsity_vs_eta.png', dpi=150)
    print(f"Saved plot: {output_path / 'sparsity_vs_eta.png'}")
    plt.close()
    
    # Plot 3: Combined metrics
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(etas, entropies, 'b-', marker='o', markersize=3, label='Entropy')
    line2 = ax2.plot(etas, sparsities, 'r-', marker='s', markersize=3, label='Sparsity')
    
    ax1.set_xlabel('Eta (Feedback Coefficient)')
    ax1.set_ylabel('Entropy', color='b')
    ax2.set_ylabel('Sparsity', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Parameter Sweep: Entropy and Sparsity vs Eta')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'combined_metrics.png', dpi=150)
    print(f"Saved plot: {output_path / 'combined_metrics.png'}")
    plt.close()


def main():
    """Main parameter sweep execution."""
    parser = argparse.ArgumentParser(description='ZKAEDI PRIME Parameter Sweep')
    parser.add_argument('--eta-steps', type=int, default=50,
                       help='Number of steps in eta sweep (default: 50)')
    parser.add_argument('--eta-min', type=float, default=0.1,
                       help='Minimum eta value (default: 0.1)')
    parser.add_argument('--eta-max', type=float, default=0.9,
                       help='Maximum eta value (default: 0.9)')
    parser.add_argument('--epsilon', type=float, nargs='+', default=[0.0, 0.05, 0.1],
                       help='Epsilon (noise) values to test (default: 0.0 0.05 0.1)')
    parser.add_argument('--gamma', type=float, nargs='+', default=[1.0],
                       help='Gamma values to test (default: 1.0)')
    parser.add_argument('--beta', type=float, nargs='+', default=[0.5],
                       help='Beta values to test (default: 0.5)')
    parser.add_argument('--h-types', type=str, nargs='+',
                       default=['pauli_z'],
                       help='Hamiltonian types to test (default: pauli_z)')
    parser.add_argument('--num-qubits', type=int, default=3,
                       help='Number of qubits (default: 3)')
    parser.add_argument('--timesteps', type=int, default=10,
                       help='Number of evolution steps (default: 10)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run in parallel mode')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    
    args = parser.parse_args()
    
    # Generate configurations
    print("Generating parameter configurations...")
    configurations = generate_configurations(
        num_qubits=args.num_qubits,
        eta_range=(args.eta_min, args.eta_max),
        eta_steps=args.eta_steps,
        epsilon_values=args.epsilon,
        gamma_values=args.gamma,
        beta_values=args.beta,
        h_types=args.h_types,
        timesteps=args.timesteps
    )
    
    print(f"Total configurations: {len(configurations)}")
    
    # Run sweep
    start_time = time.time()
    if args.parallel:
        results = run_parallel(configurations, args.workers)
    else:
        results = run_sequential(configurations)
    total_time = time.time() - start_time
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Display results
    display_results(results)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_file = output_path / f'parameter_sweep_{timestamp}.csv'
    json_file = output_path / f'parameter_sweep_{timestamp}.json'
    
    save_results_csv(results, str(csv_file))
    save_results_json(results, str(json_file))
    
    # Visualize if requested
    if args.visualize:
        visualize_results(results, args.output_dir)
    
    print("\nParameter sweep complete!")


if __name__ == "__main__":
    main()
