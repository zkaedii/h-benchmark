"""
Basic Usage Example - ZKAEDI PRIME Engine
==========================================
Demonstrates core functionality of the ZKAEDI PRIME Engine including:
- Engine initialization
- System evolution
- Diagnostics retrieval
- Result analysis and export

This example serves as a starting point for exploring quantum system simulation
with the ZKAEDI PRIME Engine.

Usage:
    python basic_usage.py [--qubits 4] [--timesteps 20] [--export]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from zkaedi_prime_engine import ZKAEDIEngine, create_example_hamiltonian


def run_basic_example(
    num_qubits: int = 4,
    timesteps: int = 20,
    h_type: str = "ising",
    eta: float = 0.4,
    gamma: float = 1.0,
    epsilon: float = 0.04,
    beta: float = 0.5,
    seed: int = 42,
    export: bool = False,
    output_dir: str = "outputs"
) -> Dict:
    """Run basic ZKAEDI PRIME Engine example.
    
    Args:
        num_qubits: Number of qubits in the system
        timesteps: Number of evolution steps
        h_type: Hamiltonian type ('ising', 'pauli_z', 'pauli_x', 'random')
        eta: Feedback coefficient
        gamma: Sharpening coefficient
        epsilon: Noise amplitude
        beta: Noise scaling
        seed: Random seed for reproducibility
        export: Whether to export results to files
        output_dir: Output directory for exported files
        
    Returns:
        Dictionary containing engine summary and diagnostics
    """
    print("=" * 70)
    print("ZKAEDI PRIME Engine - Basic Usage Example")
    print("=" * 70)
    
    # Create example Hamiltonian
    print(f"\nCreating {h_type} Hamiltonian for {num_qubits} qubits...")
    H0 = create_example_hamiltonian(num_qubits, h_type=h_type)
    print(f"  Hamiltonian shape: {H0.shape}")
    print(f"  Hamiltonian type: {h_type}")
    
    # Initialize engine
    print(f"\nInitializing ZKAEDI PRIME Engine...")
    print(f"  Parameters: eta={eta}, gamma={gamma}, epsilon={epsilon}, beta={beta}")
    
    engine = ZKAEDIEngine(
        num_qubits=num_qubits,
        H0=H0,
        eta=eta,
        gamma=gamma,
        epsilon=epsilon,
        beta=beta,
        seed=seed
    )
    
    # Evolve system
    print(f"\nEvolving system for {timesteps} timesteps...")
    start_time = time.time()
    diagnostics = engine.evolve(timesteps=timesteps, track_metrics=True)
    elapsed_time = time.time() - start_time
    
    print(f"  Evolution completed in {elapsed_time:.3f} seconds")
    print(f"  Average time per step: {elapsed_time/timesteps*1000:.3f} ms")
    
    # Get summary
    print("\n" + "=" * 70)
    print("SYSTEM SUMMARY")
    print("=" * 70)
    summary = engine.get_summary()
    
    print(f"\nSystem Configuration:")
    print(f"  Number of qubits: {summary['num_qubits']}")
    print(f"  System dimension: {summary.get('dimension', 2**num_qubits)}")
    print(f"  Evolution timesteps: {summary['timesteps']}")
    print(f"  Current backend: {summary['current_backend']}")
    
    if 'statistics' in summary:
        stats = summary['statistics']
        print(f"\nStatistics:")
        print(f"  Average entropy: {stats.get('avg_entropy', 0):.4f}")
        print(f"  Maximum entropy: {stats.get('max_entropy', 0):.4f}")
        print(f"  Average sparsity: {stats.get('avg_sparsity', 0):.4f}")
        print(f"  QEC activations: {stats.get('qec_count', 0)}")
    
    # Display latest diagnostics
    if diagnostics:
        latest = diagnostics[-1]
        print(f"\nLatest Diagnostics:")
        print(f"  Timestep: {latest.timestep}")
        print(f"  Entropy: {latest.entropy:.4f}")
        print(f"  Sparsity: {latest.sparsity:.4f}")
        print(f"  Curvature: {latest.curvature:.4f}")
        print(f"  Backend: {latest.backend.value}")
        print(f"  QEC Applied: {latest.qec_applied}")
        if latest.qec_type:
            print(f"  QEC Type: {latest.qec_type}")
        if latest.energy is not None:
            print(f"  Energy: {latest.energy:.4f}")
    
    # Export results if requested
    if export:
        export_results(engine, diagnostics, summary, output_dir, num_qubits, timesteps)
    
    return {
        'summary': summary,
        'diagnostics': [d.__dict__ for d in diagnostics],
        'elapsed_time': elapsed_time
    }


def export_results(
    engine: ZKAEDIEngine,
    diagnostics: List,
    summary: Dict,
    output_dir: str,
    num_qubits: int,
    timesteps: int
):
    """Export results to JSON and CSV files.
    
    Args:
        engine: Engine instance
        diagnostics: List of diagnostics
        summary: Summary dictionary
        output_dir: Output directory
        num_qubits: Number of qubits
        timesteps: Number of timesteps
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Export to JSON
    json_data = {
        'metadata': {
            'num_qubits': num_qubits,
            'timesteps': timesteps,
            'timestamp': timestamp
        },
        'summary': summary,
        'diagnostics': [d.__dict__ for d in diagnostics]
    }
    
    json_file = output_path / f'basic_usage_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"\nResults exported to: {json_file}")
    
    # Export diagnostics to CSV
    try:
        import csv
        csv_file = output_path / f'basic_usage_{timestamp}.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestep', 'Entropy', 'Sparsity', 'Curvature',
                'Backend', 'QEC_Applied', 'QEC_Type', 'Energy'
            ])
            for d in diagnostics:
                writer.writerow([
                    d.timestep, d.entropy, d.sparsity, d.curvature,
                    d.backend.value, d.qec_applied, d.qec_type or '', d.energy or ''
                ])
        print(f"Diagnostics exported to: {csv_file}")
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")


def visualize_diagnostics(diagnostics: List, output_dir: str = "outputs"):
    """Visualize diagnostics over time.
    
    Args:
        diagnostics: List of diagnostic objects
        output_dir: Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping visualization.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timesteps = [d.timestep for d in diagnostics]
    entropies = [d.entropy for d in diagnostics]
    sparsities = [d.sparsity for d in diagnostics]
    curvatures = [d.curvature for d in diagnostics]
    
    # Plot 1: Entropy and Sparsity over time
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(timesteps, entropies, 'b-', marker='o', markersize=4, label='Entropy')
    line2 = ax2.plot(timesteps, sparsities, 'r-', marker='s', markersize=4, label='Sparsity')
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Entropy', color='b')
    ax2.set_ylabel('Sparsity', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('System Evolution: Entropy and Sparsity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plt.savefig(output_path / f'evolution_metrics_{timestamp}.png', dpi=150)
    print(f"Saved visualization: {output_path / f'evolution_metrics_{timestamp}.png'}")
    plt.close()
    
    # Plot 2: Curvature over time
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, curvatures, 'g-', marker='^', markersize=4)
    plt.xlabel('Timestep')
    plt.ylabel('Curvature')
    plt.title('System Curvature Evolution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'curvature_{timestamp}.png', dpi=150)
    print(f"Saved visualization: {output_path / f'curvature_{timestamp}.png'}")
    plt.close()


def experiment_with_parameters():
    """Demonstrate parameter experimentation."""
    print("\n" + "=" * 70)
    print("PARAMETER EXPERIMENTATION")
    print("=" * 70)
    
    num_qubits = 3
    H0 = create_example_hamiltonian(num_qubits, h_type="pauli_z")
    
    # Test different parameter combinations
    parameter_sets = [
        {'eta': 0.3, 'gamma': 0.8, 'epsilon': 0.05, 'beta': 0.7},
        {'eta': 0.5, 'gamma': 1.0, 'epsilon': 0.04, 'beta': 0.5},
        {'eta': 0.7, 'gamma': 1.2, 'epsilon': 0.03, 'beta': 0.3},
    ]
    
    results = []
    for params in parameter_sets:
        engine = ZKAEDIEngine(num_qubits, H0, **params)
        diagnostics = engine.evolve(10, track_metrics=False)
        final = diagnostics[-1]
        results.append({
            'params': params,
            'final_entropy': final.entropy,
            'final_sparsity': final.sparsity
        })
    
    print("\nParameter Comparison:")
    print(f"{'Eta':<8} {'Gamma':<8} {'Epsilon':<10} {'Beta':<8} {'Entropy':<12} {'Sparsity':<12}")
    print("-" * 70)
    for r in results:
        p = r['params']
        print(f"{p['eta']:<8.1f} {p['gamma']:<8.1f} {p['epsilon']:<10.2f} {p['beta']:<8.1f} "
              f"{r['final_entropy']:<12.4f} {r['final_sparsity']:<12.4f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='ZKAEDI PRIME Engine Basic Usage Example')
    parser.add_argument('--qubits', type=int, default=4,
                       help='Number of qubits (default: 4)')
    parser.add_argument('--timesteps', type=int, default=20,
                       help='Number of evolution steps (default: 20)')
    parser.add_argument('--h-type', type=str, default='ising',
                       choices=['ising', 'pauli_z', 'pauli_x', 'random'],
                       help='Hamiltonian type (default: ising)')
    parser.add_argument('--eta', type=float, default=0.4,
                       help='Feedback coefficient (default: 0.4)')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Sharpening coefficient (default: 1.0)')
    parser.add_argument('--epsilon', type=float, default=0.04,
                       help='Noise amplitude (default: 0.04)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Noise scaling (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--export', action='store_true',
                       help='Export results to files')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--experiment', action='store_true',
                       help='Run parameter experimentation')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    
    args = parser.parse_args()
    
    # Run basic example
    results = run_basic_example(
        num_qubits=args.qubits,
        timesteps=args.timesteps,
        h_type=args.h_type,
        eta=args.eta,
        gamma=args.gamma,
        epsilon=args.epsilon,
        beta=args.beta,
        seed=args.seed,
        export=args.export,
        output_dir=args.output_dir
    )
    
    # Visualize if requested
    if args.visualize and results.get('diagnostics'):
        diagnostics = results['diagnostics']
        # Convert back to diagnostic objects if needed
        from zkaedi_prime_engine import ZKAEDIDiagnostics
        diag_objects = [ZKAEDIDiagnostics(**d) for d in diagnostics]
        visualize_diagnostics(diag_objects, args.output_dir)
    
    # Run parameter experimentation
    if args.experiment:
        experiment_with_parameters()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
