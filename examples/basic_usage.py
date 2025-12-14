"""Basic usage example for ZKAEDI PRIME Engine."""

from zkaedi_prime_engine import ZKAEDIEngine, create_example_hamiltonian

def main():
    """Basic example of using ZKAEDI PRIME Engine."""
    print("ZKAEDI PRIME Engine - Basic Usage Example")
    print("=" * 50)
    
    # Create example Hamiltonian
    num_qubits = 4
    H0 = create_example_hamiltonian(num_qubits, h_type="ising")
    
    print(f"Created {num_qubits}-qubit system")
    print(f"Hamiltonian shape: {H0.shape}")
    
    # Initialize engine
    engine = ZKAEDIEngine(
        num_qubits=num_qubits,
        H0=H0,
        eta=0.4,
        gamma=1.0,
        epsilon=0.04,
        beta=0.5
    )
    
    print("\nEvolving system...")
    
    # Evolve system
    diagnostics = engine.evolve(timesteps=20, track_metrics=True)
    
    # Get summary
    summary = engine.get_summary()
    
    print("\nResults:")
    print(f"  Final backend: {summary['current_backend']}")
    print(f"  Avg entropy: {summary['statistics']['avg_entropy']:.4f}")
    print(f"  Avg sparsity: {summary['statistics']['avg_sparsity']:.4f}")
    print(f"  QEC activations: {summary['statistics']['qec_count']}")

if __name__ == "__main__":
    main()

