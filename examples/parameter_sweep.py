"""Parameter sweep example."""

import numpy as np
from zkaedi_prime_engine import ZKAEDIEngine, create_example_hamiltonian

def main():
    """Example of parameter sensitivity analysis."""
    print("ZKAEDI PRIME Engine - Parameter Sweep Example")
    print("=" * 50)
    
    num_qubits = 3
    H0 = create_example_hamiltonian(num_qubits, h_type="pauli_z")
    
    print(f"\nTesting different eta values:")
    print(f"{'Eta':<10} {'Final Entropy':<15} {'Final Sparsity':<15}")
    print("-" * 40)
    
    for eta in np.linspace(0.1, 0.9, 5):
        engine = ZKAEDIEngine(
            num_qubits=num_qubits,
            H0=H0,
            eta=eta,
            epsilon=0.0  # No noise for cleaner comparison
        )
        
        diagnostics = engine.evolve(10, track_metrics=False)
        final = diagnostics[-1]
        
        print(f"{eta:<10.1f} {final.entropy:<15.3f} {final.sparsity:<15.4f}")

if __name__ == "__main__":
    main()

