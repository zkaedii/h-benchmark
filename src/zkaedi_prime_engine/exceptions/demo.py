"""Demonstration of Advanced Exception System Features.

This module provides examples and demonstrations of the various
advanced exception handling capabilities.
"""

import time
from .base import (
    ZKAEDIError,
    HamiltonianError,
    MetadataZKAEDIError,
    SelfDiagnosingZKAEDIError,
)
from .healing import SelfHealingError
from .evolution import AgeingError
from .swarm import SwarmError
from .fsm import FSMError
from .explainable import ExplainableZKAEDIError
from .diagnostics import (
    ObservabilityException,
    ExceptionTracker,
    generate_exception_hierarchy,
    print_exception_hierarchy,
)


def demo_self_healing():
    """Demonstrate self-healing exceptions."""
    print("\n" + "=" * 70)
    print("DEMO: Self-Healing Exceptions")
    print("=" * 70)
    
    def custom_healing_strategy(**context):
        """Custom healing strategy."""
        print(f"  🔧 Custom healing for: {context.get('operation', 'unknown')}")
        time.sleep(0.1)  # Simulate healing
        return True  # Healing successful
    
    error = SelfHealingError(
        "Backend connection failed",
        heal_strategy=custom_healing_strategy,
        operation="database_query",
        resource="postgres"
    )
    
    print(f"  Error created: {error}")
    print(f"  Healing status: {'✅ HEALED' if error.healed else '❌ UNHEALED'}")
    print(f"  Healing report: {error.get_healing_report()}")


def demo_evolution():
    """Demonstrate evolutionary exceptions."""
    print("\n" + "=" * 70)
    print("DEMO: Evolutionary Exceptions")
    print("=" * 70)
    
    error = AgeingError(
        "Convergence failure",
        age=0,
        mutation_rate=0.3,
        max_age=5
    )
    
    print(f"  Initial error: {error}")
    
    # Evolve through a few generations
    evolution_report = error.evolve(generations=3)
    print(f"  After evolution: {error}")
    print(f"  Evolution report: {evolution_report}")


def demo_swarm():
    """Demonstrate swarm intelligence exceptions."""
    print("\n" + "=" * 70)
    print("DEMO: Swarm Intelligence Exceptions")
    print("=" * 70)
    
    # Create multiple swarm errors
    SwarmError("Backend timeout", operation="query_1")
    SwarmError("Backend timeout", operation="query_2")
    SwarmError("Memory error", operation="allocation")
    SwarmError("Backend timeout", operation="query_3")
    
    # Analyze swarm
    analytics = SwarmError.analyze_swarm()
    print(f"  Swarm analytics: {analytics}")
    
    # Get recommendations
    recommendations = SwarmError.get_swarm_recommendations()
    print(f"  Recommendations:")
    for rec in recommendations:
        print(f"    {rec}")
    
    # Clear swarm for next demo
    SwarmError.clear_swarm()


def demo_fsm():
    """Demonstrate FSM-based exceptions."""
    print("\n" + "=" * 70)
    print("DEMO: FSM-Based Exceptions")
    print("=" * 70)
    
    error = FSMError("State computation failed")
    print(f"  Initial state: {error.get_current_state()}")
    
    error.diagnose()
    print(f"  After diagnose: {error.get_current_state()}")
    
    error.heal()
    print(f"  After heal: {error.get_current_state()}")
    
    error.resolve()
    print(f"  After resolve: {error.get_current_state()}")
    print(f"  State history: {len(error.get_state_history())} transitions")


def demo_explainable():
    """Demonstrate explainable exceptions."""
    print("\n" + "=" * 70)
    print("DEMO: Explainable Exceptions")
    print("=" * 70)
    
    error = ExplainableZKAEDIError(
        "Matrix inversion failed",
        input_vector=[1e10, 0.001, -5.0],
        feature_names=["matrix_size", "condition_number", "tolerance"]
    )
    
    explanation = error.explain()
    print(f"  Explanation summary:")
    print(f"    Causal factors: {len(explanation['causal_factors'])}")
    print(f"    Feature importance: {explanation['feature_importance']}")
    print(f"    Recommendations: {len(explanation['recommendations'])}")
    
    summary = error.get_explanation_summary()
    print(f"\n  Full summary:\n{summary}")


def demo_metadata():
    """Demonstrate metadata-rich exceptions."""
    print("\n" + "=" * 70)
    print("DEMO: Metadata-Rich Exceptions")
    print("=" * 70)
    
    error = MetadataZKAEDIError(
        "Hamiltonian computation failed",
        operation="eigen_solver",
        parameters={"matrix_dim": 3, "tolerance": 1e-6},
        backend="dense"
    )
    
    print(f"  Error: {error}")
    print(f"  Operation: {error.get_metadata('operation')}")
    print(f"  Parameters: {error.get_metadata('parameters')}")


def demo_self_diagnosing():
    """Demonstrate self-diagnosing exceptions."""
    print("\n" + "=" * 70)
    print("DEMO: Self-Diagnosing Exceptions")
    print("=" * 70)
    
    error = SelfDiagnosingZKAEDIError(
        "Backend failure",
        type="backend",
        resource_limit=95
    )
    
    print(f"  Error: {error}")
    print(f"\n  Remediation suggestions:")
    print(error.remediation_suggestions())


def demo_tracking():
    """Demonstrate exception tracking and analytics."""
    print("\n" + "=" * 70)
    print("DEMO: Exception Tracking and Analytics")
    print("=" * 70)
    
    # Clear previous tracking
    ExceptionTracker.clear_history()
    
    # Simulate some exceptions
    for i in range(5):
        ExceptionTracker.track_exception(HamiltonianError(f"Error {i}"))
    
    for i in range(3):
        ExceptionTracker.track_exception(ZKAEDIError(f"Generic error {i}"))
    
    # Get statistics
    stats = ExceptionTracker.get_statistics()
    print(f"  Statistics: {stats}")
    
    # Get trends
    trends = ExceptionTracker.get_trends()
    print(f"  Trends: {trends}")


def demo_hierarchy():
    """Demonstrate exception hierarchy generation."""
    print("\n" + "=" * 70)
    print("DEMO: Exception Hierarchy")
    print("=" * 70)
    
    hierarchy = generate_exception_hierarchy()
    print(f"  Found {len(hierarchy)} exception types")
    print("\n  Exception types:")
    for name, doc in sorted(hierarchy.items())[:5]:  # Show first 5
        print(f"    - {name}: {doc[:50]}...")


def run_all_demos():
    """Run all demonstration functions."""
    print("\n" + "🚀 " * 35)
    print("ZKAEDI PRIME Advanced Exception System - Demonstrations")
    print("🚀 " * 35)
    
    demos = [
        demo_metadata,
        demo_self_diagnosing,
        demo_self_healing,
        demo_evolution,
        demo_swarm,
        demo_fsm,
        demo_explainable,
        demo_tracking,
        demo_hierarchy,
    ]
    
    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n  ❌ Demo failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ All demonstrations completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_demos()

