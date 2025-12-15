"""Comprehensive tests for advanced exception system."""

import pytest
import time
from zkaedi_prime_engine.exceptions import (
    ZKAEDIError,
    HamiltonianError,
    MetadataZKAEDIError,
    SelfDiagnosingZKAEDIError,
    SelfHealingError,
    AgeingError,
    SwarmError,
    FSMError,
    ExplainableZKAEDIError,
    ObservabilityException,
    ExceptionTracker,
    generate_exception_hierarchy,
)


class TestBaseExceptions:
    """Test base exception functionality."""
    
    def test_basic_exception(self):
        """Test basic exception creation."""
        error = ZKAEDIError("Test error")
        assert str(error) == "[E" + str(hash("ZKAEDIError") % 10000).zfill(4) + "] Test error"
        assert error.message == "Test error"
        assert error.error_code.startswith("E")
    
    def test_exception_with_context(self):
        """Test exception with context."""
        error = ZKAEDIError("Test", operation="test_op", value=42)
        assert error.context["operation"] == "test_op"
        assert error.context["value"] == 42


class TestMetadataExceptions:
    """Test metadata-rich exceptions."""
    
    def test_metadata_exception(self):
        """Test metadata exception."""
        error = MetadataZKAEDIError(
            "Test error",
            operation="test",
            value=42
        )
        assert error.get_metadata("operation") == "test"
        assert error.get_metadata("value") == 42
        assert error.get_metadata("missing", "default") == "default"


class TestSelfDiagnosingExceptions:
    """Test self-diagnosing exceptions."""
    
    def test_self_diagnosing(self):
        """Test self-diagnosing exception."""
        error = SelfDiagnosingZKAEDIError(
            "Backend failure",
            type="backend",
            resource_limit=95
        )
        assert len(error.remediation_hints) > 0
        suggestions = error.remediation_suggestions()
        assert "Diagnostic Hint" in suggestions or "Suggested Fixes" in suggestions


class TestSelfHealingExceptions:
    """Test self-healing exceptions."""
    
    def test_self_healing_success(self):
        """Test successful self-healing."""
        def healing_strategy(**context):
            return True
        
        error = SelfHealingError(
            "Test error",
            heal_strategy=healing_strategy,
            max_healing_attempts=1
        )
        
        assert error.healed is True
        assert error.healing_attempts > 0
    
    def test_self_healing_failure(self):
        """Test failed self-healing."""
        def healing_strategy(**context):
            return False
        
        error = SelfHealingError(
            "Test error",
            heal_strategy=healing_strategy,
            max_healing_attempts=1
        )
        
        assert error.healed is False
    
    def test_healing_report(self):
        """Test healing report generation."""
        error = SelfHealingError("Test", max_healing_attempts=1)
        report = error.get_healing_report()
        assert "healed" in report
        assert "attempts" in report


class TestEvolutionaryExceptions:
    """Test evolutionary exceptions."""
    
    def test_aging_exception(self):
        """Test aging exception."""
        error = AgeingError("Test", age=0, mutation_rate=0.5)
        initial_age = error.age
        
        # Evolve
        error.mutate()
        assert error.age > initial_age
    
    def test_evolution_report(self):
        """Test evolution report."""
        error = AgeingError("Test")
        report = error.evolve(generations=2)
        assert "age" in report
        assert "mutations" in report


class TestSwarmExceptions:
    """Test swarm intelligence exceptions."""
    
    def test_swarm_registration(self):
        """Test swarm member registration."""
        SwarmError.clear_swarm()
        
        error1 = SwarmError("Error 1")
        error2 = SwarmError("Error 2")
        
        members = SwarmError.get_swarm_members()
        assert len(members) == 2
    
    def test_swarm_analytics(self):
        """Test swarm analytics."""
        SwarmError.clear_swarm()
        
        SwarmError("Error 1", swarm_id="test_swarm")
        SwarmError("Error 2", swarm_id="test_swarm")
        
        analytics = SwarmError.analyze_swarm()
        assert analytics["total_members"] == 2
        assert "type_distribution" in analytics
    
    def test_swarm_recommendations(self):
        """Test swarm recommendations."""
        SwarmError.clear_swarm()
        
        # Create multiple errors
        for i in range(6):
            SwarmError("High frequency error", swarm_id="frequent")
        
        recommendations = SwarmError.get_swarm_recommendations()
        assert len(recommendations) > 0


class TestFSMExceptions:
    """Test FSM-based exceptions."""
    
    def test_fsm_initial_state(self):
        """Test FSM initial state."""
        error = FSMError("Test")
        assert error.get_current_state() == "triggered"
    
    def test_fsm_transitions(self):
        """Test FSM state transitions."""
        error = FSMError("Test")
        
        error.diagnose()
        assert error.get_current_state() == "diagnosing"
        
        error.heal()
        assert error.get_current_state() == "healing"
        
        error.resolve()
        assert error.get_current_state() == "resolved"
    
    def test_fsm_state_history(self):
        """Test FSM state history."""
        error = FSMError("Test")
        error.diagnose()
        error.heal()
        
        history = error.get_state_history()
        assert len(history) >= 3  # At least triggered, diagnosing, healing


class TestExplainableExceptions:
    """Test explainable exceptions."""
    
    def test_explainable_exception(self):
        """Test explainable exception."""
        error = ExplainableZKAEDIError(
            "Test error",
            input_vector=[1.0, 2.0, 3.0],
            feature_names=["a", "b", "c"]
        )
        
        explanation = error.explain()
        assert "causal_factors" in explanation
        assert "feature_importance" in explanation
        assert "recommendations" in explanation
    
    def test_explanation_summary(self):
        """Test explanation summary."""
        error = ExplainableZKAEDIError(
            "Test error",
            input_vector=[1e10, 0.001]
        )
        
        summary = error.get_explanation_summary()
        assert "Explanation for" in summary


class TestExceptionTracking:
    """Test exception tracking."""
    
    def test_tracking(self):
        """Test exception tracking."""
        ExceptionTracker.clear_history()
        
        ExceptionTracker.track_exception(HamiltonianError("Error 1"))
        ExceptionTracker.track_exception(HamiltonianError("Error 2"))
        
        stats = ExceptionTracker.get_statistics()
        assert stats["total_exceptions"] == 2
        assert "HamiltonianError" in stats["exception_counts"]
    
    def test_trends(self):
        """Test trend analysis."""
        ExceptionTracker.clear_history()
        
        for i in range(10):
            ExceptionTracker.track_exception(HamiltonianError(f"Error {i}"))
        
        trends = ExceptionTracker.get_trends(window_size=5)
        assert "recent_counts" in trends


class TestExceptionHierarchy:
    """Test exception hierarchy generation."""
    
    def test_hierarchy_generation(self):
        """Test hierarchy generation."""
        hierarchy = generate_exception_hierarchy()
        assert isinstance(hierarchy, dict)
        assert len(hierarchy) > 0


class TestBackwardCompatibility:
    """Test backward compatibility with old exception API."""
    
    def test_old_imports_work(self):
        """Test that old imports still work."""
        from zkaedi_prime_engine.exceptions import (
            ZKAEDIError,
            HamiltonianError,
            StateError,
            QECError,
            BackendError,
        )
        
        # Should not raise ImportError
        assert ZKAEDIError
        assert HamiltonianError
        assert StateError
        assert QECError
        assert BackendError
    
    def test_old_exceptions_are_enhanced(self):
        """Test that old exceptions have new features."""
        error = HamiltonianError("Test")
        assert hasattr(error, "error_code")
        assert hasattr(error, "context")
        assert hasattr(error, "timestamp")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

