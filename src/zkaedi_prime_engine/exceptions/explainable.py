"""Explainable Exceptions with XAI Integration.

Exceptions that provide explainable AI (XAI) insights, causal reasoning,
and feature importance analysis for debugging and understanding failures.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from .base import ZKAEDIError

logger = logging.getLogger(__name__)

# Try to import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.debug("SHAP not available. Explainability features will be limited.")


class ExplainableZKAEDIError(ZKAEDIError):
    """Exception with explainable AI (XAI) capabilities.
    
    Provides rationale, causal explanations, and feature importance
    to help understand why the error occurred.
    """
    
    def __init__(
        self,
        message: str,
        input_vector: Optional[List[Any]] = None,
        shap_model: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None,
        **context
    ):
        """Initialize explainable exception.
        
        Args:
            message: Error message
            input_vector: Input data that triggered the error
            shap_model: Optional SHAP model for explanation
            feature_names: Names of features in input_vector
            **context: Additional context
        """
        super().__init__(message, **context)
        self.input_vector = input_vector
        self.shap_model = shap_model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(len(input_vector))] if input_vector else []
        self.explanation = None
        self.causal_factors = []
    
    def explain(self) -> Dict[str, Any]:
        """Generate explanation for the error.
        
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "message": self.message,
            "input_features": dict(zip(self.feature_names, self.input_vector)) if self.input_vector else {},
            "causal_factors": self._identify_causal_factors(),
            "feature_importance": self._compute_feature_importance(),
            "recommendations": self._generate_recommendations()
        }
        
        self.explanation = explanation
        return explanation
    
    def _identify_causal_factors(self) -> List[Dict[str, Any]]:
        """Identify potential causal factors for the error.
        
        Returns:
            List of causal factor dictionaries
        """
        factors = []
        
        if self.input_vector:
            # Check for extreme values
            for i, value in enumerate(self.input_vector):
                if isinstance(value, (int, float)):
                    if abs(value) > 1e10:
                        factors.append({
                            "feature": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                            "value": value,
                            "issue": "Extremely large value detected",
                            "severity": "high"
                        })
                    elif abs(value) < 1e-10 and value != 0:
                        factors.append({
                            "feature": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                            "value": value,
                            "issue": "Extremely small value (potential numerical instability)",
                            "severity": "medium"
                        })
        
        # Check context for known issues
        if "operation" in self.context:
            factors.append({
                "feature": "operation",
                "value": self.context["operation"],
                "issue": "Operation context available",
                "severity": "info"
            })
        
        self.causal_factors = factors
        return factors
    
    def _compute_feature_importance(self) -> Dict[str, float]:
        """Compute feature importance using SHAP or heuristics.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if SHAP_AVAILABLE and self.shap_model and self.input_vector:
            try:
                # Use SHAP for explanation
                explainer = shap.TreeExplainer(self.shap_model) if hasattr(shap, 'TreeExplainer') else None
                if explainer:
                    shap_values = explainer.shap_values(self.input_vector)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]  # Take first output
                    
                    importance = {}
                    for i, name in enumerate(self.feature_names):
                        if i < len(shap_values):
                            importance[name] = float(abs(shap_values[i]))
                    return importance
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        # Fallback: heuristic importance based on value magnitude
        importance = {}
        if self.input_vector:
            max_val = max(abs(v) for v in self.input_vector if isinstance(v, (int, float))) or 1.0
            for i, value in enumerate(self.input_vector):
                if isinstance(value, (int, float)):
                    name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                    importance[name] = abs(value) / max_val
        
        return importance
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on explanation.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze causal factors
        high_severity = [f for f in self.causal_factors if f.get("severity") == "high"]
        if high_severity:
            recommendations.append(
                f"⚠️ {len(high_severity)} high-severity issues detected. "
                "Consider input validation and normalization."
            )
        
        # Feature importance recommendations
        importance = self._compute_feature_importance()
        if importance:
            top_feature = max(importance.items(), key=lambda x: x[1])
            recommendations.append(
                f"💡 Most influential feature: '{top_feature[0]}' (importance: {top_feature[1]:.3f}). "
                "Focus debugging efforts here."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "💡 Review input data and operation context for potential issues."
            )
        
        return recommendations
    
    def visualize_explanation(self, output_file: Optional[str] = None):
        """Visualize the explanation (requires SHAP and matplotlib).
        
        Args:
            output_file: Optional file path to save visualization
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for visualization")
            return
        
        if not self.input_vector or not self.shap_model:
            logger.warning("Input vector or model not available for visualization")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            explainer = shap.TreeExplainer(self.shap_model) if hasattr(shap, 'TreeExplainer') else None
            if explainer:
                shap_values = explainer.shap_values(self.input_vector)
                shap.force_plot(
                    explainer.expected_value,
                    shap_values,
                    self.input_vector,
                    feature_names=self.feature_names,
                    show=False,
                    matplotlib=True
                )
                
                if output_file:
                    plt.savefig(output_file)
                    logger.info(f"Explanation visualization saved to {output_file}")
                else:
                    plt.show()
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def get_explanation_summary(self) -> str:
        """Get a human-readable explanation summary.
        
        Returns:
            Formatted explanation string
        """
        if not self.explanation:
            self.explain()
        
        summary = f"Explanation for: {self.message}\n"
        summary += f"Causal Factors: {len(self.causal_factors)}\n"
        
        for factor in self.causal_factors[:3]:  # Top 3 factors
            summary += f"  - {factor['feature']}: {factor['issue']}\n"
        
        recommendations = self._generate_recommendations()
        if recommendations:
            summary += "Recommendations:\n"
            for rec in recommendations:
                summary += f"  {rec}\n"
        
        return summary
    
    def __str__(self) -> str:
        """String representation with explanation hint."""
        base_str = super().__str__()
        if self.explanation:
            return f"{base_str} [Explainable: {len(self.causal_factors)} factors identified]"
        return f"{base_str} [Call explain() for detailed analysis]"

