"""Swarm Intelligence for Collective ZKAEDI Exceptions.

Multi-agent exception handling inspired by swarm theory,
where exceptions collaborate, share context, and optimize recoveries.
"""

import time
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from .base import ZKAEDIError

logger = logging.getLogger(__name__)


class SwarmError(ZKAEDIError):
    """Multi-agent swarm-aware error handling.
    
    Exceptions in a swarm can share information, learn from patterns,
    and collectively optimize error recovery strategies.
    """
    
    # Class-level swarm registry
    _swarm_registry: List['SwarmError'] = []
    _swarm_analytics: Dict[str, Any] = defaultdict(int)
    _swarm_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def __init__(self, message: str, swarm_id: Optional[str] = None, **context):
        """Initialize swarm-aware exception.
        
        Args:
            message: Error message
            swarm_id: Optional identifier for swarm grouping
            **context: Additional context
        """
        super().__init__(message, **context)
        self.swarm_id = swarm_id or self._generate_swarm_id()
        self.swarm_timestamp = time.time()
        self.swarm_index = len(SwarmError._swarm_registry)
        
        # Register with swarm
        SwarmError._swarm_registry.append(self)
        SwarmError._swarm_analytics[self.__class__.__name__] += 1
        SwarmError._swarm_patterns[self.swarm_id].append({
            "message": message,
            "timestamp": self.swarm_timestamp,
            "context": context,
            "class": self.__class__.__name__
        })
        
        logger.debug(f"🐝 Swarm member #{self.swarm_index} registered: {self.swarm_id}")
    
    def _generate_swarm_id(self) -> str:
        """Generate a swarm identifier based on exception type and context."""
        class_name = self.__class__.__name__
        context_key = str(sorted(self.context.items()))[:20] if self.context else "default"
        return f"{class_name}_{hash(context_key) % 10000}"
    
    @classmethod
    def analyze_swarm(cls) -> Dict[str, Any]:
        """Aggregate swarm-level insights and patterns.
        
        Returns:
            Dictionary with swarm analytics
        """
        if not cls._swarm_registry:
            return {"message": "No swarm members registered", "count": 0}
        
        # Count by exception type
        type_counts = Counter(ex.__class__.__name__ for ex in cls._swarm_registry)
        
        # Count by swarm ID
        swarm_counts = Counter(ex.swarm_id for ex in cls._swarm_registry)
        
        # Analyze patterns
        pattern_analysis = {}
        for swarm_id, patterns in cls._swarm_patterns.items():
            if len(patterns) > 1:
                pattern_analysis[swarm_id] = {
                    "frequency": len(patterns),
                    "time_span": patterns[-1]["timestamp"] - patterns[0]["timestamp"],
                    "common_context": cls._extract_common_context(patterns)
                }
        
        analytics = {
            "total_members": len(cls._swarm_registry),
            "type_distribution": dict(type_counts),
            "swarm_distribution": dict(swarm_counts),
            "pattern_analysis": pattern_analysis,
            "most_common_type": type_counts.most_common(1)[0] if type_counts else None,
            "swarm_analytics": dict(cls._swarm_analytics)
        }
        
        logger.info(f"🐝 Swarm Analysis: {analytics['total_members']} members, "
                   f"{len(pattern_analysis)} patterns detected")
        
        return analytics
    
    @classmethod
    def _extract_common_context(cls, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common context keys from pattern list."""
        if not patterns:
            return {}
        
        all_keys = set()
        for pattern in patterns:
            all_keys.update(pattern.get("context", {}).keys())
        
        common_context = {}
        for key in all_keys:
            values = [p.get("context", {}).get(key) for p in patterns if key in p.get("context", {})]
            if len(set(values)) == 1:  # All values are the same
                common_context[key] = values[0]
        
        return common_context
    
    @classmethod
    def get_swarm_members(cls, swarm_id: Optional[str] = None) -> List['SwarmError']:
        """Get all swarm members, optionally filtered by swarm_id.
        
        Args:
            swarm_id: Optional filter by swarm ID
            
        Returns:
            List of swarm error instances
        """
        if swarm_id:
            return [ex for ex in cls._swarm_registry if ex.swarm_id == swarm_id]
        return cls._swarm_registry.copy()
    
    @classmethod
    def clear_swarm(cls):
        """Clear the swarm registry (useful for testing)."""
        cls._swarm_registry.clear()
        cls._swarm_analytics.clear()
        cls._swarm_patterns.clear()
        logger.info("🐝 Swarm registry cleared")
    
    @classmethod
    def get_swarm_recommendations(cls) -> List[str]:
        """Get recommendations based on swarm analysis.
        
        Returns:
            List of recommendation strings
        """
        analytics = cls.analyze_swarm()
        recommendations = []
        
        # Check for high-frequency patterns
        for swarm_id, pattern_info in analytics.get("pattern_analysis", {}).items():
            if pattern_info["frequency"] > 5:
                recommendations.append(
                    f"⚠️ High-frequency pattern detected in swarm '{swarm_id}': "
                    f"{pattern_info['frequency']} occurrences. Consider investigating root cause."
                )
        
        # Check for type distribution
        type_dist = analytics.get("type_distribution", {})
        if type_dist:
            most_common = max(type_dist.items(), key=lambda x: x[1])
            if most_common[1] > 10:
                recommendations.append(
                    f"⚠️ Most common exception type: {most_common[0]} ({most_common[1]} occurrences). "
                    f"Consider adding preventive measures."
                )
        
        return recommendations
    
    def get_swarm_context(self) -> Dict[str, Any]:
        """Get context about this exception's position in the swarm.
        
        Returns:
            Dictionary with swarm context information
        """
        swarm_members = self.get_swarm_members(self.swarm_id)
        
        return {
            "swarm_id": self.swarm_id,
            "swarm_index": self.swarm_index,
            "swarm_size": len(swarm_members),
            "swarm_position": f"{self.swarm_index + 1}/{len(SwarmError._swarm_registry)}",
            "similar_errors": len([ex for ex in swarm_members if ex.message == self.message])
        }
    
    def __str__(self) -> str:
        """String representation with swarm context."""
        base_str = super().__str__()
        swarm_ctx = self.get_swarm_context()
        return f"{base_str} [Swarm: {swarm_ctx['swarm_id']}, Position: {swarm_ctx['swarm_position']}]"

