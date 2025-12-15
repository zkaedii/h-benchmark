"""Evolutionary Exception Design: Aging and Mutation.

Bio-inspired exceptions that age, mutate, and evolve over time,
providing insights into system behavior and failure patterns.
"""

import random
import time
import logging
from typing import Dict, Any, Optional
from .base import ZKAEDIError

logger = logging.getLogger(__name__)


class AgeingError(ZKAEDIError):
    """Bio-inspired exception with aging and mutation capabilities.
    
    Models exceptions as organisms under selective pressure,
    allowing them to adapt to changing system states over time.
    """
    
    def __init__(
        self,
        message: str,
        age: int = 0,
        mutation_rate: float = 0.1,
        max_age: int = 10,
        **context
    ):
        """Initialize aging exception.
        
        Args:
            message: Error message
            age: Initial age of the exception
            mutation_rate: Probability of mutation (0.0 to 1.0)
            max_age: Maximum age before forced escalation
            **context: Additional context
        """
        super().__init__(message, **context)
        self.age = age
        self.mutation_rate = mutation_rate
        self.max_age = max_age
        self.mutation_count = 0
        self.original_message = message
        self.evolution_history = []
        self.birth_time = time.time()
    
    def mutate(self) -> bool:
        """Trigger mutation with probability.
        
        Mutations can adapt the exception to new system states,
        potentially making it more or less likely to occur.
        
        Returns:
            True if mutation occurred, False otherwise
        """
        self.age += 1
        
        if random.random() < self.mutation_rate:
            self.mutation_count += 1
            mutation_type = random.choice([
                "message_adaptation",
                "context_enrichment",
                "severity_adjustment"
            ])
            
            if mutation_type == "message_adaptation":
                self.message = f"{self.original_message} [Mutated: Age {self.age}, Attempt {self.mutation_count}]"
            elif mutation_type == "context_enrichment":
                self.context["mutation_count"] = self.mutation_count
                self.context["age"] = self.age
            
            self.evolution_history.append({
                "age": self.age,
                "mutation_type": mutation_type,
                "timestamp": time.time()
            })
            
            logger.info(f"⚡ Mutation #{self.mutation_count} triggered in {self.__class__.__name__} at age {self.age}")
            return True
        
        return False
    
    def evolve(self, generations: int = 1) -> Dict[str, Any]:
        """Evolve the exception through multiple generations.
        
        Args:
            generations: Number of evolution steps
            
        Returns:
            Evolution report
        """
        for _ in range(generations):
            self.mutate()
        
        return {
            "age": self.age,
            "mutations": self.mutation_count,
            "history": self.evolution_history,
            "lifespan": time.time() - self.birth_time
        }
    
    def escalate(self):
        """Raise exception after checking age threshold.
        
        Old exceptions that have evolved too far are escalated
        to prevent infinite retry loops.
        """
        logger.info(f"🧬 Current state: Age={self.age}, Mutations={self.mutation_count}")
        
        if self.age > self.max_age:
            raise RuntimeError(
                f"❌ Evolved Exception Beyond Threshold: {self.__class__.__name__} "
                f"(age={self.age}, max_age={self.max_age})"
            )
        
        # Mutate before potentially raising
        self.mutate()
        
        # Raise the exception itself
        raise self
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """Get comprehensive evolution report.
        
        Returns:
            Dictionary with evolution statistics
        """
        return {
            "age": self.age,
            "max_age": self.max_age,
            "mutation_rate": self.mutation_rate,
            "mutation_count": self.mutation_count,
            "lifespan_seconds": time.time() - self.birth_time,
            "evolution_history": self.evolution_history,
            "is_aged": self.age > self.max_age * 0.7  # 70% of max age
        }
    
    def reset_age(self):
        """Reset the age of the exception (for testing/debugging)."""
        self.age = 0
        self.mutation_count = 0
        self.evolution_history = []
        self.birth_time = time.time()
        logger.info(f"🔄 Age reset for {self.__class__.__name__}")
    
    def __str__(self) -> str:
        """String representation with age and mutation info."""
        base_str = super().__str__()
        return f"{base_str} [Age: {self.age}, Mutations: {self.mutation_count}]"

