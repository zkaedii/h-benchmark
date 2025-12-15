"""Self-Healing Exceptions for ZKAEDI PRIME.

Bio-inspired exceptions that attempt to diagnose, repair, and retry
failed operations before escalating.
"""

import time
import logging
from typing import Callable, Optional, Dict, Any
from .base import ZKAEDIError

logger = logging.getLogger(__name__)


class SelfHealingError(ZKAEDIError):
    """Exceptions that attempt self-healing before escalation.
    
    Inspired by biological tissue repair and cellular regeneration,
    these exceptions try to resolve issues autonomously before
    propagating the error state.
    """
    
    def __init__(
        self,
        message: str,
        heal_strategy: Optional[Callable] = None,
        max_healing_attempts: int = 3,
        healing_delay: float = 1.0,
        **context
    ):
        """Initialize self-healing exception.
        
        Args:
            message: Error message
            heal_strategy: Callable that performs healing (returns bool)
            max_healing_attempts: Maximum number of healing attempts
            healing_delay: Delay between healing attempts (seconds)
            **context: Additional context for healing strategy
        """
        super().__init__(message, **context)
        self.heal_strategy = heal_strategy or self.default_healing
        self.max_healing_attempts = max_healing_attempts
        self.healing_delay = healing_delay
        self.healing_attempts = 0
        self.healed = False
        self.healing_history = []
        
        # Attempt healing immediately upon creation
        self.attempt_healing()
    
    def attempt_healing(self) -> bool:
        """Attempt to heal the exception.
        
        Returns:
            True if healing was successful, False otherwise
        """
        if self.healed:
            return True
        
        for attempt in range(self.max_healing_attempts):
            self.healing_attempts += 1
            logger.info(f"🩺 Healing attempt {self.healing_attempts}/{self.max_healing_attempts}: {self.message}")
            
            try:
                # Execute healing strategy
                result = self.heal_strategy(**self.context)
                
                if result:
                    self.healed = True
                    self.healing_history.append({
                        "attempt": self.healing_attempts,
                        "success": True,
                        "timestamp": time.time()
                    })
                    logger.info(f"✅ Healing successful for: {self.message}")
                    return True
                else:
                    self.healing_history.append({
                        "attempt": self.healing_attempts,
                        "success": False,
                        "reason": "Strategy returned False",
                        "timestamp": time.time()
                    })
                    
            except Exception as ex:
                self.healing_history.append({
                    "attempt": self.healing_attempts,
                    "success": False,
                    "error": str(ex),
                    "timestamp": time.time()
                })
                logger.warning(f"💔 Healing attempt {self.healing_attempts} failed: {ex}")
            
            # Wait before next attempt
            if attempt < self.max_healing_attempts - 1:
                time.sleep(self.healing_delay)
        
        logger.error(f"❌ All healing attempts failed for: {self.message}")
        return False
    
    @staticmethod
    def default_healing(**context) -> bool:
        """Default healing strategy.
        
        This is a simple retry mechanism. Override in subclasses
        or provide custom strategies for domain-specific healing.
        
        Args:
            **context: Context from the exception
            
        Returns:
            True if default healing should be considered successful
        """
        resource = context.get("resource", "unknown")
        operation = context.get("operation", "unknown operation")
        
        logger.info(f"🚑 Default healing attempted on resource: {resource}, operation: {operation}")
        
        # Simple retry logic - in real scenarios, this would be more sophisticated
        # For example: resetting connections, clearing caches, etc.
        time.sleep(0.5)  # Simulate healing delay
        
        # Return True to indicate healing was attempted
        # In practice, you'd check if the resource is actually available
        return True
    
    def escalate(self):
        """Raise the exception if healing fails.
        
        This should be called after attempting healing to propagate
        the error if self-healing was unsuccessful.
        """
        if not self.healed:
            if self.healing_attempts == 0:
                # Attempt healing if not already attempted
                self.attempt_healing()
            
            if not self.healed:
                logger.error(f"🚨 Escalating unhealed exception: {self.message}")
                raise self
    
    def get_healing_report(self) -> Dict[str, Any]:
        """Get a report of healing attempts.
        
        Returns:
            Dictionary with healing statistics and history
        """
        return {
            "healed": self.healed,
            "attempts": self.healing_attempts,
            "max_attempts": self.max_healing_attempts,
            "history": self.healing_history
        }
    
    def __str__(self) -> str:
        """String representation with healing status."""
        base_str = super().__str__()
        status = "✅ HEALED" if self.healed else f"❌ UNHEALED ({self.healing_attempts} attempts)"
        return f"{base_str} [{status}]"

