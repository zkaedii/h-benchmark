"""State-Aware Exceptions Using Finite State Machines.

Model exceptions as finite state machines with defined transition pathways,
enabling stateful error handling and recovery workflows.
"""

import logging
from typing import Dict, Any, Optional, List
from .base import ZKAEDIError

logger = logging.getLogger(__name__)

# Try to import transitions library, fallback to simple implementation
try:
    from transitions import Machine
    TRANSITIONS_AVAILABLE = True
except ImportError:
    TRANSITIONS_AVAILABLE = False
    logger.warning("transitions library not available. FSM functionality will be limited.")


class FSMError(ZKAEDIError):
    """Model exceptions as Finite State Machines.
    
    Exceptions transition through defined states (triggered -> diagnosing -> healing -> resolved),
    enabling structured error recovery workflows.
    """
    
    # Default state machine states
    states = ["triggered", "diagnosing", "healing", "resolved", "escalated"]
    
    def __init__(self, message: str, **context):
        """Initialize FSM-based exception.
        
        Args:
            message: Error message
            **context: Additional context
        """
        super().__init__(message, **context)
        self.state_history: List[Dict[str, Any]] = []
        self._initialize_fsm()
    
    def _initialize_fsm(self):
        """Initialize the finite state machine."""
        if TRANSITIONS_AVAILABLE:
            self.machine = Machine(
                model=self,
                states=self.states,
                initial="triggered",
                auto_transitions=False,
            )
            
            # Define state transitions
            self.machine.add_transition("diagnose", "triggered", "diagnosing")
            self.machine.add_transition("heal", "diagnosing", "healing")
            self.machine.add_transition("resolve", "healing", "resolved")
            self.machine.add_transition("escalate", "*", "escalated")
            self.machine.add_transition("reset", "*", "triggered")
        else:
            # Fallback implementation without transitions library
            self.state = "triggered"
            self._transitions = {
                "triggered": ["diagnosing", "escalated"],
                "diagnosing": ["healing", "escalated"],
                "healing": ["resolved", "escalated"],
                "resolved": [],
                "escalated": []
            }
            self._record_state("triggered")
    
    def _record_state(self, new_state: str):
        """Record state transition."""
        import time
        self.state_history.append({
            "state": new_state,
            "timestamp": time.time(),
            "message": self.message
        })
        logger.debug(f"FSM State transition: {self.state} -> {new_state}")
    
    def _transition_fallback(self, action: str, target_state: str):
        """Fallback transition method when transitions library is not available."""
        if not hasattr(self, 'state'):
            self.state = "triggered"
        
        current_state = self.state
        allowed = self._transitions.get(current_state, [])
        
        if target_state in allowed:
            self.state = target_state
            self._record_state(target_state)
            return True
        else:
            logger.warning(f"Invalid transition from {current_state} to {target_state}")
            return False
    
    def diagnose(self):
        """Transition to diagnosing state."""
        if TRANSITIONS_AVAILABLE:
            self.diagnose()
        else:
            self._transition_fallback("diagnose", "diagnosing")
        logger.info(f"🔍 Diagnosing: {self.message}")
    
    def heal(self):
        """Transition to healing state."""
        if TRANSITIONS_AVAILABLE:
            self.heal()
        else:
            self._transition_fallback("heal", "healing")
        logger.info(f"🩺 Healing: {self.message}")
    
    def resolve(self):
        """Transition to resolved state."""
        if TRANSITIONS_AVAILABLE:
            self.resolve()
        else:
            self._transition_fallback("resolve", "resolved")
        logger.info(f"✅ Resolved: {self.message}")
    
    def escalate(self):
        """Transition to escalated state and raise exception."""
        if TRANSITIONS_AVAILABLE:
            self.escalate()
        else:
            self._transition_fallback("escalate", "escalated")
        logger.error(f"🚨 Escalated: {self.message}")
        raise self
    
    def reset(self):
        """Reset FSM to initial state."""
        if TRANSITIONS_AVAILABLE:
            self.reset()
        else:
            self.state = "triggered"
            self._record_state("triggered")
        logger.info(f"🔄 Reset FSM: {self.message}")
    
    def get_current_state(self) -> str:
        """Get current FSM state."""
        if TRANSITIONS_AVAILABLE:
            return self.state
        return getattr(self, 'state', 'triggered')
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get complete state transition history.
        
        Returns:
            List of state transition records
        """
        return self.state_history.copy()
    
    def can_transition_to(self, target_state: str) -> bool:
        """Check if transition to target state is allowed.
        
        Args:
            target_state: Target state name
            
        Returns:
            True if transition is allowed
        """
        if TRANSITIONS_AVAILABLE:
            # With transitions library, check available transitions
            current_state = self.state
            available = [t.dest for t in self.machine.get_triggers(current_state)]
            return target_state in available
        else:
            current_state = getattr(self, 'state', 'triggered')
            allowed = self._transitions.get(current_state, [])
            return target_state in allowed
    
    def get_available_transitions(self) -> List[str]:
        """Get list of available transitions from current state.
        
        Returns:
            List of available transition target states
        """
        if TRANSITIONS_AVAILABLE:
            current_state = self.state
            return [t.dest for t in self.machine.get_triggers(current_state)]
        else:
            current_state = getattr(self, 'state', 'triggered')
            return self._transitions.get(current_state, [])
    
    def __str__(self) -> str:
        """String representation with current state."""
        base_str = super().__str__()
        state = self.get_current_state()
        return f"[FSMError] State={state}: {base_str}"

