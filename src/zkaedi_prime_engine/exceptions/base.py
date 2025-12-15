"""Core ZKAEDI PRIME Engine Exception Hierarchy.

Provides base exception classes with enhanced context and metadata support.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class ZKAEDIError(Exception):
    """Base exception for all ZKAEDI PRIME Engine errors.
    
    Enhanced with context tracking, timestamps, and diagnostic capabilities.
    """
    
    def __init__(self, message: str, **context):
        """Initialize exception with message and optional context.
        
        Args:
            message: Error message
            **context: Additional context metadata
        """
        super().__init__(message)
        self.message = message
        self.context = context
        self.timestamp = time.time()
        self.error_code = self._generate_error_code()
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code based on exception class name."""
        class_name = self.__class__.__name__
        # Simple hash-based code generation
        code_hash = hash(class_name) % 10000
        return f"E{code_hash:04d}"
    
    def log(self):
        """Log exception with context."""
        print(f"[{self.error_code}] {self.__class__.__name__}: {self.message}")
        if self.context:
            print(f"  Context: {self.context}")
    
    def __str__(self) -> str:
        """String representation with error code."""
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context})"


class MetadataZKAEDIError(ZKAEDIError):
    """Exception with structured metadata support.
    
    Allows attaching rich metadata that can be used for debugging,
    observability, and automated error handling.
    """
    
    def __init__(self, message: str, **metadata):
        """Initialize with message and structured metadata.
        
        Args:
            message: Error message
            **metadata: Structured metadata dictionary
        """
        super().__init__(message, **metadata)
        self.metadata = metadata
    
    def __str__(self) -> str:
        """String representation with metadata."""
        base_str = super().__str__()
        if self.metadata:
            return f"{base_str} | Metadata: {self.metadata}"
        return base_str
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)


class SelfDiagnosingZKAEDIError(ZKAEDIError):
    """Exception that performs self-diagnosis and suggests remediation.
    
    Attempts to diagnose the root cause and provide actionable suggestions.
    """
    
    def __init__(self, message: str, **diagnostics):
        """Initialize with message and diagnostic context.
        
        Args:
            message: Error message
            **diagnostics: Diagnostic information (type, resource_limit, etc.)
        """
        super().__init__(message, **diagnostics)
        self.diagnostics = diagnostics
        self.remediation_hints = []
        self._attempt_auto_diagnosis()
    
    def _attempt_auto_diagnosis(self):
        """Perform automatic diagnosis based on diagnostics context."""
        # Check for backend resource issues
        if self.diagnostics.get("type") == "backend":
            resource_limit = self.diagnostics.get("resource_limit", 0)
            if resource_limit > 80:
                self.remediation_hints.append(
                    "⚠️ Diagnostic Hint: Backend under heavy load. "
                    "Consider scaling up resources or reducing concurrent operations."
                )
        
        # Check for memory issues
        if self.diagnostics.get("type") == "memory":
            memory_usage = self.diagnostics.get("memory_usage_mb", 0)
            if memory_usage > 1000:  # 1GB threshold
                self.remediation_hints.append(
                    "⚠️ Diagnostic Hint: High memory usage detected. "
                    "Consider using sparse backends or reducing state size."
                )
        
        # Check for convergence issues
        if self.diagnostics.get("type") == "convergence":
            iterations = self.diagnostics.get("iterations", 0)
            if iterations > 1000:
                self.remediation_hints.append(
                    "⚠️ Diagnostic Hint: Convergence taking too long. "
                    "Consider adjusting tolerance or initial conditions."
                )
    
    def remediation_suggestions(self) -> str:
        """Get formatted remediation suggestions."""
        hints = "\n".join(self.remediation_hints) if self.remediation_hints else "No specific hints available."
        return f"💡 Suggested Fixes:\n{hints}\n📚 Documentation: https://zkaedi.engine/debugging"
    
    def __str__(self) -> str:
        """String representation with diagnostic hints."""
        base_str = super().__str__()
        if self.remediation_hints:
            return f"{base_str}\n{self.remediation_suggestions()}"
        return base_str


# Domain-specific exceptions

class HamiltonianError(ZKAEDIError):
    """Raised when Hamiltonian operations fail."""
    pass


class EngineInitializationError(ZKAEDIError):
    """Raised when engine initialization fails."""
    pass


class StateError(ZKAEDIError):
    """Raised when state operations fail."""
    
    def __init__(self, message: str, state_id: Optional[int] = None, **context):
        """Initialize with optional state ID.
        
        Args:
            message: Error message
            state_id: Optional state identifier
            **context: Additional context
        """
        super().__init__(message, state_id=state_id, **context)
        self.state_id = state_id


class QECError(ZKAEDIError):
    """Raised when quantum error correction fails."""
    pass


class BackendError(ZKAEDIError):
    """Raised when backend operations fail."""
    pass

