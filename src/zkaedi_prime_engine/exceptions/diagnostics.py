"""Diagnostics and Observability for ZKAEDI Exceptions.

Provides exception tracking, analytics, taxonomy generation,
and observability integrations (Sentry, Kafka, etc.).
"""

import json
import logging
import inspect
from collections import Counter
from typing import Dict, Any, List, Optional, Type
from .base import ZKAEDIError

logger = logging.getLogger(__name__)

# Try to import observability libraries
try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    logger.debug("Sentry not available. Observability features will be limited.")

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.debug("Kafka not available. Event streaming features will be limited.")


class ExceptionTracker:
    """Track and analyze exception occurrences.
    
    Provides analytics on exception frequency, patterns, and trends.
    """
    
    count = Counter()
    history: List[Dict[str, Any]] = []
    max_history = 1000  # Limit history size
    
    @classmethod
    def track_exception(cls, exception: ZKAEDIError):
        """Track an exception occurrence.
        
        Args:
            exception: Exception instance to track
        """
        exception_name = exception.__class__.__name__
        cls.count[exception_name] += 1
        
        # Add to history
        cls.history.append({
            "exception": exception_name,
            "message": str(exception),
            "error_code": getattr(exception, 'error_code', 'UNKNOWN'),
            "timestamp": getattr(exception, 'timestamp', None),
            "context": getattr(exception, 'context', {})
        })
        
        # Limit history size
        if len(cls.history) > cls.max_history:
            cls.history = cls.history[-cls.max_history:]
    
    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get exception statistics.
        
        Returns:
            Dictionary with exception statistics
        """
        return {
            "total_exceptions": sum(cls.count.values()),
            "exception_counts": dict(cls.count),
            "most_common": cls.count.most_common(10),
            "unique_types": len(cls.count),
            "history_size": len(cls.history)
        }
    
    @classmethod
    def get_trends(cls, window_size: int = 100) -> Dict[str, Any]:
        """Analyze exception trends over recent history.
        
        Args:
            window_size: Number of recent exceptions to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        recent = cls.history[-window_size:] if len(cls.history) > window_size else cls.history
        
        if not recent:
            return {"message": "Insufficient history for trend analysis"}
        
        recent_counts = Counter(item["exception"] for item in recent)
        
        return {
            "window_size": len(recent),
            "recent_counts": dict(recent_counts),
            "trending_exceptions": recent_counts.most_common(5)
        }
    
    @classmethod
    def clear_history(cls):
        """Clear tracking history (useful for testing)."""
        cls.count.clear()
        cls.history.clear()
        logger.info("Exception tracking history cleared")


class ObservabilityException(ZKAEDIError):
    """Exception with built-in observability integration.
    
    Automatically logs to observability platforms (Sentry, Kafka, etc.).
    """
    
    def __init__(
        self,
        message: str,
        observability_pipeline: Optional[Any] = None,
        sentry_dsn: Optional[str] = None,
        kafka_config: Optional[Dict[str, Any]] = None,
        **context
    ):
        """Initialize observability-aware exception.
        
        Args:
            message: Error message
            observability_pipeline: Custom observability pipeline
            sentry_dsn: Sentry DSN for error tracking
            kafka_config: Kafka configuration for event streaming
            **context: Additional context
        """
        super().__init__(message, **context)
        self.observability_pipeline = observability_pipeline
        self.sentry_dsn = sentry_dsn
        self.kafka_config = kafka_config or {}
        self._log_to_observability()
    
    def _log_to_observability(self):
        """Log exception to observability platforms."""
        # Track in local analytics
        ExceptionTracker.track_exception(self)
        
        # Sentry integration
        if SENTRY_AVAILABLE and self.sentry_dsn:
            try:
                sentry_sdk.init(dsn=self.sentry_dsn, traces_sample_rate=1.0)
                sentry_sdk.capture_exception(self)
                logger.debug("Exception logged to Sentry")
            except Exception as e:
                logger.warning(f"Failed to log to Sentry: {e}")
        
        # Kafka integration
        if KAFKA_AVAILABLE and self.kafka_config:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self.kafka_config.get('bootstrap_servers', ['localhost:9092']),
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                
                event_data = {
                    "exception": self.__class__.__name__,
                    "message": self.message,
                    "error_code": self.error_code,
                    "timestamp": self.timestamp,
                    "context": self.context
                }
                
                topic = self.kafka_config.get('topic', 'zkaedi-exceptions')
                producer.send(topic, value=event_data)
                producer.flush()
                logger.debug(f"Exception event sent to Kafka topic: {topic}")
            except Exception as e:
                logger.warning(f"Failed to send to Kafka: {e}")
        
        # Custom pipeline
        if self.observability_pipeline and callable(self.observability_pipeline):
            try:
                self.observability_pipeline(self)
            except Exception as e:
                logger.warning(f"Custom observability pipeline failed: {e}")


def generate_exception_hierarchy(base_class: Type[ZKAEDIError] = ZKAEDIError) -> Dict[str, str]:
    """Generate exception hierarchy documentation.
    
    Args:
        base_class: Base exception class to analyze
        
    Returns:
        Dictionary mapping exception names to their docstrings
    """
    hierarchy = {}
    
    # Import the exceptions module
    try:
        import zkaedi_prime_engine.exceptions as exc_module
        
        for name, obj in inspect.getmembers(exc_module, inspect.isclass):
            if issubclass(obj, base_class) and obj is not base_class:
                doc = inspect.getdoc(obj) or "No documentation available"
                hierarchy[name] = doc.strip()
    except Exception as e:
        logger.warning(f"Failed to generate exception hierarchy: {e}")
    
    return hierarchy


def print_exception_hierarchy(base_class: Type[ZKAEDIError] = ZKAEDIError):
    """Print exception hierarchy in a formatted way.
    
    Args:
        base_class: Base exception class to analyze
    """
    hierarchy = generate_exception_hierarchy(base_class)
    
    print("\n" + "=" * 70)
    print("ZKAEDI PRIME Exception Hierarchy")
    print("=" * 70)
    
    for name, doc in sorted(hierarchy.items()):
        print(f"\n{name}")
        print(f"  {doc}")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(hierarchy)} exception types")
    print("=" * 70 + "\n")

