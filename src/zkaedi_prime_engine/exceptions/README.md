# ZKAEDI PRIME Advanced Exception System

A comprehensive, novel exception handling system with cutting-edge features inspired by autonomous systems, bio-inspired computing, and explainable AI.

## Features

### 🩺 Self-Healing Exceptions
Exceptions that attempt to diagnose, repair, and retry failed operations before escalating.

```python
from zkaedi_prime_engine.exceptions import SelfHealingError

def retry_operation(**context):
    # Your healing logic here
    return True  # Return True if healing successful

error = SelfHealingError(
    "Connection failed",
    heal_strategy=retry_operation,
    operation="database_query"
)

error.escalate()  # Will attempt healing first
```

### 🧬 Evolutionary Exceptions
Bio-inspired exceptions that age, mutate, and evolve over time.

```python
from zkaedi_prime_engine.exceptions import AgeingError

error = AgeingError(
    "Convergence failure",
    age=0,
    mutation_rate=0.2,
    max_age=10
)

error.evolve(generations=5)
evolution_report = error.get_evolution_report()
```

### 🐝 Swarm Intelligence
Multi-agent exception handling where exceptions collaborate and share context.

```python
from zkaedi_prime_engine.exceptions import SwarmError

SwarmError("Backend timeout", operation="query_1")
SwarmError("Backend timeout", operation="query_2")

analytics = SwarmError.analyze_swarm()
recommendations = SwarmError.get_swarm_recommendations()
```

### 🔄 FSM-Based Exceptions
Stateful exceptions modeled as finite state machines.

```python
from zkaedi_prime_engine.exceptions import FSMError

error = FSMError("State computation failed")
error.diagnose()  # triggered -> diagnosing
error.heal()      # diagnosing -> healing
error.resolve()   # healing -> resolved
```

### 🔍 Explainable Exceptions
Exceptions with XAI integration for causal reasoning and feature importance.

```python
from zkaedi_prime_engine.exceptions import ExplainableZKAEDIError

error = ExplainableZKAEDIError(
    "Matrix inversion failed",
    input_vector=[1e10, 0.001, -5.0],
    feature_names=["size", "condition", "tolerance"]
)

explanation = error.explain()
summary = error.get_explanation_summary()
```

### 📊 Metadata-Rich Exceptions
Exceptions with structured metadata for enhanced debugging.

```python
from zkaedi_prime_engine.exceptions import MetadataZKAEDIError

error = MetadataZKAEDIError(
    "Computation failed",
    operation="eigen_solver",
    parameters={"dim": 3, "tolerance": 1e-6}
)

operation = error.get_metadata("operation")
```

### 🔬 Self-Diagnosing Exceptions
Exceptions that perform automatic diagnosis and suggest remediation.

```python
from zkaedi_prime_engine.exceptions import SelfDiagnosingZKAEDIError

error = SelfDiagnosingZKAEDIError(
    "Backend failure",
    type="backend",
    resource_limit=95
)

suggestions = error.remediation_suggestions()
```

### 📈 Exception Tracking & Analytics
Track exception occurrences and analyze patterns.

```python
from zkaedi_prime_engine.exceptions import ExceptionTracker, HamiltonianError

ExceptionTracker.track_exception(HamiltonianError("Error occurred"))
stats = ExceptionTracker.get_statistics()
trends = ExceptionTracker.get_trends()
```

### 🔔 Observability Integration
Built-in support for Sentry, Kafka, and custom observability pipelines.

```python
from zkaedi_prime_engine.exceptions import ObservabilityException

error = ObservabilityException(
    "Critical failure",
    sentry_dsn="your-sentry-dsn",
    kafka_config={
        "bootstrap_servers": ["localhost:9092"],
        "topic": "zkaedi-exceptions"
    }
)
```

### 🎨 Rich Tracebacks
Beautiful, developer-friendly traceback formatting.

```python
from zkaedi_prime_engine.exceptions import install_rich_traceback

install_rich_traceback(show_locals=True)
```

## Installation

### Required Dependencies
```bash
pip install numpy>=1.19.0
```

### Optional Dependencies
For full functionality, install optional dependencies:

```bash
# Rich tracebacks
pip install rich>=13.0.0

# FSM support
pip install transitions>=0.9.0

# Observability
pip install sentry-sdk>=1.0.0
pip install kafka-python>=2.0.0

# Explainability
pip install shap>=0.41.0

# Validation
pip install pydantic>=2.0.0
```

## Usage Examples

### Basic Usage (Backward Compatible)
```python
from zkaedi_prime_engine.exceptions import HamiltonianError

try:
    # Your code here
    raise HamiltonianError("Invalid operator")
except HamiltonianError as e:
    print(f"Error: {e}")
```

### Advanced Usage
```python
from zkaedi_prime_engine.exceptions import (
    SelfHealingError,
    SwarmError,
    FSMError,
    ExplainableZKAEDIError
)

# Self-healing
error = SelfHealingError("Connection failed", heal_strategy=my_healing_func)
error.escalate()  # Attempts healing before raising

# Swarm intelligence
SwarmError("Error 1", swarm_id="backend_errors")
SwarmError("Error 2", swarm_id="backend_errors")
analytics = SwarmError.analyze_swarm()

# FSM workflow
error = FSMError("Computation failed")
error.diagnose()
error.heal()
error.resolve()

# Explainable
error = ExplainableZKAEDIError("Failed", input_vector=[1, 2, 3])
explanation = error.explain()
```

## Exception Hierarchy

Generate the exception hierarchy:

```python
from zkaedi_prime_engine.exceptions import (
    generate_exception_hierarchy,
    print_exception_hierarchy
)

hierarchy = generate_exception_hierarchy()
print_exception_hierarchy()
```

## Running Demos

```python
from zkaedi_prime_engine.exceptions.demo import run_all_demos

run_all_demos()
```

Or from command line:
```bash
python -m zkaedi_prime_engine.exceptions.demo
```

## Architecture

```
exceptions/
├── __init__.py          # Main exports
├── base.py              # Core exception hierarchy
├── healing.py           # Self-healing logic
├── evolution.py         # Aging and mutation
├── swarm.py             # Swarm intelligence
├── fsm.py               # Finite state machines
├── explainable.py       # XAI integration
├── diagnostics.py       # Tracking and observability
├── rich_traceback.py    # Rich formatting
├── demo.py              # Demonstration examples
└── README.md            # This file
```

## Best Practices

1. **Use appropriate exception types** for your domain (HamiltonianError, StateError, etc.)
2. **Add metadata** to exceptions for better debugging
3. **Enable self-healing** for recoverable errors
4. **Track exceptions** for analytics and pattern detection
5. **Use explainable exceptions** for complex failures requiring root cause analysis
6. **Leverage swarm intelligence** for distributed systems

## Contributing

When adding new exception types:
1. Inherit from appropriate base class
2. Add to `__init__.py` exports
3. Update hierarchy documentation
4. Add tests and examples

## License

Part of the ZKAEDI PRIME Engine project.

