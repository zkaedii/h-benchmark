# Contributing to ZKAEDI PRIME Engine

Thank you for your interest in contributing!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/zkaedi-prime-engine.git
   cd zkaedi-prime-engine
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=zkaedi_prime_engine --cov-report=html

# Run specific test file
pytest tests/test_zkaedi_prime_engine.py -v
```

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all public functions/classes
- Keep functions focused and small

## Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

4. Commit your changes:
   ```bash
   git commit -m "Add feature: description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request on GitHub

## Pull Request Guidelines

- Include a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed
- Keep PRs focused and reasonably sized

## Questions?

Open an issue for discussion!

