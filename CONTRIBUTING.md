# Contributing to spatial-gpu

We welcome contributions to spatial-gpu! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/spatial-gpu.git
   cd spatial-gpu
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=spatialgpu --cov-report=html

# Run specific test file
pytest tests/test_graph.py

# Run specific test
pytest tests/test_graph.py::TestSpatialNeighbors::test_knn_graph
```

### Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black spatialgpu tests

# Lint
ruff check spatialgpu tests

# Type check
mypy spatialgpu
```

### Running Benchmarks

```bash
# Quick benchmark
python examples/quickstart.py

# Full comparison with Squidpy
python examples/benchmark_comparison.py
```

## Pull Request Guidelines

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests** for new functionality

3. **Update documentation** if needed

4. **Ensure all tests pass**:
   ```bash
   pytest tests/
   ```

5. **Format your code**:
   ```bash
   black spatialgpu tests
   ruff check --fix spatialgpu tests
   ```

6. **Write a clear commit message** describing your changes

7. **Open a pull request** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/benchmarks if applicable

## Code Organization

```
spatial-gpu/
├── spatialgpu/
│   ├── core/           # Backend, config, array utilities
│   ├── graph/          # Spatial graph operations
│   ├── segmentation/   # Cell segmentation
│   ├── visualization/  # Plotting functions
│   ├── io/             # Data readers/writers
│   └── benchmarks/     # Benchmarking utilities
├── tests/              # Test files
├── examples/           # Example scripts
└── docs/               # Documentation
```

## Adding New Features

### Adding a New Graph Operation

1. Add the function to `spatialgpu/graph/` (appropriate file)
2. Implement both CPU and GPU versions
3. Export from `spatialgpu/graph/__init__.py`
4. Add tests in `tests/test_graph.py`
5. Update documentation

### Adding a New Segmentation Model

1. Create model class in `spatialgpu/segmentation/models.py`
2. Inherit from `BaseSegmentationModel`
3. Implement `segment()` method
4. Add to `get_available_models()`
5. Add tests

## Reporting Issues

When reporting issues, please include:

- spatial-gpu version (`sp.__version__`)
- Python version
- Operating system
- GPU model (if applicable)
- Minimal reproducible example
- Full error traceback

## Questions?

Feel free to open an issue for questions or join discussions in:
- GitHub Issues
- GitHub Discussions

Thank you for contributing!
