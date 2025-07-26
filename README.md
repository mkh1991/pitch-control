## pitch-control

A very WIP but hopefully high-performance Python library for calculating pitch control in football (soccer) using NumPy vectorization and additional speed-ups upto 10x with Numba.
Based on [Laurie Shaw's](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/tree/master) implementation of the Spearman physics-based model, which is the 
only currently supported model. It also implements a simple model of acceleration if provided for player times calculation.


## Features

- **High Performance**: 🚀10x speedup using Numba JIT compilation (<1 ms per frame) 
- **Numpy Fallback**: Efficient NumPy-based calculations as fallback without Numba
- **Realistic Physics**: Proper acceleration, reaction times, position-specific parameters  
- **Configurable**: Easy to customize parameters and behavior  
- **Rich Visualization**: Built-in plotting and analysis tools  
- **Extensible**: Clean architecture for adding new models  

## Quick Start

```python
from pitch_control.core import Player, Point, Pitch, Position
from pitch_control.models import SpearmanModel

# Create players
home_player = Player("H1", "Messi", "Home", Position.FW, 10)
away_player = Player("A1", "Van Dijk", "Away", Position.CB, 4)

# Set up game state  
players = [
    home_player.create_state(Point(10, 5), Point(2, -1)),
    away_player.create_state(Point(-10, -5), Point(-1, 1))
]

# Calculate pitch control
model = SpearmanModel()
result = model.calculate(players, ball_position=Point(0, 0))

print(f"Calculated in {result.calculation_time:.3f} seconds")
print(f"Home team controls {result.get_team_control_percentage('home'):.1f}% of pitch")
```

## Installation

```bash
# Using uv (recommended)
git clone https://github.com/mkh1991/pitch-control
cd pitch-control
uv sync --all-extras

# Using pip
pip install -r requirements.txt
pip install -e .

# Install optional dependencies
uv add numba          # For JIT acceleration (recommended)
uv add psutil         # For memory usage analysis
uv add plotly bokeh   # For advanced visualizations
```

## Running Examples

### Basic Example
```bash
# Run the main example with sample players
uv run python examples/basic_example.py
```

Expected output should look something like this:

```bash
# Setting up pitch control calculation...
# Created 22 players (11 home, 11 away)
# Calculating pitch control...
# Calculation completed in 0.015 seconds
# Home team controls 52.3% of the pitch
# Away team controls 47.7% of the pitch
```

### Performance Comparison
```bash
# Compare Numba vs NumPy performance across different grid sizes
uv run python examples/performance_comparison.py

# Expected output:
# Performance Comparison: Numba vs NumPy
# =====================================
# 
# Grid Resolution: 42x27 (1134 points)
#   Numba:  0.003s
#   NumPy:  0.045s
#   Speedup: 15.2x
# 
# Grid Resolution: 84x54 (4536 points)
#   Numba:  0.010s
#   NumPy:  0.180s
#   Speedup: 18.1x
# 
# Scalability Test: Performance vs Number of Players
# =================================================
#   8 players (4v4): 0.005s (7,257,600 calculations/sec)
#  12 players (6v6): 0.007s (7,776,000 calculations/sec)
#  16 players (8v8): 0.009s (8,064,000 calculations/sec)
#  22 players (11v11): 0.012s (8,316,000 calculations/sec)
```

## Running Tests

### Complete Test Suite
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=pitch_control --cov-report=html

# Run specific test categories
uv run pytest tests/test_vectorized_model.py -v    # Model tests
uv run pytest tests/test_core.py -v               # Core functionality

# Quick smoke test
uv run pytest tests/ -x --tb=short               # Stop on first failure
```

### Expected Test Results
```bash
# Sample test output:
tests/test_core.py::TestPoint::test_distance_calculation PASSED
tests/test_core.py::TestPlayerPhysics::test_position_specific_physics PASSED
tests/test_vectorized_model.py::TestSpearmanModel::test_basic_calculation PASSED
tests/test_vectorized_model.py::TestSpearmanModel::test_numba_vs_numpy_consistency PASSED
tests/test_vectorized_model.py::TestSpearmanModel::test_performance_improvement PASSED

====================== 15 passed in 2.34s ======================
Coverage report: htmlcov/index.html (90%+ expected)
```

### Performance Benchmarks
```bash
# Run focused performance tests
uv run python -m pytest tests/test_vectorized_model.py::TestSpearmanModel::test_performance_improvement -v -s

```

## Performance

On my Macbook Air M3 laptop (16 GB RAM), with the default grid (105x68):

- **22 players**: ~0.4-1 ms with Numba,  
- **Numba speedup**: 10x vs pure NumPy


```bash
 uv run python examples/performance_comparison.py
```

```commandline
COMPARISON SUMMARY:
==================================================
Component            NumPy (ms)   Numba (ms)   Speedup   
------------------------------------------------------
Physics                   9.5          0.3       33.8x
Ball                      0.6          0.4        1.7x
Probability               2.1          0.4        4.9x
------------------------------------------------------
TOTAL CORE               12.3          1.1       11.2x
  Numba:  1.097s
  NumPy:  12.317s
  Speedup: 11.2x

```



### Benchmarking Your System
```bash
# Quick performance check
uv run python -c "
from examples.basic_example import create_sample_players
from pitch_control.models import SpearmanModel
from pitch_control.core import Point
import time

model = SpearmanModel()
players = create_sample_players()
start = time.time()
result = model.calculate(players, Point(0, 0))
print(f'Your system: {result.calculation_time:.3f}s for 22 players')
print(f'Numba enabled: {result.metadata[\"use_numba\"]}')
"
```

## Models Implemented

### Spearman (2018) - "Beyond Expected Goals"
- Physics-based player movement with realistic acceleration
- Ball travel time integration
- Configurable parameters for different playing styles
- Both Numba and NumPy backends

### Coming Soon
- Other pitch control models:
  - Fernández & Bornn (2018) - "Wide Open Spaces"
  - KNN-based methods
  - Hybrid ML-based models
- Video and streaming support

## Architecture

```
pitch_control/
├── core/           # Core abstractions (Player, Pitch, Physics)
├── models/         # Pitch control model implementations  
├── speedup/   # Performance backends (Numba, GPU (coming soon))
├── utils/          # Visualization and analysis tools
└── api/           # REST API and streaming (coming soon)
```

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Install in development mode
uv pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Numba Compilation Issues**:
```bash
# Check Numba availability
uv run python -c "import numba; print(f'Numba version: {numba.__version__}')"

# Disable Numba if needed (falls back to NumPy)
uv run python -c "
from pitch_control.models import SpearmanConfig, SpearmanModel
config = SpearmanConfig(use_numba=False)
model = SpearmanModel(config=config)
print('NumPy fallback working')
"
```

**Missing Optional Dependencies**:
```bash
uv add psutil        # For memory testing
uv add matplotlib    # For visualization
```

**Performance Issues**:
- First run is slower (Numba compilation)
- Subsequent runs should be around 10x faster
- Check `result.metadata['use_numba']` to confirm Numba is active


## Documentation

- [Quick Start Guide](docs/quickstart.md)
- [API Reference](docs/api.md) 
- [Performance Tuning](docs/performance.md)
- [Examples](examples/)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in research, please cite:

```bibtex
@software{pitch_control,
  title={Pitch Control: High-Performance Football Analytics},
  author={Mayukh Samanta},
  year={2025},
  url={https://github.com/username/pitch-control}
}
```