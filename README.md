# Pitch Control - High-Performance Football Analytics

A modern, high-performance Python library for calculating pitch control in football (soccer) using advanced vectorization and just-in-time compilation.

## Features

🚀 **High Performance**: 10-100x speedup using Numba JIT compilation  
⚡ **Vectorized Operations**: Efficient NumPy-based calculations  
🧠 **Realistic Physics**: Proper acceleration, reaction times, position-specific parameters  
🎛️ **Configurable**: Easy to customize parameters and behavior  
📊 **Rich Visualization**: Built-in plotting and analysis tools  
🔧 **Extensible**: Clean architecture for adding new models  

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
uv add pitch-control

# Using pip
pip install pitch-control

# Development installation
git clone https://github.com/username/pitch-control
cd pitch-control
uv sync --all-extras
```

## Performance

On a modern laptop with the default grid (105x68):

- **22 players**: ~0.01-0.02 seconds  
- **Numba speedup**: 10-50x vs pure NumPy
- **Memory usage**: <100MB for full game analysis

## Models Implemented

### Spearman (2018) - "Beyond Expected Goals"
- Physics-based player movement with realistic acceleration
- Ball travel time integration
- Configurable parameters for different playing styles
- Both Numba and NumPy backends

### Coming Soon
- Fernández & Bornn (2018) - "Wide Open Spaces"
- Hybrid ML-enhanced models
- Real-time streaming support

## Architecture

```
pitch_control/
├── core/           # Core abstractions (Player, Pitch, Physics)
├── models/         # Pitch control model implementations  
├── acceleration/   # Performance backends (Numba, GPU)
├── utils/          # Visualization and analysis tools
└── api/           # REST API and streaming (coming soon)
```

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
  author={Your Name},
  year={2024},
  url={https://github.com/username/pitch-control}
}
```