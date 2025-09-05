# 🌍 World Simulation – Full Documentation

This is the full project documentation moved from the repository root. For a concise overview, see the root `README.md`.

# 🌍 World Simulation

> **An Advanced Artificial Life Simulation Engine with Real-Time Visualization**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/Ilia01/World_Simulation)

## 🚀 Overview

The World Simulation is a sophisticated artificial life simulation that demonstrates advanced software engineering principles, real-time data visualization, and evolutionary algorithms. This project showcases expertise in:

- **Complex System Design**: Multi-agent simulation with emergent behaviors
- **Real-Time Visualization**: Both terminal and web-based interfaces with live data updates
- **Evolutionary Algorithms**: Genetic mutations and natural selection
- **Performance Optimization**: Efficient numpy operations and data structures
- **Modern Web UI**: Interactive Streamlit dashboard with real-time analytics

## ✨ Features

### 🧬 Evolutionary Simulation
- **Genetic Traits**: Vision, speed, metabolism, and reproduction parameters
- **Natural Selection**: Creatures evolve through survival and reproduction
- **Mutation System**: Configurable mutation rates for trait evolution
- **Lineage Tracking**: Family tree monitoring across generations

### 📊 Real-Time Analytics
- **Live Population Charts**: ASCII-based trend visualization
- **Performance Metrics**: Birth/death rates, energy consumption, evolution events
- **Creature Details**: Individual statistics and trait analysis
- **Event Logging**: Comprehensive simulation history tracking

### 🎨 Professional Interface
- **Rich Terminal UI**: Beautiful, responsive design using Rich library
- **Dynamic Layouts**: Adaptive sizing based on terminal dimensions
- **Color-Coded Events**: Intuitive visual feedback for different event types
- **Progress Indicators**: Real-time progress bars and status updates

### ⚙️ Configurable Parameters
- **World Size**: Adjustable grid dimensions (10x10 to 30x30)
- **Population Control**: Configurable initial creature count
- **Resource Management**: Food spawn rates and expiration
- **Evolution Settings**: Mutation rates and trait boundaries

## 🛠️ Technology Stack

- **Python 3.8+**: Modern Python with type hints and dataclasses
- **NumPy**: High-performance numerical computing
- **Rich**: Beautiful terminal formatting and layouts
- **Streamlit**: Modern web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **CSV Export**: Data persistence and analysis capabilities

## 📋 Requirements

```bash
Python 3.8 or higher
```

All dependencies are listed in `requirements.txt` and will be installed automatically.

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ilia01/World_Simulation.git
   cd World_Simulation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the terminal simulation**
   ```bash
   python main.py
   ```

4. **Run the web dashboard**
   ```bash
   streamlit run app.py
   ```
   Then open the provided local URL in your browser. Use the sidebar to initialize, step, run/pause, and export CSV logs.

## 🎮 Usage

### Quick Start (Terminal)
```bash
python main.py
```
Press `ENTER` to begin with default settings, or `C` to configure parameters.

### Quick Start (Web)
```bash
streamlit run app.py
```
Use the sidebar to configure parameters. Click Init to create a world, Run/Pause to animate, Step to advance one tick, and Export to save `events.csv` and `creatures.csv`.

### Configuration Options
- **World Size**: 10x10 to 50x50 grid
- **Initial Creatures**: 1-30 starting population
- **Mutation Rate**: 0.01 to 0.20 (evolution speed)
- **Refresh Rate**: 1-20 Hz (visualization speed)
- **Food Spawn**: Configurable food spawn rates and expiration

### Controls
- **Enter**: Start simulation
- **C**: Configure parameters
- **Ctrl+C**: Interrupt simulation
- **Enter**: Continue after events

## 🏗️ Architecture

### Core Components

```
World_Simulation/
├── world_sim/                  # Python package
│   ├── core/
│   │   ├── world.py            # Simulation engine (pure logic)
│   │   ├── entities/
│   │   │   ├── creature.py     # Creature entity (strategies injected)
│   │   │   └── traits.py       # Traits value object
│   │   └── behaviors/
│   │       ├── movement.py     # Movement behavior
│   │       ├── senses.py       # Sensing behavior
│   │       └── reproduction.py # Reproduction behavior
│   ├── logging/
│   │   └── logger.py           # Structured CSV logging
│   └── ui/
│       ├── cli.py              # Terminal UI controller
│       └── web.py              # Streamlit web UI
├── main.py                     # CLI entrypoint (thin wrapper)
└── app.py                      # Web entrypoint (thin wrapper)
```

### Class Structure

- **`World`**: Simulation environment and grid management (no UI concerns)
- **`Creature`**: Agent with pluggable behaviors
- **`Traits`**: Genetic characteristics
- **`MovementBehavior` / `SensingBehavior` / `ReproductionBehavior`**: Strategy classes
- **`WorldLogger`**: Event tracking and data export
- **`SimulationController`**: Terminal UI controller

### Data Flow

1. **Initialization**: World creation and creature spawning
2. **Simulation Loop**: Tick-based updates with creature behaviors
3. **Evolution**: Reproduction, mutation, and natural selection
4. **Visualization**: Real-time UI updates and data display
5. **Logging**: Event recording and performance metrics

## 📈 Performance Features

- **Vectorized Operations**: NumPy-based food finding algorithms
- **Efficient Grid Management**: Optimized spatial data structures
- **Memory Management**: Controlled object lifecycle and cleanup
- **Real-Time Updates**: Smooth visualization with configurable refresh rates

## 🎯 Key Algorithms

### Food Finding
```python
def find_closest_food(self, world_grid, rng):
    # Vectorized distance calculation using NumPy
    # Efficient scanning within vision radius
    # Manhattan distance optimization
```

### Movement System
```python
def move_towards(self, target, world_size, rng):
    # Intelligent pathfinding towards food
    # Random walk when no target available
    # Speed-based movement calculations
```

### Evolution Engine
```python
def reproduce(self, world_grid, world_size, rng, mutation_rate):
    # Genetic trait inheritance
    # Configurable mutation system
    # Energy-based reproduction costs
```

## 🔬 Scientific Applications

This simulation demonstrates concepts from:
- **Artificial Life**: Emergent behaviors and self-organization
- **Evolutionary Biology**: Natural selection and genetic drift
- **Complex Systems**: Multi-agent interactions and emergent properties
- **Ecology**: Population dynamics and resource competition

## 📊 Data Export

The simulation automatically exports:
- **Event Logs**: CSV format with timestamps and details
- **Population Data**: Historical trends and statistics
- **Performance Metrics**: Runtime analysis and efficiency data
- **Creature Histories**: Individual life cycle tracking

## 🎨 Customization

### Adding New Traits
```python
@dataclass
class Traits:
    vision: int = 5
    speed: int = 1
    metabolism: float = 0.1
    # Add new traits here
    intelligence: float = 0.5
    social_behavior: float = 0.3
```

### Modifying Behaviors
```python
def custom_behavior(self, world_grid, world_size, rng):
    # Implement custom creature logic
    # Modify movement patterns
    # Add new interaction types
```

## 🚧 Future Enhancements

- [ ] **3D Visualization**: Web-based 3D rendering
- [ ] **Machine Learning**: AI-driven creature behaviors
- [ ] **Network Effects**: Multi-world interactions
- [ ] **Advanced Genetics**: Chromosome-based inheritance
- [ ] **Environmental Factors**: Climate and terrain effects

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/Ilia01/World_Simulation.git
cd World_Simulation
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📞 Contact

- **GitHub**: [@Ilia01](https://github.com/Ilia01)
- **Project**: [World Simulation](https://github.com/Ilia01/World_Simulation)


