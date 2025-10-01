# Drop & Merge - Physics Puzzle Game

A sophisticated 2D physics-based puzzle game built with Python and Pygame, featuring realistic collision detection, particle effects, and progressive difficulty scaling.

## Features

- **Realistic Physics Simulation**: Advanced collision detection and resolution with momentum conservation
- **Particle Effects System**: Dynamic visual feedback for merges and interactions  
- **Progressive Difficulty**: Logarithmic scaling ensures balanced gameplay progression
- **High Score Persistence**: Automatic saving and loading of player achievements
- **Smooth Animations**: Interpolated scaling, glow effects, and bounce animations
- **Professional UI**: Hover effects, state management, and responsive interface
- **Performance Optimized**: Multi-pass collision detection running at 120 FPS

## Installation

### Prerequisites

- Python 3.7 or higher
- Pygame library

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ManiHasani/mergigo.git
cd mergigo
```

2. Install dependencies:
```bash
pip install pygame
```

3. Run the game:
```bash
python mergigo.py
```

## How to Play

1. **Drop Circles**: Click anywhere in the play area or press `SPACE` to drop numbered circles
2. **Merge Strategy**: Identical numbers merge when they collide, creating the next power of 2
3. **Score Points**: Higher value merges yield exponentially more points
4. **Avoid Overflow**: Keep circles below the danger line to prevent game over
5. **Beat Your Record**: High scores are automatically saved between sessions

## Game Mechanics

### Physics System
- Gravity-based movement with realistic acceleration
- Collision detection using spatial partitioning for performance
- Momentum conservation in circle-to-circle impacts
- Ground friction and bounce dampening for natural behavior

### Scoring Algorithm
```python
score_bonus = new_value * (1 + log2(new_value) * 0.1)
```

### Value Progression
Starting values: 2, 4, 8, 16 (weighted probability distribution)
Maximum merge value: 1024

## Controls

| Input | Action |
|-------|--------|
| Mouse Click | Drop circle at cursor position |
| `SPACE` | Drop circle at current position |
| `ESC` | Quit game |

## Technical Architecture

### Core Classes

- **`GameEngine`**: Main game loop and system coordination
- **`GameCircle`**: Physics object with collision detection and rendering
- **`ParticleEffect`**: Visual effects system for dynamic feedback
- **`UIButton`**: Interactive interface elements with state management
- **`ScoreManager`**: Persistent high score tracking and file I/O
- **`Config`**: Centralized configuration management

### Performance Features

- Vector-based physics using `pygame.math.Vector2`
- Multi-pass collision resolution for accuracy
- Alpha-blended particle rendering
- Optimized gradient rendering with `pygame.gfxdraw`

## Configuration

Game parameters can be modified in the `Config` class:

```python
class Config:
    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 700
    FPS = 120
    GRAVITY = 0.3
    MAX_VELOCITY = 8.0
    # ... additional parameters
```

## Development

### Code Style
- Type hints throughout codebase
- Comprehensive docstrings following Google style
- Modular architecture with clear separation of concerns
- Professional error handling and logging

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Sound effects and background music
- [ ] Multiple game modes (time attack, endless, puzzle)
- [ ] Power-ups and special circle types
- [ ] Online leaderboards
- [ ] Mobile touch controls
- [ ] Customizable themes and skins

## Acknowledgments

- Built with Python and Pygame
- Physics simulation inspired by real-world particle dynamics
- UI design following modern game interface standards

---

*Developed with ❤️ using Python and Pygame*