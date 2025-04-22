# DQN for Minigolf

This project implements a Deep Q-Network (DQN) reinforcement learning agent to play a simple 2D minigolf game built with Pygame.

## Project Structure

- `minigolf_game.py` - The game environment with Pygame for visualization
- `dqn_model.py` - DQN agent implementation with PyTorch
- `train.py` - Training script with Weights & Biases integration
- `evaluate.py` - Evaluation script with visualization tools

## Requirements

```
pygame
numpy
torch
wandb
tqdm
matplotlib
```

Install with:
```bash
pip install pygame numpy torch wandb tqdm matplotlib
```

## Features

### Game Environment
- Simple 2D minigolf game with physics
- Ball with realistic momentum and friction
- Customizable playing field
- Reinforcement learning interface

### DQN Implementation
- Experience replay buffer
- Target network for stable learning
- Double DQN implementation
- Gradient tracking and clipping
- Configurable hyperparameters

### Training
- Weights & Biases integration for experiment tracking
- Detailed metrics logging:
  - Rewards
  - Success rates
  - Q-value statistics
  - Gradient norms
- Checkpointing and model saving
- Periodic evaluation
- Visualizations of learned policy

### Evaluation
- Performance statistics
- Visualization of agent's policy
- Action direction and power visualization

## Usage

### Play the game manually
```bash
python minigolf_game.py
```

### Train the DQN agent
```bash
python train.py --epochs 1000 --save-interval 100
```

Add `--render` to see the training process visually.

### Evaluate a trained agent
```bash
python evaluate.py --model checkpoints/minigolf_dqn_final.pth --episodes 20 --render
```

Add `--visualize` to generate action direction and power visualizations.

## Hyperparameter Tuning

You can customize the DQN agent's hyperparameters via command line arguments:

```bash
python train.py --lr 0.0005 --gamma 0.98 --epsilon-decay 0.997 --batch-size 128 --hidden-size 256
```

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking. Key features:

- Automatic tracking of training metrics
- Gradient tracking to monitor learning stability
- Visualization of Q-values and policy
- Model checkpoints saved to W&B

To disable W&B logging:
```bash
python train.py --no-wandb
```

## Extending the Project

Here are some ways to extend this project:

1. Try different architectures for the DQN model
2. Implement prioritized experience replay
3. Add noise to the environment for more robust learning
4. Create more complex minigolf courses
5. Implement other RL algorithms (PPO, SAC, etc.)
6. Add multi-agent capabilities for competitive play

## Visualization Examples

During training and evaluation, the project generates visualizations of the agent's policy:

- Q-value distribution across the state space
- Action direction visualization showing how the agent would hit the ball from different positions
- Shot power heatmap showing how hard the agent would hit from different locations

These visualizations provide insights into the agent's learned strategy and help debug its behavior.