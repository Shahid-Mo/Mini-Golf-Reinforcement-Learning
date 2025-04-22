import argparse
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import math

from minigolf_game import MinigolfEnv
from dqn_model import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent on Minigolf')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between steps when rendering (seconds)')
    parser.add_argument('--visualize', action='store_true', help='Generate Q-value visualization')
    return parser.parse_args()

def visualize_action_directions(agent, env, resolution=20):
    """
    Create a visualization showing which direction the agent would hit the ball from different positions
    """
    # Define grid for visualization
    x_range = np.linspace(100, WIDTH-100, resolution)
    y_range = np.linspace(100, HEIGHT-100, resolution)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    
    # Flatten grid for processing
    states = []
    for i in range(resolution):
        for j in range(resolution):
            ball_x = grid_x[i, j] / WIDTH
            ball_y = grid_y[i, j] / HEIGHT
            hole_x = env.hole.x / WIDTH
            hole_y = env.hole.y / HEIGHT
            states.append([ball_x, ball_y, hole_x, hole_y])
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(states).to(agent.device)
    
    # Get best actions
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
        best_actions = q_values.max(1)[1].cpu().numpy()
    
    # Decode actions into angles and powers
    angles = []
    powers = []
    for action in best_actions:
        angle_idx = action // env.n_powers
        power_idx = action % env.n_powers
        
        angle = 2 * math.pi * angle_idx / env.n_angles
        power = (power_idx + 1) * env.max_power / env.n_powers
        
        angles.append(angle)
        powers.append(power)
    
    # Reshape for plotting
    angles = np.array(angles).reshape(resolution, resolution)
    powers = np.array(powers).reshape(resolution, resolution)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot directions
    ax = axes[0]
    quiver = ax.quiver(grid_x, grid_y, np.cos(angles), np.sin(angles), 
                       powers, cmap='viridis', scale=30, pivot='mid')
    
    # Add hole
    hole = Circle((env.hole.x, env.hole.y), env.hole.radius, fill=True, color='red')
    ax.add_patch(hole)
    
    # Set limits and title
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_title('Agent Shot Directions')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    # Add colorbar
    cbar = fig.colorbar(quiver, ax=ax)
    cbar.set_label('Shot Power')
    
    # Plot power heatmap
    im = axes[1].imshow(powers, cmap='plasma', origin='lower', 
                     extent=[100, WIDTH-100, 100, HEIGHT-100])
    axes[1].set_title('Shot Power Heatmap')
    axes[1].set_xlabel('X position')
    axes[1].set_ylabel('Y position')
    
    # Add hole to power plot
    hole = Circle((env.hole.x, env.hole.y), env.hole.radius, fill=True, color='black', alpha=0.7)
    axes[1].add_patch(hole)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label('Shot Power')
    
    plt.tight_layout()
    plt.savefig('action_visualization.png')
    plt.close()

def evaluate_agent(agent, env, num_episodes=20, render=False, delay=0.0):
    """
    Evaluate agent performance and print statistics
    """
    rewards = []
    successes = 0
    shots_to_hole = []
    min_distances = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        min_distance = float('inf')
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        while not done:
            # Select best action (no exploration)
            action = agent.select_action(state, eval_mode=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Calculate distance to hole
            distance = info["distance_to_hole"]
            min_distance = min(min_distance, distance)
            
            # Update metrics
            episode_reward += reward
            state = next_state
            
            # Print step info
            print(f"  Shot {info['shots']}: Distance to hole = {distance:.2f}")
            
            # Render if needed
            if render:
                env.render()
                time.sleep(delay)  # Add delay to see the movements
        
        # Record episode results
        rewards.append(episode_reward)
        min_distances.append(min_distance)
        
        if info["in_hole"]:
            successes += 1
            shots_to_hole.append(info["shots"])
            print(f"  SUCCESS! Ball in hole after {info['shots']} shots.")
        else:
            print(f"  FAILED. Minimum distance to hole: {min_distance:.2f}")
    
    # Calculate and print statistics
    success_rate = successes / num_episodes
    avg_reward = sum(rewards) / num_episodes
    avg_shots = sum(shots_to_hole) / successes if successes > 0 else float('inf')
    avg_min_distance = sum(min_distances) / num_episodes
    
    print("\n===== Evaluation Results =====")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    if successes > 0:
        print(f"Average shots for success: {avg_shots:.2f}")
    print(f"Average minimum distance to hole: {avg_min_distance:.2f}")
    
    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_shots": avg_shots,
        "avg_min_distance": avg_min_distance
    }

def main():
    args = parse_args()
    
    # Create environment
    render_mode = "human" if args.render else None
    env = MinigolfEnv(render_mode=render_mode)
    
    # Get constants from environment for visualization
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = 800, 600
    
    # Initialize agent
    agent = DQNAgent(env.state_size, env.action_size)
    
    # Load model
    if not agent.load_model(args.model):
        print(f"Error loading model from {args.model}. Exiting.")
        return
    
    # Set agent to evaluation mode
    agent.epsilon = 0  # No exploration
    
    # Create visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        visualize_action_directions(agent, env)
        print("Visualizations saved.")
    
    # Evaluate agent
    print(f"\nEvaluating agent over {args.episodes} episodes...")
    evaluate_agent(agent, env, num_episodes=args.episodes, render=args.render, delay=args.delay)

if __name__ == "__main__":
    main()