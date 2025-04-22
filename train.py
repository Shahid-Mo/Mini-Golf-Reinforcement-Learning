import argparse
import os
import time
import numpy as np
import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from minigolf_game import MinigolfEnv
from dqn_model import DQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN agent for Minigolf')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model interval')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to load checkpoint')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--update-freq', type=int, default=10, help='Target network update frequency')
    
    return parser.parse_args()

def evaluate(agent, env, num_episodes=10):
    """
    Evaluate agent performance over several episodes
    """
    total_rewards = []
    success_count = 0
    shot_counts = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, eval_mode=True)  # No exploration during evaluation
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        if info["in_hole"]:
            success_count += 1
            shot_counts.append(info["shots"])
    
    avg_reward = sum(total_rewards) / num_episodes
    success_rate = success_count / num_episodes
    avg_shots = sum(shot_counts) / success_count if success_count > 0 else float('inf')
    
    return {
        "eval/avg_reward": avg_reward,
        "eval/success_rate": success_rate,
        "eval/avg_shots": avg_shots
    }

def visualize_q_values(agent, env, num_samples=1000):
    """
    Create a visualization of Q-values across the state space
    """
    # Sample random states by randomly placing the ball on the course
    states = []
    for _ in range(num_samples):
        # Create a random position for the ball within the course boundaries
        ball_x = np.random.uniform(100, 700) / 800  # Normalize to [0,1]
        ball_y = np.random.uniform(100, 500) / 600  # Normalize to [0,1]
        
        # Use fixed hole position
        hole_x = env.hole.x / 800
        hole_y = env.hole.y / 600
        
        states.append([ball_x, ball_y, hole_x, hole_y])
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(states).to(agent.device)
    
    # Get Q-values
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
    
    # Get max Q-value and best action for each state
    max_q_values, best_actions = q_values.max(1)
    
    # Convert to numpy for plotting
    states_np = np.array(states)
    max_q_values_np = max_q_values.cpu().numpy()
    best_actions_np = best_actions.cpu().numpy()
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot max Q-values
    scatter = plt.scatter(
        states_np[:, 0] * 800,  # Convert back to pixel coordinates
        states_np[:, 1] * 600,
        c=max_q_values_np,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    
    # Add colorbar
    plt.colorbar(scatter, label='Max Q-value')
    
    # Mark the hole position
    plt.scatter(
        [env.hole.x], 
        [env.hole.y], 
        color='red', 
        marker='o', 
        s=100, 
        label='Hole'
    )
    
    # Add labels and title
    plt.title('Max Q-values across the state space')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    
    # Save plot
    plt.savefig('q_value_visualization.png')
    
    # If wandb is active, log the plot
    if wandb.run is not None:
        wandb.log({"q_value_visualization": wandb.Image('q_value_visualization.png')})
    
    plt.close()

def train():
    args = parse_args()
    
    # Initialize environment
    render_mode = "human" if args.render else None
    env = MinigolfEnv(render_mode=render_mode)
    
    # Configure DQN hyperparameters
    config = {
        "hidden_size": args.hidden_size,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "epsilon_decay": args.epsilon_decay,
        "batch_size": args.batch_size,
        "target_update_freq": args.update_freq
    }
    
    # Initialize DQN agent
    agent = DQNAgent(env.state_size, env.action_size, config)
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.isfile(args.checkpoint):
        agent.load_model(args.checkpoint)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="minigolf-dqn",
            config={
                "env": "Minigolf",
                "epochs": args.epochs,
                **config
            }
        )
        # Watch model gradients and parameters
        wandb.watch(agent.policy_net, log="all", log_freq=args.log_interval)
    
    # Track training metrics
    episode_rewards = []
    success_count = 0
    total_steps = 0
    training_start_time = time.time()
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    for episode in tqdm(range(args.epochs)):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        episode_start_time = time.time()
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action and observe next state and reward
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay memory
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update policy network
            loss = agent.update_model()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Render if enabled
            if args.render:
                env.render()
        
        # Update target network if needed
        if episode % args.update_freq == 0:
            agent.update_target_network()
        
        # Update epsilon
        current_epsilon = agent.update_epsilon()
        
        # Track if the episode was successful (ball in hole)
        success = info.get("in_hole", False)
        if success:
            success_count += 1
        
        # Calculate episode metrics
        episode_duration = time.time() - episode_start_time
        avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0
        success_rate = success_count / (episode + 1)
        
        # Add reward to history
        episode_rewards.append(episode_reward)
        
        # Log metrics
        if not args.no_wandb:
            log_dict = {
                "train/episode": episode,
                "train/reward": episode_reward,
                "train/avg_loss": avg_loss,
                "train/epsilon": current_epsilon,
                "train/episode_duration": episode_duration,
                "train/success": int(success),
                "train/success_rate": success_rate,
                "train/steps": total_steps
            }
            
            # Add moving averages
            if episode >= 10:
                log_dict["train/reward_avg10"] = sum(episode_rewards[-10:]) / 10
            if episode >= 100:
                log_dict["train/reward_avg100"] = sum(episode_rewards[-100:]) / 100
            
            wandb.log(log_dict)
        
        # Print progress
        if episode % args.log_interval == 0:
            avg_reward = sum(episode_rewards[-args.log_interval:]) / args.log_interval
            print(f"Episode: {episode}")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Epsilon: {current_epsilon:.3f}")
            print(f"  Success rate: {success_rate:.2f}")
            print(f"  Training steps: {total_steps}")
            print(f"  Avg loss: {avg_loss:.5f}")
        
        # Evaluate agent periodically
        if episode % args.eval_interval == 0:
            print(f"\nEvaluating agent at episode {episode}...")
            eval_env = MinigolfEnv()  # Create a separate environment for evaluation
            eval_metrics = evaluate(agent, eval_env)
            print(f"  Eval success rate: {eval_metrics['eval/success_rate']:.2f}")
            print(f"  Eval avg reward: {eval_metrics['eval/avg_reward']:.2f}")
            print(f"  Eval avg shots: {eval_metrics['eval/avg_shots']:.2f}")
            
            if not args.no_wandb:
                wandb.log(eval_metrics, step=total_steps)
                
                # Create Q-value visualization every few eval intervals
                if episode % (args.eval_interval * 5) == 0:
                    visualize_q_values(agent, eval_env)
        
        # Save model checkpoint
        if episode % args.save_interval == 0:
            checkpoint_path = f"checkpoints/minigolf_dqn_ep{episode}.pth"
            agent.save_model(checkpoint_path)
            
            if not args.no_wandb:
                wandb.save(checkpoint_path)  # Save to wandb as well
    
    # Save final model
    final_checkpoint_path = "checkpoints/minigolf_dqn_final.pth"
    agent.save_model(final_checkpoint_path)
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed!")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Total steps: {total_steps}")
    print(f"Final success rate: {success_count / args.epochs:.2f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_env = MinigolfEnv(render_mode="human" if args.render else None)
    final_metrics = evaluate(agent, eval_env, num_episodes=20)
    print(f"Final success rate: {final_metrics['eval/success_rate']:.2f}")
    print(f"Final avg reward: {final_metrics['eval/avg_reward']:.2f}")
    print(f"Final avg shots: {final_metrics['eval/avg_shots']:.2f}")
    
    if not args.no_wandb:
        # Log final metrics and finish wandb run
        wandb.log({"final/" + k.split("/")[1]: v for k, v in final_metrics.items()})
        wandb.finish()
    
    # Close environment
    env.close()


if __name__ == "__main__":
    train()