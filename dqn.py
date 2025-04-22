import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import wandb

# Experience tuple to store in replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQNModel, self).__init__()
        # Architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, config=None):
        # Default configuration
        self.config = {
            "hidden_size": 128,
            "learning_rate": 0.001,
            "gamma": 0.99,      # Discount factor
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "buffer_size": 10000,
            "batch_size": 64,
            "target_update_freq": 10,
            "gradient_clip": 1.0
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Q networks
        self.policy_net = DQNModel(
            state_size, 
            action_size, 
            self.config["hidden_size"]
        ).to(self.device)
        
        self.target_net = DQNModel(
            state_size, 
            action_size, 
            self.config["hidden_size"]
        ).to(self.device)
        
        # Copy weights from policy to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # Experience replay memory
        self.memory = ReplayMemory(self.config["buffer_size"])
        
        # Exploration parameters
        self.epsilon = self.config["epsilon_start"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.epsilon_end = self.config["epsilon_end"]
        
        # Training metrics for tracking
        self.training_step = 0
    
    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy
        """
        # Convert state to tensor if it's not already
        if isinstance(state, list) or isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # In evaluation mode, always select best action
        if eval_mode or random.random() > self.epsilon:
            # Exploit: select best action
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            # Explore: select random action
            return random.randrange(self.action_size)
    
    def update_epsilon(self):
        """
        Decay epsilon according to schedule
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay memory
        """
        # Convert to tensors if they're not already
        if isinstance(state, list) or isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(action, int):
            action = torch.tensor([action], device=self.device)
        if isinstance(reward, (int, float)):
            reward = torch.tensor([reward], dtype=torch.float, device=self.device)
        if isinstance(next_state, list) or isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if isinstance(done, bool):
            done = torch.tensor([done], dtype=torch.float, device=self.device)
        
        # Store in replay memory
        self.memory.push(state, action, reward, next_state, done)
    
    def update_model(self):
        """
        Update model weights using batched experience replay
        """
        if len(self.memory) < self.config["batch_size"]:
            return None  # Not enough samples for update
        
        # Sample batch of experiences
        experiences = self.memory.sample(self.config["batch_size"])
        batch = Experience(*zip(*experiences))
        
        # Extract components
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
        action_batch = torch.cat([a.unsqueeze(0) for a in batch.action])
        reward_batch = torch.cat([r.unsqueeze(0) for r in batch.reward])
        next_state_batch = torch.cat([ns.unsqueeze(0) for ns in batch.next_state])
        done_batch = torch.cat([d.unsqueeze(0) for d in batch.done])
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values using target network (double DQN approach)
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # Get Q values from target network for these actions
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            # Compute expected Q values
            expected_q_values = reward_batch + (self.config["gamma"] * next_q_values * (1 - done_batch))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to stabilize training
        if self.config["gradient_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config["gradient_clip"])
        
        # Log gradient norms to wandb
        if wandb.run is not None:
            total_norm = 0
            for p in self.policy_net.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            wandb.log({"gradient_norm": total_norm}, step=self.training_step)
            
            # Log specific layer gradients
            for name, param in self.policy_net.named_parameters():
                if param.grad is not None:
                    wandb.log({f"grad/{name}_norm": param.grad.norm().item()}, step=self.training_step)
            
            # Log Q-value statistics
            wandb.log({
                "q_values/mean": current_q_values.mean().item(),
                "q_values/max": current_q_values.max().item(),
                "q_values/min": current_q_values.min().item(),
                "loss": loss.item()
            }, step=self.training_step)
        
        self.optimizer.step()
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network with current policy network weights
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        """
        Save model checkpoint
        """
        torch.save({
            'policy_model': self.policy_net.state_dict(),
            'target_model': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
            'training_step': self.training_step
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model checkpoint
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_model'])
            self.target_net.load_state_dict(checkpoint['target_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.config = checkpoint.get('config', self.config)  # Fallback to current config
            self.training_step = checkpoint.get('training_step', 0)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False