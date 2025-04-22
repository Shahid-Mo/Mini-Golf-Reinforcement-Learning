import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
import math
import numpy as np


class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 10
        self.vel_x = 0
        self.vel_y = 0
        self.friction = 0.98
        self.moving = False
        self.in_hole = False
    
    def update(self):
        if self.moving and not self.in_hole:
            self.x += self.vel_x
            self.y += self.vel_y
            
            # Apply friction
            self.vel_x *= self.friction
            self.vel_y *= self.friction
            
            # Stop the ball if it's moving very slowly
            if abs(self.vel_x) < 0.1 and abs(self.vel_y) < 0.1:
                self.vel_x = 0
                self.vel_y = 0
                self.moving = False
                
            # Boundary collision
            if self.x - self.radius < 100:
                self.x = 100 + self.radius
                self.vel_x *= -0.8
            elif self.x + self.radius > WIDTH - 100:
                self.x = WIDTH - 100 - self.radius
                self.vel_x *= -0.8
            
            if self.y - self.radius < 100:
                self.y = 100 + self.radius
                self.vel_y *= -0.8
            elif self.y + self.radius > HEIGHT - 100:
                self.y = HEIGHT - 100 - self.radius
                self.vel_y *= -0.8
    
    def draw(self, screen):
        if not self.in_hole:
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)
    
    def hit(self, power, angle):
        if not self.moving and not self.in_hole:
            self.vel_x = power * math.cos(angle)
            self.vel_y = power * math.sin(angle)
            self.moving = True
    
    def check_hole_collision(self, hole):
        if not self.in_hole:
            # Calculate distance between ball and hole
            dx = self.x - hole.x
            dy = self.y - hole.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Check if ball is inside hole
            if distance < hole.radius - self.radius / 2:
                self.in_hole = True
                self.vel_x = 0
                self.vel_y = 0
                self.moving = False
                return True
        return False


class Hole:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 15
    
    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius)


# Constants
WIDTH, HEIGHT = 800, 600
GREEN = (34, 139, 34)
BEIGE = (245, 222, 179)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


class MinigolfEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 2000,
    }
    
    def __init__(self, render_mode=None):
        super(MinigolfEnv, self).__init__()
        
        # Initialize pygame if rendering is needed
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.ball = None
        self.hole = None
        self.shots = 0
        self.max_shots = 10
        
        # Action space discretization
        self.n_angles = 8  # 8 directions
        self.n_powers = 5  # 5 power levels
        self.max_power = 15
        
        # Define action and observation spaces
        # Action is a discrete value encoding both angle and power
        self.action_space = spaces.Discrete(self.n_angles * self.n_powers)
        
        # Observation is a normalized representation of ball and hole positions
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
    
    def _get_obs(self):
        # Return state as normalized coordinates
        return np.array([
            self.ball.x / WIDTH,
            self.ball.y / HEIGHT,
            self.hole.x / WIDTH,
            self.hole.y / HEIGHT
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            "shots": self.shots,
            "distance_to_hole": math.sqrt((self.ball.x - self.hole.x)**2 + (self.ball.y - self.hole.y)**2),
            "in_hole": self.ball.in_hole
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset the environment
        self.ball = Ball(200, HEIGHT//2)
        self.hole = Hole(WIDTH-200, HEIGHT//2)
        self.shots = 0
        self.terminated = False
        self.truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        # Only initialize renderer if needed
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Minigolf Gym Environment")
            if self.clock is None:
                self.clock = pygame.time.Clock()
        
        return observation, info
    
    def _decode_action(self, action_idx):
        # Convert action index to angle and power
        angle_idx = action_idx // self.n_powers
        power_idx = action_idx % self.n_powers
        
        # Convert indices to actual values
        angle = 2 * math.pi * angle_idx / self.n_angles
        power = (power_idx + 1) * self.max_power / self.n_powers
        
        return angle, power
    
    def step(self, action):
        # Initialize reward
        reward = 0
        self.shots += 1
        
        # Decode action into angle and power
        angle, power = self._decode_action(action)
        
        # Hit the ball
        self.ball.hit(power, angle)
        
        # Fast-forward simulation until the ball stops or goes in the hole
        max_iterations = 500  # Safety valve to prevent infinite loops
        iterations = 0
        
        while self.ball.moving and not self.ball.in_hole and iterations < max_iterations:
            self.ball.update()
            if self.ball.check_hole_collision(self.hole):
                break
            iterations += 1
            
            # If rendering, show the ball movement
            if self.render_mode == "human":
                self.render()
        
        # Get the initial distance (used for reward calculation)
        initial_distance = math.sqrt((200 - self.hole.x)**2 + (HEIGHT//2 - self.hole.y)**2)
        
        # Compute reward and check terminal conditions
        if self.ball.in_hole:
            # Success! Give positive reward inversely proportional to shots taken
            reward = 100 - (self.shots - 1) * 10  # Higher reward for fewer shots
            self.terminated = True
        elif self.shots >= self.max_shots:
            # Failed to get the ball in the hole within max shots
            reward = -50
            self.terminated = True
            self.truncated = True  # Episode is truncated due to max shots
        else:
            # Still playing, give reward based on distance to hole
            dx = self.ball.x - self.hole.x
            dy = self.ball.y - self.hole.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Reward for getting closer to the hole (normalized by initial distance)
            reward = (initial_distance - distance) / initial_distance * 5
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, self.terminated, self.truncated, info
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
        # For "human" rendering
        if self.window is None:
            return
            
        self._render_frame()
        pygame.event.pump()
        pygame.display.flip()
        
        # Add a small delay to make visualization smoother
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((WIDTH, HEIGHT))
        canvas.fill(BLUE)
        
        # Draw course
        pygame.draw.rect(canvas, GREEN, (100, 100, WIDTH-200, HEIGHT-200))
        pygame.draw.rect(canvas, BEIGE, (100, 100, WIDTH-200, HEIGHT-200), 10)
        
        # Draw game objects
        self.hole.draw(canvas)
        self.ball.draw(canvas)
        
        # Draw UI
        font = pygame.font.SysFont(None, 36)
        shots_text = font.render(f"Shots: {self.shots}", True, WHITE)
        canvas.blit(shots_text, (10, 10))
        
        if self.render_mode == "human":
            # Copy canvas to window
            self.window.blit(canvas, canvas.get_rect())
            return None
        else:
            # Return the array for "rgb_array" mode
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# Example usage for manual play
def play_manual_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simple Minigolf")
    
    ball = Ball(200, HEIGHT//2)
    hole = Hole(WIDTH-200, HEIGHT//2)
    shots = 0
    game_over = False
    aiming = False
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not ball.moving and not game_over:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    aiming = True
                elif event.type == pygame.MOUSEBUTTONUP and aiming:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    dx = mouse_x - ball.x
                    dy = mouse_y - ball.y
                    aim_angle = math.atan2(dy, dx)
                    
                    # Calculate power based on distance, but cap it
                    aim_power = min(math.sqrt(dx*dx + dy*dy) / 10, 15)
                    
                    ball.hit(aim_power, aim_angle)
                    shots += 1
                    aiming = False
        
        ball.update()
        
        if ball.check_hole_collision(hole):
            game_over = True
        
        screen.fill(BLUE)
        
        pygame.draw.rect(screen, GREEN, (100, 100, WIDTH-200, HEIGHT-200))
        pygame.draw.rect(screen, BEIGE, (100, 100, WIDTH-200, HEIGHT-200), 10)
        
        hole.draw(screen)
        ball.draw(screen)
        
        if aiming and not ball.moving and not game_over:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.line(screen, RED, (ball.x, ball.y), (mouse_x, mouse_y), 2)
        
        font = pygame.font.SysFont(None, 36)
        shots_text = font.render(f"Shots: {shots}", True, WHITE)
        screen.blit(shots_text, (10, 10))
        
        if game_over:
            win_text = font.render(f"Hole in {shots} shots! Click to play again.", True, WHITE)
            screen.blit(win_text, (WIDTH//2 - win_text.get_width()//2, 50))
            
            # Check for restart
            if pygame.mouse.get_pressed()[0]:
                ball = Ball(200, HEIGHT//2)
                shots = 0
                game_over = False
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()


# Example usage with Gymnasium
if __name__ == "__main__":
    # Choose between manual play or gym environment
    use_gym = True
    
    if use_gym:
        # Create the environment directly without registration
        env = MinigolfEnv(render_mode="human")
        
        # Run a simple random agent
        obs, info = env.reset()
        total_reward = 0
        
        for _ in range(10):  # Run for 100 episodes
            terminated = truncated = False
            obs, info = env.reset()
            episode_reward = 0
            
            while not (terminated or truncated):
                action = env.action_space.sample()  # Random action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                env.render()
            
            print(f"Episode finished with reward: {episode_reward}, Ball in hole: {info['in_hole']}, Shots: {info['shots']}")
            total_reward += episode_reward
        
        print(f"Average reward over 100 episodes: {total_reward/100}")
        env.close()
    else:
        # Play manually
        play_manual_game()