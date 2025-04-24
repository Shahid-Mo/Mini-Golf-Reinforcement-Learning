import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
import math
import numpy as np
import pymunk
import pymunk.pygame_util

# Constants
WIDTH, HEIGHT = 800, 600
GREEN = (34, 139, 34)
BEIGE = (245, 222, 179)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Physics constants
BALL_MASS = 1
BALL_RADIUS = 10
HOLE_RADIUS = 15
FRICTION = 10000
ELASTICITY = 0.8

class Ball:
    def __init__(self, x, y, space):
        self.body = pymunk.Body(BALL_MASS, pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, BALL_RADIUS)
        self.shape.friction = FRICTION
        self.shape.elasticity = ELASTICITY
        self.shape.collision_type = 1  # Ball collision type
        self.in_hole = False
        space.add(self.body, self.shape)
    
    def draw(self, screen):
        if not self.in_hole:
            pos_x, pos_y = int(self.body.position.x), int(self.body.position.y)
            pygame.draw.circle(screen, WHITE, (pos_x, pos_y), BALL_RADIUS)
    
    def hit(self, power, angle):
        # Only apply force if the ball is nearly stopped
        if self.body.velocity.length < 2 and not self.in_hole:
            impulse = power * pymunk.Vec2d(math.cos(angle), math.sin(angle))
            self.body.apply_impulse_at_local_point(impulse)
    
    @property
    def is_moving(self):
        # Check if the ball is moving significantly
        return self.body.velocity.length > 2 and not self.in_hole
    
    @property
    def position(self):
        return self.body.position.x, self.body.position.y


class Hole:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = HOLE_RADIUS
    
    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius)


class MinigolfEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60,
    }
    
    def __init__(self, render_mode=None):
        super(MinigolfEnv, self).__init__()
        
        # Initialize pygame if rendering is needed
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.draw_options = None
        
        # Physics space
        self.space = None
        self.ball = None
        self.hole = None
        self.shots = 0
        self.max_shots = 10
        
        # Action space discretization
        self.n_angles = 8  # 8 directions
        self.n_powers = 5  # 5 power levels
        self.max_power = 300  # Adjusted for PyMunk
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_angles * self.n_powers)
        
        # Observation is [ball_x, ball_y, hole_x, hole_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize physics
        self._setup_physics()
    
    def _setup_physics(self):
        # Create space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity in top-down minigolf
        
        # Create boundaries (walls)
        walls = [
            [(100, 100), (WIDTH-100, 100)],  # Top
            [(WIDTH-100, 100), (WIDTH-100, HEIGHT-100)],  # Right
            [(WIDTH-100, HEIGHT-100), (100, HEIGHT-100)],  # Bottom
            [(100, HEIGHT-100), (100, 100)]  # Left
        ]
        
        for wall in walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, wall[0], wall[1], 5)
            shape.friction = FRICTION
            shape.elasticity = ELASTICITY
            shape.collision_type = 2  # Wall collision type
            self.space.add(body, shape)
        
        # Setup collision handler for hole detection
        self.space.add_collision_handler(1, 3).begin = self._ball_hole_collision
    
    def _ball_hole_collision(self, arbiter, space, data):
        # This gets called when the ball touches the hole's sensor
        # We'll handle the "falling in hole" logic separately
        return False  # Don't generate a regular collision
    
    def _get_obs(self):
        # Return state as normalized coordinates
        ball_x, ball_y = self.ball.position
        return np.array([
            ball_x / WIDTH,
            ball_y / HEIGHT,
            self.hole.x / WIDTH,
            self.hole.y / HEIGHT
        ], dtype=np.float32)
    
    def _get_info(self):
        ball_x, ball_y = self.ball.position
        return {
            "shots": self.shots,
            "distance_to_hole": math.sqrt((ball_x - self.hole.x)**2 + (ball_y - self.hole.y)**2),
            "in_hole": self.ball.in_hole
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset the environment
        if self.space:
            # Remove old ball from space
            if self.ball:
                self.space.remove(self.ball.body, self.ball.shape)
        else:
            # First time setup
            self._setup_physics()
        
        # Create new ball
        self.ball = Ball(200, HEIGHT//2, self.space)
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
            self.draw_options = pymunk.pygame_util.DrawOptions(self.window)
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
    
    def check_hole_collision(self):
        # Check if ball is close enough to the hole to fall in
        ball_x, ball_y = self.ball.position
        dx = ball_x - self.hole.x
        dy = ball_y - self.hole.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if ball is inside hole and moving slowly enough
        if distance < self.hole.radius - BALL_RADIUS/2 and self.ball.body.velocity.length < 50:
            self.ball.in_hole = True
            # Hide the ball (we keep it in space but it won't be drawn)
            return True
        return False
    
    def step(self, action):
        # Initialize reward
        reward = 0
        
        # Only count a shot if the ball isn't moving
        if not self.ball.is_moving:
            self.shots += 1
            
            # Decode action into angle and power
            angle, power = self._decode_action(action)
            
            # Hit the ball
            self.ball.hit(power, angle)
        
        # Step the physics simulation multiple times for smoother simulation
        for _ in range(10):  # Adjust this number for physics accuracy vs. performance
            self.space.step(1/60.0)
            
            # Check if ball fell in hole
            if self.check_hole_collision():
                break
            
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
            ball_x, ball_y = self.ball.position
            dx = ball_x - self.hole.x
            dy = ball_y - self.hole.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Reward for getting closer to the hole (normalized by initial distance)
            reward = (initial_distance - distance) / initial_distance * 5
            
            # Small penalty for each step to encourage efficiency
            reward -= 0.1
            
            # Check if ball stopped moving
            if not self.ball.is_moving and not self.terminated:
                # Small penalty for stopping without reaching the hole
                reward -= 1
        
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
            self.draw_options = pymunk.pygame_util.DrawOptions(self.window)
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((WIDTH, HEIGHT))
        canvas.fill(BLUE)
        
        # Draw course
        pygame.draw.rect(canvas, GREEN, (100, 100, WIDTH-200, HEIGHT-200))
        pygame.draw.rect(canvas, BEIGE, (100, 100, WIDTH-200, HEIGHT-200), 10)
        
        # Draw hole
        self.hole.draw(canvas)
        
        # Draw ball
        self.ball.draw(canvas)
        
        # Draw physics debug info (optional)
        # self.space.debug_draw(self.draw_options)
        
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


# Manual play version
def play_manual_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Minigolf with PyMunk")
    
    # Setup physics
    space = pymunk.Space()
    space.gravity = (0, 0)
    
    # Create boundaries
    walls = [
        [(100, 100), (WIDTH-100, 100)],  # Top
        [(WIDTH-100, 100), (WIDTH-100, HEIGHT-100)],  # Right
        [(WIDTH-100, HEIGHT-100), (100, HEIGHT-100)],  # Bottom
        [(100, HEIGHT-100), (100, 100)]  # Left
    ]
    
    for wall in walls:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, wall[0], wall[1], 5)
        shape.friction = FRICTION
        shape.elasticity = ELASTICITY
        space.add(body, shape)
    
    # Create ball and hole
    ball = Ball(200, HEIGHT//2, space)
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
            
            if not ball.is_moving and not game_over:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    aiming = True
                elif event.type == pygame.MOUSEBUTTONUP and aiming:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    ball_x, ball_y = ball.position
                    dx = mouse_x - ball_x
                    dy = mouse_y - ball_y
                    aim_angle = math.atan2(dy, dx)
                    
                    # Calculate power based on distance, but cap it
                    aim_power = min(math.sqrt(dx*dx + dy*dy) * 2, 300)
                    
                    ball.hit(aim_power, aim_angle)
                    shots += 1
                    aiming = False
        
        # Step physics
        space.step(1/60.0)
        
        # Check hole collision
        ball_x, ball_y = ball.position
        dx = ball_x - hole.x
        dy = ball_y - hole.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < hole.radius - BALL_RADIUS/2 and ball.body.velocity.length < 50:
            ball.in_hole = True
            game_over = True
        
        # Draw everything
        screen.fill(BLUE)
        
        pygame.draw.rect(screen, GREEN, (100, 100, WIDTH-200, HEIGHT-200))
        pygame.draw.rect(screen, BEIGE, (100, 100, WIDTH-200, HEIGHT-200), 10)
        
        hole.draw(screen)
        ball.draw(screen)
        
        if aiming and not ball.is_moving and not game_over:
            ball_x, ball_y = ball.position
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.line(screen, RED, (ball_x, ball_y), (mouse_x, mouse_y), 2)
        
        font = pygame.font.SysFont(None, 36)
        shots_text = font.render(f"Shots: {shots}", True, WHITE)
        screen.blit(shots_text, (10, 10))
        
        if game_over:
            win_text = font.render(f"Hole in {shots} shots! Click to play again.", True, WHITE)
            screen.blit(win_text, (WIDTH//2 - win_text.get_width()//2, 50))
            
            # Check for restart
            if pygame.mouse.get_pressed()[0]:
                # Remove old ball
                space.remove(ball.body, ball.shape)
                # Create new ball
                ball = Ball(200, HEIGHT//2, space)
                shots = 0
                game_over = False
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()


# Example usage with Gymnasium
if __name__ == "__main__":
    # Choose between manual play or gym environment
    use_gym = False
    
    if use_gym:
        # Create the environment directly
        env = MinigolfEnv(render_mode="human")
        
        # Run a simple random agent
        obs, info = env.reset()
        total_reward = 0
        
        for _ in range(10):  # Run for 10 episodes
            terminated = truncated = False
            obs, info = env.reset()
            episode_reward = 0
            
            while not (terminated or truncated):
                # Wait for ball to stop before taking another action
                while env.ball.is_moving:
                    obs, reward, terminated, truncated, info = env.step(0)  # Pass a dummy action
                    env.render()
                    episode_reward += reward
                    if terminated or truncated:
                        break
                
                if not (terminated or truncated):
                    action = env.action_space.sample()  # Random action
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    env.render()
            
            print(f"Episode finished with reward: {episode_reward}, Ball in hole: {info['in_hole']}, Shots: {info['shots']}")
            total_reward += episode_reward
        
        print(f"Average reward: {total_reward/10}")
        env.close()
    else:
        # Play manually
        play_manual_game()