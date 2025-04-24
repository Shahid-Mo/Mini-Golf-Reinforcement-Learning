import pygame
import pymunk
import pymunk.pygame_util

pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 678

# game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pool")

# pymunk space
space = pymunk.Space()
static_body = space.static_body
space.gravity = (0,0)
draw_options = pymunk.pygame_util.DrawOptions(screen)

# clock
clock = pygame.time.Clock()
FPS = 120

# colours
BG = (50, 50, 50)

# load images
table_image = pygame.image.load("images/table.png").convert_alpha()

# function for creating balls
def create_ball(radius, pos):
    body = pymunk.Body()
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.mass = 5
    # use pivot joint to add friction
    pivot = pymunk.PivotJoint(static_body, body, (0,0), (0,0))
    pivot.max_bias = 0 # disable joint correction
    pivot.max_force = 1000 # emulate linera friction
    
    space.add(body, shape, pivot)
    return shape

new_ball = create_ball(25, (300,360))

cue_ball = create_ball(25, (600, 350))

# function fro creatin cushions
def create_cushion():
    

# game loop 
run = True
while run:
    
    clock.tick(FPS)
    space.step(1 / FPS)
    
    # fill background
    screen.fill(BG)
    
    # draw pool table
    screen.blit(table_image, (0,0))
    
    # event handler
    for event in pygame.event.get():
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            cue_ball.body.apply_impulse_at_local_point((-5000, 0), (0, 0))
        
        if event.type == pygame.QUIT:
            run = False
    
    space.debug_draw(draw_options)
    pygame.display.update()
    
    
pygame.quit()
