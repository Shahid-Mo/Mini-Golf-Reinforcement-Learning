"""
Mini-Golf demo with real-time aim overlay & trajectory preview
--------------------------------------------------------------
• Pull like a slingshot: click-drag away from the ball, release to shoot
• Red line  ……  pull vector
• Yellow dots …  predicted path (1.5 s ahead)
• HUD text   …  launch angle (°) & impulse units
"""

import math
import sys
import pygame
import pymunk

# ------------------------------------------------------------
# Tunables
# ------------------------------------------------------------
WIDTH, HEIGHT = 800, 600
FPS           = 60

# Colours
BLUE   = (0,   0,   255)
GREEN  = (34, 139,  34)
BEIGE  = (245, 222, 179)
WHITE  = (255, 255, 255)
RED    = (255,   0,   0)
YELLOW = (255, 255,   0)

# Ball physics
BALL_MASS       = 1.0
BALL_RADIUS     = 10
BALL_FRICTION   = 0.20        # 0.10-0.30
BALL_ELASTICITY = 0.90        # 0.80-0.95

# Wall physics
WALL_FRICTION   = 0.70        # 0.60-0.80
WALL_ELASTICITY = 0.65        # 0.60-0.75

# Global damping (rolling resistance)
SPACE_DAMPING   = 0.9999      # 0.985-0.995
MIN_VELOCITY    = 100          # px / s → snap to zero when slower

# Shot power scaling
IMPULSE_SCALE   = 1.0         # px drag → N·s
MAX_IMPULSE     = 900.0

# Trajectory preview
PREVIEW_STEPS   = 90          # 1.5 s @ 60 Hz
PREVIEW_SKIP    = 3           # draw every nth point


# ------------------------------------------------------------
# Helper: simulate future path in a throw-away space
# ------------------------------------------------------------
def preview(space, body, impulse, steps=PREVIEW_STEPS, dt=1 / FPS):
    dbg = pymunk.Space()
    dbg.gravity = (0, 0)
    dbg.damping = SPACE_DAMPING

    # clone static shapes (the walls)
    for s in space.shapes:
        if isinstance(s, pymunk.Segment):
            clone = pymunk.Segment(
                dbg.static_body, s.a, s.b, s.radius
            )
            clone.friction, clone.elasticity = s.friction, s.elasticity
            dbg.add(clone)

    # ghost ball
    ghost = pymunk.Body(BALL_MASS, pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
    ghost.position = body.position
    circ = pymunk.Circle(ghost, BALL_RADIUS)
    circ.friction, circ.elasticity = BALL_FRICTION, BALL_ELASTICITY
    dbg.add(ghost, circ)

    ghost.apply_impulse_at_local_point(impulse)

    pts = []
    for _ in range(steps):
        dbg.step(dt)
        pts.append(tuple(ghost.position))
    return pts


# ------------------------------------------------------------
# Game
# ------------------------------------------------------------
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mini-Golf Demo")
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    # ---------------- physics world ----------------
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = SPACE_DAMPING

    # ball
    ball_body = pymunk.Body(BALL_MASS, pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
    ball_body.position = 200, HEIGHT // 2
    ball_shape = pymunk.Circle(ball_body, BALL_RADIUS)
    ball_shape.friction = BALL_FRICTION
    ball_shape.elasticity = BALL_ELASTICITY
    space.add(ball_body, ball_shape)

    # walls
    rect = [(100, 100), (WIDTH - 100, 100), (WIDTH - 100, HEIGHT - 100), (100, HEIGHT - 100)]
    for a, b in zip(rect, rect[1:] + rect[:1]):
        seg = pymunk.Segment(space.static_body, a, b, 5)
        seg.friction = WALL_FRICTION
        seg.elasticity = WALL_ELASTICITY
        space.add(seg)

    # ---------------- game state ----------------
    aiming        = False
    aim_start_pos = (0, 0)

    # ---------------- main loop ----------------
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ball_body.velocity.length < MIN_VELOCITY:
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    aiming = True
                    aim_start_pos = pygame.mouse.get_pos()

                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and aiming:
                    mx, my = pygame.mouse.get_pos()
                    sx, sy = aim_start_pos
                    drag   = pymunk.Vec2d(sx - mx, sy - my)

                    if drag.length > 5:
                        power   = min(drag.length * IMPULSE_SCALE, MAX_IMPULSE)
                        impulse = drag.normalized() * power
                        ball_body.apply_impulse_at_local_point(impulse)

                    aiming = False

        # physics step
        space.step(1 / FPS)

        # snap small drift
        if 0 < ball_body.velocity.length < MIN_VELOCITY:
            ball_body.velocity = (0, 0)

        # -------------- draw --------------
        screen.fill(BLUE)
        pygame.draw.rect(screen, GREEN, (100, 100, WIDTH - 200, HEIGHT - 200))
        pygame.draw.rect(screen, BEIGE, (100, 100, WIDTH - 200, HEIGHT - 200), 10)

        # ball
        bx, by = ball_body.position
        pygame.draw.circle(screen, WHITE, (int(bx), int(by)), BALL_RADIUS)

        # aim overlay
        if aiming:
            mx, my   = pygame.mouse.get_pos()
            sx, sy   = aim_start_pos
            drag_vec = pymunk.Vec2d(sx - mx, sy - my)

            # red pull line
            pygame.draw.line(screen, RED, (bx, by), (mx, my), 2)

            if drag_vec.length > 5:
                power      = min(drag_vec.length * IMPULSE_SCALE, MAX_IMPULSE)
                impulse    = drag_vec.normalized() * power
                angle_deg  = math.degrees(math.atan2(drag_vec.y, drag_vec.x)) % 360

                # predicted path (yellow dots)
                for pt in preview(space, ball_body, impulse)[::PREVIEW_SKIP]:
                    pygame.draw.circle(screen, YELLOW, (int(pt[0]), int(pt[1])), 2)

                # HUD text
                hud = font.render(f"{angle_deg:0.1f}°  {power:.0f}", True, WHITE)
                screen.blit(hud, (bx + 15, by - 25))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
