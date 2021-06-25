import argparse
import pygame
import numpy as np
import random
from dataclasses import dataclass
from pygame import Vector2
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_r,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
from typing import List

WIDTH = 512
HEIGHT = WIDTH
TILE_SIZE = int(WIDTH // 32)
BOARD_WIDTH = int(WIDTH // TILE_SIZE)
BOARD_HEIGHT = BOARD_WIDTH
BG_COLOR = (0, 0, 0)
SCORE_COLOR = (255, 255, 255)
PADDING = 16
SNAKE_SPEED = 0.2 / TILE_SIZE
BOARD_MIDDLE = Vector2([int(BOARD_WIDTH // 2), int(BOARD_HEIGHT // 2)])
APPLE_COLOR = (0, 255, 0)
APPLE_RADIUS = int(TILE_SIZE // 2)
SNAKE_COLOR = (255, 255, 255)

def random_straight_velocity():
    vel = Vector2([0, 0])
    vel_scalar = random.choice([-SNAKE_SPEED, SNAKE_SPEED])
    horiz = random.choice([True, False])
    if horiz:
        vel.x = vel_scalar
    else:
        vel.y = vel_scalar
    return vel

@dataclass
class GameState:
    running: bool
    snake: List[Vector2]
    score: int
    snake_vel: Vector2
    apple_pos: Vector2
    dt: float

@dataclass
class RendererState:
    screen = None
    font = None
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.Font(pygame.font.get_default_font(), 18)

def random_board_pos():
    return Vector2([random.randint(0, BOARD_WIDTH), random.randint(0, BOARD_HEIGHT)])

def spawn_random_apple(S):
    S.apple_pos = random_board_pos()

def update(S):
    dt = S.dt
    if S.apple_pos == None:
        spawn_random_apple(S)
    # update player snake positions
    for i in range(len(S.snake)):
        p = S.snake[i]
        new_x = p.x + S.snake_vel.x * dt
        new_y = p.y + S.snake_vel.y * dt
        S.snake[i].x = new_x
        S.snake[i].y = new_y
    # check if player is on an apple
    for i in range(len(S.snake)):
        if S.snake[i] == S.apple_pos:
            # Eat apple
            S.apple_pos = None
            S.score += 1

def tcoord_to_real(tcoord):
    return Vector2([tcoord.x * TILE_SIZE, tcoord.y * TILE_SIZE])

def tcoord_to_real_circle(tcoord, rd):
    return Vector2([tcoord.x * TILE_SIZE + rd, tcoord.y * TILE_SIZE + rd])

def render(S, R):
    # Reset screen
    R.screen.fill(BG_COLOR)
    # Apple
    if S.apple_pos is not None:
        pygame.draw.circle(
            R.screen,
            APPLE_COLOR,
            tcoord_to_real_circle(S.apple_pos, APPLE_RADIUS),
            APPLE_RADIUS,
        )
    # Snake
    for snake_part in S.snake:
        rp = tcoord_to_real(snake_part)
        pygame.draw.rect(
            R.screen,
            SNAKE_COLOR,
            (rp.x, rp.y, TILE_SIZE, TILE_SIZE),
        )
    # Score
    score_t = R.font.render('Score: {}'.format(S.score), True, SCORE_COLOR)
    score_tr = score_t.get_rect()
    score_tr.topright = (WIDTH - PADDING, PADDING)
    R.screen.blit(score_t, score_tr)
    pygame.display.flip()

def new_state():
    return GameState(
        False,
        [BOARD_MIDDLE],
        0,
        random_straight_velocity(),
        random_board_pos(),
        0,
    )

def reset():
    return new_state()

def player_move_up(S):
    S.snake_vel.y = -SNAKE_SPEED;
    S.snake_vel.x = 0;

def player_move_down(S):
    S.snake_vel.y = SNAKE_SPEED;
    S.snake_vel.x = 0;

def player_move_left(S):
    S.snake_vel.y = 0;
    S.snake_vel.x = -SNAKE_SPEED;

def player_move_right(S):
    S.snake_vel.y = 0;
    S.snake_vel.x = SNAKE_SPEED;

def run_game():
    clock = pygame.time.Clock()
    pygame.init()
    S = new_state()
    S.running = True
    R = RendererState()
    while S.running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                S.running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    S.running = False
                elif event.key == K_r:
                    reset(S)
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            player_move_up(S)
        elif keys[K_DOWN]:
            player_move_down(S)
        elif keys[K_LEFT]:
            player_move_left(S)
        elif keys[K_RIGHT]:
            player_move_right(S)
        dt = clock.tick(10)
        S.dt = dt
        update(S)
        render(S, R)
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--play', action='store_true', default=False)
    args = parser.parse_args()
    if args.train:
        print("Training")
        # train()
    elif args.play:
        run_game()
    else:
        print("Running saved model")
        # run_model()
