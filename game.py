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
from datetime import datetime, timedelta
from enum import IntEnum
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from gym import error, spaces
import keyboard
import time
import math

WIDTH = 512
HEIGHT = WIDTH
TILE_SIZE = int(WIDTH // 128)
BOARD_WIDTH = int(WIDTH // TILE_SIZE)
BOARD_HEIGHT = BOARD_WIDTH
BG_COLOR = (0, 0, 0)
SCORE_COLOR = (255, 255, 255)
PADDING = 16
SNAKE_SPEED = 1
BOARD_MIDDLE = Vector2([int(BOARD_WIDTH // 2), int(BOARD_HEIGHT // 2)])
APPLE_COLOR = (0, 255, 0)
APPLE_RADIUS = int(TILE_SIZE // 2)
SNAKE_COLOR = (255, 255, 255)
TICK_INTERVAL = timedelta(milliseconds=100)
APPLE_SPAWN_PADDING = 1
MAX_SNAKE_LENGTH = 64
MODEL_FILE = "saved_weights.dqn"
MAX_APPLES = 3

@dataclass
class GameState:
    running: bool = None
    snake: List[Vector2] = None
    score: int = None
    snake_vel: Vector2 = None
    apple_positions: List[Vector2] = None
    dt: float = None
    lost: bool = False

@dataclass
class RendererState:
    screen = None
    font = None
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.Font(pygame.font.get_default_font(), 18)

def random_straight_velocity():
    vel = Vector2([0, 0])
    vel_scalar = random.choice([-SNAKE_SPEED, SNAKE_SPEED])
    horiz = random.choice([True, False])
    if horiz:
        vel.x = vel_scalar
    else:
        vel.y = vel_scalar
    return vel

def random_board_pos():
    return Vector2([random.randint(0, BOARD_WIDTH), random.randint(0, BOARD_HEIGHT)])

def random_board_pos_with_padding(p):
    x = random.randint(p + 1, BOARD_WIDTH - p - 1)
    y = random.randint(p + 1, BOARD_HEIGHT - p - 1)
    return Vector2([x, y])

def spawn_random_apple(S):
    S.apple_positions = []
    for i in range(random.randint(1, 3)):
        p = None
        while p is None or any(map(lambda x: x.x == p.x and x.y == p.y, S.snake)):
            p = random_board_pos_with_padding(APPLE_SPAWN_PADDING)
        S.apple_positions.append(p)

def snake_of_length_starting_at(start, n):
    snake = []
    for i in range(n):
        snake.append(Vector2([start.x + i, start.y]))
    return snake

def reset_inplace(S):
    S.snake = snake_of_length_starting_at(BOARD_MIDDLE, 2)
    S.score = 0
    S.snake_vel = random_straight_velocity()
    spawn_random_apple(S)
    S.dt = 0
    S.lost = False

def new_state():
    S = GameState()
    reset_inplace(S)
    return S

def reset():
    return new_state()

def lost(S):
    S.lost = True

def update(S):
    head_pos = S.snake[0]
    tail_pos = S.snake[len(S.snake) - 1]
    tail_pos_pre_update = Vector2([tail_pos.x, tail_pos.y])
    # update player snake parts
    for i in range(len(S.snake) - 1, 0, -1):
        # get closer to the part we're following
        follow_part = S.snake[i - 1]
        S.snake[i].x = follow_part.x
        S.snake[i].y = follow_part.y
    head_pos += S.snake_vel
    # check if player is on an apple
    for ap in S.apple_positions:
        if S.snake[0] == ap:
            # Eat apple and spawn a new one if there are no more left
            S.apple_positions.remove(ap)
            if len(S.apple_positions) == 0:
                spawn_random_apple(S)
            S.score += 1
            # Grow snake
            S.snake.append(tail_pos_pre_update)
            break
    # Check board boundaries
    if head_pos.x < 0 or head_pos.x >= BOARD_WIDTH or\
       head_pos.y < 0 or head_pos.y >= BOARD_HEIGHT:
        lost(S)
        return
    # Check snake part collision
    for i in range(len(S.snake)):
        for j in range(i + 1, len(S.snake)):
            if S.snake[i] == S.snake[j]:
                lost(S)
                return

def tcoord_to_real(tcoord):
    return Vector2([tcoord.x * TILE_SIZE, tcoord.y * TILE_SIZE])

def tcoord_to_real_circle(tcoord, rd):
    return Vector2([tcoord.x * TILE_SIZE + rd, tcoord.y * TILE_SIZE + rd])

def render(S, R):
    # Reset screen
    R.screen.fill(BG_COLOR)
    # Apple
    for ap in S.apple_positions:
        pygame.draw.circle(
            R.screen,
            APPLE_COLOR,
            tcoord_to_real_circle(ap, APPLE_RADIUS),
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

def set_velocity(S, vel):
    prev_vel = S.snake_vel
    if abs(vel.x - prev_vel.x) > 1 or abs(vel.y - prev_vel.y) > 1:
        # trying to move in the opposite direction, disallow that action
        return None
    S.snake_vel = vel

def player_move_up(S):
    set_velocity(S, Vector2([0, -SNAKE_SPEED]))

def player_move_down(S):
    set_velocity(S, Vector2([0, SNAKE_SPEED]))

def player_move_left(S):
    set_velocity(S, Vector2([-SNAKE_SPEED, 0]))

def player_move_right(S):
    set_velocity(S, Vector2([SNAKE_SPEED, 0]))

def run_game():
    clock = pygame.time.Clock()
    S = new_state()
    S.running = True
    R = RendererState()
    last_time = datetime.now()
    while S.running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                S.running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    S.running = False
                elif event.key == K_r:
                    reset_inplace(S)
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            player_move_up(S)
        elif keys[K_DOWN]:
            player_move_down(S)
        elif keys[K_LEFT]:
            player_move_left(S)
        elif keys[K_RIGHT]:
            player_move_right(S)
        dt = clock.tick(60)
        curr_time = datetime.now()
        time_diff = curr_time - last_time
        if time_diff >= TICK_INTERVAL:
            update(S)
            last_time = curr_time
        S.dt = dt
        render(S, R)
        if S.lost:
            reset_inplace(S)
    pygame.quit()

class Action(IntEnum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3

class PongEnv(gym.Env):
    ACTION_TO_FUN_MAPPING = {
        Action.Up: player_move_up,
        Action.Down: player_move_down,
        Action.Left: player_move_left,
        Action.Right: player_move_right,
    }

    def __init__(self):
        self.action_space = spaces.Discrete(len(list(PongEnv.ACTION_TO_FUN_MAPPING.keys())))
        self.S = new_state()
        self.renderer = RendererState()

    def get_state(self):
        return self.S

    def state_to_input_repr(self):
        """ Transform the game state into a neural-net input array """
        S = self.S
        snake = np.zeros((MAX_SNAKE_LENGTH*2,), dtype=int)
        for i in range(0, len(S.snake), 2):
            snake[i] = S.snake[i].x
            snake[i + 1] = S.snake[i].y
        apple_positions = np.zeros(MAX_APPLES * 2, dtype=int)
        for i in range(0, len(S.apple_positions), 2):
            apple_positions[i] = S.apple_positions[i].x
            apple_positions[i + 1] = S.apple_positions[i].y
        q = np.array([
            S.snake_vel.x,
            S.snake_vel.y,
            *apple_positions,
            *snake,
        ])
        return q

    def perform_action(self, action):
        PongEnv.ACTION_TO_FUN_MAPPING[action](self.S)

    def reset(self):
        reset_inplace(self.S)
        state = self.get_state()
        return self.state_to_input_repr()

    def render(self, mode='human'):
        render(self.S, self.renderer)

    def step(self, action):
        # Action
        self.perform_action(action)
        # Observe new state
        update(self.S)
        S = self.get_state()
        # Calculate the reward
        capd = -1000      # Distance to the closest apple
        x2 = S.snake[0].x
        y2 = S.snake[0].y
        for ap in S.apple_positions:
            x1 = ap.x
            y1 = ap.y
            mx = (x2 - x1)
            my = (y2 - y1)
            d = mx*mx + my*my
            if d > capd:
                capd = d
        capd = math.sqrt(capd)
        reward = 100 + (capd * -4) + (S.score * 100)
        done = S.lost
        return self.state_to_input_repr(), reward, done, {}

    def clone_state(self):
        return copy.deepcopy(state)

    def restore_state(self, state_):
        global state
        state = copy.deepcopy(state_)

    def clone_full_state(self):
        return copy.deepcopy(state)

    def restore_full_state(self, state_):
        global state
        state = copy.deepcopy(state_)

# Model
def setup_model(env):
    model = Sequential()
    state_param_amount = 2 + 2 * MAX_APPLES + MAX_SNAKE_LENGTH * 2
    actions_amount = env.action_space.n
    model.add(Flatten(input_shape=(1, state_param_amount)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions_amount, activation='linear'))
    return model

# Agent
def setup_agent(env, model):
    actions_amount = env.action_space.n
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                    nb_actions=actions_amount, nb_steps_warmup=10, target_model_update=1e-2)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn

def train(visualize=False):
    env = PongEnv()
    model = setup_model(env)
    dqn = setup_agent(env, model)
    dqn.fit(env, nb_steps=50000, visualize=visualize, verbose=1)
    dqn.save_weights(MODEL_FILE, overwrite=True)

def run_model(num_episodes):
    env = PongEnv()
    model = setup_model(env)
    dqn = setup_agent(env, model)
    dqn.load_weights(MODEL_FILE)
    dqn.test(env, nb_episodes=num_episodes, visualize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--play', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--episodes', default=50, type=int)
    args = parser.parse_args()
    if args.train:
        print("Training")
        train(visualize=args.render)
    elif args.play:
        run_game()
    else:
        num_episodes = args.episodes
        print("Running saved model for {} episodes".format(num_episodes))
        run_model(num_episodes)
