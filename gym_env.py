from typing import List, Tuple
import gym
import pygame
import numpy as np

from yaml import safe_load
from gym import spaces
from pygame.locals import *

from player import Player
from target import Target

def update_targets(targets, coll_idx):
    remaining_targets = [elem for idx, elem in enumerate(targets) if idx not in coll_idx]
    if not remaining_targets:
        remaining_targets = [Target(np.random.randint(100, 700, size=(2,)), (80,80))]
    return remaining_targets

class GameEnv(gym.Env):
    
    def __init__(self, config_file, render=False) -> None:
        super(GameEnv, self).__init__()

        with open(config_file, "r") as fin:
            self.config = safe_load(fin)
        
        self.screen_size = self.config["view"]["width"], self.config["view"]["height"]
        width, height = self.screen_size
        self.fps = self.config["game"]["fps"]

        if render:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.Font('freesansbold.ttf', 20)
            self.screen = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()

        self.player = Player(**self.config["player"])
        self.targets = [Target(pos=np.array([100,100]), size=(80, 80))]

        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 20

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))

    def reset(self):
        self.player = Player(**self.config["player"])
        self.targets = [Target(pos=np.array([100, 100]), size=(80, 80))]
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        
        return self.get_obs()
    
    def get_obs(self):

        angle_to_ox = self.player.angle / 180 * np.pi
        speed = self.player.speed
        distance = np.linalg.norm(self.player.pos-self.targets[0].pos)/500
        angle_to_target = np.arctan2(self.targets[0].pos[1]-self.player.pos[1],self.targets[0].pos[0]-self.player.pos[0])

        return np.array(
            [
                angle_to_ox,
                speed,
                distance,
                angle_to_target,
                self.player.angle - angle_to_target
            ]
        ).astype(np.float32)
    
    def step(self, action):

        self.reward = 0.0
        action = int(action)

        for _ in range(5):
            self.time += 1 / 60

            self.player.env_act(action)
            self.player.update(self.screen_size)

            dist = np.linalg.norm(self.player.pos-self.targets[0].pos)/500
            self.reward += 1 / 60
            self.reward -= dist / (100 * 60)

            coll_idx = self.player.check_points(self.targets)
            if coll_idx:
                self.targets = update_targets(self.targets, coll_idx)
                self.reward += 100
            
            if self.time > self.time_limit:
                done = True
                break

            # If too far from target (crash)
            elif dist > 1000:
                self.reward -= 1000
                done = True
                break
            else:
                done = False

            # self.render()

        
        info = {}

        return (
            self.get_obs(),
            self.reward,
            done,
            info
        )

    def render(self):
        pygame.event.get()
        self.screen.fill((118, 170, 176))

        self.player.draw(self.screen)
        for target in self.targets:
            target.draw(self.screen)

        text = self.font.render(f"Score: {self.player.score}", True, (255,255,255))
        text_box = text.get_rect()
        text_box.topleft = (10,20)
        self.screen.blit(text, text_box)

        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        return super().close()

        
        


