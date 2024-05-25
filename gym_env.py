import gymnasium as gym
import pygame
import numpy as np

from yaml import safe_load
from gymnasium import spaces
from pygame.locals import *

from element_manager import ElementManager

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

        self.manager = ElementManager(self.config, train=True, render=False)

        self.target_counter = 0
        self.reward = 0
        self.time = 0.0
        self.time_limit = 300

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,))

    def reset(self, seed=None, **kwargs):
        self.manager.reset(train=True)
        self.target_counter = 0
        self.reward = 0
        self.time = 0.0
        
        return self.get_obs(), {}
    
    def get_obs(self):

        return self.manager._get_obs_v2(idx=0)
    
    def step(self, action):

        self.reward = 0.0
        action = int(action)

        for _ in range(5):
            self.time += 1 / self.fps

            status, coll_idx, elem_info = self.manager.train_update(action, self.screen_size, 1000/self.fps)
            target_idx, pickup_idx = coll_idx
            target_info, pickup_info = elem_info
            if target_idx:
                self.reward += 1.0
            if pickup_idx:
                self.reward += 1.0

            x = 0
            for pos in target_info+pickup_info:
                x = max(x, np.exp(-np.sum((self.manager.players[0].pos-pos)**2)/200**2))
            self.reward += x
            done = False
            truncated = False
            if (self.time > self.time_limit):
                done = True
                break
            elif status > 0: # fuel ran out 
                truncated = True
                break
            else:
                done = False
        
        info = {}

        return (
            self.get_obs(),
            self.reward,
            done,
            truncated,
            info
        )

    def render(self):
        pygame.event.get()
        self.screen.fill((118, 170, 176))
        self.manager.draw(self.screen, self.font)
        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        return super().close()

        
        


