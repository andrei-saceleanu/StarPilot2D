import gymnasium as gym
import pygame
import numpy as np

from yaml import safe_load
from gymnasium import spaces
from pygame.locals import *

from pickup import ReplenishFuel
from player import Player
from status import StatusBar
from target import Target

def update_targets(targets, coll_idx):
    remaining_targets = [elem for idx, elem in enumerate(targets) if idx not in coll_idx]
    if not remaining_targets:
        remaining_targets = [Target(np.random.randint(100, 700, size=(2,)), (80,80))]
    return remaining_targets

def update_pickups(pickups, coll_idx):
    remaining_pickups = [elem for idx, elem in enumerate(pickups) if idx not in coll_idx]
    if not remaining_pickups:
        remaining_pickups = [ReplenishFuel(np.random.randint(100, 700, size=(2,)), (80,80))]
    return remaining_pickups

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
        self.pickups = [ReplenishFuel(pos=np.array([600, 600]), size=(80, 80))]
        self.bar = StatusBar()

        self.target_counter = 0
        self.reward = 0
        self.time = 0.0
        self.time_limit = 300
        self.time_since_last_point = 0.0

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))

    def reset(self, seed=None, **kwargs):
        self.player = Player(**self.config["player"])
        self.targets = [Target(pos=np.array([100, 100]), size=(80, 80))]
        self.pickups = [ReplenishFuel(pos=np.array([600, 600]), size=(80, 80))]
        self.target_counter = 0
        self.reward = 0
        self.time = 0.0
        self.bar.value = 100
        self.time_since_last_point = 0.0
        
        return self.get_obs(), {}
    
    def get_obs(self):

        angle_to_ox = self.player.angle / 180 * np.pi
        speed = self.player.speed
        distance_to_target = np.linalg.norm(self.player.pos-self.targets[0].pos)/500
        angle_to_target = np.arctan2(self.targets[0].pos[1]-self.player.pos[1],self.targets[0].pos[0]-self.player.pos[0])
        
        distance_to_pickup = np.linalg.norm(self.player.pos-self.pickups[0].pos)/500
        angle_to_pickup = np.arctan2(self.pickups[0].pos[1]-self.player.pos[1],self.pickups[0].pos[0]-self.player.pos[0])

        return np.array(
            [
                angle_to_ox,
                speed,
                distance_to_target,
                angle_to_target,
                distance_to_pickup,
                angle_to_pickup,
                self.player.angle - angle_to_target,
                self.player.angle - angle_to_pickup,
                self.bar.value
            ]
        ).astype(np.float32)
    
    def step(self, action):

        self.reward = 0.0
        action = int(action)

        for _ in range(5):
            self.time += 1 / self.fps
            #self.time_since_last_point += 1 / self.fps

            self.player.env_act(action)
            self.player.update(self.screen_size)

            dist = np.linalg.norm(self.player.pos-self.targets[0].pos)/500
            self.reward += 1 / self.fps
            self.reward -= dist / self.fps

            dist_pickup = np.linalg.norm(self.player.pos-self.pickups[0].pos)/500
            self.reward -= dist_pickup / self.fps

            coll_idx = self.player.check_points(self.targets)
            if coll_idx:
                #self.time_since_last_point = 0.0
                self.targets = update_targets(self.targets, coll_idx)
                self.reward += 20

            coll_idx = self.player.check_pickups(self.pickups, self.bar)
            if coll_idx:
                self.pickups = update_pickups(self.pickups, coll_idx)
                self.reward += 10

            self.bar.update(self.player.get_fuel_delta())
           
            done = False
            truncated = False
            if (self.time > self.time_limit):
                done = True
                break
            elif (self.bar.value <= 0):
                truncated = True
                break
            #elif self.time_since_last_point > 60:
            #    self.reward -= 1000
            #    truncated = True
            #    break
            else:
                done = False

            # self.render()

        
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

        self.player.draw(self.screen)
        for target in self.targets:
            target.draw(self.screen)
        self.bar.draw(self.screen)
        for pickup in self.pickups:
            pickup.draw(self.screen)

        text = self.font.render(f"Score: {self.player.score}", True, (255,255,255))
        text_box = text.get_rect()
        text_box.topleft = (10,20)
        self.screen.blit(text, text_box)

        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        return super().close()

        
        


