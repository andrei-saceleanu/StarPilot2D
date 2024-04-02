import pygame
import numpy as np

from stable_baselines3 import DQN

class Player:

    def __init__(
        self, init_pos, init_speed, init_angle, min_speed, max_speed, speed_delta, angle_delta, render=False
    ):
        self.pos = init_pos
        self.speed = init_speed
        self.angle = init_angle
        self.interval = [min_speed, max_speed]
        self.speed_delta = speed_delta
        self.angle_delta = angle_delta
        self.direction = (np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle)))
        self.score = 0

        if render:
            self.set_sprite()

    def set_sprite(self):
        self.img = pygame.image.load("assets/plane.png")
        self.img.convert_alpha()
        self.img = pygame.transform.scale(self.img, (50,50))
        self.img = pygame.transform.rotate(self.img, -90)

    def env_act(self, action):

        if action == 0:
            pass
        if action == 1:
            self.speed = min(self.interval[1], self.speed+self.speed_delta)
        if action == 2:
            self.speed = max(self.interval[0], self.speed-self.speed_delta)
        if action == 3:
            self.angle = (self.angle - self.angle_delta) % 360
        elif action == 4:
            self.angle = (self.angle + self.angle_delta) % 360
        self.direction = (np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle)))

    def act(self, obs):
        pass

    def get_fuel_delta(self):

        t = (self.speed - self.interval[0]) / (self.interval[1] - self.interval[0])
        fuel_delta = (1 - t) * 0.01 + t * 0.1
        return -fuel_delta

    def update(self, size):
        
        self.pos[0] += self.speed * self.direction[0]
        self.pos[1] += self.speed * self.direction[1]
            
        self.pos[0] = self.pos[0] % size[0]
        self.pos[1] = self.pos[1] % size[1]

    def check_points(self, targets):

        coll_idx = []
        for idx, target in enumerate(targets):
            if np.linalg.norm(np.array(self.pos) - target.pos) <= target.size[0]/2:
                self.score += 1
                coll_idx.append(idx)
        return coll_idx
    
    def check_pickups(self, pickups, bar):
        
        coll_idx = []
        for idx, pickup in enumerate(pickups):
            if np.linalg.norm(np.array(self.pos) - pickup.pos) <= pickup.size[0]/2:
                pickup.apply(bar)
                coll_idx.append(idx)
        return coll_idx
        
    
    def draw(self, screen):
        rotated_img = pygame.transform.rotate(self.img, -self.angle)
        screen.blit(
            rotated_img,
            (
                self.pos[0] - int(rotated_img.get_width()/2),
                self.pos[1] - int(rotated_img.get_height()/2)
            )    
        )


class HumanPlayer(Player):
    def __init__(
            self, init_pos, init_speed, init_angle, min_speed, max_speed, speed_delta, angle_delta, render=False
    ):
        super(HumanPlayer, self).__init__(
            init_pos, init_speed, init_angle, min_speed, max_speed, speed_delta, angle_delta, render
        )

    def act(self, obs=None):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            self.speed = min(self.interval[1], self.speed+self.speed_delta)
        if keys[pygame.K_DOWN]:
            self.speed = max(self.interval[0], self.speed-self.speed_delta)
        if keys[pygame.K_LEFT]:
            self.angle = (self.angle - self.angle_delta) % 360
        elif keys[pygame.K_RIGHT]:
            self.angle = (self.angle + self.angle_delta) % 360
        self.direction = (np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle)))

class RandomPlayer(Player):
    def __init__(
            self, init_pos, init_speed, init_angle, min_speed, max_speed, speed_delta, angle_delta, render=False
    ):
        super(RandomPlayer, self).__init__(
            init_pos, init_speed, init_angle, min_speed, max_speed, speed_delta, angle_delta, render
        )

    def act(self, obs=None):
        act_idx = np.random.randint(0,4)
        
        if act_idx==0:
            self.speed = min(self.interval[1], self.speed + self.speed_delta)
        if act_idx==1:
            self.speed = max(self.interval[0], self.speed - self.speed_delta)
        if act_idx==2:
            self.angle = (self.angle - self.angle_delta) % 360
        elif act_idx==3:
            self.angle = (self.angle + self.angle_delta) % 360
        self.direction = (np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle)))

class DQNPlayer(Player):
    def __init__(
            self, init_pos, init_speed, init_angle, min_speed, max_speed, speed_delta, angle_delta, model_path=None, render=False
    ):
        super(DQNPlayer, self).__init__(
            init_pos, init_speed, init_angle, min_speed, max_speed, speed_delta, angle_delta, render
        )
        self.model_path = model_path
        self.model = DQN.load(self.model_path)

    def act(self, obs=None):
        act_idx, _ = self.model.predict(obs)
       
        if act_idx==0:
            pass
        elif act_idx==1:
            self.speed = min(self.interval[1], self.speed + self.speed_delta)
        elif act_idx==2:
            self.speed = max(self.interval[0], self.speed - self.speed_delta)
        elif act_idx==3:
            self.angle = (self.angle - self.angle_delta) % 360
        elif act_idx==4:
            self.angle = (self.angle + self.angle_delta) % 360
        self.direction = (np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle)))
