import pygame
import numpy as np

from player import Player
from status import StatusBar


class Pickup(object):
    def __init__(self, pos, size, img_path, render=False):
        self.pos = pos
        self.size = size
        self.img_path = img_path
        
        if render:
            self.set_sprite()

    def set_sprite(self):
        self.img = pygame.image.load(self.img_path)
        self.img.convert_alpha()
        self.img = pygame.transform.scale(self.img, self.size)

    def draw(self, screen):
        screen.blit(
            self.img,
            (
                self.pos[0] - int(self.size[0]/2),
                self.pos[1] - int(self.size[1]/2)
            )    
        )

    def apply(self, subject):
        pass

    def undo(self, subject):
        pass

class ReplenishFuel(Pickup):

    def __init__(self, pos, size, img_path="assets/fuel.png", render=False):
        super(ReplenishFuel, self).__init__(pos, size, img_path, render)

    def apply(self, subject):
        if isinstance(subject, StatusBar):
            subject.update(20)
        else:
            raise Exception("ReplenishFuel is applied only on StatusBar")

class BetterPlane(Pickup):

    def __init__(self, pos, size, img_path="assets/cog.png", render=False):
        super(BetterPlane, self).__init__(pos, size, img_path, render)
        self._func = lambda x: 2*x
        self._inv_func = lambda x: x/2

    def apply(self, subject):
        if isinstance(subject, Player):
            subject.angle_delta = self._func(subject.angle_delta)
        else:
            raise Exception("BetterPlane is applied only on Player and subclasses")
        
    def undo(self, subject):
        if isinstance(subject, Player):
            subject.angle_delta = self._inv_func(subject.angle_delta)
        else:
            raise Exception("BetterPlane is applied only on Player and subclasses")
        

