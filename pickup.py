import pygame

class Pickup(object):
    def __init__(self, pos, size, img_path, render=False):
        self.pos = pos
        self.size = size
        self.img_path = img_path
        self.timeout = -1
        
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

    def apply(self, *args):
        pass

class ReplenishFuel(Pickup):

    def __init__(self, pos, size, img_path="assets/fuel.png", render=False):
        super(ReplenishFuel, self).__init__(pos, size, img_path, render)
        # self.timeout = 5

    def apply(self, *args):
        args[0].update(20)
        # return [self.timeout, args[0], "update", -20]

class BetterPlane_1(Pickup):

    def __init__(self, pos, size, img_path="assets/cog.png", render=False):
        super(BetterPlane_1, self).__init__(pos, size, img_path, render)
        self._func = lambda x: min(1.2*x, 8)
        self._inv_func = lambda x: max(x/1.2, 3)
        self.timeout = 30

    def apply(self, *args):
        args[1].angle_delta = self._func(args[1].angle_delta)
        return [self.timeout, args[1], "angle_delta", self._inv_func]

class BetterPlane_2(Pickup):

    def __init__(self, pos, size, img_path="assets/eco.png", render=False):
        super(BetterPlane_2, self).__init__(pos, size, img_path, render)
        self._func = lambda x: 0.75 * x
        self._inv_func = lambda x: 1.33 * x
        self.timeout = 15

    def apply(self, *args):
        args[1].fuel_interval = self._func(args[1].fuel_interval)
        return [self.timeout, args[1], "fuel_interval", self._inv_func]

pickup_types = [ReplenishFuel, BetterPlane_1, BetterPlane_2]