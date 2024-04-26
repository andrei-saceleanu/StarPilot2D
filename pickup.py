import pygame

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

    def apply(self, *args):
        pass

    def undo(self, *args):
        pass

class ReplenishFuel(Pickup):

    def __init__(self, pos, size, img_path="assets/fuel.png", render=False):
        super(ReplenishFuel, self).__init__(pos, size, img_path, render)

    def apply(self, *args):
        args[0].update(20)

class BetterPlane(Pickup):

    def __init__(self, pos, size, img_path="assets/cog.png", render=False):
        super(BetterPlane, self).__init__(pos, size, img_path, render)
        self._func = lambda x: min(1.2*x, 8)
        self._inv_func = lambda x: max(x/1.2, 3)

    def apply(self, *args):
        args[1].angle_delta = self._func(args[1].angle_delta)

pickup_types = [ReplenishFuel, BetterPlane]