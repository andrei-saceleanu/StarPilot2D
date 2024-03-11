import pygame

class Target:

    def __init__(self, pos, size, img_path="assets/star.png", render=False):
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