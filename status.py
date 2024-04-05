import pygame


class StatusBar:
    def __init__(self, value=100, max_value=100, topleft=(500, 20)):

        self.img = pygame.Rect(topleft[0], topleft[1], 200, 40)
        self.value = value
        self.max_value = max_value
    
    def update(self, delta):
        self.value = max(0, min(self.max_value, self.value + delta))

    def draw(self, screen):
        ratio = self.img.w * (self.value/self.max_value)
        scaled = pygame.Rect(self.img.topleft, (ratio, self.img.h))
        pygame.draw.rect(screen, (237, 143, 12), scaled, width=0)
        pygame.draw.rect(screen, (0, 0, 0), self.img, width=5)
        