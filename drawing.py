import cv2
import numpy as np
import pygame


def draw_arc(surf, color, center, radius, width, end_angle):
    circle_image = np.zeros((radius*2+4, radius*2+4, 4), dtype = np.uint8)
    circle_image = cv2.ellipse(circle_image, (radius+2, radius+2),
        (radius-width//2, radius-width//2), 0, 0, end_angle, (*color, 255), width, lineType=cv2.LINE_AA) 
    
    circle_surface = pygame.image.frombuffer(circle_image.flatten(), (radius*2+4, radius*2+4), 'RGBA')
    surf.blit(circle_surface, circle_surface.get_rect(center = center), special_flags=pygame.BLEND_PREMULTIPLIED)

def draw_timer(time_limit, font, screen, cmap, t):

    countdown = time_limit - (t / 1000)
    curr_color = tuple([int(255*elem) for elem in cmap(int(100*countdown/time_limit))[:3]])
    draw_arc(screen, curr_color, (70, 700), 50, 5, 360 * countdown/time_limit)
    
    total_time = round(countdown)
    minutes = total_time // 60
    seconds = total_time % 60
    text = font.render(f"{minutes:02}:{seconds:02}", True, (0, 0, 0))
    text_box = text.get_rect()
    text_box.center = (70, 700)
    screen.blit(text, text_box)