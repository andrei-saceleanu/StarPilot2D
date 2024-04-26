import cv2
import pygame
import matplotlib as mpl
import numpy as np

from argparse import ArgumentParser
from yaml import safe_load

from drawing import draw_timer
from element_manager import ElementManager

def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/game_config.yml"
    )

    return parser.parse_args()

def init_game(width, height):
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font('freesansbold.ttf', 20)
    font_large = pygame.font.Font('freesansbold.ttf', 28)
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    return font, font_large, screen, clock

def main():

    args = parse_args()
    with open(args.config, "r") as fin:
        config = safe_load(fin)
    
    screen_size = config["view"]["width"], config["view"]["height"]
    width, height = screen_size
    fps = config["game"]["fps"]
    time_limit = config["game"]["time_limit"]
    
    font, font_large, screen, clock = init_game(width, height)

    quit = False
    done = 0
    t = 0

    cmap = mpl.colormaps["gist_rainbow"]
    manager = ElementManager(config)

    while not quit:
        screen.fill((118, 170, 176))

        for event in pygame.event.get():
            if (
                (event.type == pygame.QUIT)
                or
                (event.type == pygame.KEYDOWN and event.key == pygame.K_q)
            ):
                quit = True

            if done > 0:
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_y):
                    done = 0
                    clock = pygame.time.Clock()
                    t = 0
                    manager.reset()
                elif (event.type == pygame.KEYDOWN and event.key == pygame.K_n):
                    quit=True
        
        if done == 0:
            
            done = manager.update(screen_size)
            manager.draw(screen, font)
            draw_timer(time_limit, font, screen, cmap, t)

        else:
            manager.game_over(screen, font_large, done)
    
        pygame.display.update()
        dt = clock.tick(fps)
        t += dt

        if (t / 1000) >= time_limit and done == 0:
            done=1

    pygame.quit()

if __name__=="__main__":
    main()
