import pygame
import numpy as np

from argparse import ArgumentParser
from yaml import safe_load

from target import Target
from player import HumanPlayer, RandomPlayer
from status import StatusBar

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
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    return font, screen, clock

def update_targets(targets, coll_idx):
    remaining_targets = [elem for idx, elem in enumerate(targets) if idx not in coll_idx]
    if not remaining_targets:
        remaining_targets = [Target(np.random.randint(100, 700, size=(2,)), (80,80),  render=True)]
    return remaining_targets

def main():

    args = parse_args()
    with open(args.config, "r") as fin:
        config = safe_load(fin)
    
    screen_size = config["view"]["width"], config["view"]["height"]
    width, height = screen_size
    fps = config["game"]["fps"]
    
    font, screen, clock = init_game(width, height)

    run = True
    targets = [Target(pos=np.array([100, 100]), size=(80, 80), render=True)]

    player = HumanPlayer(**config["player"], render=True)
    bar = StatusBar()
    
    while run:
        screen.fill((118, 170, 176))

        for event in pygame.event.get():
            if (
                (event.type == pygame.QUIT)
                or
                (event.type == pygame.KEYDOWN and event.key == pygame.K_q)
            ):
                run = False
        
        player.act()
        player.update(screen_size)

        coll_idx = player.check_points(targets)
        targets = update_targets(targets, coll_idx)

        bar.update(player.get_fuel_delta())
        
        player.draw(screen)
        for target in targets:
            target.draw(screen)
        bar.draw(screen)

        text = font.render(f"Score: {player.score}", True, (255,255,255))
        text_box = text.get_rect()
        text_box.topleft = (10,20)
        screen.blit(text, text_box)

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()

if __name__=="__main__":
    main()
