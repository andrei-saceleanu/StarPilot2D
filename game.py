import cv2
import pygame
import matplotlib as mpl
import numpy as np

from argparse import ArgumentParser
from yaml import safe_load
from math import pi
from copy import deepcopy

from target import Target
from player import HumanPlayer, RandomPlayer, DQNPlayer
from pickup import ReplenishFuel, BetterPlane
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
    font_large = pygame.font.Font('freesansbold.ttf', 28)
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    return font, font_large, screen, clock

def update_targets(targets, coll_idx):
    remaining_targets = [elem for idx, elem in enumerate(targets) if idx not in coll_idx]
    if not remaining_targets:
        remaining_targets = [Target(np.random.randint(100, 700, size=(2,)), (80,80),  render=True)]
    return remaining_targets


def update_pickups(pickups, coll_idx):
    remaining_pickups = [elem for idx, elem in enumerate(pickups) if idx not in coll_idx]
    if not remaining_pickups:
        remaining_pickups = [ReplenishFuel(np.random.randint(100, 700, size=(2,)), (80,80), render=True)]
    return remaining_pickups

def game_over_info(done, players, bar, bar2):
    if done == 1:
        msg1 = "Game over!"
    else:
        if bar.value <= 0 and bar2.value > 0:
            msg1 = "AI ran out of fuel!"
        if bar.value > 0 and bar2.value <= 0:
            msg1 = "Human ran out of fuel!"
        if bar.value <= 0 and bar2.value <= 0:
            msg1 = "Both players ran out of fuel!"
                
    if players[1].score > players[0].score:
        msg = "Human won."
    elif players[0].score > players[1].score:
        msg = "AI won."
    else:
        msg = "It's a tie."
    return msg1, msg

def get_game_elements(config):
    targets = [Target(pos=np.array([100, 100]), size=(80, 80), render=True)]
    pickups = [ReplenishFuel(pos=np.array([600, 600]), size=(80, 80), render=True)]

    players = [
        DQNPlayer(
            **deepcopy(config["player"]),
            model_path=[
                "weights/rl_model_v1_10000000_steps.zip",
                "weights/rl_model_v6_10000000_steps.zip"
            ],
            render=True
        ),
        HumanPlayer(**deepcopy(config["player"]), render=True)
    ]
    bar = StatusBar()
    bar2 = StatusBar(topleft=(500, 70), color=(214, 15, 58))
    t = 0
    return targets, pickups, players, bar, bar2, t

def get_obs_for_agent(targets, pickups, bar, player):
    angle_to_ox = player.angle / 180 * np.pi
    speed = player.speed
    distance_to_target = np.linalg.norm(player.pos-targets[0].pos)/500
    angle_to_target = np.arctan2(
        targets[0].pos[1]-player.pos[1],
        targets[0].pos[0]-player.pos[0]
    )
                
    distance_to_pickup = np.linalg.norm(player.pos-pickups[0].pos)/500
    angle_to_pickup = np.arctan2(
        pickups[0].pos[1]-player.pos[1],
        pickups[0].pos[0]-player.pos[0]
    )

    obs = np.array(
        [
            angle_to_ox,
            speed,
            distance_to_target,
            angle_to_target,
            distance_to_pickup,
            angle_to_pickup,
            player.angle - angle_to_target,
            player.angle - angle_to_pickup,
            bar.value
        ]
    ).astype(np.float32)
    
    return obs

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
    cmap = mpl.colormaps["gist_rainbow"]
    targets, pickups, players, bar, bar2, t = get_game_elements(config)

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
                    targets, pickups, players, bar, bar2, t = get_game_elements(config)
                elif (event.type == pygame.KEYDOWN and event.key == pygame.K_n):
                    quit=True
        
        if done == 0:
            for player in players:
                obs=None
                if isinstance(player, DQNPlayer):
                    obs = get_obs_for_agent(targets, pickups, bar, player)

                player.act(obs=obs)
                player.update(screen_size)
                
                coll_idx = player.check_points(targets)
                targets = update_targets(targets, coll_idx)
                if isinstance(player, DQNPlayer):
                    coll_idx = player.check_pickups(pickups, bar)
                else:
                    coll_idx = player.check_pickups(pickups, bar2)
                pickups = update_pickups(pickups, coll_idx)

            bar.update(players[0].get_fuel_delta())
            bar2.update(players[1].get_fuel_delta())

            if bar.value <= 0.0 or bar2.value <= 0.0:
                done = 2
            
            for player in players:
                player.draw(screen)
            for target in targets:
                target.draw(screen)
            bar.draw(screen)
            bar2.draw(screen)
            for pickup in pickups:
                pickup.draw(screen)

            text = font.render(f"Score (AI): {players[0].score}", True, bar.color)
            text_box = text.get_rect()
            text_box.topleft = (10,20)
            screen.blit(text, text_box)
        
            text = font.render(f"Score (Human): {players[1].score}", True, bar2.color)
            text_box = text.get_rect()
            text_box.topleft = (10,50)
            screen.blit(text, text_box)

            draw_timer(time_limit, font, screen, cmap, t)

        else:
            msg1, msg = game_over_info(done, players, bar, bar2)
                
            text = font_large.render(f"{msg1} {msg} Rematch? (y/n)", True, (255, 255, 255))
            text_box = text.get_rect()
            text_box.center = (400, 400)
            screen.blit(text, text_box)
    
        pygame.display.update()
        dt = clock.tick(fps)
        
        t += dt

        if (t / 1000) >= time_limit and done == 0:
            print("Time limit was reached")
            done=1

    pygame.quit()

if __name__=="__main__":
    main()
