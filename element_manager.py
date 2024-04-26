import numpy as np

from copy import deepcopy
from random import choice, random

from status import StatusBar
from target import Target
from player import DQNPlayer, HumanPlayer
from pickup import pickup_types

def update_targets(targets, coll_idx):
    remaining_targets = [elem for idx, elem in enumerate(targets) if idx not in coll_idx]
    if not remaining_targets or (len(remaining_targets) < 2 and random() < 0.01):
        remaining_targets.append(Target(np.random.randint(100, 700, size=(2,)), (80,80),  render=True))
    return remaining_targets


def update_pickups(pickups, coll_idx):
    remaining_pickups = [elem for idx, elem in enumerate(pickups) if idx not in coll_idx]
    if not remaining_pickups or (len(remaining_pickups) < 2 and random() < 0.005):
        pickup_type = choice(pickup_types)
        remaining_pickups.append(pickup_type(np.random.randint(100, 700, size=(2,)), (80,80), render=True))
    
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


class ElementManager:

    def __init__(self, config):
        self.targets = []
        self.pickups = []
        self.players = []
        self.huds = []
        self.config = config
        self.reset()

    def reset(self):
        self.targets = [Target(pos=np.array([100, 100]), size=(80, 80), render=True)]
        pickup_type = choice(pickup_types)
        self.pickups = [pickup_type(pos=np.array([600, 600]), size=(80, 80), render=True)]

        self.players = [
            DQNPlayer(
                **deepcopy(self.config["player"]),
                model_path=[
                    "weights/rl_model_v1_10000000_steps.zip",
                    "weights/rl_model_v6_10000000_steps.zip"
                ],
                render=True
            ),
            HumanPlayer(**deepcopy(self.config["player"]), render=True)
        ]
        self.huds = [StatusBar(), StatusBar(topleft=(500, 70), color=(214, 15, 58))]

    def draw(self, screen, font):
        for player in self.players:
            player.draw(screen)
        for target in self.targets:
            target.draw(screen)
        for bar in self.huds:
            bar.draw(screen)
        for pickup in self.pickups:
            pickup.draw(screen)

        base_pos = [10, 20]
        for bar, player in zip(self.huds, self.players):
            text = font.render(f"Score ({player.name}): {player.score}", True, bar.color)
            text_box = text.get_rect()
            text_box.topleft = base_pos
            screen.blit(text, text_box)
            base_pos[1] += 30

    def update(self, screen_size):
        
        done = 0
        for idx, (bar, player) in enumerate(zip(self.huds, self.players)):
            obs = self._get_obs(idx)
            player.act(obs=obs)
            player.update(screen_size)
                
            coll_idx = player.check_points(self.targets)
            self.targets = update_targets(self.targets, coll_idx)
            coll_idx = player.check_pickups(self.pickups, bar)
            self.pickups = update_pickups(self.pickups, coll_idx)

            bar.update(player.get_fuel_delta())
            if bar.value <= 0.0:
                done = 2

        return done
    
    def game_over(self, screen, font, done):
        msg1, msg = game_over_info(done, self.players, self.huds[0], self.huds[1])
                
        text = font.render(f"{msg1} {msg} Rematch? (y/n)", True, (255, 255, 255))
        text_box = text.get_rect()
        text_box.center = (400, 400)
        screen.blit(text, text_box)

    def _get_obs(self, idx):
        player = self.players[idx]
        bar = self.huds[idx]

        if not isinstance(player, DQNPlayer):
            return None

        angle_to_ox = player.angle / 180 * np.pi
        speed = player.speed
        distance_to_target = np.linalg.norm(player.pos-self.targets[0].pos)/500
        angle_to_target = np.arctan2(
            self.targets[0].pos[1]-player.pos[1],
            self.targets[0].pos[0]-player.pos[0]
        )
                    
        distance_to_pickup = np.linalg.norm(player.pos-self.pickups[0].pos)/500
        angle_to_pickup = np.arctan2(
            self.pickups[0].pos[1]-player.pos[1],
            self.pickups[0].pos[0]-player.pos[0]
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
    

