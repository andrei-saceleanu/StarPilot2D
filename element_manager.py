import numpy as np

from copy import deepcopy
from random import choice, random

from status import StatusBar
from target import Target
from player import DQNPlayer, HumanPlayer, Player
from pickup import pickup_types

def smart_sample(remaining_targets, remaining_pickups):
    while True:
        sample_pos = np.random.randint(100, 700, size=(2,))
        done  = all(
            [np.linalg.norm(sample_pos-elem.pos)>85 for elem in remaining_targets+remaining_pickups]
        )
        if done:
            break
    return sample_pos

def update_elements(targets, target_idx, pickups, pickup_idx, render=False):
    remaining_targets = [elem for idx, elem in enumerate(targets) if idx not in target_idx]
    remaining_pickups = [elem for idx, elem in enumerate(pickups) if idx not in pickup_idx]

    if not remaining_targets or (len(remaining_targets) < 2 and random() < 0.01):
        sample_pos = smart_sample(remaining_targets, remaining_pickups)
        remaining_targets.append(Target(sample_pos, (80,80),  render=render))

    if not remaining_pickups or (len(remaining_pickups) < 2 and random() < 0.005):
        sample_pos = smart_sample(remaining_targets, remaining_pickups)
        pickup_type = pickup_types[np.random.choice(len(pickup_types), p=[0.5, 0.3, 0.2])]
        remaining_pickups.append(pickup_type(sample_pos, (80,80), render=render))
        
    return remaining_targets, remaining_pickups

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

    def __init__(self, config, train=False, render=False):
        self.targets = []
        self.pickups = []
        self.players = []
        self.huds = []
        self.render = render
        self.config = config
        self.reset(train)

    def reset(self, train=False):
        self.targets = [Target(pos=np.array([100, 100]), size=(80, 80), render=self.render)]
        pickup_type = choice(pickup_types)
        self.pickups = [pickup_type(pos=np.array([600, 600]), size=(80, 80), render=self.render)]

        if train==False:
            self.players = [
                DQNPlayer(
                    **deepcopy(self.config["player"]),
                    model_path=[
                        "weights/rl_model_v1_10000000_steps.zip",
                        "weights/rl_model_v6_10000000_steps.zip"
                    ],
                    render=self.render
                ),
                HumanPlayer(**deepcopy(self.config["player"]), render=self.render)
            ]
            self.huds = [StatusBar(), StatusBar(topleft=(500, 70), color=(214, 15, 58))]
        else:
            self.players = [Player(**self.config["player"])]
            self.huds = [StatusBar()]

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

    def update(self, screen_size, dt):
        
        done = 0
        for idx, (bar, player) in enumerate(zip(self.huds, self.players)):
            obs = self._get_obs(idx)
            player.act(obs=obs)
            player.update(screen_size)
                
            target_idx = player.check_points(self.targets)
            pickup_idx = player.check_pickups(self.pickups, bar, dt)
            self.targets, self.pickups = update_elements(self.targets, target_idx, self.pickups, pickup_idx, render=True)
            
            bar.update(player.get_fuel_delta())
            if bar.value <= 0.0:
                done = 2

        return done
    
    def train_update(self, act, screen_size, dt):

        done = 0
        for _, (bar, player) in enumerate(zip(self.huds, self.players)):
            player.env_act(act=act)
            player.update(screen_size)
                
            target_idx = player.check_points(self.targets)
            pickup_idx = player.check_pickups(self.pickups, bar, dt)
            elem_info = self.compute_info()
            self.targets, self.pickups = update_elements(self.targets, target_idx, self.pickups, pickup_idx)
            
            bar.update(player.get_fuel_delta())
            if bar.value <= 0.0:
                done = 2

        return done, (target_idx, pickup_idx), elem_info

    def compute_dists(self):
        targets_info = []
        pickups_info = []
        for target in self.targets:
            targets_info.append(target.pos)
        
        for pickup in self.pickups:
            pickups_info.append(pickup.pos)

        return targets_info, pickups_info

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
    
    def _get_obs_v2(self, idx):
        player = self.players[idx]
        bar = self.huds[idx]

        if not isinstance(player, DQNPlayer):
            return None

        obs_elements = []
        angle_to_ox = player.angle / 180 * np.pi
        speed = player.speed
        obs_elements.extend([angle_to_ox, speed])
    
        obs_elements.append(len(self.targets))
        for target in self.targets:
            distance_to_target = np.linalg.norm(player.pos-target.pos)/500
            angle_to_target = np.arctan2(
                target.pos[1]-player.pos[1],
                target.pos[0]-player.pos[0]
            )
            obs_elements.extend([distance_to_target, angle_to_target, player.angle-angle_to_target])
        if len(self.targets) < 2:
            n_append = 2 - len(self.targets)
            for _ in range(n_append):
                obs_elements.extend([-1, -1, -1])

        obs_elements.append(len(self.pickups))
        for pickup in self.pickups:       
            distance_to_pickup = np.linalg.norm(player.pos-pickup.pos)/500
            angle_to_pickup = np.arctan2(
                pickup.pos[1]-player.pos[1],
                pickup.pos[0]-player.pos[0]
            )
            obs_elements.extend([distance_to_pickup, angle_to_pickup, player.angle-angle_to_pickup])
        if len(self.pickups) < 2:
            n_append = 2 - len(self.pickups)
            for _ in range(n_append):
                obs_elements.extend([-1, -1, -1])

        obs_elements.append(bar.value)


        obs = np.array(obs_elements).astype(np.float32)
        
        return obs
    

