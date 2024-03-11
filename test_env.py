import numpy as np

from gym_env import GameEnv


def main():

    env = GameEnv("configs/game_config.yml")
    obs = env.reset()

    print("Initial obs:")
    print(obs)    
    print("Observation space:")
    print(env.observation_space)
    print("")
    print("Action space:")
    print(env.action_space)
    print("")
    print("Action space sample:")
    print(env.action_space.sample())


    n_steps = 1000
    s = 0
    for step in range(n_steps):
        action = np.random.choice(5)
        print("Step {}".format(step + 1))
        obs, reward, done, _ = env.step(action)
        s += reward
        print("obs=", obs, "reward=", reward, "done=", done)
        # env.render()
        if done:
            print("Done!", "reward=", s)
            break

if __name__=="__main__":
    main()