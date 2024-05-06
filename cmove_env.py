import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from copy import deepcopy


gym.register(
    id="CMove-v0",
    entry_point="cmove_env:CMoveEnv",
    max_episode_steps=64,
)

class CMoveEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    def __init__(self,num_agents=64, render_mode="human"):
        super(CMoveEnv, self).__init__()
        self.pos = None
        self.velocity = None
        self.radius = 30
        self.map = None
        self.num_agents = num_agents
        self.action_space = spaces.Box(-20, 20, shape=(2, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100), dtype=np.float32)

    def step(self, action):
        self.velocity = action

        obs = np.zeros(shape=(5,5)).astype(np.float32)
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return obs , reward , done ,truncated , info

    def reset(self, *, seed=None, options=None):
        obs = np.zeros(shape=(5,5)).astype(np.float32)
        info = {}
        return obs , info


    def render(self):
        pass
    
    def close(self):
        return super().close()
    


if __name__ == "__main__":
    env = gym.make("CMove-v0")
    o, _ = env.reset()
    env.render()
    done = False
    for i in range(20):
        action = env.action_space.sample()
        print(action)
        o, r, done, _, _ = env.step(action)
        env.render()
    env.close()
    print("over")