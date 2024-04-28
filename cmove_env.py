from utils import generate_random_spawn_point , generate_random_velocity
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
    def __init__(self, render_mode="human"):
        super(CMoveEnv, self).__init__()
        self.action_space = spaces.Box(-1, 1, shape=(2, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(5, 5), dtype=np.float32)
        self.pos = None
        self.velocity = None
        self.radius = 20
        self.width, self.height = (800 , 600)
        self.obstacles = [(300, 200, 20), (500, 400, 30), (200, 500, 25)]

    def step(self, action):
        self.velocity = action

        obs = np.zeros(shape=(5,5)).astype(np.float32)
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return obs , reward , done ,truncated , info

    def reset(self, *, seed=None, options=None):
        self.pos = generate_random_spawn_point(ball_radius=self.radius , obstacles=self.obstacles, WINDOW_HEIGHT=self.height, WINDOW_WIDTH=self.width)
        self.velocity = generate_random_velocity()
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
        o, r, done, _, _ = env.step(action)
        env.render()
    env.close()
    print("over")