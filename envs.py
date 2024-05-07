from typing import Optional

import numpy as np
import gymnasium
from gymnasium.error import ResetNeeded

from grid import Grid
from grid_config import GridConfig
from wrappers.metrics import EpLengthMetric, ISRMetric, CSRMetric
from wrappers.multi_time_limit import MultiTimeLimit


class ActionsSampler:
    """
    Samples the random actions for the given number of agents using the given seed.
    """

    def __init__(self, v=1.0, seed=42):
        self._v = v
        self._rnd = None
        self.update_seed(seed)

    def update_seed(self, seed=None):
        self._rnd = np.random.default_rng(seed)

    def sample_actions(self, dim=1):
        vectors = []
        for _ in range(dim):
            vector = np.random.uniform(low=-self._v, high=self._v, size=2)
            vectors.append(vector)
        return np.array(vectors)


class CmoveBase(gymnasium.Env):
    """
    Abstract class of the Cmove environment.
    """
    metadata = {"render_modes": ["ansi"], }

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        raise NotImplementedError

    def __init__(self, grid_config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.grid_config = grid_config

        self.action_space: gymnasium.spaces.Box = gymnasium.spaces.Box(-self.grid_config.speed, self.grid_config.speed, shape=(2, ), dtype=np.float32)
        self._multi_action_sampler = ActionsSampler(self.grid_config.speed, seed=self.grid_config.seed)

    def check_reset(self):
        """
        Checks if the reset needed.
        :return:
        """
        if self.grid is None:
            raise ResetNeeded("Please reset environment first!")

    def render(self, mode='human'):
        """
        Renders the environment using ascii graphics.
        :param mode:
        :return:
        """
        self.check_reset()
        return self.grid.render(mode=mode)

    def sample_actions(self):
        """
        Samples the random actions for the given number of agents.
        :return:
        """
        return self._multi_action_sampler.sample_actions(dim=self.grid_config.num_agents)

    def get_num_agents(self):
        """
        Returns the number of agents in the environment.
        :return:
        """
        return self.grid_config.num_agents


class Cmove(CmoveBase):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.was_on_goal = None
        full_size = self.grid_config.obs_radius * 2 + 1
        self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
            obstacles=gymnasium.spaces.Box(-1.0, 1.0, shape=(full_size, full_size)),
            agents=gymnasium.spaces.Box(-1.0, 1.0, shape=(full_size, full_size)),
            xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            direction = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=int),
        )

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        terminated = []

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            terminated.append(on_goal)

        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.grid.is_active[agent_idx] = False

        infos = self._get_infos()

        observations = self._obs()
        truncated = [False] * self.grid_config.num_agents
        return observations, rewards, terminated, truncated, infos

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def update_was_on_goal(self):
        self.was_on_goal = [self.grid.on_goal(agent_idx) and self.grid.is_active[agent_idx]
                            for agent_idx in range(self.grid_config.num_agents)]

    def reset(self, seed: Optional[int] = None, return_info: bool = True, options: Optional[dict] = None, ):
        self._initialize_grid()
        self.update_was_on_goal()

        if seed is not None:
            self.grid.seed = seed

        if return_info:
            return self._obs(), self._get_infos()
        return self._obs()

    def _obs(self):
        return self._pomapf_obs()

    def _pomapf_obs(self):
        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()
        agents_speed_relative = self.grid.get_agents_speed_relative()

        for agent_idx in range(self.grid_config.num_agents):
            result = {'obstacles': self.grid.get_obstacles_for_agent(agent_idx),
                      'agents': self.grid.get_positions(agent_idx),
                      'xy': agents_xy_relative[agent_idx],
                      'target_xy': targets_xy_relative[agent_idx],
                      'speed': agents_speed_relative[agent_idx]
                      }

            results.append(result)
        return results

    def _get_infos(self):
        infos = [dict() for _ in range(self.grid_config.num_agents)]
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]
        return infos

    def move_agents(self, actions):
        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.is_active[agent_idx]:
                self.grid.move(agent_idx, actions[agent_idx])

    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()

    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)



def _make_cmove(grid_config):
    env = Cmove(grid_config=grid_config)

    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    env = ISRMetric(env)
    env = CSRMetric(env)
    env = EpLengthMetric(env)

    return env
