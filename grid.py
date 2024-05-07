from copy import deepcopy
import warnings

import numpy as np
import math
from grid_config import GridConfig
from utils import generate_points , expand_matrix , generate_random_velocities

class Grid:
    def __init__(self, grid_config: GridConfig, add_artificial_border: bool = True, num_retries=10):

        self.config = grid_config
        self.rnd = np.random.default_rng(grid_config.seed)

        self.obstacles = np.array([np.array(line) for line in self.config.map])

        self.obstacles = self.obstacles.astype(np.int32)
        self.obstacles = np.array(expand_matrix(self.obstacles , factor=100))
        self.starts_xy, self.finishes_xy = generate_points(self.obstacles, r=self.config.agents_radius, k = self.config.num_agents)
        self.starts_speed = generate_random_velocities(k = self.config.num_agents , seed=grid_config.seed)

        if not self.starts_xy or not self.finishes_xy or len(self.starts_xy) != len(self.finishes_xy):
            raise OverflowError(
                "Can't create task. Please check grid grid_config, especially density, num_agent and map.")

        if add_artificial_border:
            self.add_artificial_border()

        filled_positions = np.zeros(self.obstacles.shape)
        for x, y in self.starts_xy:
            filled_positions = self.fill_matrix_with_radius(x, y, filled_positions)

        self.positions = filled_positions
        self.agents_speed = self.starts_speed
        self.positions_xy = self.starts_xy
        self._initial_xy = deepcopy(self.starts_xy)
        self.is_active = {agent_id: True for agent_id in range(self.config.num_agents)}

    def fill_matrix_with_radius(self, x, y, matrix):
        """ Fill matrix with 1 in a circle of radius r around (x, y) """
        r = self.config.agents_radius
        for i in range(x - r, x + r + 1):
            for j in range(y - r, y + r + 1):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    matrix[i, j] = self.config.OBSTACLE
        return matrix

    def unfill_matrix_with_radius(self, x, y, matrix):
        r = self.config.agents_radius
        for i in range(x - r, x + r + 1):
            for j in range(y - r, y + r + 1):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    matrix[i, j] = self.config.FREE
        return matrix

    def add_artificial_border(self):
        gc = self.config
        r = gc.obs_radius
        if gc.empty_outside:
            filled_obstacles = np.zeros(np.array(self.obstacles.shape) + r * 2)
        else:
            filled_obstacles = np.ones(np.array(self.obstacles.shape) + r * 2)

        height, width = filled_obstacles.shape
        filled_obstacles[r - 1, r - 1:width - r + 1] = gc.OBSTACLE
        filled_obstacles[r - 1:height - r + 1, r - 1] = gc.OBSTACLE
        filled_obstacles[height - r, r - 1:width - r + 1] = gc.OBSTACLE
        filled_obstacles[r - 1:height - r + 1, width - r] = gc.OBSTACLE
        filled_obstacles[r:height - r, r:width - r] = self.obstacles

        self.obstacles = filled_obstacles

        self.starts_xy = [(x + r, y + r) for x, y in self.starts_xy]
        self.finishes_xy = [(x + r, y + r) for x, y in self.finishes_xy]

    def get_obstacles(self, ignore_borders=False):
        gc = self.config
        if ignore_borders:
            return self.obstacles[gc.obs_radius:-gc.obs_radius, gc.obs_radius:-gc.obs_radius].copy()
        return self.obstacles.copy()

    @staticmethod
    def _cut_borders_xy(positions, obs_radius):
        return [[x - obs_radius, y - obs_radius] for x, y in positions]

    @staticmethod
    def _filter_inactive(pos, active_flags):
        return [pos for idx, pos in enumerate(pos) if active_flags[idx]]

    def get_grid_config(self):
        return deepcopy(self.config)


    def _prepare_positions(self, positions, only_active, ignore_borders):
        gc = self.config

        if only_active:
            positions = self._filter_inactive(positions, [idx for idx, active in self.is_active.items() if active])

        if ignore_borders:
            positions = self._cut_borders_xy(positions, gc.obs_radius)

        return positions

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.positions_xy), only_active, ignore_borders)

    @staticmethod
    def to_relative(coordinates, offset):
        result = deepcopy(coordinates)
        for idx, _ in enumerate(result):
            x, y = result[idx]
            dx, dy = offset[idx]
            result[idx] = x - dx, y - dy
        return result

    def get_agents_speed_relative(self):
        return self.agents_speed

    def get_agents_xy_relative(self):
        return self.to_relative(self.positions_xy, self._initial_xy)

    def get_targets_xy_relative(self):
        return self.to_relative(self.finishes_xy, self._initial_xy)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self._prepare_positions(deepcopy(self.finishes_xy), only_active, ignore_borders)

    def _normalize_coordinates(self, coordinates):
        gc = self.config

        x, y = coordinates

        x -= gc.obs_radius
        y -= gc.obs_radius

        x /= gc.size - 1
        y /= gc.size - 1

        return x, y

    def get_state(self, ignore_borders=False, as_dict=False):
        agents_xy = list(map(self._normalize_coordinates, self.get_agents_xy(ignore_borders)))
        targets_xy = list(map(self._normalize_coordinates, self.get_targets_xy(ignore_borders)))

        obstacles = self.get_obstacles(ignore_borders)

        if as_dict:
            return {"obstacles": obstacles, "agents_xy": agents_xy, "targets_xy": targets_xy}

        return np.concatenate(list(map(lambda x: np.array(x).flatten(), [agents_xy, targets_xy, obstacles])))

    def get_observation_shape(self):
        full_radius = self.config.obs_radius * 2 + 1
        return 2, full_radius, full_radius


    def get_obstacles_for_agent(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        obs_obstacles = self.obstacles[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)
        return obs_obstacles

    def get_positions(self, agent_id):
        x, y = self.positions_xy[agent_id]
        r = self.config.obs_radius
        return self.positions[x - r:x + r + 1, y - r:y + r + 1].astype(np.float32)
    
    def get_target(self, agent_id):

        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        if x == fx and y == fy:
            return 0.0, 0.0
        rx, ry = fx - x, fy - y
        dist = np.sqrt(rx ** 2 + ry ** 2)
        return rx / dist, ry / dist

    def get_square_target(self, agent_id):
        c = self.config
        full_size = self.config.obs_radius * 2 + 1
        result = np.zeros((full_size, full_size))
        x, y = self.positions_xy[agent_id]
        fx, fy = self.finishes_xy[agent_id]
        dx, dy = x - fx, y - fy

        dx = min(dx, c.obs_radius) if dx >= 0 else max(dx, -c.obs_radius)
        dy = min(dy, c.obs_radius) if dy >= 0 else max(dy, -c.obs_radius)
        result[c.obs_radius - dx, c.obs_radius - dy] = 1
        return result.astype(np.float32)

    def render(self, mode='human'):
        pass

    def has_obstacle(self, x, y):
        return self.obstacles[x, y] == self.config.OBSTACLE


    def check_free(self, x, y, matrix):
        """ x and y are in the center of the agent 
            r is the radius of the agent
            matrix is the matrix of the grid
        """
        r = self.config.agents_radius
        for i in range(x - r, x + r + 1):
            for j in range(y - r, y + r + 1):
                if (i - x) ** 2 + (j - y) ** 2 <= r ** 2:
                    if matrix[i, j] == self.config.OBSTACLE:
                        return False
        return True

    def try_move(self, x, y, fake_x, fake_y):
        # 要先排除自己的体积
        fake_positions = deepcopy(self.positions)
        fake_positions = self.unfill_matrix_with_radius(x, y, fake_positions)
        if self.check_free(fake_x, fake_y, self.obstacles) and self.check_free(fake_x, fake_y, fake_positions):
            return True
        else:
            return False

    def move(self, agent_id, action):
        self.agents_speed[agent_id] = action
        x, y = self.positions_xy[agent_id]
        # 计算原始向量的模长
        magnitude = np.linalg.norm(action)
        # 计算方向相同但模长为1的向量
        unit_vector = action / magnitude
        k = int(magnitude)
        movements = []
        #先收集dx dy 后统一处理
        for i in range(k):
            """ 按单位向量方向走k步 """
            dx , dy = unit_vector
            movements.append((dx, dy))
        #先收集dx dy 后统一处理
        if magnitude % 1 != 0:
            """ 走完剩余长度不到1的部分 """
            dx , dy = (magnitude % 1)*unit_vector
            movements.append((dx, dy))
        
        for dx, dy in movements:
            fake_x, fake_y = math.ceil(x + dx), math.ceil(y + dy)
            if self.try_move(x, y, fake_x, fake_y):
                # 移动成功
                self.positions = self.unfill_matrix_with_radius(x, y, self.positions)
                self.positions = self.fill_matrix_with_radius(fake_x, fake_y, self.positions)
                self.positions_xy[agent_id] = fake_x, fake_y
                x , y = fake_x, fake_y
            else:
                # 移动失败
                break

    def on_goal(self, agent_id):
        return self.positions_xy[agent_id] == self.finishes_xy[agent_id]

    def is_active(self, agent_id):
        return self.is_active[agent_id]

    def hide_agent(self, agent_id):
        if not self.is_active[agent_id]:
            return False
        self.is_active[agent_id] = False

        x , y = self.positions_xy[agent_id]
        self.positions = self.unfill_matrix_with_radius(x, y, self.positions)
        return True

