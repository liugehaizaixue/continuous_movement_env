import random
import math
import numpy as np
from copy import deepcopy
# 定义函数来生成一个不与障碍物或边界相交的随机出生点
def generate_random_spawn_point(ball_radius, obstacles, WINDOW_WIDTH, WINDOW_HEIGHT , seed = 0):
    random.seed(seed)
    while True:
        # 生成随机的x和y坐标
        random_x = random.randint(ball_radius, WINDOW_WIDTH - ball_radius)
        random_y = random.randint(ball_radius, WINDOW_HEIGHT - ball_radius)
        
        # 检查是否与边界相交
        if random_x - ball_radius < 0 or random_x + ball_radius > WINDOW_WIDTH:
            continue
        if random_y - ball_radius < 0 or random_y + ball_radius > WINDOW_HEIGHT:
            continue
        
        # 检查是否与障碍物相交
        intersect = False
        for obstacle in obstacles:
            obstacle_x, obstacle_y, obstacle_radius = obstacle
            distance = ((random_x - obstacle_x) ** 2 + (random_y - obstacle_y) ** 2) ** 0.5
            if distance < ball_radius + obstacle_radius:
                intersect = True
                break
        
        if not intersect:
            return [random_x, random_y]


def random_point(matrix, r):
    """ 随机选择点 , 保证选择的点的r范围内不是边界 也不是 障碍1
    matrix 为矩阵
    r 为范围
    """
    height = len(matrix)
    width = len(matrix[0])

    attempts = height * width  # 最多尝试次数

    while attempts > 0:
        x = random.randint(r, width - r - 1)
        y = random.randint(r, height - r - 1)

        valid = True
        for i in range(y - r, y + r + 1):
            for j in range(x - r, x + r + 1):
                if i < 0 or i >= height or j < 0 or j >= width or matrix[i][j] == 1:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return (x, y)

        attempts -= 1

    return None

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def generate_points(matrix , r , k):
    """ 
        在matrix上
        生成k对 start与tartget
        其中starts之间满足距离大于2r，且start的r范围内没有障碍或边界外
    """
    matrix = deepcopy(matrix)
    height = len(matrix)
    width = len(matrix[0])
    start_points = []
    target_points = []
    attempts = height * width
    while len(start_points) < k and attempts > 0:
        attempts -= 1
        s_valid = True    
        start = random_point(matrix , r)
        if start:
            for point in start_points:
                if distance(start, point) < 2 * r:
                    s_valid = False
                    break
            if s_valid: # 成功找到新的start
                start_points.append(start)

    attempts = height * width
    while len(target_points) < k and attempts > 0:
        attempts -= 1
        target = random_point(matrix , r)
        if target:
            if target not in target_points:
                target_points.append(target)


    if len(start_points) == k and len(target_points) == k :
        return start_points , target_points
    else:
        return None , None


def generate_random_velocities(k , seed=42):
    """ 生成随机的速度，并确保模长为1 """
    velocities = []
    for i in range(k):
        random.seed(seed)
        angle = random.uniform(0, 2 * math.pi)  # 随机选择速度的方向
        velocity_x = math.cos(angle)
        velocity_y = math.sin(angle)
        velocities.append([velocity_x, velocity_y])
    return velocities

def expand_matrix(matrix, factor):
    """ 
    按照比例倍数扩展原矩阵
    matrix 原矩阵

    factor 扩展倍数
    """
    expanded_matrix = []
    for row in matrix:
        expanded_row = []
        for element in row:
            expanded_element = [element] * factor
            expanded_row.extend(expanded_element)
        expanded_row = [expanded_row] * factor
        expanded_matrix.extend(expanded_row)
    return expanded_matrix

def print_matrix(matrix):
    """ 打印矩阵 """
    for row in matrix:
        print("".join(row))


def get_map_array(map_str):
    map_array = np.array([[1 if c == '#' else 0 for c in line] for line in map_str.strip().split('\n')])
    return map_array

