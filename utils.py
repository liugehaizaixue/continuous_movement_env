import random
import math
import numpy as np
from copy import deepcopy

def random_point(matrix, r, seed=42):
    """ 随机选择点 , 保证选择的点的r范围内不是边界 也不是 障碍1
    matrix 为矩阵
    r 为范围
    """
    random.seed(seed)
    height = len(matrix)
    width = len(matrix[0])

    attempts = height * width  # 最多尝试次数

    while attempts > 0:
        y = random.randint(r, width - r - 1)
        x = random.randint(r, height - r - 1)

        x_min = x - r
        x_max = x + r + 1
        y_min = y - r
        y_max = y + r + 1
        # 创建一个与需要检查的区域大小相同的掩码矩阵
        mask = np.zeros((x_max - x_min, y_max - y_min), dtype=bool)
        # 计算掩码矩阵中每个元素是否在半径范围内
        r_squared = r ** 2
        i, j = np.ogrid[x_min:x_max, y_min:y_max]
        mask[((i - x) ** 2 + (j - y) ** 2) <= r_squared] = True
        # 检查需要检查的区域中的元素是否为障碍物1
        region = matrix[x_min:x_max, y_min:y_max]
        if np.any(region[mask] == 1):
            attempts -= 1
            continue
        return (x, y)        

    return None

def distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def generate_points(matrix , r , k , seed=42):
    """ 
        在matrix上
        生成k对 start与tartget
        其中starts之间满足距离大于2r，且start的r范围内没有障碍或边界外
    """
    matrix = np.array([row[:] for row in matrix])
    height = len(matrix)
    width = len(matrix[0])
    start_points = []
    target_points = []
    attempts = height * width
    while len(start_points) < k and attempts > 0:
        seed = seed + 1
        attempts -= 1
        s_valid = True    
        start = random_point(matrix , r , seed)
        if start:
            for point in start_points:
                if distance(start, point) < (2 * r)**2:
                    s_valid = False
                    break
            if s_valid: # 成功找到新的start
                start_points.append(start)

    attempts = height * width
    while len(target_points) < k and attempts > 0:
        seed = seed + 1
        attempts -= 1
        target = random_point(matrix , r, seed)
        if target:
            if target not in target_points:
                target_points.append(target)


    if len(start_points) == k and len(target_points) == k :
        return start_points , target_points
    else:
        raise ValueError("generate points error")


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

