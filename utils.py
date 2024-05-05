import random
import math

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


# 生成随机的速度，并确保模长为1
def generate_random_velocity():
    angle = random.uniform(0, 2 * math.pi)  # 随机选择速度的方向
    velocity_x = math.cos(angle)
    velocity_y = math.sin(angle)
    return [velocity_x, velocity_y]



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