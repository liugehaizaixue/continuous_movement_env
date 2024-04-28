import pygame
import sys
import math
from utils import generate_random_spawn_point , generate_random_velocity

# 初始化Pygame
pygame.init()

# 窗口大小
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 设置窗口
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("智能体连续运动")

# 小球初始位置、半径和速度
ball_radius = 20
ball_velocity = generate_random_velocity()

# 障碍物列表，每个障碍物表示为 (x, y, radius)
obstacles = [(300, 200, 20), (500, 400, 30), (200, 500, 25)]
ball_pos = generate_random_spawn_point(ball_radius, obstacles, WINDOW_WIDTH, WINDOW_HEIGHT )

# 游戏主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 清屏
    window.fill(WHITE)

    # 更新小球位置
    ball_pos[0] += ball_velocity[0]
    ball_pos[1] += ball_velocity[1]

    # 边界检测
    if ball_pos[0] - ball_radius < 0 or ball_pos[0] + ball_radius > WINDOW_WIDTH:
        ball_velocity[0] *= -1
    if ball_pos[1] - ball_radius < 0 or ball_pos[1] + ball_radius > WINDOW_HEIGHT:
        ball_velocity[1] *= -1

    ball_center = pygame.Vector2(ball_pos)
    print(ball_center)
    # 检测碰撞
    for obstacle in obstacles:
        obstacle_pos = pygame.Vector2(obstacle[0], obstacle[1])
        obstacle_radius = obstacle[2]
        if obstacle_pos.distance_to(ball_center) < ball_radius + obstacle_radius:
            # 发生碰撞，根据需要更新小球的速度或位置
            ball_velocity[0] *= -1
            ball_velocity[1] *= -1

    # 绘制障碍物
    for obstacle in obstacles:
        pygame.draw.circle(window, BLACK, (obstacle[0], obstacle[1]), obstacle[2])

    # 绘制小球
    pygame.draw.circle(window, BLUE, ball_pos, ball_radius)
    # 计算箭头位置和角度
    arrow_length = 20
    arrow_angle = math.atan2(ball_velocity[1], ball_velocity[0])
    arrow_end_x = ball_pos[0] + arrow_length * math.cos(arrow_angle)
    arrow_end_y = ball_pos[1] + arrow_length * math.sin(arrow_angle)

    # 绘制箭头
    pygame.draw.line(window, RED, ball_pos, (arrow_end_x, arrow_end_y), 2)

    # 更新窗口
    pygame.display.flip()

    # 控制帧率
    pygame.time.Clock().tick(60)

# 退出Pygame
pygame.quit()
sys.exit()
