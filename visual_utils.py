import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np

COLORS = ['#FFFFFF', '#000000']
def visual_map(map): 
    cmap = ListedColormap(COLORS)
    rendering = plt.imshow(map, cmap=cmap, interpolation='none')
    # for i in range(len(map)):
    #     for j in range(len(map[0])):
    #         rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='black', facecolor='none', linewidth=0.2)
    #         plt.gca().add_patch(rect)

    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.show()


# def visual_map_with_agents(map): 
#     cmap = mcolors.ListedColormap(COLORS + [plt.cm.jet(i) for i in np.random.rand(256)])
#     rendering = plt.imshow(map, cmap=cmap, interpolation='none')

#     plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
#     plt.show()


def visual_matrix(matrix):
    cmap = ListedColormap(COLORS)
    rendering = plt.imshow(matrix, cmap=cmap, interpolation='none')
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.show()   


def visual_state(env):
    grid = env.env.env.env.env.grid
    agents_radius = grid.config.agents_radius
    matrix = grid.obstacles
    pos_list = grid.positions_xy
    target_list = grid.finishes_xy
    cmap = ListedColormap(COLORS)
    rendering = plt.imshow(matrix, cmap=cmap, interpolation='none')
    for pos in pos_list:
        """ plt 的横轴为x，纵轴为y 与实际坐标系相反 """
        circle = patches.Circle((pos[1], pos[0]), radius=agents_radius, edgecolor='none', facecolor='blue')
        plt.gca().add_patch(circle)
    for target in target_list:
        """ plt 的横轴为x，纵轴为y 与实际坐标系相反 """
        circle = patches.Circle((target[1], target[0]), radius=agents_radius, edgecolor='r', facecolor='none')
        plt.gca().add_patch(circle)
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.show()   