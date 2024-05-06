import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
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