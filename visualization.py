from custom_maps import MAPS_REGISTRY
from utils import *

map_str = MAPS_REGISTRY["sc1-AcrosstheCape"]

_map = get_map_array(map_str)

_map = expand_matrix(_map , factor= 10)

# visual_map(_map)

# point = random_point(_map , 5)
starts , tartgets = generate_points(_map,5,128)

print(starts , tartgets)

