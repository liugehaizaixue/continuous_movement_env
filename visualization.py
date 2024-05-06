from custom_maps import MAPS_REGISTRY
from utils import get_map_array , expand_matrix , visual_map

map_str = MAPS_REGISTRY["sc1-AcrosstheCape"]

_map = get_map_array(map_str)

_map = expand_matrix(_map , factor= 5)

visual_map(_map)

