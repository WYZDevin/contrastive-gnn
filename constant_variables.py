
import torch

FEATURES_WITH_SV = [ 
    'road_275m', 'sidewalk_175m', 'building_375m',
    'wall_275m', 'fence_400m', 'pole_150m', 't_light_200m', 't_sign_175m',
    'vegetation_125m', 'terrain_125m', 'sky_175m', 'person_150m',
    'rider_375m', 'car_150m', 'bus_275m',
    'Residential_50m',
    'TransportationRelated_50m', 'Commercial_125m', 'Industrial_375m',
    'RecreationalAndGreenlands_250m', 'Mixed_125m',  'LocalStreet_25m',
    'Arterial_50m', 'Collector_25m', 
    'distance_to_airport',
    'distance_to_roads_Arterial', 'distance_to_roads_Collector',
    'distance_to_roads_Freeway', 'distance_to_roads_Local_or_Street',
    'distance_to_roads_Rapid_Transit', 'ndvi_150m'
]

FEATURES_50M = [
#     'road_50m',
#     'sidewalk_50m',
#     'building_50m',
#  'wall_50m',
#     'fence_50m',
#     'pole_50m',
#     't_light_50m',
#     't_sign_50m',
#     'vegetation_50m',
#     'terrain_50m',
#     'sky_50m',
#     'person_50m',
#  'rider_50m',
#  'car_50m',
#  'truck_50m',
#  'bus_50m',
#  'train_50m',
#  'motorcycle_50m',
#  'bicycle_50m',
 'Collector_50m',
 'LocalStreet_50m',
 'Arterial_50m',
 'Freeway_50m',
#  'RapidTransit_50m',
#  'ndvi_50m',
#  'Residential_50m',
#  'TransportationRelated_50m',
#  'Commercial_50m',
#  'Industrial_50m',
#  'Others_50m',
#  'RecreationalAndGreenlands_50m',
#  'Mixed_50m',
 'distance_to_airport',
 'distance_to_roads_Arterial',
 'distance_to_roads_Collector',
 'distance_to_roads_Freeway',
    'distance_to_roads_Local_or_Street',
    'distance_to_roads_Rapid_Transit'
]
FEATURES = FEATURES_WITH_SV
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

