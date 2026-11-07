CLASSES = {
    'potsdam': ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Background'],
    'loveda': ['ignore', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture', 'Background'],
    'deepglobe': ['Agriculture', 'Urban', 'Rangeland', 'Forest', 'Water', 'Barren', 'ignore'],
    'landslide': ['Background', 'Landslide'],
}

LANDSLIDE_ENV_CLASSES = [
    'Background',
    'Landslide',
    'Building',
    'Road',
    'Vegetation',
    'Water',
    'BareSoil',
    'Infrastructure',
]

LANDSLIDE_ENV_COLORS = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [128, 128, 128],
    [128, 0, 128],
]