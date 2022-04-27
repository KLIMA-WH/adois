import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from natsort import natsorted
from shapely.geometry import Polygon

from src.utils.package import utils

warnings.filterwarnings(action='ignore', category=FutureWarning)


class TileAggregator:
    """TileAggregator
    TODO: class documentation

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """
    DECIMAL_PLACES = 4

    def __init__(self,
                 dir_path,
                 epsg_code,
                 resolution,
                 image_size,
                 tiles_per_dimension=1):
        """Constructor method

        :param str or Path dir_path: path to the directory
        :param int epsg_code: epsg code of the coordinate reference system
        :param float resolution: resolution in meters per pixel
        :param int image_size: image size in pixels
        :param int tiles_per_dimension: number of tiles per dimension of the mask
        :returns: None
        :rtype: None
        """
        self.dir_path = Path(dir_path)
        self.epsg_code = epsg_code
        self.resolution = resolution
        self.image_size = image_size
        self.image_size_meters = self.resolution * self.image_size
        self.image_pixels = image_size ** 2
        self.tiles_per_dimension = tiles_per_dimension
        (self.dir_path.parents[0] / 'aggregated').mkdir(exist_ok=True)

    def get_imperviousness_density(self, mask):
        """Returns the imperviousness density of a mask.

        :param np.ndarray[int] mask: mask
        :returns: imperviousness density
        :rtype: float
        """
        impervious_pixels = np.count_nonzero(mask)
        return round(float(impervious_pixels / self.image_pixels), TileAggregator.DECIMAL_PLACES)

    def get_polygon(self, coordinates):
        """Returns a polygon of a tile given its coordinates of the top left corner.

        :param (float, float) coordinates: coordinates (x, y) of the top left corner
        :returns: polygon
        :rtype: Polygon
        """
        bounding_box = utils.get_bounding_box(coordinates=coordinates,
                                              image_size_meters=self.image_size_meters / self.tiles_per_dimension)
        polygon = Polygon([[bounding_box[0], bounding_box[1]],
                           [bounding_box[0], bounding_box[3]],
                           [bounding_box[2], bounding_box[3]],
                           [bounding_box[2], bounding_box[1]]])
        return polygon

    def __call__(self):
        """Exports all tiles of an area as one shape file to the aggregated directory.
        In each tiles shape is an attribute imp_dens with the value of its imperviousness density.

        :returns: None
        :rtype: None
        """
        imperviousness_density_list = []
        polygon_list = []

        dir_file_list = natsorted([x for x in self.dir_path.iterdir() if x.suffix == '.tiff'])
        for index, file in enumerate(dir_file_list):
            mask = np.array(Image.open(file))
            _, _, coordinates = utils.get_image_metadata(file)

            tile_size = int(self.image_size / self.tiles_per_dimension)
            tile_size_meters = self.image_size_meters / self.tiles_per_dimension

            for row in range(self.tiles_per_dimension):
                for column in range(self.tiles_per_dimension):
                    mask_tile = mask[row * tile_size:(row + 1) * tile_size,
                                     column * tile_size:(column + 1) * tile_size]
                    imperviousness_density = TileAggregator.get_imperviousness_density(self, mask_tile)
                    imperviousness_density_list.append(imperviousness_density)

                    coordinates_tile = (coordinates[0] + column * tile_size_meters,
                                        coordinates[1] - row * tile_size_meters)
                    polygon = TileAggregator.get_polygon(self, coordinates_tile)
                    polygon_list.append(polygon)

        shapes = gpd.GeoDataFrame({'imp_dens': imperviousness_density_list},
                                  crs={'init': f'epsg:{self.epsg_code}'},
                                  geometry=polygon_list)
        shapes.to_file(str(self.dir_path.parents[0] / 'aggregated' / 'aggregated_tiles.shp'))
