from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from natsort import natsorted
from shapely.geometry import Polygon

from src.utils.package import utils


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
                 image_size):
        """Constructor method

        :param str or Path dir_path: path to the directory
        :param int epsg_code: epsg code of the coordinate reference system
        :param float resolution: resolution in meters per pixel
        :param int image_size: image size in pixels
        :returns: None
        :rtype: None
        """
        self.dir_path = Path(dir_path)
        self.epsg_code = epsg_code
        self.resolution = resolution
        self.image_size = image_size
        self.image_size_meters = self.resolution * self.image_size
        self.image_pixels = image_size ** 2
        (self.dir_path.parents[0] / 'aggregated').mkdir(exist_ok=True)

    def get_imperviousness_density(self, mask):
        """Returns the imperviousness density of a mask.

        :param np.ndarray[int] mask: mask
        :returns: imperviousness density
        :rtype: float
        """
        impervious_pixels = np.count_nonzero(mask)
        return round(float(impervious_pixels / self.image_pixels), TileAggregator.DECIMAL_PLACES)

    def get_tile_polygon(self, coordinates):
        """Returns a polygon of a tile given its coordinates of the top left corner.

        :param (float, float) coordinates: coordinates (x, y) of the top left corner
        :returns: polygon
        :rtype: Polygon
        """
        bounding_box = utils.get_bounding_box(coordinates=coordinates, image_size_meters=self.image_size_meters)
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
        tile_polygon_list = []

        dir_file_list = natsorted([x for x in self.dir_path.iterdir() if x.suffix == '.tiff'])
        for index, file in enumerate(dir_file_list):
            mask = np.array(Image.open(file))
            imperviousness_density = TileAggregator.get_imperviousness_density(self, mask)
            imperviousness_density_list.append(imperviousness_density)

            _, _, coordinates = utils.get_image_metadata(file)
            tile_polygon = TileAggregator.get_tile_polygon(self, coordinates)
            tile_polygon_list.append(tile_polygon)

        shapes = gpd.GeoDataFrame({'imp_dens': imperviousness_density_list},
                                  crs={'init': f'epsg:{self.epsg_code}'},
                                  geometry=tile_polygon_list)
        shapes.to_file(str(self.dir_path.parents[0] / 'aggregated' / 'aggregated_tiles.shp'))
