# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from src.utils.package import utils


class GridGenerator:
    def __init__(self,
                 dir_path,
                 bounding_box,
                 tile_size_meters,
                 epsg_code):
        """Constructor method

        :param str or Path dir_path: path to the directory
        :param (float, float, float, float) bounding_box: bounding box (x_1, y_1, x_2, y_2) of the area from
            the bottom left corner to the top right corner
        :param int tile_size_meters: tile size in meters
        :param int epsg_code: epsg code of the coordinate reference system
        :returns: None
        :rtype: None
        :raises ValueError: if bounding_box is not valid (x_1 >= x_2 or y_1 >= y_2)
        """
        self.dir_path = Path(dir_path)

        if bounding_box[0] < bounding_box[2] and bounding_box[1] < bounding_box[3]:
            self.bounding_box = (round(bounding_box[0], 2),
                                 round(bounding_box[1], 2),
                                 round(bounding_box[2], 2),
                                 round(bounding_box[3], 2))
        else:
            raise ValueError('Invalid bounding_box! x_1 has to be smaller than x_2 and '
                             'y_1 has to be smaller than y_2.')

        self.tile_size_meters = tile_size_meters
        self.epsg_code = epsg_code

        (self.dir_path / f'grid_{self.tile_size_meters}').mkdir(exist_ok=True)

    def get_polygon(self, coordinates):
        """Returns a polygon of a tile given its coordinates of the top left corner.

        :param (float, float) coordinates: coordinates (x, y) of the top left corner
        :returns: polygon
        :rtype: Polygon
        """
        bounding_box = utils.get_bounding_box(coordinates=coordinates,
                                              image_size_meters=self.tile_size_meters)
        polygon = Polygon([[bounding_box[0], bounding_box[1]],
                           [bounding_box[0], bounding_box[3]],
                           [bounding_box[2], bounding_box[3]],
                           [bounding_box[2], bounding_box[1]]])
        return polygon

    def __call__(self):
        """Exports a grid of an area given its bounding box to the shape files directory.

        :returns: None
        :rtype: None
        """
        columns = int((self.bounding_box[2] - self.bounding_box[0]) // self.tile_size_meters)
        if (self.bounding_box[2] - self.bounding_box[0]) % self.tile_size_meters:
            columns += 1
        rows = int((self.bounding_box[3] - self.bounding_box[1]) // self.tile_size_meters)
        if (self.bounding_box[3] - self.bounding_box[1]) % self.tile_size_meters:
            rows += 1

        polygons = []

        for row in range(rows):
            for column in range(columns):
                coordinates = (round(self.bounding_box[0] + column * self.tile_size_meters, 2),
                               round(self.bounding_box[1] + (row + 1) * self.tile_size_meters, 2))
                polygon = self.get_polygon(coordinates)
                polygons.append(polygon)

        shapes = gpd.GeoDataFrame(crs=f'epsg:{self.epsg_code}', geometry=polygons)
        shapes.to_file(str(self.dir_path / f'grid_{self.tile_size_meters}' / f'grid_{self.tile_size_meters}.shp'))
