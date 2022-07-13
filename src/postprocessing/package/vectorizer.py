# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features
import topojson as tp
from PIL import Image
from natsort import natsorted

from src.utils.package import utils


class Vectorizer:
    def __init__(self,
                 dir_path,
                 masks_dir_path,
                 epsg_code,
                 resolution,
                 sieve_size,
                 tolerance):
        """Constructor method

        :param str or Path dir_path: path to the directory
        :param str or Path masks_dir_path: path to the directory of the masks
        :param int epsg_code: epsg code of the coordinate reference system
        :param float resolution: resolution in meters per pixel
        :param int or None sieve_size: sieve size in pixels (minimum number of pixels to retain)
        :param float or None tolerance: tolerance in meters for the simplification algorithm
        :returns: None
        :rtype: None
        """
        self.dir_path = Path(dir_path)
        self.masks_dir_path = Path(masks_dir_path)
        self.epsg_code = epsg_code
        self.resolution = resolution
        self.sieve_size = sieve_size
        self.tolerance = tolerance

        (self.dir_path / 'vectorized').mkdir(exist_ok=True)

    def __call__(self):
        """Exports the vectorized masks to the shape files directory.

        :returns: None
        :rtype: None
        """
        masks = natsorted([x.name for x in self.masks_dir_path.iterdir() if x.suffix == '.tiff'])
        features = []

        for mask in masks:
            _, _, mask_coordinates = utils.get_image_metadata(mask)
            with Image.open(self.masks_dir_path / mask) as file:
                # noinspection PyTypeChecker
                mask = np.array(file)

            if self.sieve_size is not None:
                mask = rio.features.sieve(mask, size=self.sieve_size)

            transform = rio.transform.from_origin(west=mask_coordinates[0],
                                                  north=mask_coordinates[1],
                                                  xsize=self.resolution,
                                                  ysize=self.resolution)
            vectorized_mask = rio.features.shapes(mask, transform=transform)
            mask_features = [{'properties': {'class': int(value)}, 'geometry': shape}
                             for shape, value in vectorized_mask if
                             int(value) != 0]
            features.extend(mask_features)

        shapes = gpd.GeoDataFrame.from_features(features)
        shapes.set_crs(epsg=self.epsg_code, inplace=True)

        if self.tolerance is not None:
            topo_shapes = tp.Topology(shapes, prequantize=False)
            shapes = topo_shapes.toposimplify(self.tolerance).to_gdf()

        shapes.to_file(str(self.dir_path / 'vectorized' / 'vectorized.shp'))
