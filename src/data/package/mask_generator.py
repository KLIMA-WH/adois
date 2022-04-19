import json
import logging
from datetime import datetime as DateTime  # PEP 8 compliant
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features
import rasterio.mask
from PIL import Image
from natsort import natsorted

from src.data.package import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

log_dir = Path(__file__).parents[1] / 'logs'
date_time = str(DateTime.now().isoformat(sep='_', timespec='seconds')).replace(':', '-')
file_handler = logging.FileHandler(filename=log_dir / f'{date_time}_mask_generator.log', mode='w')
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)


class MaskGenerator:
    """MaskGenerator
    TODO: class documentation

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """
    SINGLE_CLASS_MASK_VALUE = 1
    BANDS = 1

    def __init__(self,
                 metadata_path,
                 mask_shp_path,
                 shp_path=None,
                 multi_class_mask=False,
                 create_wld=False,
                 create_geotiff=False):
        """Constructor method

        :param str metadata_path: path to the metadata file created by TileGenerator
        :param str mask_shp_path: path to the shape file of the mask that needs to be rasterized
        :param str shp_path: path to the shape file for masking specific areas
        :param bool multi_class_mask: if True, the pixel value of each rasterized shape equals the value of the column
            mask_value (shape file may need to be preprocessed)
            if False, the pixel value of the rasterized shapes is 255
        :param bool create_wld: if True, a world file is created
        :param bool create_geotiff: if True, georeferencing metadata is embedded into the image
        :returns: None
        :rtype: None
        """
        self.metadata_path = Path(metadata_path)
        self.dir_path = self.metadata_path.parents[0]
        self.mask_shp_path = mask_shp_path
        if shp_path is not None:
            shapes = gpd.read_file(shp_path)
            self.shapes = list((row.geometry for _, row in shapes.iterrows()))
        else:
            self.shapes = None
        self.multi_class_mask = multi_class_mask
        self.create_wld = create_wld
        self.create_geotiff = create_geotiff
        with open(self.metadata_path, 'r') as file:
            self.metadata = json.load(file)
        self.epsg_code = self.metadata.get('epsg code')
        self.resolution = self.metadata.get('resolution')
        self.image_size = self.metadata.get('image size')
        self.image_size_meters = self.resolution * self.image_size
        (self.dir_path / 'mask').mkdir(exist_ok=True)

    def get_mask(self, path):
        """Returns an image of the mask to a corresponding tile. If necessary, the image is getting masked
        with the shapes of the optional shape file.

        :param str path: path to the image of the corresponding tile
        :returns: image and coordinates
        :rtype: (np.ndarray[int], (float, float))
        """
        image_metadata = Path(path).stem.split('_')
        coordinates = (float(image_metadata[-2]), float(image_metadata[-1]))
        bounding_box = utils.get_bounding_box(coordinates=coordinates,
                                              image_size_meters=self.image_size_meters)

        shapes = gpd.read_file(self.mask_shp_path, bbox=bounding_box)
        if self.multi_class_mask:
            shapes = list((row.geometry, row.mask_value) for _, row in shapes.iterrows())
        else:
            shapes = list((row.geometry, MaskGenerator.SINGLE_CLASS_MASK_VALUE) for _, row in shapes.iterrows())

        transform = rio.transform.from_origin(west=coordinates[0],
                                              north=coordinates[1],
                                              xsize=self.resolution,
                                              ysize=self.resolution)
        try:
            mask = rio.features.rasterize(shapes=shapes,
                                          out_shape=(self.image_size, self.image_size),
                                          transform=transform)
        except ValueError:
            mask = np.zeros(shape=(self.image_size, self.image_size), dtype=np.uint8)
        mask = np.moveaxis(np.expand_dims(mask, axis=-1), source=-1, destination=0)

        if self.shapes is not None:
            with rio.io.MemoryFile() as memory_file:
                with memory_file.open(
                        driver='GTiff',
                        width=self.image_size,
                        height=self.image_size,
                        count=MaskGenerator.BANDS,
                        crs={'init': f'epsg:{self.epsg_code}'},
                        transform=transform,
                        dtype=mask.dtype,
                        nodata=0) as dataset:
                    dataset.write(mask)
                with memory_file.open() as dataset:
                    mask_masked, _ = rio.mask.mask(dataset=dataset,
                                                   shapes=self.shapes,
                                                   crop=False)
            return mask_masked, coordinates
        return mask, coordinates

    def export_mask(self,
                    image,
                    path,
                    coordinates):
        """Exports an image from the get_mask() method. If necessary, a world file with georeferencing metadata
        is created in the same directory as the image itself or georeferencing metadata is embedded into the image.

        :param np.ndarray[int] image: image
        :param str path: path to the image
        :param (float, float) coordinates: coordinates (x, y) of the top left corner
        :returns: None
        :rtype: None
        """
        if self.create_geotiff:
            transform = rio.transform.from_origin(west=coordinates[0],
                                                  north=coordinates[1],
                                                  xsize=self.resolution,
                                                  ysize=self.resolution)
            with rio.open(path, 'w',
                          driver='GTiff',
                          width=self.image_size,
                          height=self.image_size,
                          count=MaskGenerator.BANDS,
                          crs={'init': f'epsg:{self.epsg_code}'},
                          transform=transform,
                          dtype=image.dtype,
                          nodata=0) as file:
                file.write(image)
        else:
            # noinspection PyTypeChecker
            Image.fromarray(np.moveaxis(image, source=0, destination=-1)).save(path)

        if self.create_wld:
            utils.export_wld(path=str(Path(path).with_suffix('.wld')),
                             resolution=self.resolution,
                             coordinates=coordinates)

    def __call__(self):
        """Exports all images of an area to the masks directory.
        Each image name consists of the following attributes separated by an underscore:
        'mask_id_x_y.tiff'

        :returns: None
        :rtype: None
        """
        image_name_prefix = '_'.join(self.metadata_path.stem.split('_')[:-1])
        tiles_dir_path = self.dir_path / image_name_prefix
        tiles_dir_file_list = natsorted([x.name for x in tiles_dir_path.iterdir() if x.is_file()])
        iterations = len(tiles_dir_file_list)
        logger_padding_length = len(str(len(tiles_dir_file_list)))

        for index, file in enumerate(tiles_dir_file_list):
            if str(file).endswith('.tiff'):
                mask, coordinates = self.get_mask(path=str(tiles_dir_path / file))
                image_name = Path(file).stem
                mask_name = f"mask_{'_'.join(image_name.split('_')[-3:])}.tiff"
                self.export_mask(image=mask,
                                 path=str(self.dir_path / 'mask' / mask_name),
                                 coordinates=coordinates)
                logger.info(f'iteration {index + 1:>{logger_padding_length}} / {iterations} '
                            f'-> mask with id = {index} exported')

        metadata = {'timestamp': str(DateTime.now().isoformat(sep=' ', timespec='seconds')),
                    'multi class mask': self.multi_class_mask,
                    'epsg code': self.epsg_code,
                    'resolution': self.resolution,
                    'image size': self.image_size,
                    'bounding box': self.metadata.get('bounding box'),
                    'number of columns': self.metadata.get('number of columns'),
                    'number of rows': self.metadata.get('number of rows'),
                    'number of iterations': self.metadata.get('number of iterations')}
        utils.export_metadata(path=str(self.dir_path / 'mask_metadata.json'),
                              metadata=metadata)

    @staticmethod
    def preprocess_mask_shp(path,
                            column,
                            replacement_dict,
                            delete_list=None):
        """Preprocesses the shape file. The preprocessed shape file is saved with the suffix 'preprocessed' in the
        directory of the shape file.

        :param str path: path to the shape file of the mask that needs to be rasterized
        :param str column: name of the column of the class values
        :param dict[int, int] replacement_dict: dictionary of each class value (key) and their mask value (value)
        :param list[int] or None delete_list: list of class values to delete
        :returns: None
        :rtype: None
        :raises ValueError: if value in replacement_dict is not valid (not a value between 0 and 255)
        """
        for value in list(replacement_dict.values()):
            if not 0 <= value <= 255:
                raise ValueError('Invalid value in replacement_dict! '
                                 'Values in replacement_dict have to be values between 0 and 255.')

        shapes = gpd.read_file(path)
        shapes['mask_value'] = shapes[column]
        shapes.replace({'mask_value': replacement_dict}, inplace=True)

        if delete_list is not None:
            shapes = shapes[~shapes.mask_value.isin(delete_list)]

        shapes.to_file(f'{Path(path).stem}_preprocessed.shp')
