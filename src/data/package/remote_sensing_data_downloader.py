import json
import logging
import warnings
from datetime import datetime as DateTime  # PEP 8 compliant
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.mask
from PIL import Image
from natsort import natsorted
from owslib.wms import WebMapService

from src.utils.package import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

log_dir_path = Path(__file__).parents[1] / 'logs'
date_time = str(DateTime.now().isoformat(sep='_', timespec='seconds')).replace(':', '-')
file_handler = logging.FileHandler(log_dir_path / f'{date_time}_remote_sensing_data_downloader.log', mode='w')
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)

warnings.filterwarnings(action='ignore', message='shapes are outside bounds of raster.')


class RemoteSensingDataDownloader:
    """RemoteSensingDataDownloader
    TODO: class documentation

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """
    VALID_IMAGE_SIZE = (128, 256, 512, 1024, 2048, 4096, 640, 1280, 2560, 5120)
    CHANNELS = 3

    def __init__(self,
                 dir_path,
                 image_name,
                 wms_url,
                 layer,
                 epsg_code,
                 resolution,
                 image_size,
                 coordinates_path=None,
                 bounding_box=None,
                 shp_path=None,
                 create_wld=False,
                 create_geotiff=False,
                 non_zero_ratio=.25,
                 additional_info=None):
        """Constructor method

        :param str or Path dir_path: path to the directory
        :param str image_name: prefix of the image name
        :param str wms_url: url of the web map service
        :param str layer: name of the layer
        :param int epsg_code: epsg code of the coordinate reference system
        :param float resolution: resolution in meters per pixel
        :param int image_size: image size in pixels
        :param str or Path or None coordinates_path: path to the coordinates file (.json) containing the ids and
            coordinates to download
        :param (float, float, float, float) or None bounding_box: bounding box (x_1, y_1, x_2, y_2)
            of the area from the bottom left corner to the top right corner
        :param str or Path or None shp_path: path to the shape file for masking specific areas
        :param bool create_wld: if True, a world file is created
        :param bool create_geotiff: if True, georeferencing metadata is embedded into the image
        :param float non_zero_ratio: ratio of pixels with information (pixel value > 0) to all pixels of the image
            (export filter: if the pixels with information are below the threshold of the non_zero_ratio,
            the image is skipped)
        :param str or None additional_info: additional info for metadata
        :returns: None
        :rtype: None
        :raises ValueError: if image_size is not valid (not a power of base 2, its tenfold or too small/ large) or
            if bounding_box is not valid (x_1 >= x_2 or y_1 >= y_2) or
            if the mode is not valid (neither coordinates nor a bounding box are defined) or
            if non_zero_ratio is not valid (not a value between 0 and 1)
        """
        self.dir_path = Path(dir_path)
        self.image_name = image_name
        self.wms_url = wms_url
        self.wms = WebMapService(self.wms_url)
        self.layer = layer
        self.epsg_code = epsg_code
        self.resolution = resolution

        if image_size in RemoteSensingDataDownloader.VALID_IMAGE_SIZE:
            self.image_size = image_size
        else:
            raise ValueError('Invalid image_size! image_size has to be a power of base 2 or its tenfold. Try '
                             f'{[*RemoteSensingDataDownloader.VALID_IMAGE_SIZE]}.')
        self.image_size_meters = self.resolution * self.image_size

        if coordinates_path is not None:
            coordinates_path = Path(coordinates_path)
            if coordinates_path.is_file():
                with open(coordinates_path) as file:
                    self.coordinates = json.load(file)
        else:
            self.coordinates = None

        if bounding_box is not None:
            if bounding_box[0] < bounding_box[2] and bounding_box[1] < bounding_box[3]:
                self.bounding_box = (round(bounding_box[0], 2),
                                     round(bounding_box[1], 2),
                                     round(bounding_box[2], 2),
                                     round(bounding_box[3], 2))
            else:
                raise ValueError('Invalid bounding_box! x_1 has to be smaller than x_2 and '
                                 'y_1 has to be smaller than y_2.')
        else:
            self.bounding_box = None

        if self.coordinates is None and self.bounding_box is None:
            raise ValueError('Invalid mode! Either coordinates or a bounding box have to be defined.')

        if shp_path is not None:
            shp_path = Path(shp_path)
            shapes = gpd.read_file(shp_path)
            self.shapes = [row.geometry for _, row in shapes.iterrows()]
        else:
            self.shapes = None

        self.create_wld = create_wld
        self.create_geotiff = create_geotiff

        if 0 <= non_zero_ratio <= 1:
            self.non_zero_ratio = non_zero_ratio
        else:
            raise ValueError('Invalid non_zero_ratio! non_zero_ratio has to be a value between 0 and 1.')

        self.additional_info = additional_info
        (self.dir_path / self.image_name).mkdir(exist_ok=True)

    def get_image(self, coordinates):
        """Returns an image given its coordinates of the top left corner. If necessary, the image is getting masked
        with the shapes of the optional shape file.

        :param (float, float) coordinates: coordinates (x, y) of the top left corner
        :returns: image
        :rtype: np.ndarray[int]
        """
        bounding_box = utils.get_bounding_box(coordinates=coordinates,
                                              image_size_meters=self.image_size_meters)
        response = self.wms.getmap(layers=[self.layer],
                                   srs=f'EPSG:{self.epsg_code}',
                                   bbox=bounding_box,
                                   format='image/tiff',
                                   size=(self.image_size, self.image_size),
                                   bgcolor='#000000')
        with Image.open(BytesIO(response.read())) as file:
            # noinspection PyTypeChecker
            image = np.moveaxis(np.array(file),
                                source=-1,
                                destination=0)

        if self.shapes is not None:
            transform = rio.transform.from_origin(west=coordinates[0],
                                                  north=coordinates[1],
                                                  xsize=self.resolution,
                                                  ysize=self.resolution)
            with rio.io.MemoryFile() as memory_file:
                with memory_file.open(driver='GTiff',
                                      width=self.image_size,
                                      height=self.image_size,
                                      count=RemoteSensingDataDownloader.CHANNELS,
                                      crs=f'epsg:{self.epsg_code}',
                                      transform=transform,
                                      dtype=image.dtype,
                                      nodata=0) as dataset:
                    dataset.write(image)
                with memory_file.open() as dataset:
                    image_masked, _ = rio.mask.mask(dataset=dataset,
                                                    shapes=self.shapes,
                                                    crop=False)
            return image_masked
        return image

    def export_image(self,
                     image,
                     path,
                     coordinates):
        """Exports an image from the get_image() method. If necessary, a world file with georeferencing metadata
        is created in the same directory as the image itself or georeferencing metadata is embedded into the image.

        :param np.ndarray[int] image: image
        :param str or Path path: path to the image
        :param (float, float) coordinates: coordinates (x, y) of the top left corner
        :returns: None
        :rtype: None
        """
        path = Path(path)

        if self.create_geotiff:
            transform = rio.transform.from_origin(west=coordinates[0],
                                                  north=coordinates[1],
                                                  xsize=self.resolution,
                                                  ysize=self.resolution)
            with rio.open(path,
                          mode='w',
                          driver='GTiff',
                          width=self.image_size,
                          height=self.image_size,
                          count=RemoteSensingDataDownloader.CHANNELS,
                          crs=f'epsg:{self.epsg_code}',
                          transform=transform,
                          dtype=image.dtype,
                          nodata=0) as file:
                file.write(image)
        else:
            # noinspection PyTypeChecker
            Image.fromarray(np.moveaxis(image,
                                        source=0,
                                        destination=-1)).save(path)

        if self.create_wld:
            utils.export_wld(path.with_suffix('.wld'),
                             resolution=self.resolution,
                             coordinates=coordinates)

    def export_images_coordinates(self):
        """Exports all images of an area given a dictionary of ids and coordinates to the images directory.
        Each image name consists of the following attributes separated by an underscore:
        'prefix_id_x_y.tiff'

        :returns: None
        :rtype: None
        """
        start_time = DateTime.now()

        iterations = len(self.coordinates)
        logger_padding_length = len(str(iterations))

        iteration = 1

        for image_id, coordinates in self.coordinates.items():
            coordinates = tuple(coordinates)
            image = self.get_image(coordinates)
            image_name = f'{self.image_name}_{image_id}_{coordinates[0]}_{coordinates[1]}.tiff'
            path = self.dir_path / self.image_name / image_name
            self.export_image(image,
                              path=path,
                              coordinates=coordinates)
            logger.info(f'iteration {iteration:>{logger_padding_length}} / {iterations} '
                        f'-> image with id = {image_id} exported')
            iteration += 1

        end_time = DateTime.now()
        delta = utils.chop_microseconds(end_time - start_time)

        metadata = {'timestamp': str(start_time.isoformat(sep=' ', timespec='seconds')),
                    'execution time': str(delta),
                    'wms url': self.wms_url,
                    'layer': self.layer,
                    'epsg code': self.epsg_code,
                    'resolution': self.resolution,
                    'image size': self.image_size,
                    'number of iterations/ images': iterations}
        if self.additional_info is not None:
            metadata['additional info'] = self.additional_info
        utils.export_json(self.dir_path / f'{self.image_name}_metadata.json',
                          metadata=metadata)
        utils.export_json(self.dir_path / f'{self.image_name}_coordinates.json',
                          metadata=self.coordinates)

    def export_images_bounding_box(self):
        """Exports all images of an area given its bounding box to the images directory.
        Each image name consists of the following attributes separated by an underscore:
        'prefix_id_x_y.tiff'

        :returns: None
        :rtype: None
        """
        start_time = DateTime.now()

        columns = int((self.bounding_box[2] - self.bounding_box[0]) // self.image_size_meters)
        if (self.bounding_box[2] - self.bounding_box[0]) % self.image_size_meters:
            columns += 1
        rows = int((self.bounding_box[3] - self.bounding_box[1]) // self.image_size_meters)
        if (self.bounding_box[3] - self.bounding_box[1]) % self.image_size_meters:
            rows += 1
        iterations = columns * rows
        logger_padding_length = len(str(iterations))

        iteration = 1
        image_id = 0

        non_zero_threshold = self.image_size ** 2 * RemoteSensingDataDownloader.CHANNELS * self.non_zero_ratio

        metadata_coordinates = {}

        for row in range(rows):
            for column in range(columns):
                coordinates = (round(self.bounding_box[0] + column * self.image_size_meters, 2),
                               round(self.bounding_box[1] + (row + 1) * self.image_size_meters, 2))
                image = self.get_image(coordinates)
                if np.any(image) if self.non_zero_ratio == 0 else np.count_nonzero(image) > non_zero_threshold:
                    image_name = f'{self.image_name}_{image_id}_{coordinates[0]}_{coordinates[1]}.tiff'
                    path = self.dir_path / self.image_name / image_name
                    self.export_image(image,
                                      path=path,
                                      coordinates=coordinates)
                    metadata_coordinates[image_id] = coordinates
                    logger.info(f'iteration {iteration:>{logger_padding_length}} / {iterations} '
                                f'-> image with id = {image_id} exported')
                    image_id += 1
                else:
                    logger.info(f'iteration {iteration:>{logger_padding_length}} / {iterations} '
                                f'-> image skipped')
                iteration += 1

        end_time = DateTime.now()
        delta = utils.chop_microseconds(end_time - start_time)

        metadata = {'timestamp': str(start_time.isoformat(sep=' ', timespec='seconds')),
                    'execution time': str(delta),
                    'wms url': self.wms_url,
                    'layer': self.layer,
                    'epsg code': self.epsg_code,
                    'resolution': self.resolution,
                    'image size': self.image_size,
                    'bounding box': self.bounding_box,
                    'number of iterations': iterations,
                    'number of images': image_id}
        if self.additional_info is not None:
            metadata['additional info'] = self.additional_info
        utils.export_json(self.dir_path / f'{self.image_name}_metadata.json',
                          metadata=metadata)
        utils.export_json(self.dir_path / f'{self.image_name}_coordinates.json',
                          metadata=metadata_coordinates)

    def __call__(self):
        """Exports all images of an area either given a dictionary of ids and coordinates or its bounding box
        to the images directory.
        Each image name consists of the following attributes separated by an underscore:
        'prefix_id_x_y.tiff'

        :returns: None
        :rtype: None
        """
        if self.coordinates is not None:
            self.export_images_coordinates()
        else:
            self.export_images_bounding_box()

    @staticmethod
    def print_info(wms_url):
        """Prints information about the web map service such as the type, version, provider, title and abstract.

        :param str wms_url: url of the web map service
        :returns: None
        :rtype: None
        """
        wms = WebMapService(wms_url)
        print(f'Type:         {wms.identification.type}\n'
              f'Version:      {wms.identification.version}\n'
              f'Provider:     {wms.provider.name}\n'
              f'Title:        {wms.identification.title}\n'
              f'Abstract:     {wms.identification.abstract}')

    @staticmethod
    def print_available_layers(wms_url):
        """Prints available layers of the web map service.

        :param str wms_url: url of the web map service
        :returns: None
        :rtype: None
        """
        wms = WebMapService(wms_url)
        print(f'Layers:       {[*wms.contents]}')

    @staticmethod
    def print_attributes(wms_url, layer):
        """Prints attributes of a specific layer of the web map service such as the coordinate reference system,
        bounding box and styles.

        :param str wms_url: url of the web map service
        :param str layer: name of the layer
        :returns: None
        :rtype: None
        """
        wms = WebMapService(wms_url)
        print(f'CRS:          {wms[layer].crsOptions}\n'
              f'Bounding Box: {wms[layer].boundingBox}\n'
              f'Styles:       {wms[layer].styles}')

    @staticmethod
    def create_coordinates(dir_path):
        """Creates a coordinates file (.json) in the parent directory containing the ids and coordinates
        of all images in the images directory.

        :param str or Path dir_path: path to the directory
        :returns: None
        :rtype: None
        """
        dir_path = Path(dir_path)
        images = natsorted([x.name for x in dir_path.iterdir() if x.suffix == '.tiff'])

        metadata_coordinates = {}

        for image in images:
            _, image_id, coordinates = utils.get_image_metadata(image)
            metadata_coordinates[image_id] = coordinates

        utils.export_json(dir_path.parents[0] / f'{dir_path.name}_coordinates.json',
                          metadata=metadata_coordinates)
