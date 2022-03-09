from datetime import datetime as DateTime  # PEP 8 compliant
import geopandas as gpd
from io import BytesIO
import json
import numpy as np
import os
from owslib.wms import WebMapService
from PIL import Image
import rasterio as rio
import rasterio.mask
import warnings

warnings.filterwarnings('ignore', message='shapes are outside bounds of raster.')


class TileGenerator:
    """TileGenerator
    TODO: class documentation

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """
    VALID_IMAGE_SIZE = (128, 256, 512, 1024, 2048, 4096, 1280, 2560, 5120)
    BANDS = 3
    NON_ZERO_RATIO = 0.25

    def __init__(self,
                 wms_url,
                 layer,
                 epsg_code,
                 resolution,
                 image_size,
                 dir_path='',
                 image_name_prefix='',
                 shp_path=None,
                 create_world_file=False,
                 create_geotiff=False):
        """Constructor method

        :param str wms_url: url of the web map service
        :param str layer: name of the layer
        :param int epsg_code: epsg code of the coordinate reference system
        :param float resolution: resolution in meters per pixel
        :param int image_size: image size in pixels
        :param str dir_path: relative path to the directory
        :param str image_name_prefix: prefix of the image name
        :param str or None shp_path: relative path to the shape file for masking specific areas
        :param bool create_world_file: if True, a world file is created
        :param bool create_geotiff: if True, georeferencing metadata is embedded into the image
        :returns: None
        :rtype: None
        :raises ValueError: if image_size is not valid (not a power of base 2, its tenfold or too small/ large)
        """
        self.wms_url = wms_url
        self.wms = WebMapService(self.wms_url)
        self.layer = layer
        self.epsg_code = epsg_code
        self.resolution = resolution
        if image_size not in TileGenerator.VALID_IMAGE_SIZE:
            raise ValueError('Invalid image_size! image_size has to be a power of base 2 or its tenfold. Try '
                             f'{[*TileGenerator.VALID_IMAGE_SIZE]}.')
        else:
            self.image_size = image_size
        self.image_size_meters = self.resolution * self.image_size
        self.dir_path = dir_path
        self.image_name_prefix = image_name_prefix
        if shp_path is not None:
            shapes = gpd.read_file(shp_path)
            self.shapes = list((row.geometry for _, row in shapes.iterrows()))
        else:
            self.shapes = None
        self.create_world_file = create_world_file
        self.create_geotiff = create_geotiff
        try:
            os.mkdir(os.path.join(self.dir_path, self.image_name_prefix))
        except FileExistsError:
            print(f'Directory {os.path.join(self.dir_path, self.image_name_prefix)} already exists!')

    def get_tile(self, coordinates):
        """Returns an image given its coordinates of the top left corner. If necessary, the image is getting masked
        with the shapes of the optional shape file.

        :param (float, float) coordinates: coordinates (x, y) of the top left corner
        :returns: image
        :rtype: np.ndarray of int
        """
        bounding_box = (coordinates[0],
                        round(coordinates[1] - self.image_size_meters, 2),
                        round(coordinates[0] + self.image_size_meters, 2),
                        coordinates[1])
        response = self.wms.getmap(layers=[self.layer],
                                   srs=f'EPSG:{self.epsg_code}',
                                   bbox=bounding_box,
                                   format='image/tiff',
                                   size=(self.image_size, self.image_size),
                                   bgcolor='#000000')
        image = np.rollaxis(np.array(Image.open(BytesIO(response.read()))), axis=2)

        if self.shapes is not None:
            transform = rio.transform.from_origin(west=coordinates[0],
                                                  north=coordinates[1],
                                                  xsize=self.resolution,
                                                  ysize=self.resolution)
            with rio.io.MemoryFile() as memory_file:
                with memory_file.open(
                        driver='GTiff',
                        width=self.image_size,
                        height=self.image_size,
                        count=TileGenerator.BANDS,
                        crs={'init': f'epsg:{self.epsg_code}'},
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

    def export_tile(self,
                    image,
                    path,
                    coordinates):
        """Exports an image from the get_tile() method to the images directory. If necessary, a world file
        with georeferencing metadata is created in the same directory as the image itself or georeferencing metadata
        is embedded into the image.

        :param np.ndarray of int image: image
        :param str path: relative path to the image
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
                          count=TileGenerator.BANDS,
                          crs={'init': f'epsg:{self.epsg_code}'},
                          transform=transform,
                          dtype=image.dtype,
                          nodata=0) as file:
                file.write(image)
        else:
            Image.fromarray(image).save(path)

        if self.create_world_file:
            with open(f'{os.path.splitext(path)[0]}.wld', 'w') as file:
                file.write(f'{self.resolution}\n'
                           '0.0\n'
                           '0.0\n'
                           f'-{self.resolution}\n'
                           f'{coordinates[0]}\n'
                           f'{coordinates[1]}')

    def export_tiles(self,
                     bounding_box,
                     start_index=0):
        """Exports all images of an area given its bounding box to the images directory.
        Each image name consists of the following attributes separated by an underscore:
        'prefix_id_resolution_size_x_y.tiff'

        :param (float, float, float, float) bounding_box: bounding box (x_1, y_1, x_2, y_2)
            of the area from the bottom left corner to the top right corner
        :param int start_index: if the connection to the web map service is lost during the export_tiles() method,
            change the start_index to the next id according to the last successfully downloaded image
            and rerun it manually
        :returns: None
        :rtype: None
        :raises ValueError: if bounding_box is not valid (x_1 > x_2 or y_1 > y_2)
        """
        if bounding_box[0] >= bounding_box[2] or bounding_box[1] >= bounding_box[3]:
            raise ValueError('Invalid bounding_box! x_1 has to be smaller than x_2 and '
                             'y_1 has to be smaller than y_2.')

        columns = int((bounding_box[2] - bounding_box[0]) // self.image_size_meters)
        if (bounding_box[2] - bounding_box[0]) % self.image_size_meters:
            columns += 1
        rows = int((bounding_box[3] - bounding_box[1]) // self.image_size_meters)
        if (bounding_box[3] - bounding_box[1]) % self.image_size_meters:
            rows += 1

        start_row = start_index // columns
        start_column = start_index % columns

        non_zero_threshold = self.image_size ** 2 * TileGenerator.BANDS * TileGenerator.NON_ZERO_RATIO

        coordinates_list = []

        for row in range(start_row, rows):
            for column in range(start_column, columns) if row == start_row else range(columns):
                coordinates = (round(bounding_box[0] + column * self.image_size_meters, 2),
                               round(bounding_box[1] + (row + 1) * self.image_size_meters, 2))
                image = self.get_tile(coordinates=coordinates)
                if np.count_nonzero(image) > non_zero_threshold:
                    image_name = f'{self.image_name_prefix}_{start_index}_' \
                                 f'{coordinates[0]}_{coordinates[1]}'
                    path = f'{os.path.join(self.dir_path, self.image_name_prefix, image_name)}.tiff'
                    self.export_tile(image=image, path=path, coordinates=coordinates)
                    coordinates_list.append(coordinates)
                    start_index += 1

        metadata = {'timestamp': str(DateTime.now().isoformat(sep=' ', timespec='seconds')),
                    'wms_url': self.wms_url,
                    'layer': self.layer,
                    'epsg_code': self.epsg_code,
                    'resolution': self.resolution,
                    'image_size': self.image_size,
                    'bounding_box': bounding_box,
                    'number of columns': columns,
                    'number of rows': rows,
                    'number of images': columns * rows,
                    'list of coordinates': coordinates_list}
        with open(os.path.join(self.dir_path, f'{self.image_name_prefix}_metadata.json'), 'w') as file:
            json.dump(metadata, file, indent=4)

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
