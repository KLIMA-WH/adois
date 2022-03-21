from datetime import datetime as DateTime  # PEP 8 compliant
import geopandas as gpd
import json
import numpy as np
import os
from PIL import Image
import rasterio as rio
import rasterio.features
import rasterio.mask


class MaskGenerator:
    """MaskGenerator
    TODO: class documentation
    TODO: logging

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """
    SINGLE_CLASS_MASK_VALUE = 255
    BANDS = 1

    def __init__(self,
                 metadata_path,
                 mask_shp_path,
                 shp_path=None,
                 multi_class_mask=False,
                 create_wld=False,
                 create_geotiff=False):
        """Constructor method

        :param str metadata_path: relative path to the metadata file created by TileGenerator
        :param str mask_shp_path: relative path to the shape file of the mask that needs to be rasterized
        :param str shp_path: relative path to the shape file for masking specific areas
        :param bool multi_class_mask: if True, the pixel value of each rasterized shape equals the value of the column
            mask_value (shape file may need to be preprocessed)
            if False, the pixel value of the rasterized shapes is 255
        :param bool create_wld: if True, a world file is created
        :param bool create_geotiff: if True, georeferencing metadata is embedded into the image
        """
        self.metadata_path = metadata_path
        self.dir_path = os.path.dirname(self.metadata_path)
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
        try:
            os.mkdir(os.path.join(self.dir_path, 'mask'))
        except FileExistsError:
            print(f"Directory {os.path.join(self.dir_path, 'mask')} already exists!")

    def get_mask(self, path):
        """Returns an image of the mask to a corresponding tile. If necessary, the image is getting masked
        with the shapes of the optional shape file.

        :param str path: relative path to the image of the corresponding tile
        :returns: image and coordinates
        :rtype: (np.ndarray of int, (float, float))
        """
        image_metadata = os.path.splitext(path)[0].split('/')[-1].split('_')
        coordinates = (float(image_metadata[-2]), float(image_metadata[-1]))
        bounding_box = (coordinates[0],
                        round(coordinates[1] - self.image_size_meters, 2),
                        round(coordinates[0] + self.image_size_meters, 2),
                        coordinates[1])

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
            mask = np.zeros(shape=(self.image_size, self.image_size))
        mask = np.rollaxis(np.expand_dims(mask, axis=2), axis=2)

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
                          count=MaskGenerator.BANDS,
                          crs={'init': f'epsg:{self.epsg_code}'},
                          transform=transform,
                          dtype=image.dtype,
                          nodata=0) as file:
                file.write(image)
        else:
            Image.fromarray(image).save(path)

        if self.create_wld:
            with open(f'{os.path.splitext(path)[0]}.wld', 'w') as file:
                file.write(f'{self.resolution}\n'
                           '0.0\n'
                           '0.0\n'
                           f'-{self.resolution}\n'
                           f'{coordinates[0]}\n'
                           f'{coordinates[1]}')

    def export_masks(self):
        """Exports all images of an area to the masks directory.
        Each image name consists of the following attributes separated by an underscore:
        'mask_id_x_y.tiff'

        :returns: None
        :rtype: None
        """
        image_name_prefix = '_'.join(os.path.splitext(self.metadata_path)[0].split('/')[-1].split('_')[:-1])
        tiles_dir_path = os.path.join(self.dir_path, image_name_prefix)
        for file in os.listdir(tiles_dir_path):
            if str(file).endswith('.tiff'):
                mask, coordinates = self.get_mask(path=str(os.path.join(tiles_dir_path, file)))
                image_name = str(os.path.splitext(file)[0])
                mask_name = f"mask_{'_'.join(image_name.split('_')[-3:])}.tiff"
                path = os.path.join(self.dir_path, 'mask', mask_name)
                self.export_mask(image=mask,
                                 path=path,
                                 coordinates=coordinates)

        metadata = {'timestamp': str(DateTime.now().isoformat(sep=' ', timespec='seconds')),
                    'epsg code': self.epsg_code,
                    'resolution': self.resolution,
                    'image size': self.image_size,
                    'bounding box': self.metadata.get('bounding box'),
                    'number of columns': self.metadata.get('number of columns'),
                    'number of rows': self.metadata.get('number of rows'),
                    'number of iterations': self.metadata.get('number of iterations'),
                    'list of coordinates': self.metadata.get('list of coordinates')}
        with open(os.path.join(self.dir_path, 'mask_metadata.json'), 'w') as file:
            json.dump(metadata, file, indent=4)