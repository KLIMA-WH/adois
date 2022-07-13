# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import json
import logging
from datetime import datetime as DateTime  # PEP 8 compliant
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from natsort import natsorted

from src.utils.package import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

log_dir_path = Path(__file__).parents[1] / 'logs'
date_time = str(DateTime.now().isoformat(sep='_', timespec='seconds')).replace(':', '-')
file_handler = logging.FileHandler(log_dir_path / f'{date_time}_record_generator.log', mode='w')
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)


class RecordGenerator:
    COLOR_CODES_NDSM = {(0, 0, 0): 0,  # 0.0m - 1.0m
                        (255, 255, 255): 28,  # 1.0m - 1.5m
                        (31, 120, 180): 57,  # 1.5m - 3.0m
                        (54, 214, 209): 85,  # 3.0m - 5.0m
                        (64, 207, 39): 113,  # 5.0m - 10.0m
                        (255, 255, 71): 142,  # 10.0m - 15.0m
                        (255, 206, 71): 170,  # 15.0m - 20.0m
                        (255, 127, 0): 198,  # 20.0m - 25.0m
                        (215, 25, 28): 227,  # 25.0m - 50.0m
                        (114, 0, 11): 255}  # > 50.0m

    def __init__(self,
                 dir_path,
                 record_name,
                 rgb_dir_path,
                 masks_dir_path,
                 nir_dir_path=None,
                 ndsm_dir_path=None,
                 color_codes=None,
                 skip_file_path=None,
                 additional_info=None):
        """Constructor method

        :param str or Path dir_path: path to the directory
        :param str record_name: prefix of the record name
        :param str or Path rgb_dir_path: path to the directory of the rgb images
        :param str or Path masks_dir_path: path to the directory of the masks
        :param str or Path or None nir_dir_path: path to the directory of the nir images
        :param str or Path or None ndsm_dir_path: path to the directory of the ndsm images
        :param dict[tuple[int, int, int], int] or None color_codes: color codes for the color mapping to reduce the
            dimensions of the ndsm images from 3 dimensions to 2 dimensions (the key of the dictionary is the
            rgb value and the value of the dictionary is the corresponding mapped value)
        :param str or Path or None skip_file_path: path to the skip file (.json) containing the ids to skip
        :param str or None additional_info: additional info for metadata
        :returns: None
        :rtype: None
        """
        self.dir_path = Path(dir_path)
        self.record_name = record_name
        self.rgb_dir_path = Path(rgb_dir_path)
        self.masks_dir_path = Path(masks_dir_path)

        if nir_dir_path is not None:
            self.nir_dir_path = Path(nir_dir_path)
        else:
            self.nir_dir_path = None

        if ndsm_dir_path is not None:
            self.ndsm_dir_path = Path(ndsm_dir_path)
            if color_codes is None:
                color_codes = RecordGenerator.COLOR_CODES_NDSM
            self.color_map = self.get_color_map(color_codes)
        else:
            self.ndsm_dir_path = None

        self.image_resize = None

        if skip_file_path is not None:
            skip_file_path = Path(skip_file_path)
            if skip_file_path.is_file():
                with open(skip_file_path) as file:
                    self.skip = json.load(file)
        else:
            self.skip = None

        self.additional_info = additional_info
        (self.dir_path / self.record_name).mkdir(exist_ok=True)

    @staticmethod
    def get_color_map(color_codes):
        """Returns a color map.
        Based on: https://stackoverflow.com/a/33196320

        :param dict[tuple[int, int, int], int] color_codes: color codes for the color mapping to reduce the
            dimensions of an image from 3 dimensions to 2 dimensions (the key of the dictionary is the rgb value
            and the value of the dictionary is the corresponding mapped value)
        :returns: color map
        :rtype: np.ndarray[int]
        """
        color_map = np.full(shape=(256 * 256 * 256),
                            fill_value=0,
                            dtype=np.int32)
        for rgb, index in color_codes.items():
            rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            color_map[rgb] = index
        return color_map

    def resize_image(self, image):
        """Returns an resized image. Used for manually upsampling or downsampling images to an image size without
        interpolation artefacts (nearest-neighbor interpolation is used).

        :param np.ndarray[int] image: image
        :returns: resized image
        :rtype: np.ndarray[int]
        """
        resized_image = tf.image.resize(image,
                                        size=(self.image_resize, self.image_resize),
                                        method='nearest').numpy()
        return resized_image

    def reduce_dimensions(self, image):
        """Returns an color mapped image with reduced dimensions (2 dimensions instead of 3 dimensions).
        Based on: https://stackoverflow.com/a/33196320

        :param np.ndarray[int] image: image
        :returns: color mapped image
        :rtype: np.ndarray[int]
        """
        image = np.dot(image, np.array([65536, 256, 1], dtype=np.int32))
        image = self.color_map[image].astype(np.uint8)
        return image

    def concatenate_images(self,
                           rgb_image,
                           nir_image=None,
                           ndsm_image=None):
        """Returns an concatenated image of the rgb image, the nir image and the ndsm image.

        :param np.ndarray[int] rgb_image: rgb image
        :param np.ndarray[int] or None nir_image: nir image
        :param np.ndarray[int] or None ndsm_image: ndsm image
        :returns: concatenated image
        :rtype: np.ndarray[int]
        """
        images = [rgb_image]

        if nir_image is not None:
            if nir_image.ndim == 2:
                nir_image = np.expand_dims(nir_image, axis=-1)
                images.append(nir_image)
            elif nir_image.ndim == 3:
                nir_image = np.expand_dims(nir_image[..., 0], axis=-1)
                images.append(nir_image)

        if ndsm_image is not None:
            if self.image_resize is not None:
                ndsm_image = self.resize_image(ndsm_image)
            ndsm_image = self.reduce_dimensions(ndsm_image)
            ndsm_image = np.expand_dims(ndsm_image, axis=-1)
            images.append(ndsm_image)

        concatenated_image = np.concatenate(images, axis=-1)
        return concatenated_image

    @staticmethod
    def get_tensor_feature(tensor):
        """Returns a feature.

        :param tf.Tensor tensor: tensor
        :returns: feature
        :rtype: tf.core.example.feature_pb2.Feature
        """
        feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()]))
        return feature

    def get_example(self,
                    rgb_image,
                    mask,
                    nir_image=None,
                    ndsm_image=None):
        """Returns an example with 'image' and 'mask' as features. 'image' is a concatenated image
        of the rgb image, the nir image and the ndsm image.

        :param np.ndarray[int] rgb_image: rgb image
        :param np.ndarray[int] mask: mask
        :param np.ndarray[int] or None nir_image: nir image
        :param np.ndarray[int] or None ndsm_image: ndsm image
        :returns: example
        :rtype: tf.core.example.example_pb2.Example
        """
        image = self.concatenate_images(rgb_image=rgb_image,
                                        nir_image=nir_image,
                                        ndsm_image=ndsm_image)
        feature = {'image': RecordGenerator.get_tensor_feature(tf.cast(image, dtype=tf.uint8)),
                   'mask': RecordGenerator.get_tensor_feature(tf.cast(mask, dtype=tf.uint8))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example

    def export_record(self,
                      rgb_image,
                      mask,
                      path,
                      nir_image=None,
                      ndsm_image=None):
        """Exports a record from the get_example() method.

        :param np.ndarray[int] rgb_image: rgb image
        :param np.ndarray[int] mask: mask
        :param str or Path path: path to the record
        :param np.ndarray[int] or None nir_image: nir image
        :param np.ndarray[int] or None ndsm_image: ndsm image
        :returns: None
        :rtype: None
        """
        path = str(path)

        with tf.io.TFRecordWriter(path) as writer:
            example = self.get_example(rgb_image=rgb_image,
                                       mask=mask,
                                       nir_image=nir_image,
                                       ndsm_image=ndsm_image)
            writer.write(example.SerializeToString())

    def __call__(self):
        """Exports all images (rgb, nir, ndsm, mask) of an area as a record to the records directory.
        Each record name consists of the following attributes separated by an underscore:
        'id_x_y.tfrecord'

        :returns: None
        :rtype: None
        :raises ValueError: if the image metadata is not valid (either the ids or the coordinates of the images
            (rgb, nir, ndsm, mask) do not match) or
            if the number of images is not valid (the number of images in each directory (rbg, nir, ndsm, mask)
            does not match)
        """
        start_time = DateTime.now()

        images = []
        rgb_images = natsorted([x.name for x in self.rgb_dir_path.iterdir() if x.suffix == '.tiff'])
        images.append(len(rgb_images))
        with Image.open(self.rgb_dir_path / rgb_images[0]) as file:
            # noinspection PyTypeChecker
            rgb_image_size = np.array(file).shape[0]
        if self.nir_dir_path is not None:
            nir_images = natsorted([x.name for x in self.nir_dir_path.iterdir() if x.suffix == '.tiff'])
            images.append(len(nir_images))
        if self.ndsm_dir_path is not None:
            ndsm_images = natsorted([x.name for x in self.ndsm_dir_path.iterdir() if x.suffix == '.tiff'])
            images.append(len(ndsm_images))
            with Image.open(self.ndsm_dir_path / ndsm_images[0]) as file:
                # noinspection PyTypeChecker
                ndsm_image_size = np.array(file).shape[0]
            if rgb_image_size == ndsm_image_size:
                self.image_resize = None
            else:
                self.image_resize = rgb_image_size
        masks = natsorted([x.name for x in self.masks_dir_path.iterdir() if x.suffix == '.tiff'])
        images.append(len(masks))
        iterations = len(rgb_images)
        logger_padding_length = len(str(len(rgb_images)))

        if self.skip is not None:
            valid_image_ids = set(range(iterations)) - set(self.skip)
        else:
            valid_image_ids = None

        if all(image == images[0] for image in images):
            for index, image in enumerate(rgb_images):
                image_ids = []
                coordinates = []
                _, rgb_id, rgb_coordinates = utils.get_image_metadata(rgb_images[index])
                image_ids.append(rgb_id)
                coordinates.append(rgb_coordinates)
                if self.nir_dir_path is not None:
                    # noinspection PyUnboundLocalVariable
                    _, nir_id, nir_coordinates = utils.get_image_metadata(nir_images[index])
                    image_ids.append(nir_id)
                    coordinates.append(nir_coordinates)
                if self.ndsm_dir_path is not None:
                    # noinspection PyUnboundLocalVariable
                    _, ndsm_id, ndsm_coordinates = utils.get_image_metadata(ndsm_images[index])
                    image_ids.append(ndsm_id)
                    coordinates.append(ndsm_coordinates)
                _, mask_id, mask_coordinates = utils.get_image_metadata(masks[index])
                image_ids.append(mask_id)
                coordinates.append(mask_coordinates)
                if (all(image_id == image_ids[0] for image_id in image_ids) and
                        all(coordinate == coordinates[0] for coordinate in coordinates)):
                    if self.skip is None or self.skip is not None and index in valid_image_ids:
                        with Image.open(self.rgb_dir_path / rgb_images[index]) as file:
                            # noinspection PyTypeChecker
                            rgb_image = np.array(file)
                        if self.nir_dir_path is not None:
                            with Image.open(self.nir_dir_path / nir_images[index]) as file:
                                # noinspection PyTypeChecker
                                nir_image = np.array(file)
                        else:
                            nir_image = None
                        if self.ndsm_dir_path is not None:
                            with Image.open(self.ndsm_dir_path / ndsm_images[index]) as file:
                                # noinspection PyTypeChecker
                                ndsm_image = np.array(file)
                        else:
                            ndsm_image = None
                        with Image.open(self.masks_dir_path / masks[index]) as file:
                            # noinspection PyTypeChecker
                            mask = np.array(file)
                        record_name = f'{self.record_name}_{rgb_id}_{rgb_coordinates[0]}_{rgb_coordinates[1]}.tfrecord'
                        path = self.dir_path / self.record_name / record_name
                        self.export_record(rgb_image=rgb_image,
                                           mask=mask,
                                           path=path,
                                           nir_image=nir_image,
                                           ndsm_image=ndsm_image)
                        logger.info(f'iteration {index + 1:>{logger_padding_length}} / {iterations} '
                                    f'-> record with id = {index} exported')
                    else:
                        logger.info(f'iteration {index + 1:>{logger_padding_length}} / {iterations} '
                                    f'-> record with id = {index} skipped')
                else:
                    raise ValueError('Invalid image metadata! The ids and the coordinates of the images '
                                     '(rgb, nir, mask) have to match.')
        else:
            raise ValueError('Invalid number of images! The number of images in each directory (rgb, nir, mask) '
                             'have to match.')

        end_time = DateTime.now()
        delta = utils.chop_microseconds(end_time - start_time)

        metadata = {'timestamp': str(start_time.isoformat(sep=' ', timespec='seconds')),
                    'execution time': str(delta),
                    'nir channel': True if self.nir_dir_path is not None else False,
                    'ndsm channel': True if self.ndsm_dir_path is not None else False}
        if self.skip is not None:
            metadata['number of iterations'] = iterations
            metadata['number of records'] = iterations - len(self.skip)
        else:
            metadata['number of iterations/ records'] = iterations
        if self.additional_info is not None:
            metadata['additional info'] = self.additional_info
        utils.export_json(self.dir_path / f'{self.record_name}_metadata.json',
                          metadata=metadata)
