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
    """RecordGenerator
    TODO: class documentation

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """

    def __init__(self,
                 dir_path,
                 record_name,
                 rgb_dir_path,
                 nir_dir_path,
                 masks_dir_path,
                 skip=None,
                 additional_info=None):
        """Constructor method

        :param str or Path dir_path: path to the directory
        :param str record_name: prefix of the record name
        :param str or Path rgb_dir_path: path to the directory of the rgb images
        :param str or Path nir_dir_path: path to the directory of the nir images
        :param str or Path masks_dir_path: path to the directory of the masks
        :param list[int] or None skip: list of image ids to be skipped
        :param str or None additional_info: additional info for metadata
        :returns: None
        :rtype: None
        """
        self.dir_path = Path(dir_path)
        self.record_name = record_name
        self.rgb_dir_path = Path(rgb_dir_path)
        self.nir_dir_path = Path(nir_dir_path)
        self.masks_dir_path = Path(masks_dir_path)
        self.skip = skip
        self.additional_info = additional_info
        (self.dir_path / self.record_name).mkdir(exist_ok=True)

    @staticmethod
    def concatenate_to_rgbi(rgb_image, nir_image):
        """Returns a concatenated rgbi image of the rgb image and the nir image.

        :param np.ndarray[int] rgb_image: rgb image
        :param np.ndarray[int] nir_image: nir image
        :returns: rgbi image
        :rtype: np.ndarray[int]
        :raises ValueError: if the dimensions of the nir image are not valid (not a value of 2 or 3)
        """
        if nir_image.ndim == 2:
            rgbi_image = np.concatenate((rgb_image, np.expand_dims(nir_image, axis=-1)),
                                        axis=-1)
            return rgbi_image
        elif nir_image.ndim == 3:
            rgbi_image = np.concatenate((rgb_image, np.expand_dims(nir_image[..., 0], axis=-1)),
                                        axis=-1)
            return rgbi_image
        else:
            raise ValueError('Invalid dimensions of the nir image! The dimensions of the nir_image have to be '
                             'a value of 2 or 3.')

    @staticmethod
    def get_tensor_feature(tensor):
        """Returns a feature.

        :param tf.Tensor tensor: tensor
        :returns: feature
        :rtype: tf.train.Feature
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tensor).numpy()]))

    @staticmethod
    def get_example(rgb_image,
                    nir_image,
                    mask):
        """Returns an example with 'image' and 'mask' as features. 'image' is a concatenated rgbi image.

        :param np.ndarray[int] rgb_image: rgb image
        :param np.ndarray[int] nir_image: nir image
        :param np.ndarray[int] mask: mask
        :returns: example
        :rtype: tf.train.Example
        """
        image = RecordGenerator.concatenate_to_rgbi(rgb_image=rgb_image,
                                                    nir_image=nir_image)
        feature = {'image': RecordGenerator.get_tensor_feature(tf.cast(image, dtype=tf.uint8)),
                   'mask': RecordGenerator.get_tensor_feature(tf.cast(mask, dtype=tf.uint8))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example

    @staticmethod
    def export_record(rgb_image,
                      nir_image,
                      mask,
                      path):
        """Exports a record from the get_example() method.

        :param np.ndarray[int] rgb_image: rgb image
        :param np.ndarray[int] nir_image: nir image
        :param np.ndarray[int] mask: mask
        :param str or Path path: path to the record
        :returns: None
        :rtype: None
        """
        path = str(path)

        with tf.io.TFRecordWriter(path) as writer:
            example = RecordGenerator.get_example(rgb_image=rgb_image,
                                                  nir_image=nir_image,
                                                  mask=mask)
            writer.write(example.SerializeToString())

    def __call__(self):
        """Exports all images (rgb, nir, mask) of an area as a record to the records directory.
        Each record name consists of the following attributes separated by an underscore:
        'id_x_y.tfrecord'

        :returns: None
        :rtype: None
        :raises ValueError: if the image metadata is not valid (either the ids or the coordinates of the images
            (rgb, nir, mask) do not match) or
            if the number of images is not valid (the number of images in each directory (rbg, nir, mask) do not match)
        """
        start_time = DateTime.now()

        rgb_images = natsorted([x.name for x in self.rgb_dir_path.iterdir() if x.suffix == '.tiff'])
        nir_images = natsorted([x.name for x in self.nir_dir_path.iterdir() if x.suffix == '.tiff'])
        masks = natsorted([x.name for x in self.masks_dir_path.iterdir() if x.suffix == '.tiff'])
        iterations = len(rgb_images)
        logger_padding_length = len(str(len(rgb_images)))

        if self.skip is not None:
            valid_image_ids = set(range(iterations)) - set(self.skip)
        else:
            valid_image_ids = None

        if len(rgb_images) == len(nir_images) == len(masks):
            for index, image in enumerate(rgb_images):
                _, rgb_id, rgb_coordinates = utils.get_image_metadata(rgb_images[index])
                _, nir_id, nir_coordinates = utils.get_image_metadata(nir_images[index])
                _, mask_id, mask_coordinates = utils.get_image_metadata(masks[index])
                if (rgb_id == nir_id == mask_id == index and
                        rgb_coordinates == nir_coordinates == mask_coordinates):
                    if self.skip is None or self.skip is not None and index in valid_image_ids:
                        with Image.open(self.rgb_dir_path / rgb_images[index]) as file:
                            # noinspection PyTypeChecker
                            rgb_image = np.array(file)
                        with Image.open(self.nir_dir_path / nir_images[index]) as file:
                            # noinspection PyTypeChecker
                            nir_image = np.array(file)
                        with Image.open(self.masks_dir_path / masks[index]) as file:
                            # noinspection PyTypeChecker
                            mask = np.array(file)
                        record_name = f'{self.record_name}_{rgb_id}_{rgb_coordinates[0]}_{rgb_coordinates[1]}.tfrecord'
                        path = self.dir_path / self.record_name / record_name
                        RecordGenerator.export_record(rgb_image=rgb_image,
                                                      nir_image=nir_image,
                                                      mask=mask,
                                                      path=path)
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
                    'rgb images dir': self.rgb_dir_path.stem,
                    'nir images dir': self.nir_dir_path.stem,
                    'masks dir': self.masks_dir_path.stem,
                    'number of iterations/ records': iterations}
        if self.additional_info is not None:
            metadata['additional info'] = self.additional_info
        utils.export_json(self.dir_path / f'{self.record_name}_metadata.json',
                          metadata=metadata)
