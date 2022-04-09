import logging
from natsort import natsorted
import numpy as np
import os
from PIL import Image
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

file_handler = logging.FileHandler(filename=f'{__name__}.log', mode='w')
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

    def __init__(self, dir_path):
        """Constructor method

        :param str dir_path: path to the directory
        :returns: None
        :rtype: None
        """
        self.dir_path = dir_path
        try:
            os.mkdir(os.path.join(self.dir_path, 'records'))
        except FileExistsError:
            print(f"Directory {os.path.join(self.dir_path, 'records')} already exists!")

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
            return np.concatenate((rgb_image, np.expand_dims(nir_image, axis=-1)), axis=-1)
        elif nir_image.ndim == 3:
            return np.concatenate((rgb_image, np.expand_dims(nir_image[..., 0], axis=-1)), axis=-1)
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
        image = RecordGenerator.concatenate_to_rgbi(rgb_image=rgb_image, nir_image=nir_image)
        feature = {'image': RecordGenerator.get_tensor_feature(tf.cast(image, tf.uint8)),
                   'mask': RecordGenerator.get_tensor_feature(tf.cast(mask, tf.uint8))}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def export_record(rgb_image,
                      nir_image,
                      mask,
                      path):
        """Exports a record from the get_example() method.

        :param np.ndarray[int] rgb_image: rgb image
        :param np.ndarray[int] nir_image: nir image
        :param np.ndarray[int] mask: mask
        :param str path: path to the record
        :returns: None
        :rtype: None
        """
        with tf.io.TFRecordWriter(path) as writer:
            example = RecordGenerator.get_example(rgb_image=rgb_image, nir_image=nir_image, mask=mask)
            writer.write(example.SerializeToString())

    @staticmethod
    def get_image_metadata(path):
        """Returns the image id and the coordinates.

        :param str path: path to the image
        :returns: image id and coordinates
        :rtype: (int, (float, float))
        """
        image_metadata = os.path.splitext(path)[0].split('/')[-1].split('_')
        image_id = int(image_metadata[-3])
        coordinates = (float(image_metadata[-2]), float(image_metadata[-1]))
        return image_id, coordinates

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
        rgb_dir_file_list = natsorted(os.listdir(os.path.join(self.dir_path, 'rgb')))
        nir_dir_file_list = natsorted(os.listdir(os.path.join(self.dir_path, 'nir')))
        mask_dir_file_list = natsorted(os.listdir(os.path.join(self.dir_path, 'mask')))
        iterations = len(rgb_dir_file_list)
        logger_padding_length = len(str(len(rgb_dir_file_list)))

        if len(rgb_dir_file_list) == len(nir_dir_file_list) == len(mask_dir_file_list):
            for index, file in enumerate(rgb_dir_file_list):
                rgb_id, rgb_coordinates = RecordGenerator.get_image_metadata(rgb_dir_file_list[index])
                nir_id, nir_coordinates = RecordGenerator.get_image_metadata(nir_dir_file_list[index])
                mask_id, mask_coordinates = RecordGenerator.get_image_metadata(mask_dir_file_list[index])
                if (rgb_id == nir_id == mask_id == index and
                        rgb_coordinates == nir_coordinates == mask_coordinates):
                    rgb_image = np.array(Image.open(os.path.join(self.dir_path, 'rgb', rgb_dir_file_list[index])))
                    nir_image = np.array(Image.open(os.path.join(self.dir_path, 'nir', nir_dir_file_list[index])))
                    mask = np.array(Image.open(os.path.join(self.dir_path, 'mask', mask_dir_file_list[index])))
                    path = os.path.join(self.dir_path, 'records',
                                        f'{rgb_id}_{rgb_coordinates[0]}_{rgb_coordinates[1]}.tfrecord')
                    RecordGenerator.export_record(rgb_image=rgb_image,
                                                  nir_image=nir_image,
                                                  mask=mask,
                                                  path=path)
                    logger.info(f'iteration {index + 1:>{logger_padding_length}} / {iterations} '
                                f'-> record with id = {index} exported')
                else:
                    raise ValueError('Invalid image metadata! The ids and the coordinates of the images '
                                     '(rgb, nir, mask) have to match.')
        else:
            raise ValueError('Invalid number of images! The number of images in each directory (rgb, nir, mask) '
                             'have to match.')
