import numpy as np
import os
import tensorflow as tf


class RecordGenerator:
    """RecordGenerator
    TODO: class documentation
    TODO: logging

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """

    def __init__(self, dir_path):
        """Constructor method

        :param str dir_path: relative path to the directory
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
        """Concatenates the rgb image and the nir image to a rgbi image.

        :param np.ndarray of int rgb_image: rgb image
        :param np.ndarray of int nir_image: nir image
        :returns: rgbi image
        :rtype: np.ndarray of int
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
    def tensor_feature(tensor):
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

        :param np.ndarray of int rgb_image: rgb image
        :param np.ndarray of int nir_image: nir image
        :param np.ndarray of int mask: mask
        :returns: example
        :rtype: tf.train.Example
        """
        image = RecordGenerator.concatenate_to_rgbi(rgb_image=rgb_image, nir_image=nir_image)
        feature = {'image': RecordGenerator.tensor_feature(tf.cast(image, tf.uint8)),
                   'mask': RecordGenerator.tensor_feature(tf.cast(mask, tf.uint8))}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def export_record(rgb_image,
                      nir_image,
                      mask,
                      path):
        """Exports a record from the get_example() method.

        :param np.ndarray of int rgb_image: rgb image
        :param np.ndarray of int nir_image: nir image
        :param np.ndarray of int mask: mask
        :param str path: relative path to the record
        :returns: None
        :rtype: None
        """
        with tf.io.TFRecordWriter(path) as writer:
            example = RecordGenerator.get_example(rgb_image=rgb_image, nir_image=nir_image, mask=mask)
            writer.write(example.SerializeToString())
