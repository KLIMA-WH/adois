import json
import math
from pathlib import Path
from random import shuffle
# noinspection PyUnresolvedReferences
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa
from natsort import natsorted

from src.utils.package import utils


class Pipeline:
    """Pipeline
    TODO: class documentation

    Author: Marius Maryniak (marius.maryniak@w-hs.de)
    """
    TRAIN_FILE = 'train_ids.json'
    VALIDATE_FILE = 'validate_ids.json'
    SHUFFLE_BUFFER_SIZE = 1000
    PI = tf.constant(math.pi)

    def __init__(self,
                 dir_paths,
                 mode,
                 record_size,
                 patch_size,
                 augmentation_config,
                 resize,
                 classes,
                 batch_size,
                 cache):
        """Constructor method

        :param list[str] or list[Path] dir_paths: list of paths to the directories
        :param str or None mode: 'train': records are filtered according to the record ids in train_ids.json and
            augmentation is applied to each example. 'validate': records are filtered according to the record ids
            in validate_ids.json. None: records are not filtered
        :param int record_size: record size
        :param int or None patch_size: patch size (each example is patched into patches with 50% overlap)
        :param dict[str, float] or None augmentation_config: configuration of the augmentation probabilities
            (the key of the dictionary is the augmentation type ('flip', 'rotation', 'noise', 'brightness', 'contrast',
            'hue', 'saturation') and the value of the dictionary is the corresponding probability (default is .5))
        :param int or None resize: image size to resize the image and the mask to
        :param int classes: number of numerical classes (one hot depth)
        :param int or None batch_size: batch size
        :param bool cache: if True, the dataset is cached
        :returns: None
        :rtype: None
        :raises ValueError: if mode is not valid (not 'train', 'validate' or None)
        """
        self.dir_paths = [Path(dir_path) for dir_path in dir_paths]

        if mode in ['train', 'validate', None]:
            self.mode = mode
            if self.mode == 'train':
                self.record_ids = []
                for dir_path in self.dir_paths:
                    with open(dir_path / Pipeline.TRAIN_FILE, mode='r') as file:
                        self.record_ids.extend(json.load(file))
            elif self.mode == 'validate':
                self.record_ids = []
                for dir_path in self.dir_paths:
                    with open(dir_path / Pipeline.VALIDATE_FILE, mode='r') as file:
                        self.record_ids.extend(json.load(file))
            else:
                self.record_ids = None
        else:
            raise ValueError(f"Invalid mode! mode has to be 'train', 'validate' or None.")

        self.record_size = record_size
        self.image_size = [self.record_size, self.record_size, 5]
        self.mask_size = [self.record_size, self.record_size, 1]

        if patch_size is not None:
            self.patch_size = patch_size
            self.patch_sizes = [1, self.patch_size, self.patch_size, 1]
            self.patch_strides = [1, self.patch_size / 2, self.patch_size / 2, 1]
            self.patch_reshape_image = tf.constant([-1, self.patch_size, self.patch_size, 5])
            self.patch_reshape_mask = tf.constant([-1, self.patch_size, self.patch_size, 1])
        else:
            self.patch_size = None

        if augmentation_config is not None:
            self.augmentation_config = augmentation_config
        else:
            self.augmentation_config = {'flip': .5,
                                        'rotation': .5,
                                        'noise': .5,
                                        'brightness': .5,
                                        'contrast': .5,
                                        'hue': .5,
                                        'saturation': .5}

        if resize is not None:
            self.resize = tf.constant([resize, resize])
        else:
            self.resize = None

        self.classes = tf.constant(classes)
        self.batch_size = batch_size
        self.cache = cache

    @tf.function
    def parse_record(self, record):
        """Returns an image and the corresponding mask from a parsed record.

        :param tf.Tensor[str] record: record (scalar string tensor of an single serialized example)
        :returns: image and mask
        :rtype: (tf.Tensor[int], tf.Tensor[int])
        """
        features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'mask': tf.io.FixedLenFeature([], tf.string)}
        record = tf.io.parse_single_example(record, features)
        image = tf.io.parse_tensor(record['image'], tf.uint8)
        mask = tf.io.parse_tensor(record['mask'], tf.uint8)
        return image, mask

    @tf.function
    def ensure_shape(self,
                     image,
                     mask):
        """Returns an image and the corresponding mask with valid dimensions.

        :param tf.Tensor[int] image: image
        :param tf.Tensor[int] mask: mask
        :returns: image and mask
        :rtype: (tf.Tensor[int], tf.Tensor[int])
        """
        mask = mask[..., tf.newaxis]
        image = tf.ensure_shape(image, self.image_size)
        mask = tf.ensure_shape(mask, self.mask_size)
        return image, mask

    def get_dataset(self, parsing_function):
        """Returns a dateset of all records in the directories.

        :param Callable parsing_function: parsing function
        :returns: dataset
        :rtype: tf.data.Dataset
        """
        records = []
        for dir_path in self.dir_paths:
            records_in_dir = natsorted([x.name for x in dir_path.iterdir() if x.suffix == '.tfrecord'])
            if self.record_ids is not None:
                records_in_dir = [record for record in records_in_dir if record in self.record_ids]
            records_in_dir = [str(dir_path / record) for record in records_in_dir]
            records.extend(records_in_dir)
        dataset = tf.data.TFRecordDataset(records)
        dataset = dataset.map(parsing_function)
        dataset = dataset.map(lambda image, mask: self.ensure_shape(image, mask),
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    @tf.function
    def patch_example(self,
                      image,
                      mask):
        """Returns batched image patches and mask patches.

        :param tf.Tensor[int] image: image
        :param tf.Tensor[int] mask: mask
        :returns: image patches and mask patches
        :rtype: (tf.Tensor[int], tf.Tensor[int])
        """
        image_patches = tf.image.extract_patches(image[tf.newaxis, ...],
                                                 sizes=self.patch_sizes,
                                                 strides=self.patch_strides,
                                                 rates=[1, 1, 1, 1],
                                                 padding='VALID')
        image_patches = tf.reshape(image_patches, self.patch_reshape_image)

        mask_patches = tf.image.extract_patches(mask[tf.newaxis, ...],
                                                sizes=self.patch_sizes,
                                                strides=self.patch_strides,
                                                rates=[1, 1, 1, 1],
                                                padding='VALID')
        mask_patches = tf.reshape(mask_patches, self.patch_reshape_mask)
        return image_patches, mask_patches

    @tf.function
    def augment_example(self,
                        image,
                        mask):
        """Returns an augmented image and the corresponding augmented mask. The different augmentation types are used
        according to a probability defined in the augmentation config dictionary.

        :param tf.Tensor[int] image: image
        :param tf.Tensor[int] mask: mask
        :returns: image and mask
        :rtype: (tf.Tensor[int], tf.Tensor[int])
        """

        if 'flip' in self.augmentation_config and tf.random.uniform([]) < self.augmentation_config['flip']:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if 'rotation' in self.augmentation_config and tf.random.uniform([]) < self.augmentation_config['rotation']:
            angle = tf.random.uniform([], maxval=tf.constant(2 * Pipeline.PI))
            image = tfa.image.rotate(image, angle)
            mask = tfa.image.rotate(mask, angle)

        rgb_channels = image[..., :3]
        nir_channel = image[..., 3:4]
        ndsm_channel = image[..., 4:]

        if 'noise' in self.augmentation_config and tf.random.uniform([]) < self.augmentation_config['noise']:
            rgb_noise = tf.random.uniform(shape=tf.shape(rgb_channels),
                                          minval=tf.constant(-5),
                                          maxval=tf.constant(5),
                                          dtype=tf.int16)
            nir_noise = tf.random.uniform(shape=tf.shape(nir_channel),
                                          minval=tf.constant(-5),
                                          maxval=tf.constant(5),
                                          dtype=tf.int16)
            rgb_channels = rgb_channels + rgb_noise
            nir_channel = nir_channel + nir_noise
            rgb_channels = tf.clip_by_value(rgb_channels, tf.constant(0), tf.constant(255))
            nir_channel = tf.clip_by_value(nir_channel, tf.constant(0), tf.constant(255))

        if 'brightness' in self.augmentation_config and tf.random.uniform([]) < self.augmentation_config['brightness']:
            rgb_channels = tf.image.random_brightness(rgb_channels, tf.constant(50))
            nir_channel = tf.image.random_brightness(nir_channel, tf.constant(50))

        if 'contrast' in self.augmentation_config and tf.random.uniform([]) < self.augmentation_config['contrast']:
            rgb_channels = tf.image.random_contrast(rgb_channels, tf.constant(.7), tf.constant(1.5))
            nir_channel = tf.image.random_contrast(nir_channel, tf.constant(.7), tf.constant(1.5))

        if 'hue' in self.augmentation_config and tf.random.uniform([]) < self.augmentation_config['hue']:
            rgb_channels = tf.image.random_hue(rgb_channels, tf.constant(.08))

        if 'saturation' in self.augmentation_config and tf.random.uniform([]) < self.augmentation_config['saturation']:
            rgb_channels = tf.image.random_saturation(rgb_channels, tf.constant(.7), tf.constant(1.3))

        image = tf.concat([rgb_channels, nir_channel, ndsm_channel], axis=tf.constant(-1))
        image = tf.clip_by_value(image, tf.constant(0), tf.constant(255))
        return image, mask

    @tf.function
    def resize_and_normalize_example(self,
                                     image,
                                     mask):
        """Returns an resized and normalized image and the corresponding resized and normalized mask.

        :param tf.Tensor[int] image: image
        :param tf.Tensor[int] mask: mask
        :returns: image and mask
        :rtype: (tf.Tensor[float], tf.Tensor[int])
        """
        if self.resize is not None:
            rgbi_channels = image[..., :4]
            ndsm_channel = image[..., 4:]
            rgbi_channels = tf.image.resize(rgbi_channels,
                                            size=self.resize,
                                            method='bilinear')
            ndsm_channel = tf.image.resize(ndsm_channel,
                                           size=self.resize,
                                           method='nearest')
            ndsm_channel = tf.cast(ndsm_channel, tf.float32)
            image = tf.concat([rgbi_channels, ndsm_channel], axis=tf.constant(-1))

            mask = tf.image.resize(mask,
                                   size=self.resize,
                                   method='nearest')
        image = tf.cast(image, tf.float32) / 255.
        return image, mask

    @tf.function
    def one_hot_encode_example(self,
                               image,
                               mask):
        """Returns a one hot encoded mask.

        :param tf.Tensor[float] image: image (for mapping purposes, the image is not affected)
        :param tf.Tensor[int] mask: mask
        :returns: image and mask
        :rtype: (tf.Tensor[float], tf.Tensor[int])
        """
        mask = tf.cast(mask, tf.uint8)
        mask = tf.squeeze(mask)
        mask = tf.one_hot(mask, depth=self.classes)
        return image, mask

    def __call__(self):
        """Returns a dataset.

        :returns: dataset
        :rtype: tf.data.Dataset
        """
        dataset = self.get_dataset(parsing_function=self.parse_record)

        if self.patch_size is not None:
            dataset = dataset.map(lambda image, mask: self.patch_example(image, mask),
                                  num_parallel_calls=tf.data.AUTOTUNE)
            data_options = tf.data.Options()
            data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            dataset = dataset.with_options(data_options).unbatch()

        if self.mode == 'train':
            if self.cache:
                dataset = dataset.cache()
            dataset = dataset.shuffle(Pipeline.SHUFFLE_BUFFER_SIZE).repeat()
            dataset = dataset.map(lambda image, mask: self.augment_example(image, mask),
                                  num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(lambda image, mask: self.resize_and_normalize_example(image, mask),
                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(lambda image, mask: self.one_hot_encode_example(image, mask),
                              num_parallel_calls=tf.data.AUTOTUNE)

        if self.batch_size is not None:
            dataset = dataset.batch(self.batch_size,
                                    num_parallel_calls=tf.data.AUTOTUNE)

        if self.mode != 'train' and self.cache:
            dataset = dataset.cache()

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def split_records(dir_paths, ratio):
        """Exports split files (train_ids.json and validate_ids.json) for each directory containing the record ids.

        :param list[str] or list[Path] dir_paths: list of paths to the directories
        :param float ratio: ratio of number of train records to number of records
        :returns: None
        :rtype: None
        :raises FileExistsError: if split files (train_ids.json and validate_ids.json) already exist
        """
        dir_paths = [Path(dir_path) for dir_path in dir_paths]
        for dir_path in dir_paths:
            if not (dir_path / Pipeline.TRAIN_FILE).is_file() and not (dir_path / Pipeline.VALIDATE_FILE).is_file():
                records = natsorted([x.name for x in dir_path.iterdir() if x.suffix == '.tfrecord'])
                shuffle(records)
                train_ids = natsorted(records[:int(len(records) * ratio)])
                validate_ids = natsorted(records[int(len(records) * ratio):])
                utils.export_json(dir_path / Pipeline.TRAIN_FILE,
                                  metadata=train_ids)
                utils.export_json(dir_path / Pipeline.VALIDATE_FILE,
                                  metadata=validate_ids)
            else:
                raise FileExistsError('Split files (train_ids.json and validate_ids.json) already exist!')
