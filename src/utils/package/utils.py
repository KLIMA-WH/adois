import json
from datetime import timedelta as TimeDelta  # PEP 8 compliant
from pathlib import Path


def get_bounding_box(coordinates, image_size_meters):
    """Returns a bounding box given its coordinates of the top left corner and the image size in meters.

    :param (float, float) coordinates: coordinates (x, y) of the top left corner
    :param float image_size_meters: image size in meters
    :returns: bounding box (x_1, y_1, x_2, y_2) of the area from the bottom left corner to the top right corner
    :rtype: (float, float, float, float)
    """
    bounding_box = (coordinates[0],
                    round(coordinates[1] - image_size_meters, 2),
                    round(coordinates[0] + image_size_meters, 2),
                    coordinates[1])
    return bounding_box


def export_wld(path,
               resolution,
               coordinates):
    """Exports a world file (.wld).

    :param str or Path path: path to the world file
    :param float resolution: resolution in meters per pixel
    :param (float, float) coordinates: coordinates (x, y) of the top left corner
    :returns: None
    :rtype: None
    :raises ValueError: if path is not valid (file extension is not .wld)
    """
    path = Path(path)
    if path.suffix == '.wld':
        with open(path, mode='w') as file:
            file.write(f'{resolution}\n'
                       '0.0\n'
                       '0.0\n'
                       f'-{resolution}\n'
                       f'{coordinates[0]}\n'
                       f'{coordinates[1]}')
    else:
        raise ValueError('Invalid path! The file extension of the path has to be .wld.')


def export_json(path, metadata):
    """Exports a metadata file (.json).

    :param str or Path path: path to the metadata file
    :param dict or list metadata: metadata
    :returns: None
    :rtype: None
    :raises ValueError: if path is not valid (file extension is not .json)
    """
    path = Path(path)
    if path.suffix == '.json':
        with open(path, mode='w') as file:
            json.dump(metadata, file, indent=4)
    else:
        raise ValueError('Invalid path! The file extension of the path has to be .json.')


def chop_microseconds(delta):
    """Returns a timedelta without the microseconds.

    :param TimeDelta delta: timedelta (difference of two datetimes)
    :returns: timedelta without the microseconds
    :rtype: TimeDelta
    """
    return delta - TimeDelta(microseconds=delta.microseconds)


def get_image_metadata(path):
    """Returns the metadata of an image.
    Due to this projects convention, image names consist of the following attributes separated by an underscore:
    'prefix_id_x_y.tiff'

    :param str or Path path: path to the image
    :returns: image metadata (prefix, id, (x, y))
    :rtype: (str, int, (float, float))
    """
    path = Path(path)
    image_metadata_list = path.stem.split('_')
    image_metadata = ('_'.join(image_metadata_list[:-3]),
                      int(image_metadata_list[-3]),
                      (float(image_metadata_list[-2]),
                      float(image_metadata_list[-1])))
    return image_metadata
