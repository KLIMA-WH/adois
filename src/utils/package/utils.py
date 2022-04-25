import json
from datetime import timedelta as TimeDelta  # PEP 8 compliant


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

    :param str path: path to the world file
    :param float resolution: resolution in meters per pixel
    :param (float, float) coordinates: coordinates (x, y) of the top left corner
    :returns: None
    :rtype: None
    :raises ValueError: if path is not valid (file extension is not .wld)
    """
    if path.endswith('.wld'):
        with open(path, 'w') as file:
            file.write(f'{resolution}\n'
                       '0.0\n'
                       '0.0\n'
                       f'-{resolution}\n'
                       f'{coordinates[0]}\n'
                       f'{coordinates[1]}')
    else:
        raise ValueError('Invalid path! The file extension of the path has to be .wld.')


def export_metadata(path, metadata):
    """Exports a metadata file (.json).

    :param str path: path to the metadata file
    :param dict metadata: metadata
    :returns: None
    :rtype: None
    :raises ValueError: if path is not valid (file extension is not .json)
    """
    if path.endswith('.json'):
        with open(path, 'w') as file:
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
