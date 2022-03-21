import geopandas as gpd
import os


def preprocess_mask_shp(path,
                        column,
                        replacement_dict):
    """Preprocesses the shape file, so it can be used with MaskGenerator.

    :param str path: relative path to the shape file
    :param str column: name of the column of the class values
    :param dict replacement_dict: dictionary of each class value (key) and their mask value (value)
    :returns: None
    :rtype: None
    :raises ValueError: if value in replacement_dict is not valid (not a value between 0 and 255)
    """
    for value in list(replacement_dict.values()):
        if 0 <= value <= 255:
            pass
        else:
            raise ValueError('Invalid value in replacement_dict! '
                             'Values in replacement_dict have to be values between 0 and 255.')
    shapes = gpd.read_file(path)
    shapes['mask_value'] = shapes[column]
    shapes.replace({'mask_value': replacement_dict}, inplace=True)
    shapes.to_file(os.path.join(os.path.dirname(path), f"{os.path.splitext(path)[0].split('/')[-1]}_preprocessed.shp"))
