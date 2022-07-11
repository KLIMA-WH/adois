# @author: Maryniak, Marius - Fachbereich Elektrotechnik, Westfälische Hochschule Gelsenkirchen

import logging
from collections import OrderedDict
from datetime import datetime as DateTime  # PEP 8 compliant
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

log_dir_path = Path(__file__).parents[1] / 'logs'
date_time = str(DateTime.now().isoformat(sep='_', timespec='seconds')).replace(':', '-')
file_handler = logging.FileHandler(log_dir_path / f'{date_time}_aggregator.log', mode='w')
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)


class Aggregator:
    DECIMAL_PLACES = 4

    def __init__(self,
                 mask_shp_path,
                 aggregated_shp_path,
                 shp_path,
                 epsg_code):
        """Constructor method

        :param str or Path mask_shp_path: path to the shape file of the mask
        :param str or Path aggregated_shp_path: path to the shape file for aggregation
        :param str or Path or None shp_path: path to the shape file for masking specific areas
        :param int epsg_code: epsg code of the coordinate reference system
        :returns: None
        :rtype: None
        """
        self.mask_shp_path = Path(mask_shp_path)
        self.aggregated_shp_path = Path(aggregated_shp_path)

        if shp_path is not None:
            shp_path = Path(shp_path)
            self.masking_shapes = gpd.read_file(shp_path)
        else:
            self.masking_shapes = None

        self.epsg_code = epsg_code

        (self.mask_shp_path.parents[1] / f'{self.mask_shp_path.parents[0].name}_aggregated').mkdir(exist_ok=True)

    def __call__(self):
        """Exports a aggregated shape file with the following attributes for each polygon:
        'area', 'imp_area', 'imp_dens', 'bui_area', 'bui_dens', 'sur_area', 'sur_dens', 'bui_imp_r', 'sur_imp_r'

        :returns: None
        :rtype: None
        """
        aggregated_shapes = gpd.read_file(self.aggregated_shp_path)
        aggregated_shapes = aggregated_shapes[['geometry']]
        iterations = len(aggregated_shapes)
        logger_padding_length = len(str(len(aggregated_shapes)))

        for index in aggregated_shapes.index:
            polygon = gpd.GeoDataFrame(geometry=[aggregated_shapes.geometry.iloc[index]],
                                       crs=f'epsg:{self.epsg_code}')

            if self.masking_shapes is not None:
                polygon = gpd.overlay(df1=polygon,
                                      df2=self.masking_shapes,
                                      how='intersection')

            mask_shapes = gpd.read_file(self.mask_shp_path, mask=polygon)
            mask = gpd.overlay(df1=mask_shapes,
                               df2=polygon,
                               how='intersection',
                               keep_geom_type=False)

            area = float(polygon.area)
            imp_area = float(mask.area.sum())
            imp_density = imp_area / area
            bui_area = float(mask.loc[mask['class'] == 1].area.sum())
            bui_density = bui_area / area
            sur_area = float(mask.loc[mask['class'] == 2].area.sum())
            sur_density = sur_area / area

            if round(imp_area, 4) != 0:
                bui_imp_ratio = bui_area / imp_area
                sur_imp_ratio = sur_area / imp_area
            else:
                bui_imp_ratio = 1.
                sur_imp_ratio = 1.

            aggregated_shapes.at[index, 'area'] = area
            aggregated_shapes.at[index, 'imp_area'] = imp_area
            aggregated_shapes.at[index, 'imp_dens'] = imp_density
            aggregated_shapes.at[index, 'bui_area'] = bui_area
            aggregated_shapes.at[index, 'bui_dens'] = bui_density
            aggregated_shapes.at[index, 'sur_area'] = sur_area
            aggregated_shapes.at[index, 'sur_dens'] = sur_density
            aggregated_shapes.at[index, 'bui_imp_r'] = bui_imp_ratio
            aggregated_shapes.at[index, 'sur_imp_r'] = sur_imp_ratio
            logger.info(f'iteration {index + 1:>{logger_padding_length}} / {iterations}')

        attributes = OrderedDict()
        attributes['area'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['imp_area'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['imp_dens'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['bui_area'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['bui_dens'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['sur_area'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['sur_dens'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['bui_imp_r'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        attributes['sur_imp_r'] = f'float:10.{Aggregator.DECIMAL_PLACES}'
        schema = {'properties': attributes,
                  'geometry': 'Polygon'}

        aggregated_shapes.to_file(self.mask_shp_path.parents[1] / f'{self.mask_shp_path.parents[0].name}_aggregated' /
                                  f'{self.mask_shp_path.parents[0].name}_aggregated.shp',
                                  schema=schema)
