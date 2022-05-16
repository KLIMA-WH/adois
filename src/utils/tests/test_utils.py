import json
import tempfile
import unittest
from datetime import datetime as DateTime  # PEP 8 compliant
from datetime import timedelta as TimeDelta  # PEP 8 compliant
from pathlib import Path

from src.utils.package import utils


class TestUtils(unittest.TestCase):
    def test_get_bounding_box(self):
        test_coordinates = (363193.98, 5715477.01)
        test_image_size_meters = 256
        result = (363193.98, 5715221.01, 363449.98, 5715477.01)
        test_result = utils.get_bounding_box(coordinates=test_coordinates,
                                             image_size_meters=test_image_size_meters)
        self.assertTupleEqual(test_result, result)

    def test_export_wld(self):
        test_path = 'test.wld'
        test_resolution = .1
        test_coordinates = (363193.98, 5715477.01)
        content = '0.1\n0.0\n0.0\n-0.1\n363193.98\n5715477.01'
        with tempfile.TemporaryDirectory(dir='') as temp_dir:
            utils.export_wld(path=str(Path(temp_dir) / test_path),
                             resolution=test_resolution,
                             coordinates=test_coordinates)
            with open(Path(temp_dir) / test_path) as file:
                test_content = file.read()
            self.assertEqual(test_content, content)

        test_path = 'test.txt'
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory(dir='') as temp_dir:
                utils.export_wld(path=str(Path(temp_dir) / test_path),
                                 resolution=test_resolution,
                                 coordinates=test_coordinates)

        test_path = Path('test.wld')
        test_resolution = .1
        test_coordinates = (363193.98, 5715477.01)
        content = '0.1\n0.0\n0.0\n-0.1\n363193.98\n5715477.01'
        with tempfile.TemporaryDirectory(dir='') as temp_dir:
            utils.export_wld(path=str(Path(temp_dir) / test_path),
                             resolution=test_resolution,
                             coordinates=test_coordinates)
            with open(Path(temp_dir) / test_path) as file:
                test_content = file.read()
            self.assertEqual(test_content, content)

        test_path = Path('test.txt')
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory(dir='') as temp_dir:
                utils.export_wld(path=str(Path(temp_dir) / test_path),
                                 resolution=test_resolution,
                                 coordinates=test_coordinates)

    def test_export_metadata(self):
        test_path = 'test.json'
        test_metadata = {'a': 1, 'b': 2, 'c': 3}
        content = {'a': 1, 'b': 2, 'c': 3}
        with tempfile.TemporaryDirectory(dir='') as temp_dir:
            utils.export_json(path=str(Path(temp_dir) / test_path),
                              metadata=test_metadata)
            with open(Path(temp_dir) / test_path) as file:
                test_content = json.load(file)
            self.assertEqual(test_content, content)

        test_path = 'test.txt'
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory(dir='') as temp_dir:
                utils.export_json(path=str(Path(temp_dir) / test_path),
                                  metadata=test_metadata)

        test_path = Path('test.json')
        test_metadata = {'a': 1, 'b': 2, 'c': 3}
        content = {'a': 1, 'b': 2, 'c': 3}
        with tempfile.TemporaryDirectory(dir='') as temp_dir:
            utils.export_json(path=str(Path(temp_dir) / test_path),
                              metadata=test_metadata)
            with open(Path(temp_dir) / test_path) as file:
                test_content = json.load(file)
            self.assertEqual(test_content, content)

        test_path = Path('test.txt')
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory(dir='') as temp_dir:
                utils.export_json(path=str(Path(temp_dir) / test_path),
                                  metadata=test_metadata)

    def test_chop_microseconds(self):
        test_datetime_1 = DateTime(year=2020,
                                   month=1,
                                   day=1,
                                   hour=10,
                                   minute=10,
                                   second=10,
                                   microsecond=0)
        test_datetime_2 = DateTime(year=2020,
                                   month=1,
                                   day=2,
                                   hour=20,
                                   minute=20,
                                   second=20,
                                   microsecond=123456)
        result = TimeDelta(days=1,
                           hours=10,
                           minutes=10,
                           seconds=10)
        test_result = utils.chop_microseconds(delta=test_datetime_2 - test_datetime_1)
        self.assertEqual(test_result, result)

    def test_get_image_metadata(self):
        test_path = 'test_dir/test_abc_0_363193.98_5715477.01.tiff'
        result = ('test_abc', 0, (363193.98, 5715477.01))
        test_result = utils.get_image_metadata(path=test_path)
        self.assertTupleEqual(test_result, result)

        test_path = Path('test_dir/test_abc_0_363193.98_5715477.01.tiff')
        result = ('test_abc', 0, (363193.98, 5715477.01))
        test_result = utils.get_image_metadata(path=test_path)
        self.assertTupleEqual(test_result, result)


if __name__ == '__main__':
    unittest.main()
