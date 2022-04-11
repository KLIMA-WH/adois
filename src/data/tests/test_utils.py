import json
import os
from src.data.package import utils
import tempfile
import unittest


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
            utils.export_wld(path=os.path.join(temp_dir, test_path),
                             resolution=test_resolution,
                             coordinates=test_coordinates)
            with open(os.path.join(temp_dir, test_path), 'r') as file:
                test_content = file.read()
            self.assertEqual(test_content, content)

        test_path = 'test.txt'
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory(dir='') as temp_dir:
                utils.export_wld(path=os.path.join(temp_dir, test_path),
                                 resolution=test_resolution,
                                 coordinates=test_coordinates)

    def test_export_metadata(self):
        test_path = 'test.json'
        test_metadata = {'a': 1, 'b': 2, 'c': 3}
        content = {'a': 1, 'b': 2, 'c': 3}
        with tempfile.TemporaryDirectory(dir='') as temp_dir:
            utils.export_metadata(path=os.path.join(temp_dir, test_path),
                                  metadata=test_metadata)
            with open(os.path.join(temp_dir, test_path), 'r') as file:
                test_content = json.load(file)
            self.assertEqual(test_content, content)

        test_path = 'test.txt'
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory(dir='') as temp_dir:
                utils.export_metadata(path=os.path.join(temp_dir, test_path),
                                      metadata=test_metadata)


if __name__ == '__main__':
    unittest.main()