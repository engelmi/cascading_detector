#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import unittest

from cascading_detector.detectable import Detectable


class DetectableTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.curr_dir = os.getcwd()
        self.temp_dir = os.path.join(self.curr_dir, "temp")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

        self.default_detectable = Detectable(1, 200, 200)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.temp_dir)

    def test_to_json_str_without_children(self):
        json_str = self.default_detectable.to_json_str()

        expected = '{"bounding_box": {"height": 200, "width": 200}, ' \
                   '"children": [], "class_id": 1, "py/object": "cascading_detector.detectable.Detectable"}'

        self.assertEqual(json_str, expected)

    def test_to_json_dict_without_children(self):
        det_dict = self.default_detectable.to_json_dict()

        expected = {
                    "bounding_box": {"height": 200, "width": 200},
                    "children": [],
                    "class_id": 1,
                    "py/object": "cascading_detector.detectable.Detectable"
        }

        self.assertEqual(det_dict, expected)

    @classmethod
    def write_detectable(cls, detectable, filename):
        detectable.write_to_file(filename)

    def test_write_to_file_without_children(self):
        filename = os.path.join(self.temp_dir, "detectable_write_test")
        self.write_detectable(self.default_detectable, filename)

        # expected: file exists
        self.assertTrue(os.path.exists(filename))

    def test_read_from_file_without_children(self):
        filename = os.path.join(self.temp_dir, "detectable_read_test")
        self.write_detectable(self.default_detectable, filename)
        obj = Detectable.read_from_file(filename)

        attributes = [a for a in dir(self.default_detectable) if not a.startswith('__') and not callable(getattr(self.default_detectable, a))]
        for attr in attributes:
            default_attr = getattr(self.default_detectable, attr)
            obj_attr = getattr(obj, attr)
            self.assertIsNotNone(obj_attr)
            self.assertEqual(obj_attr, default_attr)


if __name__ == "__main__":
    unittest.main()
