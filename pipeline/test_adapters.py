"""Verify the boundary between independent pipelines and the viewer."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .run import write_csv


class ViewerExportTests(unittest.TestCase):
    def test_dead_and_ungraded_frames_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mall.csv'
            write_csv(np.array([2, 0, 1, 2, 2]), np.array([0, 1, 1, 1, 0]), path, expected=5)
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['frame'] for r in rows], ['0', '1', '2', '3', '4'])
            self.assertEqual([r['label'] for r in rows], ['X', 'H', 'A', 'D', 'X'])

    def test_invalid_output_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            for labels, half in [([0], [1]), ([0, 9], [1, 1]), ([0, 1], [1])]:
                with self.assertRaises(ValueError):
                    write_csv(labels, half, path, expected=2)
            self.assertFalse(path.exists())

    def test_cache_directories_do_not_overlap(self):
        from .lenny import config
        from .mall.common import CACHE
        self.assertNotEqual(config.CACHE, Path(CACHE))
        self.assertEqual(config.ROOT, Path(__file__).resolve().parents[1])


class ComparisonInputTests(unittest.TestCase):
    def test_baseline_dead_frames_and_abstention_are_valid(self):
        from .compare import read
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'baseline.csv'
            path.write_text('frame,label\n0,D\n1,H\n2,X\n')
            with patch('pipeline.compare.source_frames', return_value=3):
                np.testing.assert_array_equal(read(path), ['D', 'H', 'X'])

    def test_duplicate_frame_rejected(self):
        from .compare import read
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'duplicate.csv'
            path.write_text('frame,label\n0,H\n0,A\n')
            with patch('pipeline.compare.source_frames', return_value=2):
                with self.assertRaises(ValueError):
                    read(path)


if __name__ == '__main__':
    unittest.main()
