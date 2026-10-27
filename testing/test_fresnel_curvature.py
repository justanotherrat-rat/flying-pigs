"""Regression tests for Earth curvature in the Fresnel obstruction check."""
import unittest
from unittest.mock import patch

import numpy as np

from simulation import losGenerationBands


class FresnelCurvatureTest(unittest.TestCase):
    def test_ship_link_range_includes_four_thirds_earth_bulge(self):
        sea_level_profile = [{'elevation': 0.0} for _ in range(101)]
        ships_20_km = np.array([[0.0, 0.0, 25.0], [20_000.0, 0.0, 25.0]])
        ships_21_km = np.array([[0.0, 0.0, 25.0], [21_000.0, 0.0, 25.0]])
        ships_34_km = np.array([[0.0, 0.0, 25.0], [34_000.0, 0.0, 25.0]])

        with patch.object(
                losGenerationBands, 'get_elevation_profile',
                return_value=sea_level_profile):
            graph_20_km = losGenerationBands.build_los_graph(
                ships_20_km, object(), 0.0, 0.0)
            graph_21_km = losGenerationBands.build_los_graph(
                ships_21_km, object(), 0.0, 0.0)
            graph_34_km = losGenerationBands.build_los_graph(
                ships_34_km, object(), 0.0, 0.0)

        self.assertGreater(graph_20_km[0, 1], 0.0)
        self.assertEqual(graph_21_km[0, 1], 0.0)
        self.assertEqual(graph_34_km[0, 1], 0.0)


if __name__ == '__main__':
    unittest.main()
