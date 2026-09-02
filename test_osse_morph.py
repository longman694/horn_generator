import unittest
import numpy as np
import pandas as pd
from lib import (
    generate_osse_horn,
    calculate_target_mouth_radius,
    generate_osse_morphed_horn
)

class TestOSSEMorphing(unittest.TestCase):

    def test_unmorphed_osse_formula(self):
        """Verify basic OS-SE calculation matches Eq 5 from Ath paper."""
        throat_r = 12.7  # 1 inch diameter
        length = 100.0
        alpha = 45.0
        alpha_0 = 0.0
        k = 1.0
        s = 0.5
        q = 0.996
        n = 4.0

        df = generate_osse_horn(throat_r, length, alpha, alpha_0, k, s, q, n, num_points=11, plot=False)
        
        # At z = 0: r_GOS = throat_r, r_TERM = 0 -> r_OSSE = throat_r
        self.assertAlmostEqual(df.iloc[0]['y (mm)'], throat_r, places=4)

        # Verify z values linear spacing
        self.assertEqual(len(df), 11)
        self.assertAlmostEqual(df.iloc[0]['x (mm)'], 0.0)
        self.assertAlmostEqual(df.iloc[-1]['x (mm)'], length)

    def test_target_mouth_radius_circle(self):
        """Verify circle target shape radius at various angles."""
        width = 200.0
        height = 200.0
        corner_r = 0.0

        for phi in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
            r_m = calculate_target_mouth_radius(phi, 'circle', width, height, corner_r)
            self.assertAlmostEqual(r_m, 100.0, places=4)

    def test_target_mouth_radius_ellipse(self):
        """Verify ellipse target shape radius at major and minor axes."""
        width = 300.0   # semi-major a = 150
        height = 200.0  # semi-minor b = 100
        corner_r = 0.0

        r_major = calculate_target_mouth_radius(0.0, 'ellipse', width, height, corner_r)
        self.assertAlmostEqual(r_major, 150.0, places=4)

        r_minor = calculate_target_mouth_radius(np.pi/2, 'ellipse', width, height, corner_r)
        self.assertAlmostEqual(r_minor, 100.0, places=4)

    def test_target_mouth_radius_rounded_rectangle(self):
        """Verify rounded rectangle radius at axes and corner."""
        width = 400.0   # w = 200
        height = 200.0  # h = 100
        corner_r = 20.0 # Rc = 20

        # At phi = 0 (right edge x = 200)
        r_right = calculate_target_mouth_radius(0.0, 'rectangle', width, height, corner_r)
        self.assertAlmostEqual(r_right, 200.0, places=4)

        # At phi = pi/2 (top edge y = 100)
        r_top = calculate_target_mouth_radius(np.pi/2, 'rectangle', width, height, corner_r)
        self.assertAlmostEqual(r_top, 100.0, places=4)

        # Corner center is at (180, 80). Ray through corner center has angle theta = arctan2(80, 180)
        phi_corner = np.arctan2(80.0, 180.0)
        r_corner = calculate_target_mouth_radius(phi_corner, 'rectangle', width, height, corner_r)
        expected_r = np.sqrt(180.0**2 + 80.0**2) + corner_r
        self.assertAlmostEqual(r_corner, expected_r, places=4)

    def test_morphed_osse_fixed_part_and_mouth(self):
        """Verify morphed OS-SE preserves fixed part and reaches target shape at mouth."""
        throat_r = 12.7
        length = 100.0
        fixed_part = 0.3  # z_f = 30mm
        morph_rate = 3.0
        target_width = 300.0
        target_height = 200.0

        res = generate_osse_morphed_horn(
            throat_radius=throat_r,
            length=length,
            alpha=45,
            alpha_0=0,
            k=1.0,
            s=0.5,
            q=0.996,
            n=4.0,
            target_shape='ellipse',
            target_width=target_width,
            target_height=target_height,
            corner_radius=0,
            fixed_part=fixed_part,
            morph_rate=morph_rate,
            allow_shrinkage=True,
            num_points=101,
            num_angles=36
        )

        z = res['z']
        z_f = fixed_part * length

        # 1. For z <= z_f, major and minor profiles should match raw OS-SE profile exactly
        raw_r = res['raw_r']
        mask_fixed = z <= z_f
        np.testing.assert_allclose(res['r_major'][mask_fixed], raw_r[mask_fixed], atol=1e-4)
        np.testing.assert_allclose(res['r_minor'][mask_fixed], raw_r[mask_fixed], atol=1e-4)

        # 2. At z = length (mouth), major radius should be target_width / 2 = 150
        self.assertAlmostEqual(res['r_major'][-1], target_width / 2.0, places=3)
        self.assertAlmostEqual(res['r_minor'][-1], target_height / 2.0, places=3)

    def test_allow_shrinkage_scaling(self):
        """Verify target shape is auto-scaled up if allow_shrinkage is False and raw shape is larger."""
        throat_r = 25.0
        length = 150.0
        # Large flare creates raw mouth radius > 100
        raw_df = generate_osse_horn(throat_r, length, alpha=60, s=1.2, num_points=10, plot=False)
        raw_mouth_r = raw_df.iloc[-1]['y (mm)']

        # Request small target width/height (e.g. 100x100)
        res = generate_osse_morphed_horn(
            throat_radius=throat_r,
            length=length,
            alpha=60,
            s=1.2,
            target_shape='circle',
            target_width=100.0,
            target_height=100.0,
            allow_shrinkage=False,
            num_points=10,
            num_angles=16
        )

        # When allow_shrinkage is False, mouth radius should not shrink below raw_mouth_r
        self.assertGreaterEqual(res['r_major'][-1], raw_mouth_r - 1e-4)

if __name__ == '__main__':
    unittest.main()
