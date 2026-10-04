"""
test_math.py

Ensure that all calculations are mathematically correct, including scaling,
activations, and confidence intervals.
"""

import unittest
import numpy as np
from scipy import stats

from models import sigmoid, softmax
from preprocess import scale_features
from stats import compute_replication_stats, compute_asymptotic_ci, format_ci_latex_block


class TestMath(unittest.TestCase):
    def test_standard_normal_scaling(self):
        """Verify that scaled features have zero mean and unit variance."""
        raw_data = np.array([
            [1.0, 5.0, 10.0],
            [2.0, 8.0, 20.0],
            [3.0, 11.0, 30.0],
            [4.0, 14.0, 40.0],
        ])
        scaled, scaler = scale_features(raw_data)

        # Means should be 0 and variance should be 1 (within numerical tolerance)
        np.testing.assert_allclose(np.mean(scaled, axis=0), np.zeros(3), atol=1e-7)
        np.testing.assert_allclose(np.std(scaled, axis=0), np.ones(3), atol=1e-7)

    def test_sigmoid_function_math(self):
        """Verify sigmoid activation mathematical properties."""
        # sigma(0) = 0.5
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5, places=7)

        # Symmetry: sigma(-z) = 1 - sigma(z)
        z_vals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        sig_z = sigmoid(z_vals)
        sig_neg_z = sigmoid(-z_vals)
        np.testing.assert_allclose(sig_z, 1.0 - sig_neg_z, atol=1e-7)

        # Extreme clipping stability
        self.assertAlmostEqual(float(sigmoid(-1000.0)), 0.0, places=5)
        self.assertAlmostEqual(float(sigmoid(1000.0)), 1.0, places=5)

    def test_softmax_function_math(self):
        """Verify softmax partition function and shift invariance."""
        logits = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [1000.0, 1001.0, 1002.0],  # test numerical overflow protection
        ])
        probs = softmax(logits, axis=-1)

        # Every row must sum to 1.0
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(3), atol=1e-7)

        # Shift invariance: softmax(z + c) == softmax(z)
        shifted = logits + 42.0
        probs_shifted = softmax(shifted, axis=-1)
        np.testing.assert_allclose(probs, probs_shifted, atol=1e-7)

    def test_confidence_interval_math(self):
        """Verify confidence interval margin of error and asymptotic intervals."""
        # 1. Replication CI math check
        fake_coefs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # N=5
        feat_names = ["feat1"]
        rep_stats = compute_replication_stats(fake_coefs, feat_names, confidence_level=0.90)

        mean_val = np.mean(fake_coefs)
        std_val = np.std(fake_coefs, ddof=1)
        se_val = std_val / np.sqrt(5)
        expected_t = stats.t.ppf(0.95, df=4)
        expected_margin = expected_t * se_val

        self.assertAlmostEqual(rep_stats["feat1"]["mean"], mean_val, places=6)
        self.assertAlmostEqual(rep_stats["feat1"]["std"], std_val, places=6)
        self.assertAlmostEqual(rep_stats["feat1"]["margin_of_error"], expected_margin, places=6)

        # 2. Asymptotic CI math check
        X_dummy = np.array([[-1.0], [0.0], [1.0], [2.0]])
        y_dummy = np.array([0, 0, 1, 1])
        asymp_ci = compute_asymptotic_ci(
            X_dummy, y_dummy, np.array([0.5]), 0.1, ["feat1"], confidence_level=0.90
        )
        self.assertIn("feat1", asymp_ci)
        self.assertIn("Intercept", asymp_ci)
        self.assertGreater(asymp_ci["feat1"]["standard_error"], 0.0)
        self.assertEqual(
            asymp_ci["feat1"]["ci_lower"] < asymp_ci["feat1"]["ci_upper"], True
        )

    def test_format_ci_latex_block(self):
        """Verify format_ci_latex_block properly formats features and intercept."""
        ci_dict = {
            "Intercept": {"coefficient": 0.0614, "ci_lower": -0.3863, "ci_upper": 0.5092},
            "Monster": {"coefficient": -0.0246, "ci_lower": -0.4862, "ci_upper": 0.4370},
        }
        latex = format_ci_latex_block(ci_dict)
        self.assertTrue(latex.startswith("\\begin{aligned}"))
        self.assertTrue(latex.endswith("\\end{aligned}"))
        self.assertIn("w_{\\text{Monster}} &= -0.0246 \\quad (90\\% \\text{ CI}: [-0.4862, 0.4370]) \\\\", latex)
        self.assertIn("b_{\\text{Intercept}} &= 0.0614 \\quad (90\\% \\text{ CI}: [-0.3863, 0.5092])", latex)


if __name__ == "__main__":
    unittest.main()
