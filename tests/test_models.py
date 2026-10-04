"""
test_models.py

Tests the logic and interfaces of the logistic regression models for both phases.
"""

import unittest
import numpy as np

from models import BinaryLogisticRegression, MultinomialSoftmaxRegression
from preprocess import FEATURE_NAMES, ALBUM_ORDER


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 30
        self.n_features = len(FEATURE_NAMES)
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y_binary = np.random.randint(0, 2, size=self.n_samples)
        self.y_multi = np.random.choice(ALBUM_ORDER, size=self.n_samples)

    def test_binary_model_fit_and_shapes(self):
        """Test binary logistic regression coefficient shapes and fitting."""
        model = BinaryLogisticRegression(random_state=42)
        model.fit(self.X, self.y_binary)

        self.assertTrue(model.is_fitted)
        self.assertEqual(model.coefficients.shape, (self.n_features,))
        self.assertIsInstance(model.intercept, float)
        eq = model.get_equation_string(FEATURE_NAMES)
        self.assertIn("P(y=1 \\mid \\mathbf{x}) = \\sigma(", eq)

    def test_binary_predict_and_probabilities(self):
        """Test binary predictions and probability bounds."""
        model = BinaryLogisticRegression(random_state=42)
        model.fit(self.X, self.y_binary)

        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (self.n_samples,))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

        probs = model.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_samples, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(self.n_samples), rtol=1e-5)

    def test_multinomial_model_fit_and_classes(self):
        """Test multinomial softmax regression classes and shapes."""
        model = MultinomialSoftmaxRegression(random_state=42)
        model.fit(self.X, self.y_multi)

        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.classes), len(ALBUM_ORDER))
        self.assertEqual(model.coefficients.shape, (len(ALBUM_ORDER), self.n_features))
        self.assertEqual(model.intercepts.shape, (len(ALBUM_ORDER),))

        equations = model.get_album_equations(FEATURE_NAMES, album_order=ALBUM_ORDER)
        self.assertEqual(len(equations), len(ALBUM_ORDER))
        for album in ALBUM_ORDER:
            self.assertIn(album, equations)
            self.assertTrue(equations[album].startswith(f"z_{{\\text{{{album}}}}} ="))

    def test_multinomial_predict_probabilities(self):
        """Test multinomial prediction probabilities sum to 1.0."""
        model = MultinomialSoftmaxRegression(random_state=42)
        model.fit(self.X, self.y_multi)

        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (self.n_samples,))
        for p in preds:
            self.assertIn(p, ALBUM_ORDER)

        probs = model.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_samples, len(ALBUM_ORDER)))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(self.n_samples), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
