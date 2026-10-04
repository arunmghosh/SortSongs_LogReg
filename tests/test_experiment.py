"""
test_experiment.py

Ensure that the pipeline runs correctly and outputs are in the correct format.
"""

import os
import json
import tempfile
import unittest

from experiment import run_phase1, run_phase2, run_experiment
from preprocess import FEATURE_NAMES, ALBUM_ORDER


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.data_path = "artist_comparison.xlsx"
        self.assertTrue(os.path.exists(self.data_path), "Dataset missing for tests")

    def test_run_phase1(self):
        """Test Phase 1 pipeline execution and output schema."""
        results = run_phase1(data_path=self.data_path, trials=3, random_state=42)

        self.assertEqual(results["phase"], 1)
        self.assertEqual(results["trials"], 3)
        self.assertGreater(results["accuracy"], 0.6)
        self.assertEqual(len(results["coefficients"]), len(FEATURE_NAMES))
        self.assertIn("P(y=1 \\mid \\mathbf{x}) = \\sigma(", results["equation"])
        self.assertIn("replication_stats", results)
        self.assertIn("asymptotic_confidence_intervals", results)

    def test_run_phase2(self):
        """Test Phase 2 pipeline execution and output schema."""
        results = run_phase2(data_path=self.data_path, trials=3, random_state=42)

        self.assertEqual(results["phase"], 2)
        self.assertEqual(results["trials"], 3)
        self.assertGreater(results["accuracy"], 0.35)
        self.assertEqual(len(results["album_equations"]), len(ALBUM_ORDER))
        for album in ALBUM_ORDER:
            self.assertIn(album, results["album_equations"])
        self.assertIn("variability", results)

    def test_run_experiment_all(self):
        """Test full experiment orchestration across all phases."""
        results = run_experiment(phase="all", trials=2, data_path=self.data_path)

        self.assertIn("phase_1", results)
        self.assertIn("phase_2", results)
        self.assertEqual(results["phase_1"]["phase"], 1)
        self.assertEqual(results["phase_2"]["phase"], 2)

    def test_experiment_json_serialization(self):
        """Test structured experimental results JSON export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            results = run_experiment(
                phase="all",
                trials=2,
                data_path=self.data_path,
                output_path=tmp_path,
            )
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertIn("phase_1", loaded)
            self.assertIn("phase_2", loaded)
            self.assertAlmostEqual(loaded["phase_1"]["accuracy"], results["phase_1"]["accuracy"], places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
