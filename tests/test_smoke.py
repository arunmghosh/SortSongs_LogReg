"""
test_smoke.py

Rapid end-to-end integration smoke tests for CLI runner and experiment execution.
"""

import os
import sys
import json
import tempfile
import subprocess
import unittest


class TestSmoke(unittest.TestCase):
    def setUp(self):
        self.python_exec = sys.executable

    def test_smoke_cli_flag(self):
        """Verify runner.py --smoke-test exits cleanly with code 0."""
        cmd = [self.python_exec, "runner.py", "--smoke-test"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            f"Smoke test failed with return code {result.returncode}:\n{result.stderr}",
        )
        self.assertIn("SMOKE TEST PASSED", result.stdout)

    def test_smoke_phase1_cli(self):
        """Verify runner.py --phase 1 --trials 2 executes cleanly."""
        cmd = [self.python_exec, "runner.py", "--phase", "1", "--trials", "2"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            f"Phase 1 CLI run failed:\n{result.stderr}",
        )
        self.assertIn("Phase 1 Accuracy:", result.stdout)

    def test_smoke_phase2_cli(self):
        """Verify runner.py --phase 2 --trials 2 executes cleanly."""
        cmd = [self.python_exec, "runner.py", "--phase", "2", "--trials", "2"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            f"Phase 2 CLI run failed:\n{result.stderr}",
        )
        self.assertIn("Phase 2 Accuracy:", result.stdout)

    def test_smoke_all_phases_json_output(self):
        """Verify full runner execution exports structured JSON successfully."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                self.python_exec,
                "runner.py",
                "--phase",
                "all",
                "--trials",
                "2",
                "--output",
                tmp_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode,
                0,
                f"Full run failed:\n{result.stderr}",
            )
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("phase_1", data)
            self.assertIn("phase_2", data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
