"""
runner.py

CLI interface for custom trials, rapid smoke tests, individual phases, and full runs.
"""

import sys
import argparse
from typing import Optional

from experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Logistic Regression Song Classifier Runner (Olivia Rodrigo vs Gracie Abrams)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run a rapid end-to-end integration check across both phases",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default=None,
        help="Run a specific phase (1: binary artist, 2: multinomial album, or all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of replications of each regression (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export structured experimental results as JSON",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="artist_comparison.xlsx",
        help="Path to dataset Excel file (default: artist_comparison.xlsx)",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.smoke_test:
        print("=== Running Rapid End-to-End Smoke Test ===")
        results = run_experiment(
            phase="all",
            trials=1,
            data_path=args.data_path,
            output_path=args.output,
        )
        assert "phase_1" in results and "phase_2" in results, "Smoke test failed missing phases!"
        assert results["phase_1"]["accuracy"] > 0.5, "Smoke test Phase 1 accuracy too low!"
        assert results["phase_2"]["accuracy"] > 0.2, "Smoke test Phase 2 accuracy too low!"
        print("\n[OK] SMOKE TEST PASSED: Both phases completed successfully.")
        return 0

    target_phase = args.phase if args.phase is not None else "all"

    print(f"=== Starting Experiment: Phase={target_phase}, Trials={args.trials} ===")
    run_experiment(
        phase=target_phase,
        trials=args.trials,
        data_path=args.data_path,
        output_path=args.output,
    )
    print("\n[OK] Experiment completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
