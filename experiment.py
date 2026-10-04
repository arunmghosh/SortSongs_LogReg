"""
experiment.py

Orchestrates Phase 1 (binary artist classification) and Phase 2 (multinomial
album classification) experiments, tracking equations, metrics, and replications.
"""

from typing import Dict, Any, Optional
import json
import numpy as np

from preprocess import (
    get_preprocessed_data,
    shuffle_dataset,
    FEATURE_NAMES,
    ALBUM_ORDER,
)
from models import BinaryLogisticRegression, MultinomialSoftmaxRegression
from stats import (
    compute_replication_stats,
    compute_asymptotic_ci,
    compute_multiclass_replication_variance,
    format_ci_latex_block,
)


def _serialize_numpy(obj: Any) -> Any:
    """Helper to convert NumPy structures to standard Python types for JSON export."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_numpy(i) for i in obj]
    return obj


def run_phase1(
    data_path: str = "artist_comparison.xlsx",
    trials: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute Phase 1: Binary classification of songs by artist.
    Class 0 = Olivia Rodrigo, Class 1 = Gracie Abrams.
    """
    data = get_preprocessed_data(filepath=data_path)
    X = data["X_scaled"]
    y = data["y_artist"]

    # Base model fit on complete dataset
    base_model = BinaryLogisticRegression(random_state=random_state)
    base_model.fit(X, y)

    equation_str = base_model.get_equation_string(FEATURE_NAMES)
    accuracy = base_model.score(X, y)

    coef_history = []
    intercept_history = []

    num_replications = max(1, trials)
    for t in range(num_replications):
        seed = random_state + t
        X_shuffled, y_shuffled, _ = shuffle_dataset(X, y, random_state=seed)
        trial_model = BinaryLogisticRegression(random_state=seed)
        trial_model.fit(X_shuffled, y_shuffled)
        coef_history.append(trial_model.coefficients)
        intercept_history.append(trial_model.intercept)

    all_history = np.column_stack([intercept_history, coef_history])
    replication_stats = compute_replication_stats(
        all_history, ["Intercept"] + list(FEATURE_NAMES), confidence_level=0.90
    )

    asymptotic_cis = compute_asymptotic_ci(
        X,
        y,
        base_model.coefficients,
        base_model.intercept,
        FEATURE_NAMES,
        confidence_level=0.90,
    )

    ci_latex = format_ci_latex_block(asymptotic_cis)

    return {
        "phase": 1,
        "trials": num_replications,
        "accuracy": accuracy,
        "coefficients": {k: float(v) for k, v in zip(FEATURE_NAMES, base_model.coefficients)},
        "intercept": float(base_model.intercept),
        "equation": equation_str,
        "replication_stats": replication_stats,
        "asymptotic_confidence_intervals": asymptotic_cis,
        "ci_latex": ci_latex,
    }


def run_phase2(
    data_path: str = "artist_comparison.xlsx",
    trials: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute Phase 2: Multinomial classification of songs across the 6 albums.
    """
    data = get_preprocessed_data(filepath=data_path)
    X = data["X_scaled"]
    y = data["y_album"]

    base_model = MultinomialSoftmaxRegression(random_state=random_state)
    base_model.fit(X, y)

    album_equations = base_model.get_album_equations(FEATURE_NAMES, album_order=ALBUM_ORDER)
    accuracy = base_model.score(X, y)

    replications = []
    num_replications = max(1, trials)
    for t in range(num_replications):
        seed = random_state + t
        X_shuffled, y_shuffled, _ = shuffle_dataset(X, y, random_state=seed)
        trial_model = MultinomialSoftmaxRegression(random_state=seed)
        trial_model.fit(X_shuffled, y_shuffled)
        replications.append(trial_model.get_album_coefficients())

    variability = compute_multiclass_replication_variance(
        replications, FEATURE_NAMES, ALBUM_ORDER
    )

    album_coef_summary = {}
    for album, details in base_model.get_album_coefficients().items():
        album_coef_summary[album] = {
            "intercept": float(details["intercept"]),
            "coefficients": {
                name: float(val) for name, val in zip(FEATURE_NAMES, details["coef"])
            },
        }

    return {
        "phase": 2,
        "trials": num_replications,
        "accuracy": accuracy,
        "album_equations": album_equations,
        "album_coefficients": album_coef_summary,
        "variability": variability,
    }


def run_experiment(
    phase: str = "all",
    trials: int = 1,
    data_path: str = "artist_comparison.xlsx",
    output_path: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Orchestrate running Phase 1, Phase 2, or both.
    """
    results: Dict[str, Any] = {
        "config": {
            "phase": phase,
            "trials": trials,
            "data_path": data_path,
        }
    }

    if phase in ["1", "all"]:
        print(f"\n[Phase 1] Running Binary Artist Classification ({trials} trial(s))...")
        p1_res = run_phase1(data_path=data_path, trials=trials, random_state=random_state)
        results["phase_1"] = p1_res
        print(f"Phase 1 Accuracy: {p1_res['accuracy']:.4f}")
        print("Learned Equation:")
        print(f"  {p1_res['equation']}")

    if phase in ["2", "all"]:
        print(f"\n[Phase 2] Running Multinomial Album Classification ({trials} trial(s))...")
        p2_res = run_phase2(data_path=data_path, trials=trials, random_state=random_state)
        results["phase_2"] = p2_res
        print(f"Phase 2 Accuracy: {p2_res['accuracy']:.4f}")
        print("Learned Equations for Albums:")
        for album, eq in p2_res["album_equations"].items():
            print(f"  {album}: {eq}")

    if output_path is not None:
        serializable_results = _serialize_numpy(results)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n[Output] Results successfully saved to {output_path}")

    return results
