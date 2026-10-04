"""
stats.py

Statistical tracking, replication variance, and confidence interval estimation
for logistic regression models.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy import stats


def compute_replication_stats(
    coef_history: np.ndarray,
    feature_names: List[str],
    confidence_level: float = 0.90,
) -> Dict[str, Any]:
    """
    Calculate summary statistics and confidence intervals across experimental replications.

    Args:
        coef_history: np.ndarray of shape (N_trials, N_features)
        feature_names: list of feature names corresponding to columns
        confidence_level: target confidence level (e.g. 0.90 for 90% CI)
    """
    coef_history = np.asarray(coef_history)
    n_trials, n_features = coef_history.shape

    alpha = 1.0 - confidence_level
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=max(1, n_trials - 1))
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    results = {}
    for i, name in enumerate(feature_names):
        vals = coef_history[:, i]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n_trials > 1 else 0.0
        se_val = std_val / np.sqrt(n_trials) if n_trials > 1 else 0.0
        margin_of_error = t_crit * se_val

        ci_lower_t = mean_val - margin_of_error
        ci_upper_t = mean_val + margin_of_error
        ci_lower_pct = float(np.percentile(vals, lower_pct))
        ci_upper_pct = float(np.percentile(vals, upper_pct))

        results[name] = {
            "mean": mean_val,
            "std": std_val,
            "se": se_val,
            "margin_of_error": margin_of_error,
            "ci_lower": ci_lower_t,
            "ci_upper": ci_upper_t,
            "empirical_ci_lower": ci_lower_pct,
            "empirical_ci_upper": ci_upper_pct,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "range": float(np.max(vals) - np.min(vals)),
        }

    return results


def compute_asymptotic_ci(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    feature_names: List[str],
    confidence_level: float = 0.90,
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute asymptotic parametric standard errors and confidence intervals
    derived from the Fisher Information (negative Hessian of the log-likelihood).
    """
    X_design = np.column_stack([np.ones(len(X)), X])
    all_names = ["Intercept"] + list(feature_names)
    all_coefs = np.concatenate([[intercept], coef])

    z = X_design @ all_coefs
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    W = p * (1.0 - p)

    # Hessian = X^T W X + (1/C) * I (excluding intercept regularization if standard)
    Hessian = X_design.T @ (X_design * W[:, None])
    reg = np.eye(len(all_coefs)) / C
    reg[0, 0] = 1e-6  # unpenalized or weakly penalized intercept
    Hessian_reg = Hessian + reg

    try:
        cov = np.linalg.inv(Hessian_reg)
        se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Hessian_reg)
        se = np.sqrt(np.maximum(0.0, np.diag(cov)))

    alpha = 1.0 - confidence_level
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    results = {}
    for name, w, s in zip(all_names, all_coefs, se):
        margin = z_crit * s
        results[name] = {
            "coefficient": float(w),
            "standard_error": float(s),
            "margin_of_error": float(margin),
            "ci_lower": float(w - margin),
            "ci_upper": float(w + margin),
        }

    return results


def compute_multiclass_replication_variance(
    replications: List[Dict[str, Dict[str, Any]]],
    feature_names: List[str],
    album_names: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Assess variability of coefficients across replications for multinomial classification.
    """
    n_trials = len(replications)
    variations: Dict[str, Dict[str, Dict[str, float]]] = {}

    for album in album_names:
        variations[album] = {}
        # Collect intercept
        intercepts = [rep[album]["intercept"] for rep in replications]
        variations[album]["Intercept"] = {
            "mean": float(np.mean(intercepts)),
            "std": float(np.std(intercepts, ddof=1)) if n_trials > 1 else 0.0,
            "min": float(np.min(intercepts)),
            "max": float(np.max(intercepts)),
        }

        # Collect features
        coef_matrix = np.array([rep[album]["coef"] for rep in replications])
        for j, feat in enumerate(feature_names):
            vals = coef_matrix[:, j]
            variations[album][feat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if n_trials > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    return variations


def format_ci_latex_block(
    ci_dict: Dict[str, Dict[str, float]],
    precision: int = 4,
    include_intercept: bool = True,
) -> str:
    """
    Format confidence intervals into LaTeX aligned block for markdown.
    """
    body_lines: List[str] = []

    # Format feature weights
    for feat, stats_item in ci_dict.items():
        if feat == "Intercept":
            continue
        ci_l = stats_item.get("ci_lower", 0.0)
        ci_u = stats_item.get("ci_upper", 0.0)
        coef_val = stats_item.get("coefficient", stats_item.get("mean", 0.0))
        body_lines.append(
            f"w_{{\\text{{{feat}}}}} &= {coef_val:.{precision}f} \\quad (90\\% \\text{{ CI}}: [{ci_l:.{precision}f}, {ci_u:.{precision}f}])"
        )

    # Format intercept at the bottom if present and requested
    if include_intercept and "Intercept" in ci_dict:
        stats_item = ci_dict["Intercept"]
        ci_l = stats_item.get("ci_lower", 0.0)
        ci_u = stats_item.get("ci_upper", 0.0)
        coef_val = stats_item.get("coefficient", stats_item.get("mean", 0.0))
        body_lines.append(
            f"b_{{\\text{{Intercept}}}} &= {coef_val:.{precision}f} \\quad (90\\% \\text{{ CI}}: [{ci_l:.{precision}f}, {ci_u:.{precision}f}])"
        )

    lines = ["\\begin{aligned}"]
    for i, line in enumerate(body_lines):
        if i < len(body_lines) - 1:
            lines.append(f"{line} \\\\")
        else:
            lines.append(line)
    lines.append("\\end{aligned}")
    return "\n".join(lines)
