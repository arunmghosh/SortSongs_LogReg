"""
models.py

Defines binary logistic regression and multinomial softmax regression models.
"""

from typing import List, Dict, Optional, Union
import numpy as np
from sklearn.linear_model import LogisticRegression


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute standard sigmoid activation: sigma(z) = 1 / (1 + exp(-z)).
    """
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable softmax: exp(z - max(z)) / sum(exp(z - max(z))).
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


class BinaryLogisticRegression:
    """
    Phase 1: Binary Logistic Regression model to classify songs by artist
    (0 = Olivia Rodrigo, 1 = Gracie Abrams).
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: Optional[str] = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
    ):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        kwargs = {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        if self.penalty is not None:
            kwargs["penalty"] = self.penalty

        self.model = LogisticRegression(**kwargs)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryLogisticRegression":
        """
        Fit model on standardized feature matrix X and binary labels y.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return float(self.model.score(X, y))

    @property
    def coefficients(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.coef_[0]

    @property
    def intercept(self) -> float:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return float(self.model.intercept_[0])

    def get_equation_string(
        self, feature_names: List[str], precision: int = 4
    ) -> str:
        """
        Return the LaTeX/math formatted learned logistic regression equation.
        """
        coefs = self.coefficients
        b = self.intercept

        terms = []
        for name, c in zip(feature_names, coefs):
            sign = "+" if c >= 0 else "-"
            terms.append(f"{sign} {abs(c):.{precision}f} \\cdot x_{{\\text{{{name}}}}}")

        b_sign = "+" if b >= 0 else "-"
        linear_combination = " ".join(terms)
        if linear_combination.startswith("+ "):
            linear_combination = linear_combination[2:]

        eq = (
            f"P(y=1 \\mid \\mathbf{{x}}) = \\sigma({linear_combination} "
            f"{b_sign} {abs(b):.{precision}f})"
        )
        return eq


class MultinomialSoftmaxRegression:
    """
    Phase 2: Multinomial Softmax Logistic Regression model to classify songs
    across the 6 albums.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: Optional[str] = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
    ):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        kwargs = {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        if self.penalty is not None:
            kwargs["penalty"] = self.penalty

        self.model = LogisticRegression(**kwargs)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialSoftmaxRegression":
        """
        Fit multinomial softmax model on standardized features X and album labels y.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return float(self.model.score(X, y))

    @property
    def classes(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.classes_

    @property
    def coefficients(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.coef_

    @property
    def intercepts(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.intercept_

    def get_album_coefficients(self) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return mapping from each album to its weights and intercept.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        results = {}
        for cls_name, coef, intercept in zip(self.classes, self.coefficients, self.intercepts):
            results[cls_name] = {
                "coef": coef,
                "intercept": float(intercept),
            }
        return results

    def get_album_equations(
        self,
        feature_names: List[str],
        album_order: Optional[List[str]] = None,
        precision: int = 4,
    ) -> Dict[str, str]:
        """
        Return formatted linear score equations for each album: z_k = w_k^T x + b_k.
        """
        album_dict = self.get_album_coefficients()
        order = album_order if album_order is not None else list(self.classes)

        equations = {}
        for album in order:
            if album not in album_dict:
                continue
            coef = album_dict[album]["coef"]
            b = album_dict[album]["intercept"]

            terms = []
            for name, c in zip(feature_names, coef):
                sign = "+" if c >= 0 else "-"
                terms.append(f"{sign} {abs(c):.{precision}f} \\cdot x_{{\\text{{{name}}}}}")

            b_sign = "+" if b >= 0 else "-"
            linear_comb = " ".join(terms)
            if linear_comb.startswith("+ "):
                linear_comb = linear_comb[2:]

            eq = f"z_{{\\text{{{album}}}}} = {linear_comb} {b_sign} {abs(b):.{precision}f}"
            equations[album] = eq

        return equations
