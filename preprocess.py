"""
preprocess.py

Data loading, extraction, and standardization for song plot archetype analysis.
"""

from typing import Tuple, Dict, Any, List, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES: List[str] = [
    "Monster",
    "RagsRiches",
    "Quest",
    "VoyageReturn",
    "Comedy",
    "Tragedy",
    "Rebirth",
]

ALBUM_ORDER: List[str] = [
    "SOUR",
    "GUTS",
    "You seem pretty sad for a girl so in love",
    "Good Riddance",
    "The Secret Of Us",
    "Daughter From Hell",
]

ARTIST_TO_LABEL: Dict[str, int] = {
    "Olivia Rodrigo": 0,
    "Gracie Abrams": 1,
}

LABEL_TO_ARTIST: Dict[int, str] = {v: k for k, v in ARTIST_TO_LABEL.items()}


def load_dataset(
    filepath: str = "artist_comparison.xlsx",
    sheet_name: str = "gabrams_orodrigo",
) -> pd.DataFrame:
    """
    Load dataset from Excel file and clean column names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract feature matrix X, binary artist labels y_artist,
    multinomial album labels y_album, and metadata dataframe.
    """
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = df[FEATURE_NAMES].values.astype(float)

    if "Artist" not in df.columns:
        raise ValueError("Column 'Artist' missing from dataset")
    y_artist = df["Artist"].map(ARTIST_TO_LABEL).values.astype(int)

    if "Album" not in df.columns:
        raise ValueError("Column 'Album' missing from dataset")
    y_album = df["Album"].values.astype(str)

    metadata = df[["Song", "Artist", "Album"]].copy()
    return X, y_artist, y_album, metadata


def scale_features(
    X: np.ndarray, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features to standard normal distribution (zero mean, unit variance).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def shuffle_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Permute dataset observations using a random state.
    """
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    meta_shuffled = metadata.iloc[indices].reset_index(drop=True) if metadata is not None else None
    return X_shuffled, y_shuffled, meta_shuffled


def get_preprocessed_data(
    filepath: str = "artist_comparison.xlsx",
    sheet_name: str = "gabrams_orodrigo",
) -> Dict[str, Any]:
    """
    High-level convenience function to load and preprocess the entire dataset.
    """
    df = load_dataset(filepath, sheet_name)
    X_raw, y_artist, y_album, metadata = extract_features_and_targets(df)
    X_scaled, scaler = scale_features(X_raw)

    return {
        "raw_df": df,
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "y_artist": y_artist,
        "y_album": y_album,
        "metadata": metadata,
        "feature_names": FEATURE_NAMES,
        "album_order": ALBUM_ORDER,
        "scaler": scaler,
    }
