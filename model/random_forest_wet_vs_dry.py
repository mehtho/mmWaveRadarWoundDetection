#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score


OUTPUT_DIR = Path("../output/official_features")


def load_data():
    X_path = OUTPUT_DIR / "X.npy"
    y_path = OUTPUT_DIR / "y.npy"

    if not X_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"Could not find X.npy and y.npy in {OUTPUT_DIR}.\n"
            f"Expected:\n  {X_path}\n  {y_path}"
        )

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Loaded X: {X.shape}, dtype={X.dtype}")
    print(f"Loaded y: {y.shape}, dtype={y.dtype}")

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")

    y = y.reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
        )

    return X, y


def build_wet_dry_labels(y: np.ndarray) -> np.ndarray:
    """
    Map original labels to:
      dry (0.0) -> 0
      wet (non-zero) -> 1
    """
    y_bin = (y != 0.0).astype(int)
    print("\nWet vs Dry label distribution:")
    unique, counts = np.unique(y_bin, return_counts=True)
    for u, c in zip(unique, counts):
        label = "dry" if u == 0 else "wet"
        print(f"  {label} ({u}): {c} samples")
    return y_bin


def run_wet_dry_rf(X: np.ndarray, y_bin: np.ndarray, n_splits: int = 9):
    n_samples = X.shape[0]
    if n_samples < n_splits:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {n_splits}-fold CV."
        )

    # Class-balanced RF so dry mistakes matter more
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    print(f"\nRunning {n_splits}-fold stratified CV (wet vs dry)...")

    # Plain accuracy (still useful to see, but can be misleading)
    acc_scores = cross_val_score(rf, X, y_bin, cv=cv, scoring="accuracy")
    print(f"Per-fold *accuracy*:        {np.round(acc_scores, 3)}")
    print(
        f"Mean accuracy:              {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    # Balanced accuracy (more meaningful with class imbalance)
    # Do this manually using cross_val_predict
    y_pred = cross_val_predict(rf, X, y_bin, cv=cv)
    bal_acc = balanced_accuracy_score(y_bin, y_pred)
    print(f"Balanced accuracy (global): {bal_acc:.3f}")

    print("\nClassification report (wet vs dry):")
    target_names = ["dry (0.0)", "wet (!=0.0)"]
    print(classification_report(y_bin, y_pred,
          target_names=target_names, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_bin, y_pred))


def main():
    X, y = load_data()
    y_bin = build_wet_dry_labels(y)
    run_wet_dry_rf(X, y_bin, n_splits=9)


if __name__ == "__main__":
    main()
