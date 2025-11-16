#!/usr/bin/env python3

from pathlib import Path
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier  # <-- NEW

OUTPUT_DIR = Path("../output/official_features")

# -----------------------------
# Subsets to evaluate
# -----------------------------
WATER_LABELS = {0.0, 2.5, 5.0}
JELLY_LABELS = {0.0, 3.75, 7.5}
ALL_LABELS = {0.0, 2.5, 3.75, 5.0, 7.5}

# For water vs jelly binary task (exclude dry)
WATER_ONLY = {2.5, 5.0}
JELLY_ONLY = {3.75, 7.5}


# -----------------------------
# Load X and y
# -----------------------------
def load_data():
    X = np.load(OUTPUT_DIR / "X.npy")
    y = np.load(OUTPUT_DIR / "y.npy")

    print(f"Loaded X: {X.shape}")
    print(f"Loaded y: {y.shape}")

    y = y.reshape(-1)
    return X, y


# -----------------------------
# Filter dataset to chosen labels
# -----------------------------
def filter_by_labels(X, y, allowed_labels):
    mask = np.isin(y, list(allowed_labels))
    return X[mask], y[mask]


# -----------------------------
# Model evaluation (XGBoost)
# -----------------------------
def run_xgb_cv(X, y, title="Experiment"):
    print("\n=====================================")
    print(f"   {title}")
    print("=====================================")

    # Encode labels to class indices 0..N
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("\nLabel encoding:")
    for idx, cls in enumerate(le.classes_):
        print(f"  class_index={idx} <- label={cls}")

    n_classes = len(le.classes_)

    # Choose objective based on number of classes
    if n_classes == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
        extra_params = {}
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        extra_params = {"num_class": n_classes}

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective=objective,
        eval_metric=eval_metric,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        verbosity=0,
        **extra_params,
    )

    n_splits = min(9, len(y_enc))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(xgb, X, y_enc, cv=cv, scoring="accuracy")
    print(f"\nPer-fold accuracy: {np.round(scores, 3)}")
    print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    y_pred = cross_val_predict(xgb, X, y_enc, cv=cv)

    print("\nClassification report:")
    print(
        classification_report(
            y_enc,
            y_pred,
            digits=3,
            target_names=[str(c) for c in le.classes_],
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_enc, y_pred))


# -----------------------------
# Water vs Jelly (binary, exclude dry)
# -----------------------------
def run_water_vs_jelly(X, y):
    allowed = WATER_ONLY | JELLY_ONLY
    X_sub, y_sub = filter_by_labels(X, y, allowed)

    if X_sub.shape[0] == 0:
        raise ValueError("No samples left after filtering for water vs jelly!")

    y_bin = np.empty_like(y_sub, dtype=object)
    y_bin[np.isin(y_sub, list(WATER_ONLY))] = "water"
    y_bin[np.isin(y_sub, list(JELLY_ONLY))] = "jelly"

    print("\nWater vs Jelly distribution:")
    unique, counts = np.unique(y_bin, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")

    run_xgb_cv(X_sub, y_bin, title="WATER VS JELLY (2.5/5.0 vs 3.75/7.5)")


# -----------------------------
# Generic pairwise experiment
# -----------------------------
def run_pairwise_experiment(
    X,
    y,
    label_a: float,
    name_a: str,
    label_b: float,
    name_b: str,
    title: str,
):
    allowed = {label_a, label_b}
    X_sub, y_sub = filter_by_labels(X, y, allowed)

    if X_sub.shape[0] == 0:
        raise ValueError(f"No samples left after filtering for {title}!")

    y_pair = np.empty_like(y_sub, dtype=object)
    y_pair[y_sub == label_a] = name_a
    y_pair[y_sub == label_b] = name_b

    print(f"\n{title} distribution:")
    unique, counts = np.unique(y_pair, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")

    run_xgb_cv(X_sub, y_pair, title=title)


# -----------------------------
# 3-class: dry vs water 5.0 vs jelly 7.5
# -----------------------------
def run_dry_water5_jelly75(X, y):
    allowed = {0.0, 5.0, 7.5}
    X_sub, y_sub = filter_by_labels(X, y, allowed)

    if X_sub.shape[0] == 0:
        raise ValueError(
            "No samples left after filtering for dry/water5/jelly75!")

    y_tri = np.empty_like(y_sub, dtype=object)
    y_tri[y_sub == 0.0] = "dry_0.0"
    y_tri[y_sub == 5.0] = "water_5.0"
    y_tri[y_sub == 7.5] = "jelly_7.5"

    print("\nDRY 0.0 vs WATER 5.0 vs JELLY 7.5 distribution:")
    unique, counts = np.unique(y_tri, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} samples")

    run_xgb_cv(X_sub, y_tri, title="DRY (0.0) vs WATER 5.0 vs JELLY 7.5")


# -----------------------------
# Main
# -----------------------------
def main():
    X, y = load_data()

    # 1) WATER ONLY (0, 2.5, 5.0)
    Xw, yw = filter_by_labels(X, y, WATER_LABELS)
    run_xgb_cv(Xw, yw, title="WATER ONLY (0, 2.5, 5.0)")

    # 2) JELLY ONLY (0, 3.75, 7.5)
    Xj, yj = filter_by_labels(X, y, JELLY_LABELS)
    run_xgb_cv(Xj, yj, title="JELLY ONLY (0, 3.75, 7.5)")

    # 3) ALL LABELS (original)
    run_xgb_cv(X, y, title="ALL LABELS (0, 2.5, 3.75, 5.0, 7.5)")

    # 4) WATER VS JELLY (binary, exclude dry)
    run_water_vs_jelly(X, y)

    # 5) DRY vs WATER 5.0 (0.0 vs 5.0)
    run_pairwise_experiment(
        X,
        y,
        label_a=0.0,
        name_a="dry_0.0",
        label_b=5.0,
        name_b="water_5.0",
        title="DRY (0.0) vs WATER 5.0",
    )

    # 6) DRY vs JELLY 7.5 (0.0 vs 7.5)
    run_pairwise_experiment(
        X,
        y,
        label_a=0.0,
        name_a="dry_0.0",
        label_b=7.5,
        name_b="jelly_7.5",
        title="DRY (0.0) vs JELLY 7.5",
    )

    # 7) WATER 5.0 vs JELLY 7.5 (5.0 vs 7.5)
    run_pairwise_experiment(
        X,
        y,
        label_a=5.0,
        name_a="water_5.0",
        label_b=7.5,
        name_b="jelly_7.5",
        title="WATER 5.0 vs JELLY 7.5",
    )

    # 8) DRY 0.0 vs WATER 5.0 vs JELLY 7.5 (3-class)
    run_dry_water5_jelly75(X, y)


if __name__ == "__main__":
    main()
