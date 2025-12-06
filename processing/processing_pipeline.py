from pathlib import Path
import numpy as np

from ablation_vars import ABLATION_VARS
from features import extract_features
from filtering import mean_middle
from range_angle import range_angle_matrix_for_9_files

DATASET_DIR = Path("../datasets/Official Testing Data")
TRAINING_DIRS = ["j", "w"]
OUTPUT_DIR = Path("../output/official_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: restrict to a known set of label values
VALID_LABELS = {"0.0", "2.5", "3.75", "5.0", "7.5"}

def get_numeric_label(sample_dir: Path) -> float:
    """
    Given a sample directory like:
        ../datasets/Official Testing Data/w/2.5/w_2.5_8
    or     ../datasets/Official Testing Data/j/3.75/j_3.75_2
    return the numeric label from the '2.5' or '3.75' part.
    """
    rel = sample_dir.relative_to(DATASET_DIR)
    parts = rel.parts  # e.g. ('w', '2.5', 'w_2.5_8')
    if len(parts) < 2:
        raise ValueError(f"Cannot derive label from path: {sample_dir}")

    label_str = parts[1]
    if VALID_LABELS and label_str not in VALID_LABELS:
        raise ValueError(
            f"Unexpected label directory: {label_str} in {sample_dir}")

    return float(label_str)


def iter_sample_dirs():
    """
    Yield (label, sample_dir, npy_files) for each sample directory.

    A 'sample' is defined as any directory under TRAINING_DIRS
    that contains exactly 9 .npy files (regardless of subdirectories).
    """
    for top in TRAINING_DIRS:
        root = DATASET_DIR / top
        if not root.is_dir():
            print(f"[WARN] Training root not found: {root}")
            continue

        for subdir in root.rglob("*"):
            if not subdir.is_dir():
                continue

            npy_files = sorted(subdir.glob("*.npy"))
            if not npy_files:
                continue

            if len(npy_files) != 9:
                print(
                    f"[WARN] {subdir} has {len(npy_files)} .npy files (expected 9)"
                )
                continue

            label = get_numeric_label(subdir)
            yield label, subdir, npy_files


def filter_sample(rams: np.ndarray) -> np.ndarray:
    """
    For feature extraction / transformation logic.

    rams: np.ndarray of shape (9, 50, RANGE_BINS, ANGLE_BINS)

    Returns:
        (9, RANGE_BINS, ANGLE_BINS) after temporal averaging
        across the middle frames
    """
    transformed = mean_middle(rams)
    return transformed

def build_dry_baseline_global() -> np.ndarray | None:
    """
    Compute a single global dry baseline.

    Returns:
        baseline: np.ndarray of shape (9, RANGE_BINS, ANGLE_BINS)
                  or None if no dry samples found.
    """
    baseline_sum = None
    count = 0

    for label, sample_dir, npy_files in iter_sample_dirs():
        if label != 0.0:
            continue

        arrays = [np.load(f) for f in npy_files]
        rams = range_angle_matrix_for_9_files(
            arrays,
            ABLATION_VARS.NORM_PER_FRAME
        )  # (9, 50, R, A)
        filtered = filter_sample(rams)        # (9, R, A)

        if baseline_sum is None:
            baseline_sum = filtered.astype(np.float64)
        else:
            baseline_sum += filtered
        count += 1

    if count == 0:
        print("[WARN] No dry baselines (label 0.0) found at all")
        return None

    baseline = baseline_sum / count
    print(f"[INFO] Global dry baseline built from {count} dry sample(s), shape {baseline.shape}")
    return baseline.astype(np.float32)

def normalize_per_position_with_baseline(arr: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    arr: (9, R, A)
    baseline: (9, R, A)
    """
    eps = 1e-12
    out = np.empty_like(arr)

    for pos in range(arr.shape[0]):
        frame = arr[pos]
        base = baseline[pos]
        base_max = base.max()
        out[pos] = (frame - base) / (base_max + eps)

    return out

def process_all_samples():
    X = []
    y = []

    baseline = None
    if ABLATION_VARS.SUBTRACT_BASELINE:
        print("[INFO] Building GLOBAL dry baseline...")
        baseline = build_dry_baseline_global()

    for label, sample_dir, npy_files in iter_sample_dirs():
        print(sample_dir)
        arrays = [np.load(f) for f in npy_files]
        rams = range_angle_matrix_for_9_files(
            arrays,
            ABLATION_VARS.NORM_PER_FRAME
        )  # (9, 50, R, A)

        filtered = filter_sample(rams)   # (9, R, A)

        # baseline subtraction
        if ABLATION_VARS.SUBTRACT_BASELINE and label != 0.0:
            filtered = filtered - baseline

        # new: per-position normalization
        if ABLATION_VARS.NORM_PER_POSITION:
            if ABLATION_VARS.SUBTRACT_BASELINE:
                filtered = normalize_per_position_with_baseline(filtered, baseline)

        features = extract_features(filtered)
        X.append(features.flatten())
        y.append(label)

    # Convert to arrays
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    print("Final dataset:")
    print("  X shape:", X_arr.shape)
    print("  y shape:", y_arr.shape)

    # Save to OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X.npy", X_arr)
    np.save(OUTPUT_DIR / "y.npy", y_arr)

    print(f"Saved X.npy and y.npy to {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all_samples()