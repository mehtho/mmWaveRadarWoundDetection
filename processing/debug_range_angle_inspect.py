from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from range_angle import range_angle_matrix_for_9_files
from filtering import mean_middle
from ablation_vars import ABLATION_VARS

# File just to plot some heat maps for feature brainstorming

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets" / "Official Testing Data"

TRAINING_DIRS = ["j", "w"]
VALID_LABELS = {"0.0", "2.5", "3.75", "5.0", "7.5"}
DESIRED_LABELS = [0.0, 2.5, 3.75, 5.0, 7.5]

def get_numeric_label(sample_dir: Path) -> float:
    """
    Given a sample directory like:
        ../datasets/Official Testing Data/w/2.5/w_2.5_8
    or     ../datasets/Official Testing Data/j/3.75/j_3.75_2
    return the numeric label from the '2.5' or '3.75' part.
    """
    rel = sample_dir.relative_to(DATASET_DIR)
    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"Cannot derive label from path: {sample_dir}")

    label_str = parts[1]
    if VALID_LABELS and label_str not in VALID_LABELS:
        raise ValueError(
            f"Unexpected label directory: {label_str} in {sample_dir}"
        )

    return float(label_str)


def iter_sample_dirs():
    """
    Yield (label, sample_dir, npy_files) for each sample directory.

    A 'sample' is defined as any directory under TRAINING_DIRS
    that contains exactly 9 .npy files.
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
                continue

            label = get_numeric_label(subdir)
            yield label, subdir, npy_files


def pick_one_sample_per_label():
    """
    Return a dict: label -> (sample_dir, npy_files)
    choosing the first sample we find for each label in DESIRED_LABELS.
    """
    chosen = {}
    label_set = set(DESIRED_LABELS)

    for label, sample_dir, npy_files in iter_sample_dirs():
        if label not in label_set:
            continue
        if label in chosen:
            continue
        chosen[label] = (sample_dir, npy_files)
        print(f"[INFO] Chose sample for label {label}: {sample_dir}")

        if len(chosen) == len(label_set):
            break

    missing = label_set - set(chosen.keys())
    if missing:
        print(f"[WARN] Missing labels with samples: {missing}")

    return chosen


def main():
    sample_map = pick_one_sample_per_label()
    if not sample_map:
        print("No samples found. Check DATASET_DIR / TRAINING_DIRS path.")
        return

    # Which grid positions to visualize
    positions_to_plot = [0, 4, 8]

    for label in sorted(sample_map.keys()):
        sample_dir, npy_files = sample_map[label]
        print(f"\n=== Label {label} from {sample_dir} ===")

        # Load the 9 files for this sample
        arrays_9 = [np.load(f) for f in npy_files]
        print("  Example file shape:", arrays_9[0].shape)

        # Compute range-angle matrices for all 9 positions
        # rams shape: (9, N_frames, range_bins, angle_bins)
        rams = range_angle_matrix_for_9_files(
            arrays_9,
            ABLATION_VARS.NORM_PER_FRAME
        )
        print("  range_angle_matrix_for_9_files ->", rams.shape)

        # Temporal averaging over middle frames -> (9, range_bins, angle_bins)
        filtered = mean_middle(rams)
        print("  mean_middle ->", filtered.shape)

        # Basic stats
        print("  filtered min/max:", filtered.min(), filtered.max())

        # Plot some positions
        fig, axes = plt.subplots(
            1, len(positions_to_plot),
            figsize=(4 * len(positions_to_plot), 4),
            constrained_layout=True
        )

        if len(positions_to_plot) == 1:
            axes = [axes] 

        for ax, pos_idx in zip(axes, positions_to_plot):
            if pos_idx < 0 or pos_idx >= filtered.shape[0]:
                continue

            mat = filtered[pos_idx]
            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower"
            )
            ax.set_title(f"Label {label}, pos {pos_idx}")
            ax.set_xlabel("Angle beam index")
            ax.set_ylabel("Range bin index")
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"Range–angle maps for label {label}")
        plt.show()


if __name__ == "__main__":
    main()
