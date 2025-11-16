import numpy as np


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    arr: numpy array of shape (9, 5, 11)
         [samples, ranges, angles]

    For each sample (0..8) and each range bin (0..4), compute:
      - index of the largest angle bin  (argmax over axis=2)
      - amplitude at that angle index   (max over axis=2)
      - standard deviation over angles
      - skewness over angles

    Returns:
        features: np.ndarray of shape (9, 5, 4)
        feature order: [max_angle_idx, max_amplitude, std, skewness]
    """
    if arr.ndim != 3 or arr.shape[1] != 5 or arr.shape[2] != 11:
        raise ValueError(f"Expected shape (9, 5, 11), got {arr.shape}")

    # 1) Index of largest angle bin (int originally)
    max_idx = np.argmax(arr, axis=2)          # (9, 5)

    # 2) Amplitude at that bin
    max_val = np.max(arr, axis=2)             # (9, 5)

    # 3) Standard deviation over the 11 angles
    mean = np.mean(arr, axis=2)               # (9, 5)
    std = np.std(arr, axis=2)                 # (9, 5)

    # 4) Skewness over the 11 angles (per sample, per range)
    # skew = E[(x - mean)^3] / std^3
    diff = arr - mean[..., None]              # (9, 5, 11)
    m3 = np.mean(diff ** 3, axis=2)           # (9, 5)
    skew = m3 / (std ** 3 + 1e-12)            # (9, 5), avoid divide-by-zero

    # Stack features along last axis: (9, 5, 4)
    # Cast everything to float so we have a homogeneous array
    features = np.stack(
        [
            max_idx.astype(np.float32),
            max_val.astype(np.float32),
            std.astype(np.float32),
            skew.astype(np.float32),
        ],
        axis=-1,
    )

    return features
