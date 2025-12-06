import numpy as np


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    arr: numpy array of shape (9, RANGE_BINS, ANGLE_BINS)
         [positions, ranges, angles]

    For each position and each range bin, compute:

      ORIGINAL FEATURES:
        - max_angle_idx     (argmax over angles)
        - max_amplitude     (max over angles)
        - std               (std over angles)
        - skewness          (manual skew over angles)
        - peak_to_mean      (max / mean)
        - dynamic_range     (max - min)

      EXTRA FEATURES:
        - energy            (mean of squared values over angles)
        - frac_above_half   (fraction of angle bins >= 0.5 * max)
        - center_edge_ratio (center-mean / edge-mean)

    Returns:
        features: (9, RANGE_BINS, F)
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3 dims got {arr.shape}")

    num_positions, num_ranges, num_angles = arr.shape
    eps = 1e-12

    # ---------- Original features ----------

    # 1) Index of largest angle bin
    max_idx = np.argmax(arr, axis=2).astype(np.float32)   # (9, R)

    # 2) Amplitude at that bin
    max_val = np.max(arr, axis=2).astype(np.float32)      # (9, R)

    # 3) Standard deviation over the angles
    mean = np.mean(arr, axis=2)                           # (9, R)
    std = np.std(arr, axis=2)                             # (9, R)

    # 4) Skewness over the angles (manual)
    # skew = E[(x - mean)^3] / std^3
    diff = arr - mean[..., None]                          # (9, R, A)
    m3 = np.mean(diff ** 3, axis=2)                       # (9, R)
    skew = (m3 / (std ** 3 + eps)).astype(np.float32)     # (9, R)

    # 5) Peak-to-mean ratio
    peak_to_mean = (max_val / (mean + eps)).astype(np.float32)  # (9, R)

    # 6) Dynamic range over angles: max - min
    min_val = np.min(arr, axis=2)
    dynamic_range = (max_val - min_val).astype(np.float32)      # (9, R)

    # 7) Energy over angles: mean of squared amplitude
    energy = np.mean(arr ** 2, axis=2).astype(np.float32)       # (9, R)

    # 8) Fraction of angles above half-max (lobe width)
    half_max = 0.5 * max_val[..., None]                         # (9, R, 1)
    above_half = (arr >= half_max).sum(axis=2)                  # (9, R)
    frac_above_half = (above_half / (num_angles + eps)).astype(np.float32)

    # 9) Center vs edges ratio
    third = num_angles // 3
    if third >= 1:
        center = arr[:, :, third:2 * third]                     # middle third
        edges = np.concatenate(
            [arr[:, :, :third], arr[:, :, 2 * third:]],
            axis=2,
        )
        center_mean = np.mean(center, axis=2)
        edges_mean = np.mean(edges, axis=2)
        center_edge_ratio = (center_mean / (edges_mean + eps)).astype(np.float32)
    else:
        # Fallback if angle dimension is tiny
        center_edge_ratio = np.ones_like(max_val, dtype=np.float32)

    # ---------- Stack all features ----------

    features = np.stack(
        [
            max_idx,
            max_val,
            std.astype(np.float32),
            skew,
            peak_to_mean,
            dynamic_range,
            energy,
            frac_above_half,
            center_edge_ratio,
        ],
        axis=-1,  # (9, R, F)
    )

    return features