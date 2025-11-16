def mean_middle(arr, middle=30):
    """
    arr: numpy array of shape (9, 50, 5, 11)

    Operation:
      - drop first 10 frames
      - drop last 10 frames
      - mean over the remaining 30

    Returns:
      shape (9, 5, 11)
    """
    if arr.shape[1] != 50:
        raise ValueError(
            f"Expected axis 1 to be length 50, got {arr.shape[1]}")

    # Slice out frames 10..39 (inclusive) -> 30 frames
    mid = arr[:, (50-middle)//2:(50 - (50-middle)) //
              2, :, :]

    # Mean over the middle frame dimension (axis=1)
    return mid.mean(axis=1)
