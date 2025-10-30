import numpy as np
import time

from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp

from pathlib import Path

NUM_SAMPLES = 10
DATA_DIR = "output"


def num_rx_antennas_from_rx_mask(rx_mask):
    # popcount for rx_mask
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c


if __name__ == '__main__':
    num_beams = 27
    max_angle_degrees = 40

    config = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=0.08,
        chirp_repetition_time_s=0.00035,
        num_chirps=128,
        tdm_mimo=False,
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=58.0e9,
            end_frequency_Hz=63.5e9,
            sample_rate_Hz=2_000_000,
            num_samples=64,
            rx_mask=0b111,
            tx_mask=0b001,
            tx_power_level=18,
            lp_cutoff_Hz=500_000,
            hp_cutoff_Hz=20_000,
            if_gain_dB=20
        )
    )

    with DeviceFmcw() as device:
        # configure device
        sequence = device.create_simple_sequence(config)
        device.set_acquisition_sequence(sequence)

        # get metrics and print them
        chirp_loop = sequence.loop.sub_sequence.contents
        metrics = device.metrics_from_sequence(chirp_loop)

        # get maximum range
        max_range_m = metrics.max_range_m

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        num_rx_antennas = num_rx_antennas_from_rx_mask(chirp.rx_mask)

        frames = []

        for _ in range(NUM_SAMPLES):
            # frame has dimension num_rx_antennas x num_chirps_per_frame x num_samples_per_chirp
            frame_contents = device.get_next_frame()
            frame = frame_contents[0]
            frames.append(frame)

        # Combine all frames into one ndarray
        data = np.stack(frames, axis=0)

        # Prepare timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.npy"

        # Use pathlib for path operations
        DATA_DIR = Path(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = DATA_DIR / filename

        # Save to file
        np.save(filepath, data)

        print(f"Saved {data.shape} to {filepath}")
