"""TVSD MUA: stimulus ordering, optional baseline subtraction, cropping and per-trial z-score.

Adapted from the project's TVSD preprocessing, with streamed HDF5 reading.
The two animals are processed independently; no electrode correspondence is assumed.
"""

import argparse
from pathlib import Path
import numpy as np
from preprocessing.common import output_directory, metadata, save_ids


def normalize_trial(trial, baseline_samples=100, subtract_baseline=False):
    x = np.asarray(trial, dtype=np.float64)
    if (
        x.ndim != 2
        or not 0 < baseline_samples < x.shape[-1]
        or not np.isfinite(x).all()
    ):
        raise ValueError(
            "Expected finite [channels,time] data with a nonempty baseline and poststimulus interval"
        )
    if subtract_baseline:
        x = x - x[:, :baseline_samples].mean(axis=-1, keepdims=True)
    x = x[:, baseline_samples:]
    std = x.std()
    if std == 0:
        raise ValueError("Constant trial cannot be z-scored")
    return ((x - x.mean()) / std).astype(np.float32)


def preprocess(
    source,
    output,
    baseline="none",
    train_count=22248,
    test_count=100,
    repeats=30,
    channels=1024,
):
    import h5py

    with h5py.File(source, "r") as f:
        data, labels = f["ALLMUA"], np.asarray(f["ALLMAT"])
        if data.ndim != 3 or data.shape[0] != 300 or data.shape[2] != channels:
            raise ValueError(
                "ALLMUA must be [300 samples, trials, channels] in the original HDF5 layout"
            )
        if labels.ndim != 2 or labels.shape[0] < 3 or labels.shape[1] != data.shape[1]:
            raise ValueError(
                "ALLMAT must have attributes on rows and trials on columns"
            )
        train_idx, test_idx = labels[1], labels[2]
        out = output_directory(output)
        x = np.lib.format.open_memmap(
            out / "train.npy",
            mode="w+",
            dtype="float32",
            shape=(train_count, 1, channels, 200),
        )
        y = np.lib.format.open_memmap(
            out / "test.npy",
            mode="w+",
            dtype="float32",
            shape=(test_count, repeats, channels, 200),
        )
        for kind, ids, target, count, expected_repeats in [
            ("train", train_idx, x, train_count, 1),
            ("test", test_idx, y, test_count, repeats),
        ]:
            valid = ids[np.isfinite(ids) & (ids > 0)]
            if not np.array_equal(np.unique(valid), np.arange(1, count + 1)):
                raise ValueError(f"{kind} IDs do not cover exactly 1..{count}")
            for i in range(count):
                rows = np.flatnonzero(ids == i + 1)
                if len(rows) != expected_repeats:
                    raise ValueError(
                        f"{kind} image {i + 1}: expected {expected_repeats} trials, got {len(rows)}"
                    )
                for j, row in enumerate(rows):
                    target[i, j] = normalize_trial(
                        data[:, int(row), :].T, subtract_baseline=baseline == "subtract"
                    )
            target.flush()
    save_ids(out, train_count, test_count)
    metadata(
        out,
        modality="mua",
        source=Path(source).name,
        channels=channels,
        sample_rate_hz=1000,
        time_samples="100:300 of original 300-sample trial",
        baseline=baseline,
        normalization="per-trial z-score over all poststimulus channels and times",
        train_repeats=1,
        test_repeats=repeats,
        image_ids="one-based ALLMAT stimulus IDs, local to each split",
    )


def main():
    p = argparse.ArgumentParser(
        description="Prepare one TVSD animal from THINGS_MUA_trials.mat"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", choices=["none", "subtract"], default="none")
    a = p.parse_args()
    preprocess(a.input, a.output, a.baseline)


if __name__ == "__main__":
    main()
