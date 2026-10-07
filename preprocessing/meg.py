"""THINGS-MEG epoch preparation and canonical stimulus ordering.

The cache conversion follows the local get_data.py's category/exemplar mapping
and 201-to-200 sample crop, without additional signal scaling.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from preprocessing.common import output_directory, metadata, save_ids


def order_trials(values, labels, train_count=22248, test_count=200, repeats=12):
    """Return row indices; reject missing/duplicate experimental images instead of overwriting."""
    if len(values) != len(labels):
        raise ValueError("Neural trial count differs from metadata rows")
    train_rows = np.flatnonzero(labels.trial_type.eq("exp"))
    test_rows = np.flatnonzero(labels.trial_type.eq("test"))
    fields = labels.iloc[train_rows][
        ["things_category_nr", "things_exemplar_nr"]
    ].to_numpy(float)
    if not np.isfinite(fields).all() or not np.equal(fields, np.floor(fields)).all():
        raise ValueError(
            "Training category and exemplar labels must be finite integers"
        )
    if np.any((fields[:, 1] < 1) | (fields[:, 1] > 12)):
        raise ValueError("Expected exemplar IDs 1..12")
    destinations = 12 * (fields[:, 0].astype(int) - 1) + fields[:, 1].astype(int) - 1
    if len(destinations) != train_count or not np.array_equal(
        np.sort(destinations), np.arange(train_count)
    ):
        raise ValueError("Each canonical training image must appear exactly once")
    train_order = train_rows[np.argsort(destinations)]
    ids = labels.iloc[test_rows].test_image_nr.to_numpy(float)
    if not np.array_equal(np.unique(ids), np.arange(1, test_count + 1)):
        raise ValueError("Test image IDs must cover the complete one-based gallery")
    test_order = [test_rows[ids == i] for i in range(1, test_count + 1)]
    if any(len(x) != repeats for x in test_order):
        raise ValueError("Unexpected number of repetitions for a test stimulus")
    return train_order, np.stack(test_order)


def prepare_epochs(
    input_path,
    labels_path,
    output,
    fmt="cosmo",
    channels=271,
    train_count=22248,
    test_count=200,
    repeats=12,
):
    labels = pd.read_csv(labels_path)
    if fmt == "cosmo":
        import h5py

        with h5py.File(input_path, "r") as f:
            flat = np.asarray(f["ds/samples"])
        if flat.shape[0] != 281 * channels:
            raise ValueError("CoSMo samples must be [281*channels, trials]")
        values = flat.reshape(281, channels, -1).transpose(2, 1, 0)
        times = -0.1 + np.arange(281) / 200
    else:
        import mne

        epochs = mne.read_epochs(input_path, preload=True, verbose="ERROR")
        epochs.pick("mag")
        if len(epochs.ch_names) != channels or not np.isclose(
            epochs.info["sfreq"], 200
        ):
            raise ValueError("Expected 271 MEG channels sampled at 200 Hz")
        values, times = epochs.get_data(), epochs.times
    # The original loader discarded t=0 from the 0..1 s inclusive cache.
    crop = np.flatnonzero((times > 1e-9) & (times <= 1 + 1e-9))
    if len(crop) != 200:
        raise ValueError("Expected 200 samples at t=0.005..1.000 s")
    train_rows, test_rows = order_trials(
        values, labels, train_count, test_count, repeats
    )
    out = output_directory(output)
    for name, indices in [("train", train_rows[:, None]), ("test", test_rows)]:
        target = np.lib.format.open_memmap(
            out / f"{name}.npy",
            mode="w+",
            dtype="float32",
            shape=(*indices.shape, channels, 200),
        )
        for i, row in enumerate(indices):
            target[i] = values[row][:, :, crop]
            if not np.isfinite(target[i]).all():
                raise ValueError("Nonfinite input MEG values")
        target.flush()
    save_ids(out, train_count, test_count)
    labels.iloc[train_rows].to_csv(out / "train_metadata.csv", index=False)
    labels.iloc[test_rows[:, 0]].to_csv(out / "test_metadata.csv", index=False)
    metadata(
        out,
        modality="meg",
        source=Path(input_path).name,
        format=fmt,
        channels=channels,
        sample_rate_hz=200,
        crop_seconds=[0.005, 1.0],
        normalization="inherited from input epochs; no additional scaling",
        ordering="12*(category_id-1)+(exemplar_id-1); test_image_nr-1",
    )


def epoch_ctf(input_path, events_path, output, low=0.1, high=95.0):
    """Epoch one CTF run with explicit event metadata and filter before downsampling.

    events.tsv onset is in seconds relative to raw-file start. All rows must be
    explicitly selected stimulus events; code does not guess photodiode latency.
    """
    import mne

    if Path(output).exists() or Path(output).with_suffix(".csv").exists():
        raise FileExistsError(output)
    raw = mne.io.read_raw_ctf(input_path, preload=True, verbose="ERROR")
    if "MRO11-1609" in raw.ch_names:
        raw.drop_channels(["MRO11-1609"])
    raw.pick("mag")
    if len(raw.ch_names) != 271:
        raise ValueError("Expected 271 channels after removing MRO11-1609")
    raw.filter(low, high, verbose="ERROR")
    labels = pd.read_csv(events_path, sep="\t")
    onsets = labels.onset.to_numpy(float)
    if (
        not np.isfinite(onsets).all()
        or len(np.unique(onsets)) != len(onsets)
        or np.any(np.diff(onsets) <= 0)
    ):
        raise ValueError(
            "Provide unique, increasing stimulus onsets, with latency corrections already applied"
        )
    events = np.column_stack(
        [
            np.rint(onsets * raw.info["sfreq"]).astype(int) + raw.first_samp,
            np.zeros(len(labels), int),
            np.ones(len(labels), int),
        ]
    )
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.1,
        tmax=1.3,
        baseline=None,
        preload=True,
        metadata=labels,
        verbose="ERROR",
    )
    if len(epochs) != len(labels):
        raise ValueError(
            "Epochs dropped at boundaries/annotations; resolve event alignment before export"
        )
    data = epochs.get_data()
    baseline = data[:, :, epochs.times <= 0]
    scale = baseline.std(-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Zero-variance baseline")
    normalized = (data - baseline.mean(-1, keepdims=True)) / scale
    result = mne.EpochsArray(
        normalized,
        epochs.info,
        events=epochs.events,
        tmin=epochs.tmin,
        metadata=labels,
        verbose="ERROR",
    )
    result.resample(200, verbose="ERROR")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.save(output, overwrite=False)
    labels.to_csv(Path(output).with_suffix(".csv"), index=False)


def main():
    p = argparse.ArgumentParser(
        description="THINGS-MEG raw-run epoching or canonical image-ordered cache preparation"
    )
    modes = p.add_subparsers(dest="mode", required=True)
    cache = modes.add_parser("prepare")
    cache.add_argument("--input", type=Path, required=True)
    cache.add_argument("--labels", type=Path, required=True)
    cache.add_argument("--format", choices=["cosmo", "fif"], default="cosmo")
    cache.add_argument("--output", type=Path, required=True)
    raw = modes.add_parser("epoch-ctf")
    raw.add_argument("--input", type=Path, required=True)
    raw.add_argument("--events", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare_epochs(a.input, a.labels, a.output, a.format)
    else:
        epoch_ctf(a.input, a.events, a.output)


if __name__ == "__main__":
    main()
