"""THINGS-EEG2: epoching, baseline correction, 250-Hz resampling and session MVNN.

Portable adaptation of the raw-data implementation in this workspace. The
original historical cache-generation script was not available for exact recovery.
"""

import argparse
from pathlib import Path
import numpy as np
from preprocessing.common import output_directory, metadata, save_ids


def auto_covariance(trial):
    """Public-API equivalent of sklearn's shrinkage='auto' LDA covariance."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import ledoit_wolf

    scaler = StandardScaler()
    z = scaler.fit_transform(trial.T)
    covariance = ledoit_wolf(z)[0]
    return scaler.scale_[:, None] * covariance * scaler.scale_[None, :]


def whitening(covariance):
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * 1e-12, np.finfo(float).tiny)
    return (vectors * np.maximum(eigenvalues, floor) ** -0.5) @ vectors.T


def epoch_session(path, channels, repetitions):
    import mne

    source = np.load(path, allow_pickle=True).item()
    raw = mne.io.RawArray(
        source["raw_eeg_data"],
        mne.create_info(source["ch_names"], source["sfreq"], source["ch_types"]),
        verbose="ERROR",
    )
    events = mne.find_events(raw, stim_channel="stim", verbose="ERROR")
    events = events[events[:, 2] != 99999]
    raw.pick(channels)
    raw.reorder_channels(channels)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.2,
        tmax=1.0,
        baseline=(None, 0),
        preload=True,
        verbose="ERROR",
    )
    epochs.resample(250, verbose="ERROR")
    crop = np.flatnonzero((epochs.times >= -1e-9) & (epochs.times < 1 - 1e-9))
    if len(crop) != 250:
        raise ValueError("Expected exactly 250 samples in [0,1) seconds")
    values, labels = epochs.get_data()[:, :, crop], epochs.events[:, 2]
    result = {}
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) < repetitions:
            raise ValueError(f"Insufficient repetitions for image {label}")
        rows = np.random.RandomState(20200220).permutation(rows)[:repetitions]
        result[int(label) - 1] = values[rows]
    return result


def preprocess(raw_root, subject, output, channels, fit_indices=None, mvnn=True):
    if len(channels) != 63 or len(set(channels)) != 63:
        raise ValueError(
            "Supply exactly 63 unique EEG channel names in original feature-compatible order"
        )
    eligible = (
        np.arange(16540)
        if fit_indices is None
        else np.load(fit_indices, allow_pickle=False)
    )
    if (
        eligible.ndim != 1
        or not np.issubdtype(eligible.dtype, np.integer)
        or np.any((eligible < 0) | (eligible >= 16540))
    ):
        raise ValueError(
            "MVNN fit indices must be zero-based training image rows in 0..16539"
        )
    out = output_directory(output)
    tr = np.lib.format.open_memmap(
        out / "train.npy", mode="w+", dtype="float32", shape=(16540, 4, 63, 250)
    )
    te = np.lib.format.open_memmap(
        out / "test.npy", mode="w+", dtype="float32", shape=(200, 80, 63, 250)
    )
    counts = {"training": np.zeros(16540, int), "test": np.zeros(200, int)}
    audit = []
    for session in range(1, 5):
        folder = Path(raw_root) / f"sub-{subject:02d}" / f"ses-{session:02d}"
        training = epoch_session(folder / "raw_eeg_training.npy", channels, 2)
        fit = sorted(set(training).intersection(eligible.tolist()))
        if mvnn:
            if not fit:
                raise ValueError(f"No covariance fitting images in session {session}")
            cov = np.mean(
                [
                    np.mean([auto_covariance(x) for x in training[i]], axis=0)
                    for i in fit
                ],
                axis=0,
            )
            transform = whitening(cov)
        else:
            transform = np.eye(63)
        np.save(out / f"whitener_session_{session:02d}.npy", transform)
        for part, trials, dest, expected in [
            ("training", training, tr, 4),
            ("test", epoch_session(folder / "raw_eeg_test.npy", channels, 20), te, 80),
        ]:
            for row, x in trials.items():
                if row < 0 or row >= len(dest) or counts[part][row] + len(x) > expected:
                    raise ValueError(
                        f"Out-of-range stimulus or excess repetitions: {part}, {row}"
                    )
                start = counts[part][row]
                dest[row, start : start + len(x)] = np.einsum(
                    "ij,rjt->rit", transform, x
                ).astype(np.float32)
                counts[part][row] += len(x)
            dest.flush()
        audit.append({"session": session, "fit_rows": fit if mvnn else []})
    if not np.all(counts["training"] == 4) or not np.all(counts["test"] == 80):
        raise ValueError("Incomplete image/repetition coverage after four sessions")
    save_ids(out, 16540, 200)
    np.save(out / "channel_names.npy", np.asarray(channels, dtype=str))
    metadata(
        out,
        modality="eeg",
        subject=subject,
        sample_rate_hz=250,
        baseline_seconds=[-0.2, 0],
        poststimulus_interval="[0,1) seconds",
        mvnn=mvnn,
        sessions=audit,
        mvnn_fit_rows=np.unique(eligible).tolist() if mvnn else [],
        fit_scope=(
            "explicit optimization rows"
            if fit_indices
            else "all nominal training images (may include downstream validation)"
        ),
        test_data_used_for_covariance=False,
        train_repeats=4,
        test_repeats=80,
    )


def main():
    p = argparse.ArgumentParser(
        description="Prepare THINGS-EEG2 raw sessions; use explicit channel order"
    )
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--channels",
        type=Path,
        required=True,
        help="63 channel names as a string NPY or one name per text line",
    )
    p.add_argument(
        "--fit-indices",
        type=Path,
        help="Zero-based optimization image rows used to fit MVNN",
    )
    p.add_argument("--no-mvnn", action="store_true")
    p.add_argument(
        "--trust-pickle",
        action="store_true",
        help="Required to read official raw NumPy dictionary files",
    )
    a = p.parse_args()
    if not a.trust_pickle:
        p.error("Official raw dictionary files require --trust-pickle")
    names = (
        np.load(a.channels, allow_pickle=False).tolist()
        if a.channels.suffix == ".npy"
        else a.channels.read_text().splitlines()
    )
    preprocess(a.raw_root, a.subject, a.output, names, a.fit_indices, not a.no_mvnn)


if __name__ == "__main__":
    main()
