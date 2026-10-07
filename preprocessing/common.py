"""Prepared-array format and explicit legacy-cache conversion."""

import argparse
import json
from pathlib import Path
import pickle
import numpy as np


def output_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def metadata(folder, **details):
    (Path(folder) / "preprocessing.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )


def save_ids(folder, train_count, test_count):
    # These are split-local row IDs, not semantic concept IDs.
    np.save(Path(folder) / "train_ids.npy", np.arange(1, train_count + 1))
    np.save(Path(folder) / "test_ids.npy", np.arange(1, test_count + 1))


def trusted_array(path, trust_pickle=False):
    path = Path(path)
    if path.suffix == ".pkl":
        if not trust_pickle:
            raise ValueError(
                "Historical pickle caches require --trust-pickle; use only trusted dataset files"
            )
        with path.open("rb") as f:
            return pickle.load(f)
    obj = np.load(path, allow_pickle=trust_pickle)
    if isinstance(obj, np.lib.npyio.NpzFile):
        with obj:
            return obj["preprocessed_eeg_data"].copy()
    if obj.shape == () and obj.dtype == object:
        return obj.item()["preprocessed_eeg_data"]
    return obj


def copy_array(path, values):
    dest = np.lib.format.open_memmap(
        path, mode="w+", dtype="float32", shape=values.shape
    )
    for i in range(len(values)):
        v = np.asarray(values[i], dtype=np.float32)
        if not np.isfinite(v).all():
            raise ValueError(f"Nonfinite values in row {i}")
        dest[i] = v
    dest.flush()


def convert_cache(modality, train, test, output, trust_pickle=False):
    """Retain cache normalization, repeat order and channel order; crop MEG zero-time sample."""
    x, y = trusted_array(train, trust_pickle), trusted_array(test, trust_pickle)
    if x.ndim == 3:
        x = x[:, None]
    if y.ndim == 3:
        y = y[:, None]
    if modality == "meg":
        if x.shape[-1] == 201:
            x = x[..., 1:]
        if y.shape[-1] == 201:
            y = y[..., 1:]
    expected = {"eeg": (63, 250), "meg": (271, 200), "mua": (1024, 200)}[modality]
    if (
        x.ndim != 4
        or y.ndim != 4
        or x.shape[-2:] != expected
        or y.shape[-2:] != expected
    ):
        raise ValueError(
            f"{modality}: expected image/repeat axes followed by {expected}; got {x.shape}, {y.shape}"
        )
    out = output_directory(output)
    copy_array(out / "train.npy", x)
    copy_array(out / "test.npy", y)
    save_ids(out, len(x), len(y))
    metadata(
        out,
        modality=modality,
        source="existing cache",
        normalization="inherited unchanged",
        train_source=Path(train).name,
        test_source=Path(test).name,
        train_shape=list(x.shape),
        test_shape=list(y.shape),
        image_ids="one-based split-local row IDs",
        note="Cache conversion does not reconstruct upstream filtering or normalization-fit provenance.",
    )


def main():
    p = argparse.ArgumentParser(
        description="Convert original EEG/MEG/MUA caches into portable memory-mappable arrays"
    )
    p.add_argument("--modality", choices=["eeg", "meg", "mua"], required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trust-pickle", action="store_true")
    a = p.parse_args()
    convert_cache(a.modality, a.train, a.test, a.output, a.trust_pickle)


if __name__ == "__main__":
    main()
