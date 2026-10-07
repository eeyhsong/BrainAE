"""Small utilities for explicit splits, aligned visual features and concept exclusions."""

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    split = modes.add_parser("split")
    split.add_argument("--count", type=int, required=True)
    split.add_argument("--validation-size", type=int, required=True)
    split.add_argument("--seed", type=int, default=2023)
    split.add_argument("--exclude-indices", type=Path)
    split.add_argument("--output", type=Path, required=True)
    bundle = modes.add_parser("features")
    bundle.add_argument("--features", type=Path, required=True)
    bundle.add_argument(
        "--image-ids",
        type=Path,
        required=True,
        help="Actual feature row IDs; do not invent IDs from the neural ordering",
    )
    bundle.add_argument("--output", type=Path, required=True)
    exclude = modes.add_parser("exclude-concepts")
    exclude.add_argument("--train-concepts", type=Path, required=True)
    exclude.add_argument("--test-concepts", type=Path, required=True)
    exclude.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "split":
        rows = np.arange(a.count)
        if a.exclude_indices:
            indices = np.load(a.exclude_indices, allow_pickle=False)
            if (
                indices.ndim != 1
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any((indices < 0) | (indices >= a.count))
            ):
                raise ValueError("Invalid exclusion indices")
            rows = np.setdiff1d(rows, indices)
        if not 0 < a.validation_size < len(rows):
            raise ValueError("Validation size must leave optimization rows")
        with a.output.open("wb") as f:
            np.save(
                f, np.random.RandomState(a.seed).permutation(rows)[a.validation_size :]
            )
    elif a.mode == "features":
        features = np.load(a.features, allow_pickle=False)
        ids = np.load(a.image_ids, allow_pickle=False)
        if (
            len(features) != len(ids)
            or ids.ndim != 1
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError(
                "Features and unique image IDs must have identical row counts"
            )
        with a.output.open("wb") as f:
            np.savez(f, features=features, image_ids=ids)
    else:
        tr = np.load(a.train_concepts, allow_pickle=False).astype(str)
        te = np.load(a.test_concepts, allow_pickle=False).astype(str)
        if tr.ndim != 1 or te.ndim != 1:
            raise ValueError("Concept identifiers must be one-dimensional arrays")
        with a.output.open("wb") as f:
            np.save(f, np.flatnonzero(np.isin(tr, te)))


if __name__ == "__main__":
    main()
