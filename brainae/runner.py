"""Shared I/O, training and evaluation for the original per-recording BrainAE models."""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SPECS = {
    "eeg": (63, 250, 200, 800, 740, 2023),
    "meg": (271, 200, 200, 600, 748, 2023),
    "mua": (1024, 200, 100, 100, 748, 2024),
}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite(array, name):
    for start in range(0, len(array), 128):
        if not np.isfinite(array[start : start + 128]).all():
            raise ValueError(f"{name} contains nonfinite entries")


def feature_array(path, expected_ids, assume_order):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if not {"features", "image_ids"}.issubset(loaded.files):
                raise ValueError(f"{Path(path).name}: NPZ must contain features and image_ids")
            data = loaded["features"].copy()
            ids = loaded["image_ids"].copy()
        if not np.array_equal(ids.astype(str), expected_ids.astype(str)):
            raise ValueError(
                f"Image order differs between neural data and {Path(path).name}"
            )
    else:
        if not assume_order:
            raise ValueError(
                "Use an NPZ with features/image_ids, or explicitly --assume-feature-order for native NPY caches"
            )
        data = loaded
    # Native files may have a singleton axis between image and embedding.
    if data.ndim == 3 and data.shape[1] == 1:
        data = data[:, 0]
    if data.shape != (len(expected_ids), 1024):
        raise ValueError(
            f"{Path(path).name}: expected {(len(expected_ids), 1024)}, got {data.shape}"
        )
    finite(data, "visual features")
    return np.asarray(data, dtype=np.float32)


def load_signals(folder, modality, splits=("train", "test")):
    channels, samples, *_ = SPECS[modality]
    arrays = {}
    for split in splits:
        x = np.load(folder / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        if x.ndim != 4 or x.shape[2:] != (channels, samples) or x.shape[1] < 1:
            raise ValueError(
                f"{split}: expected [images, repeats, {channels}, {samples}], got {x.shape}"
            )
        finite(x, split)
        ids = np.load(folder / f"{split}_ids.npy", allow_pickle=False)
        if ids.ndim != 1 or len(ids) != len(x) or len(np.unique(ids)) != len(ids):
            raise ValueError(
                f"{split} image IDs must be unique and match the first array dimension"
            )
        arrays[split], arrays[split + "_ids"] = x, ids
    return arrays


class Responses(Dataset):
    """Average repeats on demand, retaining the original stimulus-level training unit."""

    def __init__(self, signals, features, indices):
        self.signals, self.features, self.indices = (
            signals,
            features,
            np.asarray(indices),
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        x = np.asarray(self.signals[index], dtype=np.float32).mean(
            axis=0, keepdims=True
        )
        return torch.from_numpy(x.copy()), torch.from_numpy(self.features[index].copy())


def four_losses(encoder, decoder, response, visual):
    """Original four equally weighted losses; temperature fixed to 0.07."""
    z = encoder(response)
    reconstructed = decoder(z)
    encoded = decoder(visual)
    logits = F.normalize(z, dim=1) @ F.normalize(visual, dim=1).T / 0.07
    labels = torch.arange(len(z), device=z.device)
    contrastive = (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    ) / 2
    return torch.stack(
        [
            contrastive,
            F.mse_loss(z, visual),
            F.mse_loss(reconstructed, response),
            F.mse_loss(encoded, response),
        ]
    )


def epoch(encoder, decoder, loader, device, optimizer=None):
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    losses = []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            parts = four_losses(encoder, decoder, x, y)
            if not torch.isfinite(parts).all():
                raise FloatingPointError(
                    "Nonfinite loss; check neural amplitudes and visual features"
                )
            if training:
                parts.sum().backward()
                optimizer.step()
            losses.append(parts.detach().cpu().numpy())
    if not losses:
        raise ValueError("Empty training or validation loader")
    # Preserve original equal-batch averaging for validation checkpoint selection.
    return np.stack(losses).mean(0)


def correlation(x, y):
    x, y = x.flatten(1), y.flatten(1)
    x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    denom = x.norm(dim=1) * y.norm(dim=1)
    return torch.where(denom > 0, (x * y).sum(1) / denom, torch.nan)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def evaluate(
    encoder, decoder, signals, features, prototypes, out, device, batch_size, metric
):
    encoder.eval()
    decoder.eval()
    embeddings, encoding_r, reconstruction_r, mse, reconstruction_mse = (
        [],
        [],
        [],
        [],
        [],
    )
    loader = DataLoader(
        Responses(signals, features, np.arange(len(signals))), batch_size=batch_size
    )
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = encoder(x)
            pred, reconstructed = decoder(y), decoder(z)
            embeddings.append(z.cpu().numpy())
            encoding_r.extend(correlation(x, pred).cpu().tolist())
            reconstruction_r.extend(correlation(x, reconstructed).cpu().tolist())
            mse.extend((x - pred).square().flatten(1).mean(1).cpu().tolist())
            reconstruction_mse.extend(
                (x - reconstructed).square().flatten(1).mean(1).cpu().tolist()
            )
    z = torch.from_numpy(np.concatenate(embeddings))
    z = F.normalize(z, dim=1)
    p = torch.from_numpy(prototypes.astype(np.float32))
    if metric == "cosine":
        p = F.normalize(p, dim=1)
    classification = (z @ p.T).numpy()
    retrieval = (z @ F.normalize(torch.from_numpy(features), dim=1).T).numpy()
    labels = np.arange(len(features))
    result = {
        "classification_metric": metric,
        "queries": len(labels),
        "gallery_size": len(prototypes),
    }
    for name, scores in [("classification", classification), ("retrieval", retrieval)]:
        ranking = np.argsort(-scores, axis=1, kind="stable")
        for k in [1, 3, 5]:
            result[f"{name}_top{k}"] = float(
                np.mean(
                    (ranking[:, : min(k, len(prototypes))] == labels[:, None]).any(1)
                )
            )
    for name, values in [
        ("encoding_r", encoding_r),
        ("reconstruction_r", reconstruction_r),
        ("encoding_mse", mse),
        ("reconstruction_mse", reconstruction_mse),
    ]:
        valid = np.isfinite(values)
        result[name] = float(np.asarray(values)[valid].mean()) if valid.any() else None
        result[name + "_valid_queries"] = int(valid.sum())
    np.savez(
        out / "test_scores.npz",
        embeddings=np.concatenate(embeddings),
        classification=classification,
        retrieval=retrieval,
        labels=labels,
        encoding_r=encoding_r,
        reconstruction_r=reconstruction_r,
        encoding_mse=mse,
        reconstruction_mse=reconstruction_mse,
    )
    write_json(out / "test_results.json", result)
    return result


def make_parser(modality):
    _, _, _, batch, validation, seed = SPECS[modality]
    p = argparse.ArgumentParser(
        description=f"BrainAE v0.2: original {modality.upper()} architecture and four-loss training"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Prepared sub-XX directory containing train/test.npy and IDs",
    )
    p.add_argument("--train-features", type=Path, help="Default: train_features.npz or .npy in --data-dir")
    p.add_argument("--test-features", type=Path, help="Default: test_features.npz or .npy in --data-dir")
    p.add_argument(
        "--prototypes",
        type=Path,
        help="Default: prototypes.npz or .npy in --data-dir; rows match test IDs",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Default: a new timestamped outputs/<modality>/sub-XX run directory",
    )
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--epochs", "--epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=batch)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--validation-size", type=int, default=validation)
    p.add_argument("--seed", type=int, default=seed)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA if available, otherwise CPU), cpu, or cuda:N",
    )
    p.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use all externally visible GPUs; BatchNorm uses local device batches",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--prototype-metric", choices=["original-dot", "cosine"], default="original-dot"
    )
    p.add_argument(
        "--assume-feature-order",
        action="store_true",
        help="Acknowledge row alignment for native NPY feature files",
    )
    p.add_argument(
        "--exclude-train-indices",
        type=Path,
        help="Zero-based rows removed before train/validation splitting",
    )
    p.add_argument(
        "--checkpoint", "--evaluate-only",
        dest="evaluate_only",
        type=Path,
        metavar="CHECKPOINT",
        help="Strictly load a release best.pt and evaluate",
    )
    p.add_argument("--check-data", action="store_true", help="Validate inputs and split without training, GPU allocation, or output creation")
    return p


def resolve_inputs(args, parser, training):
    """Fail before model allocation; deterministic filenames avoid hidden choices."""
    args.data_dir = args.data_dir.expanduser().resolve()
    for key in (["train_features"] if training else []) + ["test_features", "prototypes"]:
        path = getattr(args, key)
        if path is None:
            candidates = [args.data_dir / (key + ext) for ext in (".npz", ".npy")]
            existing = [p for p in candidates if p.is_file()]
            if len(existing) != 1:
                parser.error(f"Provide --{key.replace('_', '-')} or exactly one of " + ", ".join(str(p) for p in candidates))
            path = existing[0]
        path = path.expanduser().resolve()
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
        setattr(args, key, path)
    if args.evaluate_only:
        args.evaluate_only = args.evaluate_only.expanduser().resolve()
        if not args.evaluate_only.is_file():
            parser.error(f"Checkpoint not found: {args.evaluate_only}")
    if args.output_dir is None:
        stage = "evaluate" if args.evaluate_only else "train"
        args.output_dir = Path("outputs") / parser.modality / f"sub-{args.subject:02d}" / (stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.check_data and args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}. Choose a new --output-dir.")


def run(modality, Encoder, Decoder, initializer, argv=None, mode=None):
    parser = make_parser(modality)
    parser.modality = modality
    if mode:
        parser.prog = f"run.py {modality} {mode}"
        parser.description = f"BrainAE {modality.upper()}: {mode} prepared recording inputs"
        for action in parser._actions:
            if mode == "evaluate" and action.dest in {
                "train_features", "epochs", "batch_size", "validation_size", "lr", "exclude_train_indices"
            }:
                action.help = argparse.SUPPRESS
            if mode == "evaluate" and action.dest == "data_dir":
                action.help = "Prepared directory containing test.npy and test_ids.npy"
    args = parser.parse_args(argv)
    if mode == "evaluate" and args.evaluate_only is None:
        parser.error("evaluate requires --checkpoint PATH")
    if mode == "train" and args.evaluate_only:
        parser.error("Use the evaluate command with --checkpoint")
    if mode == "check":
        args.check_data = True
    training = args.evaluate_only is None
    resolve_inputs(args, parser, training)
    if (
        min(
            args.epochs,
            args.batch_size,
            args.eval_batch_size,
            args.validation_size,
            args.threads,
        )
        < 1
        or args.num_workers < 0
        or args.lr <= 0
    ):
        raise ValueError(
            "Epochs, batches, threads, validation size and learning rate must be positive"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    # Data checking stays CPU-only even on machines with active CUDA jobs.
    print("Checking inputs (full neural-array scan)...", file=sys.stderr, flush=True)
    data = load_signals(args.data_dir, modality, ("train", "test") if training else ("test",))
    features = feature_array(args.train_features, data["train_ids"], args.assume_feature_order) if training else None
    test_features = feature_array(args.test_features, data["test_ids"], args.assume_feature_order)
    # Prototype NPY order remains explicit in the original interface; NPZ IDs
    # are checked, and conventional NPY visual features still require opt-in.
    prototypes = feature_array(args.prototypes, data["test_ids"], True)
    if training:
        keep = np.arange(len(features))
        exclude = args.exclude_train_indices
        if modality == "mua" and exclude is None:
            exclude = (
                Path(__file__).resolve().parents[1]
                / "assets/mua_test_concept_train_indices.npy"
            )
            if len(keep) != 22248:
                raise ValueError(
                    "For a non-native MUA corpus supply --exclude-train-indices explicitly (an empty array is allowed)"
                )
        if exclude is not None:
            removed = np.load(exclude, allow_pickle=False)
            if (
                removed.ndim != 1
                or not np.issubdtype(removed.dtype, np.integer)
                or np.any((removed < 0) | (removed >= len(keep)))
            ):
                raise ValueError(
                    "Excluded rows must be a one-dimensional integer array within the original training range"
                )
            keep = np.setdiff1d(keep, removed)
        order = np.random.permutation(keep)
        if len(order) <= args.validation_size:
            raise ValueError("Validation size must leave at least one optimization example")
        val_rows, train_rows = order[: args.validation_size], order[args.validation_size :]
    else:
        train_rows = val_rows = np.array([], dtype=int)
    if args.check_data:
        print(json.dumps({"modality": modality, "mode": "training" if training else "evaluation",
                          "input_shapes": {k: list(data[k].shape) for k in ("train", "test") if k in data},
                          "optimization_images": len(train_rows), "validation_images": len(val_rows),
                          "test_queries": len(test_features), "gallery_size": len(prototypes),
                          "status": "Inputs validated; no model training or output files created"}, indent=2))
        return
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if args.data_parallel and (device.type != "cuda" or device.index not in (None, 0)):
        parser.error("--data-parallel requires --device cuda:0 (or auto selecting it)")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable; use --device cpu or install a compatible PyTorch build")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"BrainAE {modality.upper()} | {'training' if training else 'evaluation'} | device={device}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        modality=modality,
        resolved_device=str(device),
        mode="training" if training else "evaluation",
        losses=["contrastive", "feature_mse", "neural_mse", "image_neural_mse"],
        loss_weights=[1, 1, 1, 1],
        temperature=0.07,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        gallery_order="test_ids.npy",
        feature_order_checked=not args.assume_feature_order,
    )
    write_json(args.output_dir / "config.json", config)
    if training:
        write_json(
            args.output_dir / "split.json",
            {
                "train_rows": train_rows.tolist(),
                "validation_rows": val_rows.tolist(),
                "excluded_rows": np.setdiff1d(np.arange(len(features)), keep).tolist(),
                "seed": args.seed,
            },
        )
    np.save(args.output_dir / "test_ids.npy", data["test_ids"])
    enc, dec = Encoder().to(device), Decoder().to(device)
    enc.apply(initializer)
    dec.apply(initializer)
    if args.data_parallel:
        if device.type != "cuda" or device.index not in (None, 0):
            raise ValueError("--data-parallel requires --device cuda:0")
        enc, dec = nn.DataParallel(enc), nn.DataParallel(dec)
    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=args.lr, betas=(0.5, 0.999)
    ) if training else None
    best_path = args.evaluate_only or args.output_dir / "best.pt"
    if not args.evaluate_only:
        train = DataLoader(
            Responses(data["train"], features, train_rows),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val = DataLoader(
            Responses(data["train"], features, val_rows),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        best, history = float("inf"), []
        names = ["contrastive", "feature_mse", "neural_mse", "image_neural_mse"]
        for number in range(1, args.epochs + 1):
            tl = epoch(enc, dec, train, device, optimizer)
            vl = epoch(enc, dec, val, device)
            row = {
                "epoch": number,
                "train_loss": float(tl.sum()),
                "validation_loss": float(vl.sum()),
            }
            row.update(
                {
                    f"{prefix}_{name}": float(v)
                    for prefix, values in [("train", tl), ("validation", vl)]
                    for name, v in zip(names, values)
                }
            )
            history.append(row)
            if row["validation_loss"] <= best:
                best = row["validation_loss"]
                state = {
                    "encoder": unwrap(enc).state_dict(),
                    "decoder": unwrap(dec).state_dict(),
                    "epoch": number,
                    "validation_loss": best,
                    "modality": modality,
                }
                temp = best_path.with_suffix(".tmp")
                torch.save(state, temp)
                temp.replace(best_path)
            with (args.output_dir / "history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(
                f"Epoch {number}/{args.epochs}: training={tl.sum():.6f}, validation={vl.sum():.6f}",
                flush=True,
            )
    saved = torch.load(best_path, map_location=device, weights_only=True)
    if saved["modality"] != modality:
        raise ValueError("Checkpoint modality does not match this entry point")
    unwrap(enc).load_state_dict(saved["encoder"], strict=True)
    unwrap(dec).load_state_dict(saved["decoder"], strict=True)
    result = evaluate(
        enc,
        dec,
        data["test"],
        test_features,
        prototypes,
        args.output_dir,
        device,
        args.eval_batch_size,
        args.prototype_metric,
    )
    result.update(
        selected_epoch=int(saved["epoch"]),
        selected_validation_loss=float(saved["validation_loss"]),
        strict_reload=True,
    )
    write_json(args.output_dir / "test_results.json", result)
    print(json.dumps(result, indent=2))
