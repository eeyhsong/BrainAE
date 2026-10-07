"""Small plotting and inference helpers shared by the four figure notebooks."""

from pathlib import Path
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BLUE, RED, GRAY = "#2066a8", "#a00000", "#888888"
MODALITIES = ["eeg", "meg", "mua"]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(
            directory / f"{name}.{extension}", bbox_inches="tight", pad_inches=0.06
        )


def panel(ax, label):
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=11)


def pearson_rows(a, b, mask=None):
    values = []
    for i in range(len(a)):
        x, y = a[i].ravel(), b[i].ravel()
        if mask is not None:
            x, y = x[mask[i].ravel()], y[mask[i].ravel()]
        values.append(
            np.corrcoef(x, y)[0, 1]
            if len(x) > 1 and x.std() > 0 and y.std() > 0
            else np.nan
        )
    return np.asarray(values)


def load_model(checkpoint, modality, device="cpu"):
    spec = importlib.util.spec_from_file_location(
        "release_model", ROOT / f"BAE_v0.2_{modality}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state["modality"] != modality:
        raise ValueError("Checkpoint modality mismatch")
    enc, dec = module.Enc_eeg().to(device), module.Dec_eeg().to(device)
    enc.load_state_dict(state["encoder"], strict=True)
    dec.load_state_dict(state["decoder"], strict=True)
    return enc.eval(), dec.eval()


def responses(data_dir):
    x = np.load(Path(data_dir) / "test.npy", mmap_mode="r", allow_pickle=False)
    return np.stack([np.asarray(trial, dtype=np.float32).mean(0) for trial in x])


def infer(model, x, device="cpu", batch=16):
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            result.append(
                model(
                    torch.as_tensor(
                        x[start : start + batch], dtype=torch.float32, device=device
                    )
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(result)


def accuracy(embeddings, prototypes, metric="original-dot"):
    z = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    p = prototypes
    if metric == "cosine":
        p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    elif metric != "original-dot":
        raise ValueError(metric)
    return np.mean(np.argmax(z @ p.T, axis=1) == np.arange(len(z)))


def curve(ax, x, subject_values, label, color=BLUE, marker="o"):
    values = np.asarray(subject_values, float)
    mean = np.nanmean(values, axis=0)
    ax.plot(x, mean, color=color, marker=marker, ms=3, label=label)
    if len(values) > 1:
        sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.isfinite(values).sum(0))
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.14, linewidth=0)
