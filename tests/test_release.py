"""Small CPU checks; no dataset downloads or GPU allocation."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainae.runner import four_losses, feature_array, SPECS
from preprocessing.meg import order_trials
from preprocessing.mua import normalize_trial


def model_module(modality):
    spec = importlib.util.spec_from_file_location(
        "model_" + modality, ROOT / f"BAE_v0.2_{modality}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("modality", ["eeg", "meg", "mua"])
def test_original_shapes_and_gradients(modality):
    torch.set_num_threads(2)
    m = model_module(modality)
    enc, dec = m.Enc_eeg(), m.Dec_eeg()
    c, t, *_ = SPECS[modality]
    x, y = torch.randn(2, 1, c, t), torch.randn(2, 1024)
    losses = four_losses(enc, dec, x, y)
    assert losses.shape == (4,) and torch.isfinite(losses).all()
    losses.sum().backward()
    assert enc.patch_embed.shallownet[0].weight.grad is not None
    assert dec.dec_eeg[0].weight.grad is not None
    enc.eval()
    dec.eval()
    with torch.no_grad():
        assert enc(x).shape == (2, 1024)
        assert dec(y).shape == x.shape


@pytest.mark.parametrize("modality", ["eeg", "meg", "mua"])
def test_cli_train_reload_evaluate(tmp_path, modality):
    c, t, *_ = SPECS[modality]
    data = tmp_path / modality
    data.mkdir()
    rng = np.random.default_rng(9)
    for split, n in [("train", 6), ("test", 5)]:
        np.save(
            data / f"{split}.npy", rng.standard_normal((n, 1, c, t)).astype("float32")
        )
        np.save(data / f"{split}_ids.npy", np.arange(1, n + 1))
        np.savez(
            data / f"{split}_features.npz",
            features=rng.standard_normal((n, 1024)).astype("float32"),
            image_ids=np.arange(1, n + 1),
        )
    np.save(data / "prototypes.npy", rng.standard_normal((5, 1024)).astype("float32"))
    np.save(data / "exclude.npy", np.array([], dtype="int64"))
    run = data / "run"
    cmd = [
        sys.executable,
        str(ROOT / f"BAE_v0.2_{modality}.py"),
        "--data-dir",
        str(data),
        "--train-features",
        str(data / "train_features.npz"),
        "--test-features",
        str(data / "test_features.npz"),
        "--prototypes",
        str(data / "prototypes.npy"),
        "--output-dir",
        str(run),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--eval-batch-size",
        "2",
        "--validation-size",
        "2",
        "--device",
        "cpu",
        "--threads",
        "2",
        "--exclude-train-indices",
        str(data / "exclude.npy"),
    ]
    check = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), modality, "check",
         "--data-dir", str(data), "--validation-size", "2",
         "--exclude-train-indices", str(data / "exclude.npy"),
         "--device", "cuda:999", "--output-dir", str(run)],
        capture_output=True, text=True, timeout=60,
    )
    assert check.returncode == 0, check.stderr
    assert not run.exists()
    assert json.loads(check.stdout)["optimization_images"] == 4
    if modality == "eeg":
        cmd = [sys.executable, str(ROOT / "run.py"), modality, "train"] + cmd[2:]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((run / "test_results.json").read_text())
    assert (
        metrics["strict_reload"]
        and metrics["selected_epoch"] == 1
        and metrics["queries"] == 5
    )
    old = np.load(run / "test_scores.npz")
    evaluation = data / "evaluation"
    # Evaluation needs no training signals, features, exclusions or split.
    for name in ["train.npy", "train_ids.npy", "train_features.npz", "exclude.npy"]:
        (data / name).unlink()
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), modality, "evaluate",
         "--data-dir", str(data), "--checkpoint", str(run / "best.pt"),
         "--output-dir", str(evaluation), "--device", "cpu", "--threads", "2",
         "--eval-batch-size", "2"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert not (evaluation / "split.json").exists()
    new = np.load(evaluation / "test_scores.npz")
    np.testing.assert_array_equal(old["classification"], new["classification"])


def test_identity_mismatch_rejected(tmp_path):
    p = tmp_path / "features.npz"
    np.savez(p, features=np.ones((2, 1024)), image_ids=[2, 1])
    with pytest.raises(ValueError, match="Image order"):
        feature_array(p, np.array([1, 2]), False)


def test_mua_preprocessing_matches_original_formula():
    x = np.random.default_rng(0).normal(size=(4, 300))
    for subtract in [False, True]:
        y = x - x[:, :100].mean(-1, keepdims=True) if subtract else x
        y = y[:, 100:]
        expected = (y - y.mean()) / y.std()
        np.testing.assert_allclose(
            normalize_trial(x, subtract_baseline=subtract), expected, atol=2e-7
        )
    with pytest.raises(ValueError, match="Constant"):
        normalize_trial(np.ones((4, 300)))


def test_meg_stimulus_mapping_rejects_duplicates():
    import pandas as pd

    table = pd.DataFrame(
        {
            "trial_type": ["exp"] * 12 + ["test"] * 4,
            "things_category_nr": [1] * 12 + [np.nan] * 4,
            "things_exemplar_nr": list(range(1, 13)) + [np.nan] * 4,
            "test_image_nr": [np.nan] * 12 + [2, 1, 2, 1],
        }
    )
    train, test = order_trials(
        np.zeros((16, 2, 281)), table, train_count=12, test_count=2, repeats=2
    )
    np.testing.assert_array_equal(train, np.arange(12))
    np.testing.assert_array_equal(test, [[13, 15], [12, 14]])
    table.loc[1, "things_exemplar_nr"] = 1
    with pytest.raises(ValueError, match="exactly once"):
        order_trials(
            np.zeros((16, 2, 281)), table, train_count=12, test_count=2, repeats=2
        )


def test_eeg_covariance_matches_workspace_method():
    from preprocessing.eeg import auto_covariance, whitening
    from sklearn.discriminant_analysis import _cov

    x = np.random.default_rng(0).normal(size=(5, 100))
    cov = auto_covariance(x)
    np.testing.assert_allclose(cov, _cov(x.T, shrinkage="auto"), atol=1e-12)
    w = whitening(cov)
    np.testing.assert_allclose(w @ cov @ w.T, np.eye(5), atol=1e-10)


def test_ambiguous_feature_paths_are_actionable(tmp_path):
    for suffix in ["npy", "npz"]:
        (tmp_path / ("train_features." + suffix)).touch()
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), "eeg", "check",
                             "--data-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "exactly one" in result.stderr and "--train-features" in result.stderr
