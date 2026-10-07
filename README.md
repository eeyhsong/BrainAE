# BrainAE v0.2

Core preprocessing, training and evaluation code for EEG, MEG and multi-unit
activity (MUA). The three entry files preserve the author's original encoder,
projection and decoder definitions and the four-loss objective. Rebuttal
experiments, data and trained weights are not included. Four focused notebooks
provide plotting code for the main manuscript analyses.

## Contents

| File | Purpose |
|---|---|
| `BAE_v0.2_eeg.py` | EEG model and training entry point |
| `BAE_v0.2_meg.py` | MEG model and training entry point |
| `BAE_v0.2_mua.py` | MUA model and training entry point |
| `run.py` | Common `check`, `train` and `evaluate` commands for every modality |
| `brainae/runner.py` | Shared data loading, four-loss training and evaluation |
| `preprocessing/eeg.py` | EEG epoching, baseline correction, resampling and MVNN |
| `preprocessing/meg.py` | MEG epoch preparation and stimulus ordering |
| `preprocessing/mua.py` | MUA stimulus ordering, cropping and normalization |
| `preprocessing/common.py` | Conversion of original preprocessed caches |
| `brainae/prepare_inputs.py` | Split, feature-order and concept-exclusion utilities |

## Installation

Use Python 3.10 or newer and a PyTorch installation compatible with your hardware.
Run the following commands from this folder:

```bash
pip install -r requirements.txt
# Additional dependencies for raw-data preprocessing:
pip install -r requirements-preprocessing.txt
```

Obtain neural data and corresponding image features separately. Dataset links
and source acknowledgements are in [NOTICE.md](NOTICE.md).

## Quick start: check → train → evaluate

First prepare the neural arrays using the instructions below. Place (or link)
the corresponding feature files alongside them with these names:

```text
data/prepared/eeg/sub-01/
├── train.npy
├── train_ids.npy
├── test.npy
├── test_ids.npy
├── train_features.npz
├── test_features.npz
└── prototypes.npz
```

Each feature/prototype NPZ contains `features` and `image_ids`, in neural row
order. Existing NPY files also work: name them `train_features.npy`,
`test_features.npy` and `prototypes.npy`, and add `--assume-feature-order` to
acknowledge the visual-feature ordering. Prototype NPY ordering is the user's
responsibility; NPZ prototype IDs are checked automatically. If both NPY and
NPZ versions exist, specify the intended file explicitly.

```bash
# Check dimensions, image IDs, finite values and the train/validation split.
# This performs a full array scan, which can take time on large datasets.
# No GPU allocation, training, or output directory is created.
python run.py eeg check --data-dir data/prepared/eeg/sub-01

# Original per-recording training; also evaluates the selected checkpoint.
python run.py eeg train --data-dir data/prepared/eeg/sub-01 \
  --output-dir outputs/eeg/sub-01/run-01

# Evaluate without loading any training data or constructing a new split.
python run.py eeg evaluate --data-dir data/prepared/eeg/sub-01 \
  --checkpoint outputs/eeg/sub-01/run-01/best.pt \
  --output-dir outputs/eeg/sub-01/evaluation-01
```

Replace `eeg` with `meg` or `mua` and select that recording's prepared directory.
`--device auto` chooses CUDA when available and CPU otherwise; use `--device cpu`
or `--device cuda:1` to select explicitly. Omit `--output-dir` to create a new
timestamped run under `outputs/<modality>/sub-XX/`; the resolved device and output
path are printed. Use `--subject` to set the identifier used in that default path.
Only new output directories are accepted.

Evaluation needs just `test.npy`, `test_ids.npy`, test features, prototypes and
a release checkpoint. It does not need training arrays, training features,
exclusion files, or the original validation size. To check evaluation inputs
without running inference, use `check --checkpoint PATH`. This validates input
files and checkpoint existence; strict weight loading occurs during evaluation.

Use `python run.py --help` or `python run.py meg train --help` for options.
Explicit `--train-features`, `--test-features` and `--prototypes` paths override
discovery, so existing directories need not be reorganized. The original
`BAE_v0.2_eeg.py`, `BAE_v0.2_meg.py` and `BAE_v0.2_mua.py` commands remain supported.

## Prepare neural data

The training scripts read `train.npy`, `test.npy`, `train_ids.npy` and
`test_ids.npy` from a prepared directory. Neural arrays are float32 with shape
`[images, repetitions, channels, samples]`; repetitions are averaged during
training and evaluation. IDs are one-based, split-local image-row identifiers,
not semantic category labels.

| Modality | Channels | Samples | Training images × repetitions | Test images × repetitions |
|---|---:|---:|---:|---:|
| EEG | 63 | 250 | 16,540 × 4 | 200 × 80 |
| MEG | 271 | 200 | 22,248 × 1 | 200 × 12 |
| MUA | 1,024 | 200 | 22,248 × 1 | 100 × 30 |

**Existing caches.** This is the simplest route when using the original
preprocessed data. Replace the example filenames with your local files:

```bash
python -m preprocessing.common --modality eeg \
  --train data/original/eeg_training.npy --test data/original/eeg_test.npy \
  --output data/prepared/eeg/sub-01 --trust-pickle
```

Use `--modality meg` or `mua` for the other datasets. The converter preserves
normalization and channel order; MEG caches containing 201 samples lose their
first sample, as in the original loader. `--trust-pickle` is required only for
trusted dictionary/pickle caches. Prefer MUA caches without baseline correction.

**EEG raw sessions.** Supply the original ordered list of 63 channel names,
one per line. The raw directory must contain
`sub-01/ses-01/raw_eeg_training.npy` and `raw_eeg_test.npy`, through session 4.

```bash
python -m brainae.prepare_inputs split --count 16540 --validation-size 740 \
  --seed 2023 --output data/eeg_fit_rows.npy
python -m preprocessing.eeg --raw-root data/eeg_raw --subject 1 \
  --channels data/eeg_channels.txt --fit-indices data/eeg_fit_rows.npy \
  --output data/prepared/eeg/sub-01 --trust-pickle
```

EEG processing uses a −0.2–0 s baseline, 250 Hz resampling, the [0,1) s response,
and session-wise multivariate noise normalization (MVNN). Keep the split seed,
validation size and exclusions consistent with training. Without `--fit-indices`,
MVNN uses all nominal training images, including downstream validation images.
This implementation comes from the current project pipeline; it is not an
exact recovery of the historical EEG cache generator.

**MEG epochs.** Prepare the original preprocessed CoSMo HDF5 file and its
trial metadata CSV:

```bash
python -m preprocessing.meg prepare --input data/meg_epochs.mat \
  --labels data/meg_trials.csv --output data/prepared/meg/sub-01
```

The HDF5 dataset `ds/samples` must have shape `[281*271, trials]`. Metadata must
contain `trial_type` (`exp` or `test`), `things_category_nr`,
`things_exemplar_nr` and `test_image_nr`. Training images are ordered by
category then exemplar; test repetitions by image ID. The retained interval
is 0.005–1.000 s at 200 Hz. Alternatively use `--format fif` with combined,
preprocessed MNE epochs and matching metadata. An optional `epoch-ctf`
subcommand handles a single raw run with explicitly corrected event onsets;
dataset-specific artifact repairs and combining runs remain upstream steps.

**MUA raw trials.** Process each monkey independently:

```bash
python -m preprocessing.mua --input data/monkey_1/THINGS_MUA_trials.mat \
  --baseline none --output data/prepared/mua/sub-01
```

The original HDF5 layout is `ALLMUA[300, trials, 1024]`, with training/test
image IDs in rows 1/2 of `ALLMAT` (zero-based indexing). Processing retains
samples 100:300 and z-scores each trial across channels and time. The default
omits baseline subtraction; `--baseline subtract` reproduces the original
baseline-corrected variant. Use a separate prepared directory and model run
for each monkey; channel correspondence is not assumed.

## Train and evaluate

Supply visual features with shape `[images,1024]` and test-concept prototypes
with shape `[test images,1024]`. Features must use the original backbone and
preprocessing, and exactly match neural image order. This release does not
extract visual features. Native feature NPY files require an explicit
`--assume-feature-order` acknowledgement. Alternatively, supply NPZ files with
`features` and `image_ids`; IDs are checked against the prepared neural data.
Prototype NPY rows must follow `test_ids.npy` order.

```bash
python BAE_v0.2_eeg.py --data-dir data/prepared/eeg/sub-01 \
  --train-features data/features/eeg_train.npy \
  --test-features data/features/eeg_test.npy \
  --prototypes data/features/eeg_center.npy \
  --assume-feature-order --subject 1 --device cuda:0 \
  --output-dir outputs/eeg/sub-01
```

For MEG or MUA, use `BAE_v0.2_meg.py` or `BAE_v0.2_mua.py` and the corresponding
prepared data, features and prototypes. Run each participant/animal separately.
The `--subject` option records the identifier; `--data-dir` selects the data.

| Default | EEG | MEG | MUA |
|---|---:|---:|---:|
| Epochs | 50 | 50 | 50 |
| Batch size | 800 | 600 | 100 |
| Validation images | 740 | 748 | 748 |
| Seed | 2023 | 2023 | 2024 |

All modalities use Adam (learning rate 0.001; betas 0.5, 0.999) and equally
weighted symmetric contrastive, feature-MSE, neural-reconstruction and
image-to-neural reconstruction losses. Selection uses validation loss. The
default classification score preserves the original dot product with normalized
neural embeddings; `--prototype-metric cosine` also normalizes prototypes and
therefore changes the evaluation protocol.

MUA automatically excludes the supplied 1,200 original training-image indices
belonging to test concepts. The original MEG protocol includes training
exemplars of test concepts. For a different concept-exclusion protocol, supply
`--exclude-train-indices` (zero-based NPY rows before splitting).

Each training output directory contains `best.pt`, `history.csv`, `config.json`,
`split.json`, `test_results.json` and `test_scores.npz`. Accuracy values are
fractions. Encoding/reconstruction correlations are computed over flattened
channels and time per averaged test response, then averaged across valid
queries. Existing output directories are never overwritten. Evaluation-only runs contain
configuration, test IDs, scores and metrics, without a newly generated training split.
The local `config.json` records resolved input paths for reproducibility; review
these machine-specific paths before sharing your own run outputs.

To evaluate a saved release checkpoint, use the short `run.py ... evaluate`
command above. The original entry files also accept `--evaluate-only CHECKPOINT`
with only test inputs and a new output directory. Use `--device cpu`
for CPU execution. GPUs are selected automatically, externally or with `--device`;
`--data-parallel` is optional and changes BatchNorm microbatch behavior.

## Release changes and checks

Model layers and the four losses are retained. Private paths and fixed GPU
assignments are removed. Checkpoints now reload strictly, output names are
isolated per run, and input shape/order checks fail explicitly. Seeds are
specified per run rather than drawn from the original multi-subject loop;
identical manuscript scores are not guaranteed by this cleanup.

```bash
pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

CPU checks cover all three architectures, gradients, short synthetic training
runs, exact checkpoint reload, GPU-free input checks, test-only evaluation and
actionable errors for ambiguous feature filenames. All four figure notebooks were also executed
with synthetic inputs; optional scalp maps require real channel metadata.
Synthetic checks cover EEG epoching and MEG/MUA HDF5 ordering.
Full raw-dataset preprocessing and manuscript
benchmarks have not been rerun as part of release preparation.

## Manuscript figure notebooks

```bash
pip install -r requirements-figures.txt
jupyter lab notebooks
```

Edit the configuration cell in each notebook to point to your data and release
checkpoints. Outputs are PDF, SVG and PNG panels using the manuscript's blue
(`#2066a8`) and red (`#a00000`) palette. Notebooks contain no saved outputs or
hard-coded experimental results.

| Notebook | Panels |
|---|---|
| `01_encoding.ipynb` | Waveforms, spectra, EEG band-power maps, sensor-correlation and phase-lag matrices |
| `02_decoding.ipynb` | Classification/retrieval bars with individual recordings; representation similarity |
| `03_signal_reconstruction.ipynb` | Masked/reconstructed waveforms and masking-performance curves |
| `04_spatial_temporal.ipynb` | Temporal windows, regional occlusion and encoding–decoding associations |

These are portable versions of the analysis panels, not an automatic rebuild of
historical composite figures. Provide verified channel groups for regional
plots. Reconstruction uses the checkpoint you supply: a clean-trained checkpoint
is not a mask-trained model. Plotting inputs and computed metrics are explicit;
no manuscript scores are substituted. Image synthesis requires its separate
external generation pipeline and is outside this core release.

## License

See [LICENCE](LICENCE) (MIT). External datasets, image features and dependencies
retain their own licenses. Source provenance is recorded in
`SOURCE_MANIFEST.json` and [NOTICE.md](NOTICE.md).

## Common setup problems

- **Feature order mismatch:** align rows with `train_ids.npy` / `test_ids.npy`;
  do not change IDs merely to suppress the check. These split-local row IDs are
  only useful if assigned to the corresponding images correctly.
- **Out of GPU memory:** the original logical batches can require substantial
  memory. `--batch-size` can be reduced, but this changes the contrastive
  objective and BatchNorm behavior; document that setting when comparing scores.
  `--eval-batch-size` controls test inference separately.
- **Small/custom dataset:** select a smaller `--validation-size`. For non-native
  MUA data, provide your own `--exclude-train-indices`; the bundled exclusion
  list applies only to the original image ordering.
- **Existing output directory:** choose another `--output-dir` or omit it to use
  a timestamped directory. This release does not implement resume-training.
