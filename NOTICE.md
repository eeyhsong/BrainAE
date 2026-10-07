# Sources and acknowledgements

The three `BAE_v0.2_*.py` model definitions come from the author's original
`bae_v0.2_base.py`, `bae_v0.2_meg_base.py`, and `bae_v0.2_spike_base.py`.
`SOURCE_MANIFEST.json` records their relative filenames and SHA-256 hashes.
Their encoder, projection, decoder and weight-initialization definitions are
preserved; the shared training/I/O implementation is cleaned up for this release.
The original files describe a NICE-style shallow neural encoder. Cite the
original BrainAE manuscript and the upstream methods and datasets used in your
work; this release does not claim that the shallow encoder design is novel.

EEG raw preprocessing is adapted from this project's `rebuttal_cbg/train_r1q2.py`,
which implements session-wise training-fitted normalization. It is not a claim
that the unavailable historical EEG cache generator has been recovered.
MEG trial ordering follows the project's `Things-MEG1/preprocessing/get_data.py`.
MUA ordering, cropping and z-scoring follow `TVSD/preprocessing/preprocessing.py`.
The optional CTF epoching utility implements the filtering/epoching/baseline
steps described in the local THINGS-MEG workflow. The upstream workflow credits
Lina Teichmann. Dataset-specific artifact repairs and photodiode corrections
must be applied and documented by the user; they are not guessed by the utility.
No upstream raw-data script is copied into this distribution.

The four release notebooks adapt the analysis structure of the author's
`draw_pic/fig2.ipynb`, `draw_pic/fig3.ipynb`, `draw_pic/fig4.ipynb` and
`draw_pic/signal_recon.ipynb`. Private paths, embedded outputs and hard-coded
performance values are removed.

Dataset reference resources:

- THINGS-EEG / THINGS-EEG2: https://github.com/gifale95/eeg_encoding
- THINGS-MEG: https://openneuro.org/datasets/ds004212
- THINGS Ventral Stream Dataset (TVSD): https://gin.g-node.org/paolo_papale/TVSD

Data, stimulus images, visual-backbone weights and third-party dependencies are
not included and retain their own licenses. `LICENCE` covers the BrainAE release
code supplied here; it does not relicense those external resources. MIT is the
default prepared license for this local release folder. Confirm the copyright
holder and licensing choice before public publication.

The release contains no raw neural recordings, trained weights, manuscript
reviews, personal paths, account credentials or private experiment logs.
