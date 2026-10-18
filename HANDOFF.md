# MemoCMT MELD Reproduction — Handoff Notes

For a Claude instance picking this up on a new machine with no prior context.
Written 2026-10-19 from the machine that did all the work described below
(Windows, AMD RX 9060 XT, no CUDA/ROCm — DirectML only).

## 1. What this project is

Reproducing Table 5 (MELD dataset, CMT+MIN fusion) from the paper
*"MemoCMT: multimodal emotion recognition using cross-modal transformer-based
feature fusion"* (Khan, Tran, Pham, El Saddik, Othmani — *Scientific Reports*
2025). Paper's own code: https://github.com/tpnam0901/MemoCMT (this repo's
`upstream` remote). This repo (`origin`, `kylewkchuan/MemoCMT`) is a forked
working copy with bug fixes applied — **the upstream repo's MELD path is
non-functional as published; see §3.**

Paper's reported MELD numbers (CMT+MIN, Table 5):

| | Acc | F1 | Prec | Rec |
|---|---|---|---|---|
| Validation | 60.38% | 57.99% | 58.29% | 60.38% |
| Test | 64.18% | 62.52% | 63.82% | 64.18% |

## 2. What's been done so far (on the DirectML machine)

1. Set up conda env, installed `torch-directml` (this machine has no CUDA/ROCm
   GPU — an AMD card). Hit and fixed a `torch_directml` bug: on Python 3.8 it
   calls a `@staticmethod` as a default-arg value, which only works on
   Python 3.10+ (patched the installed package directly, not a repo fix).
2. Downloaded MELD (`MELD.Raw.tar.gz`, official HuggingFace mirror linked
   from https://affective-meld.github.io/), extracted it to `data/MELD.Raw/`.
3. Found and fixed the bugs in §3, got `preprocess.py` + `train.py` +
   `eval.py` actually running end-to-end for MELD for the first time.
4. Ran two 15-epoch training demos with `src/configs/hubert_base_meld_short*.py`
   (CMT+MIN, batch_size=1, seed=0 — same hyperparameters the paper states),
   differing only in how stereo audio is downmixed to mono (`audio_channel_mode`
   in config, consumed by `dataloader.py`'s `__paudio__`):
   - **mean_downmix**: average both channels (the sensible default)
   - **left_channel**: keep only the left channel (ablation, to test whether
     channel-handling — which the paper's code never specifies, since the
     public repo never even handles stereo — explains the gap vs. the paper)
5. Verified the local MELD copy's utterance/label counts match the official
   published statistics exactly (`scripts/dataset_statistics.py`).

### Results obtained (DirectML, RTX 9060 XT)

| | Val Acc | Val F1 | Val Prec | Test Acc | Test F1 | Test Prec |
|---|---|---|---|---|---|---|
| Paper | 60.38% | 57.99% | 58.29% | 64.18% | 62.52% | 63.82% |
| mean_downmix | 68.08% | 64.85% | 66.28% | 72.08% | 69.87% | 70.21% |
| left_channel | 68.08% | 64.45% | 66.74% | 72.22% | 69.66% | 70.86% |

Both runs **beat the paper by ~7-8 points on every metric, on both splits**,
and the two runs agree with each other within ~1 point (val accuracy is
identical to 4 sig figs). **Conclusion: the stereo-downmix choice is not a
meaningful driver of the gap vs. the paper.**

Training curves on both runs show the same classic overfitting shape: train
accuracy climbs to ~87% by epoch 15 while validation loss bottoms out around
epoch 4-5 then rises. We evaluate from the `save_best_val`-selected
best-validation-accuracy checkpoint (epoch 4 in both runs), not the final
epoch, which is standard practice and matches what the paper describes doing
("stopped after 100 cycles... when it performs best on a separate validation
dataset").

### Hypotheses for the ~7-8 point gap vs. the paper — status

| Hypothesis | Status |
|---|---|
| Wrong/corrupted MELD copy | **Ruled out** — utterance/label counts match official stats exactly |
| torchaudio resample algorithm changed between versions (we're on 2.4.1, paper's requirements.txt says 2.0.2) | **Ruled out** — compared `_get_sinc_resample_kernel` source between torchaudio v2.0.1 and v2.4.0 on GitHub; identical formulas and defaults |
| Our own bug fixes (§3) changing results | **Ruled out** — all are crash-only fixes (NameError, NotImplementedError, AttributeError); none change what gets computed once they're fixed |
| Stereo→mono downmix method (mean vs. left-channel-only) | **Ruled out** — ablation above, <1pt difference between the two |
| **Hardware/backend RNG divergence (DirectML vs. CUDA)** | **UNTESTED — this is why the project is moving to this machine.** `torch.manual_seed(0)` does not guarantee identical results across backends; `batch_size=1` training is itself unstable/seed-sensitive (the paper states this explicitly) |
| Paper's MELD numbers are a k-fold average vs. our single run | Plausible, unverifiable — paper has a possibly-contradictory statement about k=5 CV "across all datasets" vs. a separate statement that MELD specifically used the fixed train/dev/test split |
| Paper's actual (undisclosed) MELD preprocessing differs from what's in the public repo | Likely true at some level, given the repo's MELD path doesn't run at all as published — can't verify further without the authors' original code |
| Paper used a different audio *source* than re-extracting from `MELD.Raw.tar.gz`'s mp4 clips (e.g. MELD's other download, `MELD.Features.Models.tar.gz`) | **Ruled out** — that archive holds precomputed openSMILE feature vectors (1611/1422-dim) and GloVe/bcLSTM outputs for the *original* MELD paper's (Poria et al. 2019) own baseline models, not raw audio. HuBERT needs raw waveform input, not hand-engineered feature vectors, so this archive isn't a viable alternative source for this architecture regardless. Audio must come from the mp4 clips either way — only the *extraction tool/version* could plausibly differ, same small-magnitude category as the already-ruled-out torchaudio version question |

### What running on this (CUDA) machine should establish

Run the **exact same config** (`src/configs/hubert_base_meld_short.py`,
`audio_channel_mode="mean"`, seed=0, 15 epochs — or more if you want to let
it run further) and compare:
- **If this CUDA run lands close to the paper's numbers** (not ~68/72% but
  closer to ~60/64%): hardware/backend RNG divergence is confirmed as a real
  driver.
- **If this CUDA run also beats the paper by a similar ~7-8 points**:
  hardware isn't it either — the remaining candidates are the k-fold
  question and/or an undisclosed difference in the paper's actual MELD
  preprocessing, neither of which can be fully resolved from outside.

Either outcome is a useful, reportable result — update this file (or a new
one) with what you find.

## 3. Bugs found and fixed in this fork (vs. `upstream`)

All in the commit `Fix MELD pipeline bugs, add DirectML support, add MELD
ablation configs` — look at that commit for exact diffs.

- **`scripts/preprocess.py`**: `LABEL_MAP` was reassigned inside an
  `if args.all_classes:` block. Because Python decides a name is local to a
  function the moment it's assigned *anywhere* in that function body — even
  inside a branch that never executes — every reference to `LABEL_MAP` inside
  the nested `_preprocess_data()` closure raised `NameError` on **every
  single row**, for every split, whenever `--all_classes` was *not* passed
  (i.e. always, in the default/documented usage). Preprocessing MELD with
  the unmodified repo silently produces `train.pkl`/`val.pkl`/`test.pkl`
  with 0 samples and exits 0 — no crash, no warning read carefully enough,
  just empty data. Fixed by not shadowing the module-level name.
- **`src/data/dataloader.py`**: `DATASET_MAP` in `build_train_test_dataset()`
  only registered `"IEMOCAP"` and `"ESD"` — `"MELD"` was missing, so training
  raised `NotImplementedError` immediately.
- **`src/data/dataloader.py`**: `__paudio__()` never downmixed multi-channel
  audio. IEMOCAP/ESD are mono at the source so this was never hit before;
  MELD's clips (extracted from TV video) are stereo, and HuBERT's first
  conv1d layer is sized for mono input — feeding it stereo crashes with a
  confusing `RuntimeError` about a kernel/input-size mismatch. Added
  `audio_channel_mode` (`mean`/`left`/`right`, default `mean`) to
  `configs/base.py` and the downmix logic in `__paudio__()`.
- **`src/data/dataloader.py`**: `torchaudio.functional.resample()` was
  called on a raw numpy array. Works on old torchaudio; raises
  `AttributeError: 'numpy.ndarray' object has no attribute 'device'` on
  current versions, which expect a tensor. Fixed by converting to a tensor
  first.
- **`src/torchvggish/torchvggish.py`** + **`dataloader.py`**: an unused
  `from torchvggish.vggish_input import waveform_to_examples` import pulled
  in `numba`/`resampy` transitively even though this repo's actual audio
  path (`audio_encoder_type="hubert_base"`) never touches VGGish at all —
  `modules.py`'s `build_audio_encoder()` only registers `"hubert_base"`.
  Removed the dead import. (On the Windows dev machine this import also
  crashed outright under an application-control policy blocking numba's
  native DLL — may not reproduce on other machines, but the import was dead
  code either way.)
- **`scripts/eval.py`** / **`scripts/auc_roc.py`**: wrapped the per-sample
  eval loop in try/except so one oversized-audio sample causing a GPU OOM
  (attention is O(n²) in sequence length; `batch_size=1` + no length cap
  means outlier-length clips can blow up memory) skips and logs instead of
  crashing the whole eval run. Also fixed `auc_roc.py`'s `dataFrame["y_true"]
  == cls` bug (comparing a Python list to a scalar is always `False` — needed
  `np.array(...)`), and generalized its hardcoded author-specific checkpoint
  paths/label lists into a normal `-ckpt` CLI arg.

## 4. Repo structure — what's in git vs. what you need to (re)generate

```
MemoCMT/
  src/               tracked normally (all .py)
  scripts/*.py        tracked (scripts/.gitignore: "* / !.gitignore / !*.py / !*.ps1")
  scripts/*.ps1        tracked (Windows setup script)
  data/                NOT tracked (root .gitignore) — ~36GB, copyrighted
                        TV clips, re-download yourself, see §5
  scripts/MELD_preprocessed/   NOT tracked (scripts/.gitignore blanket rule)
                        — regenerate via preprocess.py, see §5. Also: the
                        .pkl files store ABSOLUTE file paths baked in at
                        preprocess time, so they would not be portable
                        between machines even if you copied them.
  scripts/working/      NOT tracked (checkpoints/ in root .gitignore)
                        — training output, multi-GB per run, regenerate by
                        training
  scripts/results/      NOT tracked (scripts/.gitignore blanket rule) — the
                        DirectML-machine's output artifacts (charts, t-SNE,
                        confusion matrices) exist only on that machine, not
                        in this repo. Ask the user if they want them if you
                        need the exact numbers beyond what's in §2's tables.
```

## 5. Setting up on this machine

A one-shot Windows+NVIDIA setup script already exists:
`scripts/setup_windows.ps1` (installs Miniconda, a `python=3.10` env named
`MemoCMT-cuda`, `torch==2.7.1+cu118`, and `requirements.txt`). Check whether
it already ran (look for `%USERPROFILE%\miniconda3\envs\MemoCMT-cuda`) before
re-running — it's idempotent but there's no need to repeat finished steps.

No Linux-equivalent setup script exists yet — only portability groundwork
(moviepy pinned to `1.0.3` since `preprocess.py` uses the removed
`moviepy.editor` namespace in 2.x; `dataset_statistics.py`'s `DATA_ROOT`
reads a `MELD_ROOT` env var instead of a hardcoded Windows path). If this
machine is Linux, you'll need to adapt the Windows script's logic (conda env
creation, pip installs) yourself — same package versions should work.

**Get the dataset** (same process used on the DirectML machine):
```
curl -L -o MELD.Raw.tar.gz https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
tar -xzf MELD.Raw.tar.gz -C data/
tar -xzf data/MELD.Raw/train.tar.gz -C data/MELD.Raw/
tar -xzf data/MELD.Raw/dev.tar.gz -C data/MELD.Raw/
tar -xzf data/MELD.Raw/test.tar.gz -C data/MELD.Raw/
```
Expect the download to drop/need resuming on a flaky connection — use
`curl -C -` to resume; it happened multiple times on the DirectML machine.

**Preprocess:**
```
cd scripts
python preprocess.py -ds MELD -dr <path to data/MELD.Raw>
```
Expect ~8,244 / 896 / 2,211 valid samples (train/val/test) after the 7→4
class filter (anger/joy/neutral/sadness kept; disgust/fear/surprise
dropped) — matches `dataset_statistics.py`'s output on the DirectML machine
exactly. Note: the `output_repeated_splits_test` folder contains roughly 2x
more `.mp4` files than `test_sent_emo.csv` has rows — this is a known MELD
release quirk (the folder name says "repeated"), not a bug; preprocess.py
correctly ignores the extras since it only looks up files the CSV
references.

**Train** (exact config used for the DirectML `mean_downmix` run, for direct
comparison):
```
python train.py -cfg ../src/configs/hubert_base_meld_short.py
```
This uses `audio_channel_mode="mean"`, `fusion_head_output_type="min"`,
`batch_size=1`, `num_epochs=15`, seed 0 (hardcoded in `train.py`). On
DirectML this took ~15-20 min/epoch (~3.3 hours total for 15 epochs); CUDA
should be substantially faster.

**Evaluate:**
```
python eval.py -ckpt <checkpoint_dir> -cm            # test set (default)
python eval.py -ckpt <checkpoint_dir> -t val.pkl -cm # val set
python auc_roc.py -ckpt <checkpoint_dir>
```
`<checkpoint_dir>` is the timestamped folder under
`working/checkpoints/MELD/MemoCMT_bert_hubert_base_short/`. Compare the
printed `BACC | ACC | MACRO_F1 | WEIGHTED_F1` line against §2's table —
`ACC` and `WEIGHTED_F1` map to the paper's `Acc`/`F1` columns; weighted
`Prec`/`Rec` aren't printed directly but can be derived from the printed
confusion matrix the same way the DirectML machine did (weighted recall
always equals accuracy in multi-class classification, so that's a free
sanity check).

## 5b. The 3 specific MELD samples that get dropped (cross-verified on 2 machines)

Two independent machines (this one + a separate RTX 3070 session) both ran
into exactly these 3 files. Root causes confirmed:

| Sample | Split | When it's dropped | Root cause |
|---|---|---|---|
| `dia125_utt3.mp4` | train | preprocessing | Corrupt MP4 container — ffmpeg: `moov atom not found` / `Invalid data found when processing input`. The file is truncated (moov atom, which holds track/codec/timing metadata, is missing — a known MELD-release data-quality issue, not something we caused). File exists, 1.9MB, not literally empty, but undecodable. |
| `dia110_utt7.mp4` | dev | preprocessing | **File does not exist at all** in the official `dev.tar.gz` release — confirmed missing on two independent downloads of the same official HuggingFace mirror, so this isn't a download/extraction error on either machine, it's absent from the archive itself. |
| `dia38_utt4.mp4` | test | eval-time (GPU OOM) | Mislabeled clip boundary: transcript is one short line ("Oh its great, its a role on") but the extracted clip is **304.97 seconds** long (vs. ~3.3-3.6s average) — the source timestamps/clip extraction for this utterance are wrong in the official release. Still present and valid in `test.pkl` (2,211 samples); only dropped *at evaluation time* by the OOM guard in `eval.py`/`auc_roc.py` (see §3), not during preprocessing. |

These 3 are expected and accounted for — don't spend time trying to "fix"
them; the counts in §2 already reflect them being dropped.

## 6. Known gotchas (from the DirectML machine, may or may not apply here)

- MLflow's `mlflow ui` on Windows needs `<conda env>\Scripts` on `PATH` to
  find `waitress-serve.exe`, or it fails with a garbled `WinError 2`.
- `mlflow==2.4.0` needs `setuptools<70` (pins `pkg_resources`, removed from
  newer setuptools) — already pinned in `requirements.txt`.
- A GPU OOM on one abnormally-long audio clip during eval is expected and
  handled (skip + log, see §3) — don't be alarmed by one `Skipped N
  sample(s)` warning.
