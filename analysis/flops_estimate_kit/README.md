# FLOPs Estimate Kit

This kit reproduces the selected SFT-to-RL scaling curves used by RETU.  It is
designed for public release: the expensive hyperparameter-search stage has been
removed, and every released script directly reuses the selected fitting
hyperparameters stored under `fit_res/`.

The kit works from cached FLOPs-performance dictionaries.  It does not parse raw
training parquet files or RL progress logs during public curve reproduction.

## What This Kit Does

Given an SFT checkpoint and its downstream RL run, the fitting pipeline:

1. Estimates the cumulative SFT FLOPs at the chosen checkpoint.
2. Loads the cached SFT performance curve from `cache/sft/`.
3. Loads the cached RL branch curve from `cache/sft_then_rl/`.
4. Aligns the SFT branch point with historical cache keys.
5. Converts the branch into relative RL EFLOPs versus performance gain.
6. Fits the selected scaling curve with fixed, released hyperparameters.
7. Saves the figure, fit metrics, and run log under `fit_result/`.

The public scripts reproduce the chosen scaling curves only.  They intentionally
do not run hyperparameter search.

## Directory Layout

```text
flops_estimate_kit/
├── cache/
│   ├── sft/                 # Cached SFT FLOPs -> validation performance maps.
│   └── sft_then_rl/         # Cached RL branch maps for each SFT scene/checkpoint.
├── config/
│   ├── model/               # Model constants used by the FLOPs estimator.
│   └── train/               # SFT and RL metadata used by the fitter.
├── fit_res/                 # Selected reference hyperparameters and figures.
├── fit_result/              # Generated figures, metrics, and logs.
├── run_sh/                  # One reproduction script per released curve.
├── utils/                   # Cache loading, FLOPs estimation, and metric helpers.
├── visualize/               # Curve fitting and plotting utilities.
├── curve_fit_prepare.py     # Builds one cached SFT->RL fitting dataset.
└── fitting_curves.py        # CLI entry point used by all run scripts.
```

## Supported Curves

The released scripts cover the following SFT scenes and branch checkpoints:

| Script folder | Cache scene | SFT checkpoints |
| --- | --- | --- |
| `run_sh/SFT889K/` | `general` | `0`, `360`, `720`, `1080`, `1440`, `1800`, `3600`, `5400`, `7200`, `9000`, `10800`, `12600`, `14080` |
| `run_sh/easy102K/` | `easy102K` | `0`, `360`, `720`, `1080`, `1440`, `1800` |
| `run_sh/hard102K/` | `hard102K` | `0`, `360`, `720`, `1080`, `1440`, `1800` |
| `run_sh/s1K/` | `s1K` | `0`, `62`, `124`, `186`, `248`, `310` |
| `run_sh/uniform102K/` | `uniform102K` | `0`, `360`, `720`, `1080`, `1440`, `1800` |

`general` is the cache name for the `SFT889K` setting.

## Quick Start

Run all released curve reproductions:

```bash
cd /mnt/public/dingbowen/RETU/analysis/flops_estimate_kit
bash run_sh/run_all.sh
```

Run one curve:

```bash
cd /mnt/public/dingbowen/RETU/analysis/flops_estimate_kit
bash run_sh/SFT889K/SFT889K_fit_ckpt14080.sh
```

By default, outputs are written to:

```text
fit_result/<scene>/ckpt<step>/
```

You can redirect generated outputs without editing scripts:

```bash
FIT_OUTPUT_ROOT=/tmp/flops_fit_result bash run_sh/run_all.sh
```

## Output Files

Each fitted curve directory contains:

```text
fit_result/<scene>/ckpt<step>/
├── *.png          # Reproduced scaling-curve figure.
├── metrics.json   # Fit quality, selected hyperparameters, and predictions.
└── run.log        # Full command output for this curve.
```

The top-level runner also writes:

```text
fit_result/run_all.log
```

## How To Read The Figures

- The x-axis is relative RL compute after the selected SFT checkpoint, measured
  in EFLOPs.
- The y-axis is validation performance gain over the SFT checkpoint performance.
- `Psft` in a panel title is the validation performance at the SFT branch point.
- Individual benchmark panels show per-task scaling behavior.
- `overall` is the weighted aggregate over benchmark validation-set sizes.
- The plotted fitted curve is selected from the released hyperparameters rather
  than searched during this reproduction run.

The default benchmark weights are:

| Benchmark | Count |
| --- | ---: |
| `gsm8k` | 1317 |
| `olympiad_bench` | 291 |
| `math` | 237 |
| `minerva` | 262 |
| `aime` | 25 |
| `aime25` | 25 |

## Fitting Hyperparameters

Each script embeds the selected hyperparameters recovered from `fit_res/`.  The
most important options are:

| Option | Meaning |
| --- | --- |
| `fit_points_num` | Number of early branch points used for fitting. |
| `train_split` | Index range used as the fitting split for `overall`. |
| `val_split` | Index range used as the validation split for `overall`. |
| `free_c0` | Lets the curve fit its own lower asymptote instead of fixing it at zero. |
| `use_robust_reg` | Enables least-trimmed-squares robust regression. |
| `lts_alpha` | Fraction of points retained by robust regression. |
| `outlier_threshold` | Residual threshold used by outlier detection. |
| `val_most` | Upper index bound considered by validation. |

For public reproduction, these values should normally stay unchanged.

## Cache Notes

The fitter expects cached pickle files instead of raw training artifacts:

- `cache/sft/*.pkl` stores SFT FLOPs to performance dictionaries.
- `cache/sft_then_rl/<scene>/*.pkl` stores RL branch dictionaries for each SFT
  checkpoint.

Historical caches mix one-decimal and two-decimal EFLOPs keys.  For example, the
FLOPs estimator may produce `349.18` while the cache stores `349.2`.  The fitter
therefore performs a small nearest-key alignment with a `0.2` EFLOPs tolerance
when joining an SFT checkpoint with its RL branch.  This only handles cache key
rounding and does not change the underlying curve values.

## Implementation Notes

- `fitting_curves.py` is the only CLI used by the released scripts.
- `curve_fit_prepare.py` builds the in-memory fitting data from cached SFT and
  RL dictionaries.
- `utils/flops_est.py` contains the Qwen2.5-7B SFT FLOPs estimator.
- `visualize/scaling_law_v2.py` performs the candidate curve fitting and figure
  generation.
- Hyperparameter-search code and raw-log parsing utilities are intentionally not
  included in this cleaned release path.
