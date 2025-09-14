# SNR x CQI → MCS Lookup Tables (LUTs)

This document explains the two LUT exporters in this repo, how they work, and practical recipes to generate, evaluate, and tune tables.

Contents
- What is a LUT and why use it
- Model‑driven LUT: `export_lut.py`
- Empirical LUT (no model): `export_lut_empirical.py`
- Common options and tuning for conservatism
- Dense CQI coverage and grid fill
- Evaluation against the model: `compare_baselines.py`
- Examples and quick recipes
- Troubleshooting and FAQs

## What is a LUT and why use it

A LUT maps operating conditions to a recommended MCS without running the ML model online. Here, the operating conditions are SNR (quantized, e.g., 0.1 dB steps) and CQI (integer 0..15). LUTs are useful for:

- Simple, predictable runtime behavior (no model dependency at inference)
- Easy auditing and testing against policy constraints
- Fast A/B comparisons vs. the learned policy

We provide two ways to generate LUTs:

1) Model‑driven (uses the trained model’s P(pass | context, MCS))
2) Empirical (uses observed pass rates and throughput in the dataset)

Both can be evaluated against the model to compare throughput and reliability.

## Model‑driven LUT — `export_lut.py`

Builds a (SNR, CQI) → MCS table by aggregating the trained model’s pass probabilities over rows in each bin, then selecting the MCS by a chosen objective.

Key steps:
- Load `features` from `models/model_meta.json` and the trained model `models/xgb_mcs_pass.json`.
- Load input dataset (`--data`, Parquet). Must contain at least `snr` and `cqi`.
- Predict P(pass | row, mcs) for all MCS, all rows.
- Bin rows by `(round(snr/--snr-bin), round(cqi/--cqi-bin))`.
- Aggregate per bin across rows:
  - default: mean probability; or a quantile via `--pass-quantile` for conservative aggregation
- Select an MCS per bin via objective and rules:
  - `--objective threshold` picks an MCS whose aggregated P(pass) ≥ τ (`--threshold`), with fallback if none meets it
  - `--objective throughput` picks argmax of P(pass)×spectral_efficiency, with optional pass guardrail
  - Tie‑breaking / feasibility rules via `--select-rule` (highest_feasible, max_prob_feasible, lowest_feasible)

Important options:
- `--threshold` (or `-1` to use calibrated τ from model meta)
- `--pass-quantile <q>`: use low quantile (e.g., 0.1) to be pessimistic
- `--prob-margin <m>`: require extra margin above threshold/guardrail
- `--min-pass-guardrail <g>`: throughput objective safety filter
- `--select-rule`: behavior among feasible candidates
- `--mcs-min/--mcs-max`: clamp recommended range
- `--snr-bin/--cqi-bin`: bin sizes (0.1 and 1.0 are common)
- `--min-count`: minimum rows per bin to include
- `--fill-cqi-grid`: fill all CQIs 0..15 per SNR from nearest CQI when sparse

Example (conservative, dense CQI grid):

```
uv run python export_lut.py \
  --data features/all.parquet \
  --out data/snr_cqi_lut_dense_safe.csv \
  --objective threshold --threshold -1 \
  --snr-bin 0.1 --cqi-bin 1 --min-count 100 \
  --fill-cqi-grid --cqi-min 0 --cqi-max 15 \
  --pass-quantile 0.1 --prob-margin 0.05 --select-rule max_prob_feasible
```

Very conservative variant:

```
uv run python export_lut.py \
  --data features/all.parquet \
  --out data/snr_cqi_lut_dense_ultra_safe.csv \
  --objective threshold --threshold 0.95 \
  --snr-bin 0.1 --cqi-bin 1 --min-count 100 \
  --fill-cqi-grid --cqi-min 0 --cqi-max 15 \
  --pass-quantile 0.01 --prob-margin 0.1 --select-rule lowest_feasible \
  --mcs-max 5
```

## Empirical LUT — `export_lut_empirical.py`

Builds a (SNR, CQI) → MCS table directly from logged outcomes; no model predictions are used.

Key steps:
- Requires dataset with `snr`, `cqi`, `mcs`, `label_pass` (and optionally `tbs`).
- Bin by `(snr_code, cqi_code)` as above.
- Aggregate per `(snr_code, cqi_code, mcs)` cell:
  - count `n`
  - `pass_rate = mean(label_pass)`
  - `obs_tput = mean(tbs × label_pass)` if `tbs` exists; else `pass_rate × spectral_efficiency(mcs)`
- Select per bin via `--objective threshold` or `--objective throughput`, with the same tie/guardrail knobs as the model exporter.

Example (threshold objective):

```
uv run python export_lut_empirical.py \
  --data features/all.parquet \
  --out data/snr_cqi_lut_empirical.csv \
  --objective threshold --threshold 0.9 \
  --snr-bin 0.1 --cqi-bin 1 --min-count 50 \
  --fill-cqi-grid --cqi-min 0 --cqi-max 15 \
  --prob-margin 0.05 --select-rule max_prob_feasible
```

Example (throughput with guardrail):

```
uv run python export_lut_empirical.py \
  --data features/all.parquet \
  --out data/snr_cqi_lut_empirical_tput.csv \
  --objective throughput --min-pass-guardrail 0.85 \
  --snr-bin 0.1 --cqi-bin 1 --min-count 50 \
  --fill-cqi-grid --cqi-min 0 --cqi-max 15
```

Notes and caveats:
- Observational bias: only MCS actually tried in a bin are evaluated.
- Sparse bins are filtered by `--min-count`; grid fill copies nearest CQI choice.

## Dense CQI coverage and grid fill

Both exporters support `--fill-cqi-grid` to fill all CQIs in [cqi_min..cqi_max] for each SNR by nearest‑neighbor CQI. This makes a rectangular LUT but can “replicate” a single CQI’s choice when data are sparse. Pair with conservative settings so fills don’t become overly aggressive.

## Evaluation against the model — `compare_baselines.py`

Regardless of how a LUT was created, you can evaluate it against the trained model (used as an oracle) for consistent comparison of reliability and throughput.

Example:

```
uv run python compare_baselines.py \
  --data features/test.parquet \
  --output-dir reports/model_lut \
  --snr-cqi-lut data/snr_cqi_lut_dense_safe.csv \
  --lut-snr-bin 0.1 --lut-cqi-bin 1.0 \
  --baseline-guardrail 0.9 --sample 200000
```

Artifacts:
- `baseline_summary.json` (key `baseline_snr_cqi_table`)
- `baseline_summary.png` (bars: expected throughput, violation)
- `baseline_vs_model_elev.png` (BLER and Avg MCS vs elevation)

## Recipes

1) Build combined Parquet of features:

```
uv run python featurize.py \
  --data-dir data \
  --out features/all.parquet \
  --train-out features/train.parquet \
  --test-out features/test.parquet \
  --include-glob '*'
```

2) Export a conservative model‑driven LUT and evaluate:

```
uv run python export_lut.py --data features/all.parquet --out data/snr_cqi_lut_dense_safe.csv \
  --objective threshold --threshold -1 --snr-bin 0.1 --cqi-bin 1 --min-count 100 \
  --fill-cqi-grid --cqi-min 0 --cqi-max 15 --pass-quantile 0.1 --prob-margin 0.05 \
  --select-rule max_prob_feasible

uv run python compare_baselines.py --data features/test.parquet --output-dir reports/model_lut \
  --snr-cqi-lut data/snr_cqi_lut_dense_safe.csv --lut-snr-bin 0.1 --lut-cqi-bin 1.0 \
  --baseline-guardrail 0.9
```

3) Export an empirical LUT and compare side‑by‑side:

```
uv run python export_lut_empirical.py --data features/all.parquet --out data/snr_cqi_lut_empirical.csv \
  --objective threshold --threshold 0.9 --snr-bin 0.1 --cqi-bin 1 --min-count 50 \
  --fill-cqi-grid --cqi-min 0 --cqi-max 15 --prob-margin 0.05 --select-rule max_prob_feasible

uv run python compare_baselines.py --data features/test.parquet --output-dir reports/empirical_lut \
  --snr-cqi-lut data/snr_cqi_lut_empirical.csv --lut-snr-bin 0.1 --lut-cqi-bin 1.0 \
  --baseline-guardrail 0.9
```

## Troubleshooting & FAQs

Q: My LUT recommends high MCS everywhere (e.g., 27).
- If using throughput objective, add `--min-pass-guardrail` (e.g., 0.85) and/or switch to threshold objective.
- Use conservative aggregation: `--pass-quantile 0.1` and add `--prob-margin`.

Q: LUT entries look flat (same MCS across many CQIs).
- You likely used `--fill-cqi-grid` with sparse CQI data; consider exporting without fill or include CQI‑rich logs in `features/all.parquet`.
- Increase SNR resolution (`--snr-bin 0.05`) and lower `--min-count` moderately to capture variation.

Q: How do I make it much more conservative?
- Raise `--threshold` (e.g., 0.95), `--prob-margin` (e.g., 0.1), and use `--select-rule lowest_feasible` or `max_prob_feasible`.
- Apply a global cap: `--mcs-max 5`.
