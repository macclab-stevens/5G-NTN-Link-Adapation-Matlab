# Evaluation & Visualization Guide

This guide explains how to evaluate the XGBoost MCS model and generate analysis charts from the artifacts in `ML/`. It covers commands, key parameters, expected inputs/outputs, and quick examples.

## Prerequisites
- A working Python/uv environment: `uv venv && uv sync`
- Feature Parquet files produced by the featurizer: `features/train.parquet`, `features/test.parquet` (see `ML/featurize.py`)
- A trained model and metadata in `ML/models/` (run `ML/train_xgb.py`)

---

## Quick Workflow

```bash
# 1) Featurize (if you haven’t already)
uv run python ML/featurize.py --data-dir ML/data --include-glob "**/*.csv"

# 2) Train model
uv run python ML/train_xgb.py --train features/train.parquet --test features/test.parquet --output-dir ML/models --calibrate-target 0.1

# 3) Evaluate model, write predictions + metrics
uv run python ML/evaluate_xgb.py --model ML/models/xgb_mcs_pass.json --meta ML/models/model_meta.json --test features/test.parquet --output-dir ML/reports --tradeoff

# 4) Optional plots and additional analyses
uv run python ML/eval_violation_curves.py --predictions ML/reports/predictions.csv --output-dir ML/reports
uv run python ML/plot_threshold_sweep.py --test-data features/test.parquet --output ML/reports/threshold_sweep.png
uv run python ML/plot_throughput_pareto.py --test-data features/test.parquet --output ML/reports/throughput_pareto.png
uv run python ML/plot_bler_vs_elevation.py --bin-width 1.0 --smooth 1 --output ML/reports/bler_vs_elevation.png
```

Outputs are written under `ML/reports/` (CSV + PNG).

---

## Script Reference

### evaluate_xgb.py
- File: `ML/evaluate_xgb.py`
- Purpose: Compute core metrics, export predictions, feature importance, SHAP summaries; optionally sweep threshold tradeoffs.
- Inputs: model (`--model`), metadata (`--meta`), features (`--test`).
- Outputs (in `--output-dir`):
  - `metrics.json` (e.g., `logloss`, `accuracy`)
  - `predictions.csv` (columns: `pred_prob`, `label_pass` if available, `tbs`, and features)
  - `feature_importance_gain.csv/png`, `feature_importance_weight.csv`
  - `shap_summary.csv`, `shap_top_contribs.csv`
  - If `--tradeoff`: `tradeoff_overall.csv/png`, `tradeoff_by_<slice>.csv/png`
- Key parameters:
  - `--model`: path to `xgb_mcs_pass.json`
  - `--test`: Parquet dataset to evaluate
  - `--meta`: `model_meta.json` (contains feature order)
  - `--output-dir`: destination for CSV/PNGs (default: `reports`)
  - `--sample`: cap rows to load (int; 0 = all)
  - `--shap-sample`: rows for SHAP pred_contribs (default 10k)
  - `--topk`: top features to plot by gain (default 20)
  - `--tradeoff`: add acceptance/violation/throughput sweep vs threshold τ
  - `--slice-by`: column to slice tradeoffs (e.g., `cqi`, `snr_round`)
  - `--min-slice-count`: minimum rows per slice to include (default 1000)
  - `--max-slices`: maximum slices to plot (default 8)
  - `--grid-steps`: number of τ values (0.99→0.01) to evaluate
- Examples:
  - Overall eval only:
    ```bash
    uv run python ML/evaluate_xgb.py --test features/test.parquet --output-dir ML/reports
    ```
  - With tradeoffs and per‑CQI slices:
    ```bash
    uv run python ML/evaluate_xgb.py --test features/test.parquet --output-dir ML/reports --tradeoff --slice-by cqi --grid-steps 49
    ```

### Calibration with Confidence Bands
- File: `ML/compute_metrics.py`
- Purpose: Core probability metrics and reliability diagrams; can overlay 95% CI bands for empirical pass rate per bin.
- Example:
  ```bash
  uv run python ML/compute_metrics.py \
    --test features/test.parquet \
    --model ML/models/xgb_mcs_pass.json \
    --meta ML/models/model_meta.json \
    --output-dir ML/reports \
    --sample 200000 \
    --calibration-bins 15 \
    --calibration-ci
  ```
  Outputs: `metrics_prob.json`, `calibration_overall.csv/.png` (with CI), and optional `metrics_by_<slice>.csv` / `calibration_by_<slice>.png` when using `--slice-by`.

### eval_violation_curves.py
- File: `ML/eval_violation_curves.py`
- Purpose: ROC/PR curves for “violation” detection with score = 1 − P(pass).
- Inputs: `--predictions` from `evaluate_xgb.py` (needs `pred_prob`, `label_pass`).
- Outputs: `violation_roc.png`, `violation_pr.png`, `violation_curves.json`.
- Parameters:
  - `--predictions`: path to `predictions.csv`
  - `--output-dir`: destination folder (default: `reports`)
- Example:
  ```bash
  uv run python ML/eval_violation_curves.py --predictions ML/reports/predictions.csv --output-dir ML/reports
  ```

### plot_threshold_sweep.py
- File: `ML/plot_threshold_sweep.py`
- Purpose: Three panels vs τ (acceptance rate, expected throughput among accepted, aggregate throughput).
- Inputs: Parquet test set (`--test-data`); uses the trained model via `MCSRecommender`.
- Outputs: chart `threshold_sweep.png` and `threshold_sweep.csv`.
- Parameters:
  - `--test-data`: Parquet path (default `features/test.parquet`)
  - `--sample-size`: number of rows to sample (int; 0 = all)
  - `--output`, `--csv-out`: output paths
  - `--device`: `cpu` or `gpu` for inference
  - `--grid`: number of τ values from 0.1..0.95 (default 25)
- Example:
  ```bash
  uv run python ML/plot_threshold_sweep.py --test-data features/test.parquet --grid 31 --device cpu
  ```

### plot_throughput_pareto.py
- File: `ML/plot_throughput_pareto.py`
- Purpose: Throughput–Reliability Pareto plot for the threshold sweep and the throughput policy point.
- Inputs: Parquet test set (`--test-data`).
- Outputs: `throughput_pareto.png`, `throughput_pareto_data.csv`.
- Parameters:
  - `--test-data`: Parquet path (default `features/test.parquet`)
  - `--sample-size`: rows to sample (int; 0 = all)
  - `--output`, `--csv-out`: output paths
  - `--device`: `cpu` or `gpu`
  - `--grid`: number of τ values (default 25)
- Example:
  ```bash
  uv run python ML/plot_throughput_pareto.py --test-data features/test.parquet --device cpu --grid 25
  ```

### plot_bler_vs_elevation.py
- File: `ML/plot_bler_vs_elevation.py`
- Purpose: BLER vs Elevation with Avg MCS overlays per BLER target; two stacked panels sharing the elevation x‑axis. Uses elevation binning for stability.
- Inputs: Case‑style CSV (e.g., `data/Case9_MCS_ThroughputCalulation_*.csv`) and the trained model.
- Outputs: `bler_vs_elevation_<windowTag>.png`, `bler_vs_elevation_data_<windowTag>.csv`.
- Parameters:
  - `--input`: CSV path; if omitted, the script auto‑selects a Case9 file
  - `--output`, `--csv-out`: output paths; a window tag (e.g., `blerw2000`) is appended
  - `--device`: `cpu` or `gpu` for model inference
  - `--sample-size`: rows to sample for speed (0 = all)
  - `--bin-width`: elevation bin width in degrees (default 1.0)
  - `--smooth`: optional post‑bin rolling window in bins (default 1 = off)
- Example:
  ```bash
  uv run python ML/plot_bler_vs_elevation.py --bin-width 1.0 --smooth 1 --device cpu
  ```

---

## Interpreting Results
- `metrics.json`: sanity checks for probability quality (log loss) and rough accuracy.
- `feature_importance_gain.png`: top drivers; cross‑check with domain knowledge.
- `shap_summary.csv`: global mean |SHAP| for interpretability.
- `violation_roc.png` / `violation_pr.png`: ranking quality for failure detection.
- `threshold_sweep.png`: pick τ that achieves your target violation while preserving acceptance/throughput.
- `throughput_pareto.png`: compare throughput policy vs threshold sweep trade‑offs.
- `bler_vs_elevation_*.png`: behavior across geometry; BLER (solid) and Avg MCS (dashed) share colors per target.

---

## Tips & Troubleshooting
- GPU inference: pass `--device gpu` on plotting scripts if XGBoost GPU predictor is available.
- Large datasets: use `--sample` or `--sample-size` to speed up iteration.
- Slices: change `--slice-by` (e.g., `snr_round`, `cqi`, `ele_angle`) to inspect fairness/robustness.
- Windows in filenames: `plot_bler_vs_elevation.py` appends a window tag (`blerw<val>` or `blerw_binsX_smoothY`).

---

## File Map
- Training: `ML/train_xgb.py`
- Evaluation: `ML/evaluate_xgb.py`, `ML/eval_violation_curves.py`
- Plotting: `ML/plot_threshold_sweep.py`, `ML/plot_throughput_pareto.py`, `ML/plot_bler_vs_elevation.py`
- Data prep: `ML/featurize.py`
