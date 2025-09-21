# Model and Method Inventory

## Monotonic Classifiers (`snr`, `cqi` features unless noted)
| Model directory | Training source | Notes | Key metrics |
| --- | --- | --- | --- |
| `models/mcs15_xgb/` | `ML/data/snr_cqi_lut_mcs15.csv` (clipped empirical LUT) | Baseline monotonic booster for MCS≤15. | acc 0.897, MAE 0.233, top-2 0.984 (`reports/mcs15_xgb/metrics.json`). |
| `models/mcs15_xgb_empirical_low/` | `ML/data/snr_cqi_lut_empirical_thresh095_low_mcs15.csv` (conservative LUT with MCS 1–11) | Captures low-index policy; feeds classifier LUTs when safety is priority. | acc 0.873, MAE 0.224, top-2 0.968. |
| `models/mcs_elev_snrcqi/` | `features/all.parquet` filtered to `{ele_angle, snr, cqi, mcs≤15}` (`train_mcs_elev_snrcqi.py`) | Adds elevation angle; monotonic on snr & cqi only. | acc 0.676, MAE 0.364, top-2 0.923. |
| `models/mcs_snr_cqi_dataset/` | `ML/data/snr_cqi_dataset_mcs15.csv` | Directly trained on raw observations (no LUT smoothing); used for the classifier LUT below. | acc 0.631, MAE 0.412, top-2 0.915. |

**Concept** Monotonic boosters constrain the tree learner so that improving a helpful feature never lowers the predicted class, and increasing a harmful feature never raises it. XGBoost enforces the constraint while growing each tree: splits that would violate the requested sign pattern are rejected or their leaf scores reordered, guaranteeing the aggregated score is monotone along every root-to-leaf path.

**Scripts**
- `ML/train_mcs_mono_xgb.py` trains the LUT-backed multi-class model with `monotone_constraints="(1,1)"`, locking the class probabilities so higher `snr` and/or `cqi` can only maintain or increase the chosen MCS index.
- `ML/train_mcs_elev_snrcqi.py` extends the classifier to include elevation; the optional `--monotone` flag applies `(0,1,1)` so `ele_angle` remains free while `snr` and `cqi` stay non-decreasing.
- `ML/train_xgb.py` produces the pass-probability booster. With `--monotone`, the logistic model obeys `(+snr, +cqi, -mcs)`, reflecting that cleaner channels raise `P(pass)` whereas more aggressive coding lowers it. The calibrated decision threshold saved in `models/.../model_meta.json` feeds the recommender policies.

## Pass-Probability XGBoost Models
| Model directory | Feature set | Threshold | Metrics |
| --- | --- | --- | --- |
| `models/xgb_reduced/` | `[snr, pathloss, cqi, ele_angle, mcs]` | 0.85 (calibrated) | logloss 0.0396, acc 0.993 (`reports/xgb_reduced/metrics.json`). |
| `models/xgb_snr_cqi/` | `[snr, cqi, mcs]` | 0.99 (calibrated) | logloss 0.0411, acc 0.993 (`reports/xgb_snr_cqi/metrics.json`). |


## Polynomial Models & LUTs
`ML/polynomial_models.py` trains two complementary surrogates for the LUT space:

- **Polynomial regression**: builds a fixed design matrix with `PolynomialFeatures(degree=3)` and solves a least-squares fit (`LinearRegression`). The result is a closed-form polynomial in `snr`, `cqi`, (optionally) `ele_angle`. Evaluation is a simple dot product, making the model extremely fast for on-device inference. Metrics (R², RMSE, MAE) and formula strings are written to `ML/reports/polynomial/polynomial_models.json`; diagnostic surfaces/contours (e.g., `poly_surface_snr_cqi.png`) visualize bias against the empirical dataset.
- **Newton interpolant**: constructs a sparse grid by sampling the feature domain and uses SymPy’s multivariate Newton interpolation. Each grid vertex value is produced by a distance-weighted average over the nearest raw samples (k-NN smoothing), then the symbolic interpolant is expanded to a polynomial. This preserves local curvature while remaining monotone-friendly when evaluated on a mesh. The expanded expressions feed LUT generation without needing runtime SymPy.

Both surrogates export meshes to `ML/data/polynomial_luts/` (`*_poly_lut.csv`, `*_newton_lut.csv`). Values are pre-clamped to MCS 0–27; downstream scripts can re-cap at ≤15 for conservative operation.

## Classifier LUTs (`ML/data/luts/`)
- `snr_cqi_lut_classifier.csv`: 0.2 dB × 1 CQI grid produced by `export_lut.py --mode classifier` against `models/mcs_snr_cqi_dataset`. Each row lists the quantized `(snr, cqi)` pair and the monotonic classifier’s recommended MCS (2–9) after mean aggregation. Use this as the default table when wiring MATLAB policies that expect the full mid-band range.
- `snr_cqi_lut_classifier_low.csv`: variant exported with an `mcs_min` guardrail so entries stay within 4–9. This avoids the most robust MCS levels and is the table referenced in `Matlab/MCS_Allo_Algo2.m` for a higher-throughput operating point.

See `ML/docs/LUT_Catalog.md` for reproduction commands and additional LUT variants.
