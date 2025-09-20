# Model and Method Inventory

## Monotonic Classifiers (`snr`, `cqi` features unless noted)
| Model directory | Training source | Notes | Key metrics |
| --- | --- | --- | --- |
| `models/mcs15_xgb/` | `ML/data/snr_cqi_lut_mcs15.csv` (clipped empirical LUT) | Baseline monotonic booster for MCS≤15. | acc 0.897, MAE 0.233, top-2 0.984 (`reports/mcs15_xgb/metrics.json`). |
| `models/mcs15_xgb_empirical_low/` | `ML/data/snr_cqi_lut_empirical_thresh095_low_mcs15.csv` (conservative LUT with MCS 1–11) | Captures low-index policy; feeds classifier LUTs when safety is priority. | acc 0.873, MAE 0.224, top-2 0.968. |
| `models/mcs_elev_snrcqi/` | `features/all.parquet` filtered to `{ele_angle, snr, cqi, mcs≤15}` (`train_mcs_elev_snrcqi.py`) | Adds elevation angle; monotonic on snr & cqi only. | acc 0.676, MAE 0.364, top-2 0.923. |
| `models/mcs_snr_cqi_dataset/` | `ML/data/snr_cqi_dataset_mcs15.csv` | Directly trained on raw observations (no LUT smoothing); used for the classifier LUT below. | acc 0.631, MAE 0.412, top-2 0.915. |

## Pass-Probability XGBoost Models
| Model directory | Feature set | Threshold | Metrics |
| --- | --- | --- | --- |
| `models/xgb_reduced/` | `[snr, pathloss, cqi, ele_angle, mcs]` | 0.85 (calibrated) | logloss 0.0396, acc 0.993 (`reports/xgb_reduced/metrics.json`). |
| `models/xgb_snr_cqi/` | `[snr, cqi, mcs]` | 0.99 (calibrated) | logloss 0.0411, acc 0.993 (`reports/xgb_snr_cqi/metrics.json`). |


## Polynomial Models & LUTs
- `ML/polynomial_models.py` – Fits cubic polynomial regressions and Newton-style interpolants for both `(snr, cqi)` and `(snr, cqi, ele_angle)`, exporting:
  - Analytics (`ML/reports/polynomial/polynomial_models.json`, plus heatmaps/contours).
  - LUTs in `ML/data/polynomial_luts/` with separate files per method (`*_poly_lut.csv`, `*_newton_lut.csv`, MCS already clamped 0–27; clip to ≤15 if needed).

