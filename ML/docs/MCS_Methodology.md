# MCS Recommendation Methodology and Implementation

## Overview

Goal: choose the highest Modulation and Coding Scheme (MCS) that satisfies a BLER constraint to maximize throughput while maintaining acceptable quality. We frame this as constraint‑aware decision making and learn a predictive model to estimate the probability a transmission will pass the BLER target given context and a candidate MCS.

## Terminology & Acronyms (Plain English)

- MCS (Modulation and Coding Scheme): a number that picks a pair of modulation and code rate for transmission. Higher MCS → higher data rate but requires cleaner channels (more likely to fail if conditions are poor).
- Modulation (MOD): how bits are mapped to radio symbols. Common values: QPSK, 16QAM, 64QAM, 256QAM. The modulation order Qm is the bits per symbol (2, 4, 6, 8 respectively).
- Code rate (R): fraction of useful data vs redundancy (error‑correction). Higher R means fewer protection bits (faster, but less robust).
- Spectral efficiency: how many bits you can push per symbol; roughly Qm x R. We use it to rank MCS for throughput.
- BLER (Block Error Rate): fraction of transmitted blocks that fail to decode. Lower is better. Systems often target ~10% BLER for best throughput vs reliability trade‑off.
- Target BLER (target_bler): the BLER limit we want to stay under (e.g., 0.1).
- SNR (Signal‑to‑Noise Ratio): loudness of the signal vs noise (in dB). Higher SNR is better.
- Path Loss (pathloss): how much the signal is weakened by distance/obstacles (in dB). Higher path loss is worse.
- CQI (Channel Quality Indicator): a compact number reported by the device to indicate channel quality; typically maps to a recommended MCS.
- TBS (Transport Block Size): the number of bits in one scheduled data block; a proxy for throughput per time unit.
- TCR (Target Code Rate): code rate value associated with an MCS (0..1).
- Window (window): a configuration describing how many recent transmissions or samples to consider for statistics; also present as a column in the dataset.
- Slot / slot_percent: time indexing for transmissions (slot) and a normalized position in time (slot_percent).
- Elevation angle (ele_angle): geometry of the link (e.g., satellite angle); affects path loss and SNR.
- PDSCH (Physical Downlink Shared Channel): downlink data channel in 5G.
- CSI (Channel State Information): measurements/indicators (such as CQI) describing the radio channel.
- OLLA (Outer Loop Link Adaptation): a feedback controller that nudges the decision threshold to keep BLER near the target.
- Violation rate: among transmissions we “accept” at inference, how often BLER exceeds the target (lower is better).
- Threshold τ (tau): the minimum pass probability required to accept a given MCS under the threshold‑based policy.
- label_pass: the training label we derive from data: 1 if (BLER ≤ Targetbler) and (BLKErr == 1), else 0.

Dataset column cheat‑sheet (after cleaning/canonicalization):
- slot_percent: normalized time index (from `slotPercnt`)
- slot: transmission slot index (from `slot`)
- ele_angle: elevation angle (from `eleAnge`)
- pathloss: path loss in dB (from `PathLoss`)
- snr: signal‑to‑noise ratio in dB (from `SNR`)
- cqi: channel quality indicator (from `CQI`)
- window: configured window length (from `window`)
- target_bler: BLER target (from `Targetbler`)
- mcs: MCS index used in the record (from `MCS`)
- tbs: transport block size (from `TBS`)
- tcr: code rate (from `TCR`)
- bler: measured block error rate (from `BLER`)
- blkerr: success flag in the logs (per spec here: 1 = success)

## Problem Formulation

- Inputs (context): slot_percent, slot, ele_angle, pathloss, snr, cqi, window, target_bler
- Decision variable: mcs (integer index)
- Outcome: label_pass = 1 if (BLER <= Targetbler) and (BLKErr == 1), else 0
- Objective at inference:
  - Threshold objective: pick the highest MCS with P(pass | context, MCS) ≥ τ
  - Throughput objective: pick MCS maximizing spectral_efficiency(MCS) x P(pass | context, MCS)

This mirrors 3GPP link adaptation: choose an MCS that meets a BLER constraint, and among feasible options, prefer higher throughput. It also aligns with MATLAB helper logic (hMCSSelect, MCS_Allo_Algo1), which uses CQI/SINR, MCS tables, and a BLER target to select MCS conservatively.

## Why Not Direct “Optimal MCS” Classification?

Logs typically show one tried MCS per context, not the outcome for all alternatives; true “optimal” is unobserved. Training a classifier to match the logged MCS embeds policy bias and cannot safely simulate counterfactual choices. Learning P(pass | context, MCS) enables us to evaluate any MCS for the same context, apply a BLER constraint via calibration, and pick the highest feasible MCS.

## Data & Feature Engineering

- CSV parsing: input files often have quoted numeric fields with whitespace; we normalize and cast safely (Polars).
- Canonicalization: slotPercnt→slot_percent, eleAnge→ele_angle, Targetbler→target_bler.
- Core features: snr, cqi, pathloss, target_bler, slot_percent, slot, ele_angle, window, and the candidate mcs.
- Derived features (current baseline): snr bins and clipping, mod_code, simple interactions (snrxcqi, snrxpathloss). Further windowed statistics can be added (rolling pass rate, snr/cqi trends) when sequence data are used.
- Label: label_pass = 1 if (BLER ≤ Targetbler) & (BLKErr == 1).

## Model & Training

- Model: XGBoost binary classifier trained on [context + mcs] → label_pass.
- Rationale: non‑linear, fast, supports feature importance and pred_contribs for interpretability, and can use CPU/GPU.
- Training split: streaming featurization writes separate Parquet files for train/test. Optionally cap rows for dev speed.

## Calibration (Constraint Control)

We calibrate a pass‑probability threshold τ on a validation set to meet a target violation rate (e.g., 10%). This provides an operational link between the model’s probability and the BLER constraint, enabling the “highest MCS meeting τ” policy. We store τ in `models/model_meta.json` and the calibration curve in `models/threshold_calibration.csv`.

Notes:
- Calibration uses predictions for the logged MCS on validation (offline approximation). For production, monitor real violation and adjust τ.
- Stratified or per‑SNR/CQI calibration can further improve adherence.

## Recommendation Policies

1) Threshold objective (constraint‑first): pick the highest MCS with P(pass) ≥ τ.
2) Throughput objective (reward‑first): pick argmax of spectral_efficiency(MCS) x P(pass). We compute spectral efficiency via NR MCS Table 1 (TS 38.214), see `mcs_tables.py`.

Both objectives are implemented in `recommend_mcs.py`; threshold uses τ from meta by default.

## Implementation

- Featurization: `featurize.py` streams all CSVs via Polars lazy I/O, cleans/casts columns, creates engineered features and label, and writes `features/train.parquet` and `features/test.parquet` (optionally a combined file). Command knobs support subsetting and chunk sizes for scale.
- Training: `train_xgb.py` reads train/test Parquets, trains XGBoost with early stopping, computes metrics, performs threshold calibration and saves artifacts under `models/`.
- Evaluation: `evaluate_xgb.py` writes predictions, feature importance (gain/weight), and SHAP summaries (pred_contribs) to `reports/`.
- Recommendation: `recommend_mcs.py` loads the model and recommends MCS given context only. Two objectives: threshold and throughput. Also includes a CQI→MCS baseline for comparison.

## Validation & Metrics

- Predictive metrics: logloss, accuracy (less meaningful than operational metrics).
- Operational metrics:
  - Violation rate (BLER > target) under the selected policy
  - Achieved throughput proxy (e.g., TBS on passing transmissions) or expected throughput (efficiency x P(pass))
  - Trade‑off curves: violation vs throughput by threshold τ
  - Sliced analysis by SNR/CQI/pathloss windows

## Real‑World Deployment

Integration into a link adaptation loop:
- Inputs: per‑TTI context (SNR/CQI/pathloss, etc.) and target_bler.
- Inference: for each context, evaluate candidate MCS values with the trained model; select based on the chosen objective (usually threshold with calibrated τ).
- Hysteresis & smoothing: introduce minimum/maximum step changes and dwell time to avoid frequent toggling.
- OLLA‑like adjustments: track a small offset to τ or per‑SNR bin thresholds to maintain desired long‑term BLER.
- Monitoring: log P(pass), chosen MCS, realized BLER, and throughput; compute live violation rates and adjust τ if drift occurs.
- Safety: fall back to CQI baseline if model is unavailable or inputs are out of range.

Latency & Scale:
- The model is compact; evaluating all MCS candidates (e.g., 0..27) per context is cheap on CPU and trivial on GPU.
- Use batched inference if serving at very high rates.

## Repro & Commands (Summary)

1) Featurize all CSVs (streaming split):
```
uv run python featurize.py --data-dir data --train-out features/train.parquet --test-out features/test.parquet --test-frac 0.2 --seed 42 --chunksize 200000
```
2) Train + calibrate:
```
uv run python train_xgb.py --train features/train.parquet --test features/test.parquet --output-dir models --device auto --num-rounds 1500 --early-stopping 100 --calibrate-target 0.1
```
3) Evaluate & interpret:
```
uv run python evaluate_xgb.py --model models/xgb_mcs_pass.json --meta models/model_meta.json --test features/test.parquet --output-dir reports --sample 100000 --shap-sample 20000
```
4) Recommend MCS from context (
threshold objective uses calibrated τ):
```
uv run python recommend_mcs.py --input features/test.parquet --model models/xgb_mcs_pass.json --meta models/model_meta.json --objective threshold --threshold -1 --output reports/recommendations_threshold.csv
```
Throughput objective:
```
uv run python recommend_mcs.py --input features/test.parquet --model models/xgb_mcs_pass.json --meta models/model_meta.json --objective throughput --output reports/recommendations_throughput.csv
```
