#!/usr/bin/env python3
"""
Baseline CQI mapping and OLLA comparison against the trained model.

This script simulates per-row decisions for several policies and compares them
on reliability and throughput. It uses the trained model as a counterfactual
oracle to estimate P(pass | context, MCS), and optionally samples outcomes to
approximate realized performance.

Policies:
  - model_threshold: highest MCS with P(pass) ≥ τ (τ from model_meta.json)
  - model_throughput: argmax_m P(pass) x spectral_efficiency(m)
  - baseline_cqi: CQI→MCS mapping (configurable: round/linear/table)
  - baseline_cqi_olla: like baseline_cqi but with a running offset (OLLA)

Outputs:
  - reports/baseline_summary.json: overall metrics per policy
  - reports/baseline_summary.png: bar chart (throughput, violation)
  - reports/baseline_vs_model_elev_blerw<window>.png: BLER & Avg MCS vs elevation

Notes on baselines:
- Real logs often have CQI clustered at very small values. A naïve
  `round(CQI)` mapping can degenerate to MCS 0 almost always and therefore
  show near-zero throughput under a counterfactual oracle. This script exposes
  tunables to produce more realistic baselines:
    * `--baseline-map`: choose mapping style (round, linear, or lookup table)
    * `--baseline-linear-scale/offset`: linear rescaling into MCS domain
    * `--baseline-min-mcs/--baseline-max-mcs`: clamp recommended range
    * `--olla-init`: initialize OLLA offset (helps avoid getting stuck at 0)
    * `--baseline-guardrail`: optional minimum P(pass) (degrade MCS until met)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo import MCSRecommender
from mcs_tables import spectral_efficiency


BASE_COLS = [
    "slot_percent","slot","ele_angle","pathloss","snr","cqi","window","target_bler",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cqi_to_mcs(cqi: float) -> int:
    return int(max(0, min(27, round(float(cqi)))))

def _map_cqi_array(cqi_vals: np.ndarray, args) -> np.ndarray:
    """Map CQI array to MCS according to CLI options."""
    if args.baseline_map == "round":
        m = np.rint(cqi_vals)
    elif args.baseline_map == "linear":
        m = args.baseline_linear_scale * cqi_vals + args.baseline_linear_offset
    else:  # lookup table with gentle increase
        table = np.array([0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27], dtype=np.int32)
        idx = np.clip(np.rint(cqi_vals).astype(np.int32), 0, len(table) - 1)
        mapped = table[idx]
        return np.clip(mapped, args.baseline_min_mcs, args.baseline_max_mcs).astype(np.int32)
    return np.clip(np.rint(m), args.baseline_min_mcs, args.baseline_max_mcs).astype(np.int32)


# --- SNR→MCS lookup table support ---
class SNRLookup:
    def __init__(self, lut: dict[int, int], bin_size: float, default_mcs: int = 0):
        self.lut = dict(lut)
        self.bin = float(bin_size)
        self.default = int(default_mcs)
        self.keys = np.array(sorted(self.lut.keys()), dtype=np.int64)

    def map_array(self, snr_vals: np.ndarray) -> np.ndarray:
        if self.bin <= 0:
            raise ValueError("snr_lut_bin must be > 0")
        # Quantize SNR to integer bin codes, e.g., 0.1 → code = round(snr*10)
        codes = np.rint(snr_vals / self.bin).astype(np.int64)
        out = np.full(codes.shape, self.default, dtype=np.int32)
        if self.keys.size == 0:
            return out
        # Exact hits
        mask_exact = np.isin(codes, self.keys)
        if mask_exact.any():
            # Build fast dict access via vectorized take
            # Map codes to indices in self.keys
            pos = np.searchsorted(self.keys, codes[mask_exact])
            # Correct positions that are not exact matches
            exact_codes = codes[mask_exact]
            in_range = (pos < self.keys.size) & (self.keys[pos] == exact_codes)
            # For exact in-range, assign
            if in_range.any():
                out_idx = np.nonzero(mask_exact)[0][in_range]
                key_idx = pos[in_range]
                out[out_idx] = np.array([self.lut[int(k)] for k in self.keys[key_idx]], dtype=np.int32)
        # Fallback: nearest key
        mask_miss = ~mask_exact
        if mask_miss.any():
            c = codes[mask_miss]
            pos = np.searchsorted(self.keys, c)
            pos_clamped = np.clip(pos, 0, self.keys.size - 1)
            left = np.maximum(pos_clamped - 1, 0)
            right = pos_clamped
            # Choose nearer of left/right
            dist_left = np.abs(c - self.keys[left])
            dist_right = np.abs(self.keys[right] - c)
            choose_right = dist_right < dist_left
            nearest_idx = np.where(choose_right, right, left)
            out_idx = np.nonzero(mask_miss)[0]
            out[out_idx] = np.array([self.lut[int(k)] for k in self.keys[nearest_idx]], dtype=np.int32)
        return out


def load_snr_lut(path: Optional[str], bin_size: float, default_mcs: int) -> Optional[SNRLookup]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"SNR LUT file not found: {p}")
    try:
        df = pl.read_csv(p)
    except Exception as e:
        raise SystemExit(f"Failed to read SNR LUT CSV: {e}")
    # Accept columns: 'snr' or 'snr_bin' and 'mcs'
    cols = [c.lower() for c in df.columns]
    norm = {c.lower(): c for c in df.columns}
    if 'mcs' not in cols:
        raise SystemExit("SNR LUT CSV must have a 'mcs' column")
    snr_col = 'snr' if 'snr' in cols else ('snr_bin' if 'snr_bin' in cols else None)
    if snr_col is None:
        raise SystemExit("SNR LUT CSV must have 'snr' or 'snr_bin' column")
    snr_vals = df[norm[snr_col]].to_numpy()
    mcs_vals = df[norm['mcs']].to_numpy()
    # Build mapping: quantized code -> mcs
    codes = np.rint(snr_vals.astype(np.float64) / float(bin_size)).astype(np.int64)
    lut: dict[int, int] = {}
    for code, m in zip(codes, mcs_vals):
        try:
            lut[int(code)] = int(m)
        except Exception:
            continue
    return SNRLookup(lut, bin_size=bin_size, default_mcs=default_mcs)


# --- 2D SNR×CQI → MCS lookup table support ---
class SNR_CQI_Lookup:
    def __init__(self, lut: dict[tuple[int, int], int], snr_bin: float, cqi_bin: float, default_mcs: int = 0):
        self.lut = dict(lut)
        self.snr_bin = float(snr_bin)
        self.cqi_bin = float(cqi_bin)
        self.default = int(default_mcs)
        # Precompute sorted unique snr codes and within-snr cqi codes and mcs arrays
        snr_codes = sorted({k[0] for k in self.lut.keys()})
        self.snr_keys = np.array(snr_codes, dtype=np.int64)
        self.cqi_keys_by_snr: dict[int, np.ndarray] = {}
        self.mcs_by_snr: dict[int, np.ndarray] = {}
        from collections import defaultdict
        buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (s, c), m in self.lut.items():
            buckets[s].append((c, m))
        for s, pairs in buckets.items():
            pairs.sort(key=lambda x: x[0])
            self.cqi_keys_by_snr[int(s)] = np.array([c for c, _ in pairs], dtype=np.int64)
            self.mcs_by_snr[int(s)] = np.array([m for _, m in pairs], dtype=np.int32)

    def _nearest_snr(self, codes: np.ndarray) -> np.ndarray:
        # Choose nearest snr key in 1D
        pos = np.searchsorted(self.snr_keys, codes)
        pos = np.clip(pos, 0, self.snr_keys.size - 1)
        left = np.maximum(pos - 1, 0)
        right = pos
        dist_left = np.abs(codes - self.snr_keys[left])
        dist_right = np.abs(self.snr_keys[right] - codes)
        choose_right = dist_right < dist_left
        return np.where(choose_right, self.snr_keys[right], self.snr_keys[left])

    def map_arrays(self, snr_vals: np.ndarray, cqi_vals: np.ndarray) -> np.ndarray:
        if self.snr_bin <= 0 or self.cqi_bin <= 0:
            raise ValueError("snr_bin and cqi_bin must be > 0")
        snr_codes = np.rint(snr_vals / self.snr_bin).astype(np.int64)
        cqi_codes = np.rint(cqi_vals / self.cqi_bin).astype(np.int64)
        out = np.full(snr_codes.shape, self.default, dtype=np.int32)
        if self.snr_keys.size == 0:
            return out
        # Nearest snr key per row
        snr_near = self._nearest_snr(snr_codes)
        # Process rows grouped by chosen snr key
        uniq_snr = np.unique(snr_near)
        for s in uniq_snr:
            mask = snr_near == s
            cqi_keys = self.cqi_keys_by_snr.get(int(s))
            mcs_arr = self.mcs_by_snr.get(int(s))
            if cqi_keys is None or mcs_arr is None or cqi_keys.size == 0:
                continue
            c = cqi_codes[mask]
            pos = np.searchsorted(cqi_keys, c)
            pos = np.clip(pos, 0, cqi_keys.size - 1)
            left = np.maximum(pos - 1, 0)
            right = pos
            dist_left = np.abs(c - cqi_keys[left])
            dist_right = np.abs(cqi_keys[right] - c)
            choose_right = dist_right < dist_left
            idx = np.where(choose_right, right, left)
            out_idx = np.nonzero(mask)[0]
            out[out_idx] = mcs_arr[idx]
        return out


def load_snr_cqi_lut(path: Optional[str], snr_bin: float, cqi_bin: float, default_mcs: int) -> Optional[SNR_CQI_Lookup]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"SNR×CQI LUT file not found: {p}")
    try:
        df = pl.read_csv(p)
    except Exception as e:
        raise SystemExit(f"Failed to read SNR×CQI LUT CSV: {e}")
    cols = {c.lower(): c for c in df.columns}
    need = ["snr", "cqi", "mcs"]
    for n in need:
        if n not in cols:
            raise SystemExit("SNR×CQI LUT CSV must have columns: snr,cqi,mcs")
    snr_vals = df[cols["snr"]].to_numpy()
    cqi_vals = df[cols["cqi"]].to_numpy()
    mcs_vals = df[cols["mcs"]].to_numpy()
    snr_codes = np.rint(snr_vals.astype(np.float64) / float(snr_bin)).astype(np.int64)
    cqi_codes = np.rint(cqi_vals.astype(np.float64) / float(cqi_bin)).astype(np.int64)
    lut: dict[tuple[int, int], int] = {}
    for s, c, m in zip(snr_codes, cqi_codes, mcs_vals):
        try:
            lut[(int(s), int(c))] = int(m)
        except Exception:
            continue
    return SNR_CQI_Lookup(lut, snr_bin=float(snr_bin), cqi_bin=float(cqi_bin), default_mcs=int(default_mcs))

def batch_predict_all_mcs(df: pl.DataFrame, rec: MCSRecommender) -> Tuple[np.ndarray, np.ndarray]:
    """Predict P(pass) for all rows and all 0..27 MCS in a single batch.

    Returns:
        preds: shape (n_rows, 28)
        eff:   shape (28,) spectral efficiencies per MCS
    """
    feats = rec.features
    n = df.height
    d = len(feats)
    # Build base feature matrix in model feature order (without mcs column)
    Xbase = np.zeros((n, d), dtype=np.float32)
    mcs_pos = feats.index("mcs")
    for j, c in enumerate(feats):
        if c == "mcs":
            continue
        if c in df.columns:
            Xbase[:, j] = df[c].to_numpy().astype(np.float32, copy=False)
        else:
            Xbase[:, j] = 0.0

    # Stack all candidates vertically once: [row1 x 28; row2 x 28; ...]
    Xall = np.repeat(Xbase, 28, axis=0)
    # Fill MCS column as a repeating 0..27 pattern per row
    mseq = np.tile(np.arange(28, dtype=np.float32), n)
    Xall[:, mcs_pos] = mseq
    dmat = xgb.DMatrix(Xall)
    y_all = rec.model.predict(dmat).astype(np.float32)
    preds = y_all.reshape(n, 28)
    eff = np.array([spectral_efficiency(m) for m in range(28)], dtype=np.float32)
    return preds, eff


def choose_model_threshold(rec: MCSRecommender, ctx: Dict[str, float]) -> Tuple[int, float, float]:
    tau = float(rec.threshold)
    best = 0
    best_p = 0.0
    best_score = -1.0
    for m in range(28):
        p = rec.predict_pass_probability(ctx, m)
        if p >= tau and m >= best:
            best = m
            best_p = p
            best_score = p * spectral_efficiency(m)
        elif best_score < 0.0:
            # remember argmax p for fallback if nothing meets tau
            if p > best_p:
                best = m
                best_p = p
    if best_score < 0.0:
        best_score = best_p * spectral_efficiency(best)
    return best, best_p, best_score


def choose_model_throughput(rec: MCSRecommender, ctx: Dict[str, float]) -> Tuple[int, float, float]:
    best = 0
    best_p = 0.0
    best_score = 0.0
    for m in range(28):
        p = rec.predict_pass_probability(ctx, m)
        s = p * spectral_efficiency(m)
        if s > best_score:
            best = m
            best_p = p
            best_score = s
    return best, best_p, best_score


def simulate_policies(
    df: pl.DataFrame,
    rec: MCSRecommender,
    step: float,
    target_bler: float,
    rng: np.random.Generator,
    sample: bool,
    args,
) -> Dict[str, Dict[str, float]]:
    """Vectorized simulation using a single batch of model predictions.

    This avoids per-row x per-MCS calls which are extremely slow.
    """
    # Predict probabilities for all MCS in a single large batch
    preds, eff = batch_predict_all_mcs(df, rec)
    n = df.height

    # Model threshold policy (with fallback to argmax p if none meet tau)
    tau = float(rec.threshold)
    m_idx = np.arange(28, dtype=np.int32)[None, :]
    mask = preds >= tau
    best = np.where(mask, m_idx, -1).max(axis=1)
    fallback = preds.argmax(axis=1)
    best = np.where(best < 0, fallback, best).astype(np.int32)
    p_best = preds[np.arange(n), best]
    s_best = p_best * eff[best]

    # Model throughput policy
    scores = preds * eff[None, :]
    tp_best = scores.argmax(axis=1).astype(np.int32)
    p_tp = preds[np.arange(n), tp_best]
    s_tp = scores[np.arange(n), tp_best]

    # Baseline CQI policy (with configurable mapping)
    cqi_vals = df["cqi"].to_numpy().astype(np.float32, copy=False) if "cqi" in df.columns else np.full(n, 9.0, dtype=np.float32)
    m_cqi = _map_cqi_array(cqi_vals, args)
    if args.baseline_guardrail >= 0.0:
        tau_b = float(args.baseline_guardrail)
        row_idx = np.arange(n)
        # Try highest feasible MCS meeting the guardrail; fallback to current mapping
        m_idx = np.arange(28, dtype=np.int32)[None, :]
        mask = preds >= tau_b
        best_feas = np.where(mask, m_idx, -1).max(axis=1)
        m_cqi = np.where(best_feas >= 0, best_feas, m_cqi).astype(np.int32)
    p_cqi = preds[np.arange(n), m_cqi]
    s_cqi = p_cqi * eff[m_cqi]

    # Baseline SNR→MCS LUT policy (optional)
    p_snr = None
    s_snr = None
    m_snr = None
    if args.snr_lut is not None:
        snr_vals = df["snr"].to_numpy().astype(np.float32, copy=False) if "snr" in df.columns else np.zeros(n, dtype=np.float32)
        lut = load_snr_lut(args.snr_lut, args.snr_lut_bin, args.snr_lut_default_mcs)
        if lut is None:
            raise SystemExit("Failed to load SNR LUT")
        m_snr = np.clip(lut.map_array(snr_vals), args.baseline_min_mcs, args.baseline_max_mcs).astype(np.int32)
        if args.baseline_guardrail >= 0.0:
            tau_b = float(args.baseline_guardrail)
            m_idx2 = np.arange(28, dtype=np.int32)[None, :]
            mask2 = preds >= tau_b
            best_feas2 = np.where(mask2, m_idx2, -1).max(axis=1)
            m_snr = np.where(best_feas2 >= 0, best_feas2, m_snr).astype(np.int32)
        p_snr = preds[np.arange(n), m_snr]
        s_snr = p_snr * eff[m_snr]

    # Baseline SNR×CQI→MCS LUT policy (optional)
    p_sc = None
    s_sc = None
    m_sc = None
    if args.snr_cqi_lut is not None:
        snr_vals = df["snr"].to_numpy().astype(np.float32, copy=False) if "snr" in df.columns else np.zeros(n, dtype=np.float32)
        cqi_vals = df["cqi"].to_numpy().astype(np.float32, copy=False) if "cqi" in df.columns else np.zeros(n, dtype=np.float32)
        lut2d = load_snr_cqi_lut(args.snr_cqi_lut, args.lut_snr_bin, args.lut_cqi_bin, args.lut_default_mcs)
        if lut2d is None:
            raise SystemExit("Failed to load SNR×CQI LUT")
        m_sc = np.clip(lut2d.map_arrays(snr_vals, cqi_vals), args.baseline_min_mcs, args.baseline_max_mcs).astype(np.int32)
        if args.baseline_guardrail >= 0.0:
            tau_b = float(args.baseline_guardrail)
            m_idx3 = np.arange(28, dtype=np.int32)[None, :]
            mask3 = preds >= tau_b
            best_feas3 = np.where(mask3, m_idx3, -1).max(axis=1)
            m_sc = np.where(best_feas3 >= 0, best_feas3, m_sc).astype(np.int32)
        p_sc = preds[np.arange(n), m_sc]
        s_sc = p_sc * eff[m_sc]

    # OLLA policy (sequential offset)
    delta = float(args.olla_init)
    up = step * target_bler
    down = step * (1.0 - target_bler)
    m_olla = np.empty(n, dtype=np.int32)
    p_olla = np.empty(n, dtype=np.float32)
    for i in range(n):
        m = int(np.clip(round((cqi_vals[i] if i < cqi_vals.shape[0] else 9.0) + delta), 0, 27))
        m_olla[i] = m
        p = preds[i, m]
        p_olla[i] = p
        succ = (rng.random() < p) if sample else p >= (1.0 - target_bler)
        delta = delta + (up if succ else -down)
    s_olla = p_olla * eff[m_olla]

    def summarize(p: np.ndarray, s: np.ndarray, m: np.ndarray) -> Dict[str, float]:
        return {
            "accept_rate": 1.0,
            "violation": float(1.0 - p.mean()),
            "throughput_expected": float(s.mean()),
            "throughput_realized": float(s.mean()) if not sample else float((eff[m] * (rng.random(n) < p)).mean()),
            "avg_mcs": float(m.mean()),
        }

    out = {
        "model_threshold": summarize(p_best, s_best, best),
        "model_throughput": summarize(p_tp, s_tp, tp_best),
        "baseline_cqi": summarize(p_cqi, s_cqi, m_cqi),
        "baseline_cqi_olla": summarize(p_olla, s_olla, m_olla),
    }
    if p_snr is not None and s_snr is not None and m_snr is not None:
        out["baseline_snr_table"] = summarize(p_snr, s_snr, m_snr)
    if p_sc is not None and s_sc is not None and m_sc is not None:
        out["baseline_snr_cqi_table"] = summarize(p_sc, s_sc, m_sc)
    return out


def upd(store: Dict[str, float], m: int, p: float, s: float, rng: np.random.Generator, sample: bool, ret_success: bool=False):
    store["n"] += 1
    store["mcs_avg"] += m
    store["throughput_exp"] += s
    if sample:
        succ = float(rng.random() < p)
    else:
        succ = p  # use expectation if not sampling
    store["succ"] += succ
    store["throughput_real"] += succ * spectral_efficiency(m)
    if ret_success:
        return bool(succ)


def elevation_series(
    df: pl.DataFrame,
    rec: MCSRecommender,
    step: float,
    target_bler: float,
    rng: np.random.Generator,
    sample: bool,
    bin_width: float,
    args,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # Vectorized predictions
    preds, eff = batch_predict_all_mcs(df, rec)
    n = df.height
    cqi_vals = df["cqi"].to_numpy().astype(np.float32, copy=False) if "cqi" in df.columns else np.full(n, 9.0, dtype=np.float32)
    ele_vals = df["ele_angle"].to_numpy().astype(np.float32, copy=False) if "ele_angle" in df.columns else np.zeros(n, dtype=np.float32)

    # Threshold and throughput policies
    tau = float(rec.threshold)
    m_idx = np.arange(28, dtype=np.int32)[None, :]
    mask = preds >= tau
    best_tau = np.where(mask, m_idx, -1).max(axis=1)
    fb = preds.argmax(axis=1)
    best_tau = np.where(best_tau < 0, fb, best_tau).astype(np.int32)
    p_tau = preds[np.arange(n), best_tau]

    scores = preds * eff[None, :]
    best_tp = scores.argmax(axis=1).astype(np.int32)
    p_tp = preds[np.arange(n), best_tp]

    # Baseline CQI and OLLA
    m_cqi = _map_cqi_array(cqi_vals, args)
    if args.baseline_guardrail >= 0.0:
        tau_b = float(args.baseline_guardrail)
        bfeas = np.where(preds >= tau_b, m_idx, -1).max(axis=1)
        m_cqi = np.where(bfeas >= 0, bfeas, m_cqi).astype(np.int32)
    p_cqi = preds[np.arange(n), m_cqi]

    # Baseline SNR LUT (optional)
    have_snr_lut = args.snr_lut is not None
    if have_snr_lut:
        lut = load_snr_lut(args.snr_lut, args.snr_lut_bin, args.snr_lut_default_mcs)
        m_snr = np.clip(lut.map_array(df["snr"].to_numpy().astype(np.float32, copy=False)), args.baseline_min_mcs, args.baseline_max_mcs).astype(np.int32)
        if args.baseline_guardrail >= 0.0:
            tau_b = float(args.baseline_guardrail)
            bfeas2 = np.where(preds >= tau_b, m_idx, -1).max(axis=1)
            m_snr = np.where(bfeas2 >= 0, bfeas2, m_snr).astype(np.int32)
        p_snr = preds[np.arange(n), m_snr]

    # Baseline SNR×CQI LUT (optional)
    have_snr_cqi_lut = args.snr_cqi_lut is not None
    if have_snr_cqi_lut:
        lut2d = load_snr_cqi_lut(args.snr_cqi_lut, args.lut_snr_bin, args.lut_cqi_bin, args.lut_default_mcs)
        m_sc = np.clip(lut2d.map_arrays(
            df["snr"].to_numpy().astype(np.float32, copy=False),
            df["cqi"].to_numpy().astype(np.float32, copy=False) if "cqi" in df.columns else np.zeros(n, dtype=np.float32)),
            args.baseline_min_mcs, args.baseline_max_mcs).astype(np.int32)
        if args.baseline_guardrail >= 0.0:
            tau_b = float(args.baseline_guardrail)
            bfeas3 = np.where(preds >= tau_b, m_idx, -1).max(axis=1)
            m_sc = np.where(bfeas3 >= 0, bfeas3, m_sc).astype(np.int32)
        p_sc = preds[np.arange(n), m_sc]

    delta = float(args.olla_init)
    up = step * target_bler
    down = step * (1.0 - target_bler)
    m_olla = np.empty(n, dtype=np.int32)
    p_olla = np.empty(n, dtype=np.float32)
    for i in range(n):
        m = int(np.clip(round((cqi_vals[i] if i < cqi_vals.shape[0] else 9.0) + delta), 0, 27))
        m_olla[i] = m
        p = preds[i, m]
        p_olla[i] = p
        succ = (rng.random() < p) if sample else p >= (1.0 - target_bler)
        delta = delta + (up if succ else -down)

    # Build aggregation frame
    plot_df = pl.DataFrame({
        "ele": ele_vals,
        "p_model_tau": p_tau, "m_model_tau": best_tau.astype(np.float32),
        "p_model_tp": p_tp,  "m_model_tp": best_tp.astype(np.float32),
        "p_cqi": p_cqi,      "m_cqi": m_cqi.astype(np.float32),
        "p_olla": p_olla,    "m_olla": m_olla.astype(np.float32),
        **({"p_snr": p_snr, "m_snr": m_snr.astype(np.float32)} if have_snr_lut else {}),
        **({"p_snr_cqi": p_sc, "m_snr_cqi": m_sc.astype(np.float32)} if have_snr_cqi_lut else {}),
    }).with_columns(((pl.col("ele") / bin_width).floor() * bin_width).alias("ele_bin"))
    def agg(prefix: str) -> List[pl.Expr]:
        return [
            (1 - pl.col(f"p_{prefix}")).mean().alias(f"bler_{prefix}"),
            pl.col(f"m_{prefix}").mean().alias(f"mcs_{prefix}"),
        ]
    aggs = agg("model_tau") + agg("model_tp") + agg("cqi") + agg("olla")
    if have_snr_lut:
        aggs += agg("snr")
    if have_snr_cqi_lut:
        aggs += agg("snr_cqi")
    gb = plot_df.group_by("ele_bin").agg(aggs).sort("ele_bin")
    elev = gb["ele_bin"].to_numpy()
    keys = ["model_tau","model_tp","cqi","olla"] + (["snr"] if have_snr_lut else []) + (["snr_cqi"] if have_snr_cqi_lut else [])
    bler = {k: gb[f"bler_{k}"].to_numpy() for k in keys}
    mcs  = {k: gb[f"mcs_{k}"].to_numpy() for k in keys}
    return elev, bler, mcs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CQI/OLLA baselines vs model")
    ap.add_argument("--data", type=str, default="features/test.parquet")
    ap.add_argument("--output-dir", type=str, default="reports")
    ap.add_argument("--sample", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--step", type=float, default=0.25, help="OLLA step (CQI units). Up=step×target, down=step×(1-target)")
    ap.add_argument("--target-bler", type=float, default=0.1, help="Target BLER for OLLA")
    ap.add_argument("--sample-outcomes", action="store_true", help="Sample success with Bernoulli(p) instead of using expectations")
    ap.add_argument("--bin-width", type=float, default=1.0, help="Elevation bin width (degrees)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"], help="Inference device for XGBoost model")
    # Baseline mapping knobs
    ap.add_argument("--baseline-map", type=str, default="round", choices=["round", "linear", "table"], help="CQI→MCS mapping for baselines")
    ap.add_argument("--baseline-linear-scale", type=float, default=2.7, help="Scale for linear CQI→MCS (m = scale*cqi + offset)")
    ap.add_argument("--baseline-linear-offset", type=float, default=0.0, help="Offset for linear CQI→MCS (m = scale*cqi + offset)")
    ap.add_argument("--baseline-min-mcs", type=int, default=0, help="Minimum MCS for baselines")
    ap.add_argument("--baseline-max-mcs", type=int, default=27, help="Maximum MCS for baselines")
    ap.add_argument("--baseline-guardrail", type=float, default=-1.0, help="Optional min P(pass) for baselines; if >=0, degrade MCS until met")
    ap.add_argument("--olla-init", type=float, default=0.0, help="Initial OLLA offset (CQI units)")
    # SNR LUT baseline
    ap.add_argument("--snr-lut", type=str, default=None, help="CSV file with SNR→MCS table (columns: snr,mcs). snr is rounded to --snr-lut-bin.")
    ap.add_argument("--snr-lut-bin", type=float, default=0.1, help="SNR bin size for lookup table rounding (e.g., 0.1 for x.x)")
    ap.add_argument("--snr-lut-default-mcs", type=int, default=0, help="Default MCS if an SNR key is missing (nearest key used otherwise)")
    # 2D LUT baseline (SNR×CQI)
    ap.add_argument("--snr-cqi-lut", type=str, default=None, help="CSV file with SNR×CQI→MCS table (columns: snr,cqi,mcs). snr,cqi rounded to bins.")
    ap.add_argument("--lut-snr-bin", type=float, default=0.1, help="SNR bin size for 2D LUT rounding")
    ap.add_argument("--lut-cqi-bin", type=float, default=1.0, help="CQI bin size for 2D LUT rounding")
    ap.add_argument("--lut-default-mcs", type=int, default=0, help="Default MCS for missing SNR×CQI keys (nearest along snr, then nearest cqi)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    ensure_dir(out)

    # Load data
    lf = pl.scan_parquet(args.data)
    cols = lf.columns
    need = [c for c in BASE_COLS if c in cols]
    if args.sample and args.sample > 0:
        lf = lf.head(args.sample)
    df = lf.select(need).collect(streaming=True)
    if df.height == 0:
        raise SystemExit("No rows available in input data")

    # Recommender/model
    # Recommender/model with device hint
    if args.device == "gpu":
        rec = MCSRecommender(device="cuda")
    elif args.device == "cpu":
        rec = MCSRecommender(device="cpu")
    else:
        # auto: try CUDA, fall back to CPU inside MCSRecommender
        rec = MCSRecommender()
        try:
            rec.set_device("cuda")
        except Exception:
            rec.set_device("cpu")
    rng = np.random.default_rng(args.seed)

    # Overall metrics
    summary = simulate_policies(df, rec, args.step, args.target_bler, rng, args.sample_outcomes, args)
    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2))

    # Summary bar plot
    labels = list(summary.keys())
    thr_exp = [summary[k]["throughput_expected"] for k in labels]
    viol = [summary[k]["violation"] for k in labels]
    fig, ax1 = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, thr_exp, width=0.4, label="Expected Throughput", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, viol, width=0.4, label="Violation", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Expected Throughput")
    ax2.set_ylabel("Violation Rate")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Policy Comparison: Throughput vs Violation")
    fig.tight_layout()
    fig.savefig(out / "baseline_summary.png", dpi=160)
    plt.close(fig)

    # Elevation series (0..90) — note this still does a single pass over rows
    elev, bler, mcs = elevation_series(df, rec, args.step, args.target_bler, rng, args.sample_outcomes, args.bin_width, args)
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(9.8, 6.0), sharex=True, gridspec_kw={"hspace": 0.08})
    colors = {
        "model_tau": "#2ca02c",
        "model_tp": "#9467bd",
        "cqi": "#8c564b",
        "olla": "#ff7f0e",
    }
    names = {
        "model_tau": "Model τ",
        "model_tp": "Model Throughput",
        "cqi": "Baseline CQI",
        "olla": "CQI + OLLA",
    }
    for k in ["model_tau","model_tp","cqi","olla"]:
        ax_t.plot(elev, bler[k], color=colors[k], label=names[k], linewidth=1.5)
        ax_b.plot(elev, mcs[k], color=colors[k], label=names[k], linewidth=1.5, linestyle='--')
    ax_t.set_ylabel("Avg BLER (1 − p)")
    ax_b.set_ylabel("Avg MCS")
    ax_b.set_xlabel("Elevation Angle")
    ax_t.grid(True, alpha=0.3)
    ax_b.grid(True, alpha=0.3)
    ax_t.set_xlim(0.0, 90.0)
    ax_b.set_xlim(0.0, 90.0)
    leg1 = ax_t.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    leg2 = ax_b.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
    fig.subplots_adjust(right=0.78, top=0.92)
    fig.suptitle("Policies vs Elevation (Expected)")
    fig.savefig(out / "baseline_vs_model_elev.png", dpi=160)
    plt.close(fig)

    print(f"Wrote baseline comparison to {out}")


if __name__ == "__main__":
    main()
