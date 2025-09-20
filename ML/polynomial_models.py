"""Fit polynomial regression and Newton interpolation models for MCS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sympy as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

FEATURE_SETS = {
    "snr_cqi": ["snr", "cqi"],
    "snr_cqi_ele": ["snr", "cqi", "ele_angle"],
}


def load_dataset(path: Path, sample: int | None) -> pl.DataFrame:
    lf = pl.scan_parquet(path)
    cols = set(lf.collect_schema().names())
    required = {"snr", "cqi", "ele_angle", "mcs"}
    missing = required - cols
    if missing:
        raise SystemExit(f"Dataset missing columns: {sorted(missing)}")
    lf = lf.select(["snr", "cqi", "ele_angle", "mcs"])
    lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in ["snr", "cqi", "ele_angle", "mcs"]]))
    if sample and sample > 0:
        lf = lf.head(sample)
    return lf.collect()


def fit_polynomial_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    degree: int,
) -> Tuple[LinearRegression, PolynomialFeatures, Dict[str, float]]:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
    }
    return model, poly, metrics


def polynomial_formula(
    poly: PolynomialFeatures, model: LinearRegression, feature_names: List[str]
) -> str:
    terms = poly.get_feature_names_out(feature_names)
    coefs = model.coef_
    intercept = model.intercept_
    pieces = [f"{intercept:.6f}"]
    for coef, term in zip(coefs, terms):
        if term == "1" or abs(coef) < 1e-12:
            continue
        pieces.append(f"({coef:.6f})*{term}")
    return " + ".join(pieces)


def newton_interpolation(
    data: pl.DataFrame,
    feature_names: List[str],
    target_name: str,
    grid_counts: List[int],
    neighbors: int = 200,
) -> Tuple[sp.Expr, Dict[str, float]]:
    symbols = sp.symbols(" ".join(feature_names))
    arrays = []
    stats: Dict[str, float] = {}
    feature_matrix = np.column_stack([data[f].to_numpy() for f in feature_names]).astype(np.float64)
    target = data[target_name].to_numpy().astype(np.float64)
    scales = feature_matrix.max(axis=0) - feature_matrix.min(axis=0)
    scales[scales == 0] = 1.0
    for name, count in zip(feature_names, grid_counts):
        values = np.linspace(float(data[name].min()), float(data[name].max()), count)
        arrays.append(values)
        stats[f"{name}_min"] = float(data[name].min())
        stats[f"{name}_max"] = float(data[name].max())

    def estimate(coord: np.ndarray) -> float:
        deltas = (feature_matrix - coord) / scales
        dist = np.linalg.norm(deltas, axis=1)
        k = min(neighbors, dist.size)
        idx = np.argpartition(dist, k - 1)[:k]
        chosen = dist[idx]
        weights = 1.0 / (chosen + 1e-6)
        return float(np.average(target[idx], weights=weights))

    def build(level: int, prefix: List[float]) -> sp.Expr:
        if level == len(feature_names):
            coord = np.array(prefix, dtype=np.float64)
            return sp.Float(estimate(coord))
        points = []
        symbol = symbols[level]
        for value in arrays[level]:
            expr = build(level + 1, prefix + [float(value)])
            points.append((float(value), expr))
        return sp.interpolate(points, symbol)

    expr = build(0, [])
    return sp.expand(expr), stats


def plot_surface(
    X: np.ndarray,
    y_true: np.ndarray,
    model: LinearRegression,
    poly: PolynomialFeatures,
    feature_names: List[str],
    out_dir: Path,
    suffix: str,
) -> None:
    if X.shape[1] == 2:
        snr, cqi = X[:, 0], X[:, 1]
        grid_snr = np.linspace(snr.min(), snr.max(), 50)
        grid_cqi = np.linspace(cqi.min(), cqi.max(), 50)
        mesh = np.meshgrid(grid_snr, grid_cqi)
        grid = np.column_stack([mesh[0].ravel(), mesh[1].ravel()])
        preds = model.predict(poly.transform(grid)).reshape(mesh[0].shape)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(mesh[0], mesh[1], preds, cmap="viridis", alpha=0.8)
        ax.scatter(snr, cqi, y_true, s=5, c="red", alpha=0.3)
        ax.set_xlabel("SNR")
        ax.set_ylabel("CQI")
        ax.set_zlabel("MCS")
        ax.set_title(f"Polynomial regression surface ({suffix})")
        fig.tight_layout()
        fig.savefig(out_dir / f"poly_surface_{suffix}.png", dpi=200)
        plt.close(fig)
    elif X.shape[1] == 3:
        fig = plt.figure(figsize=(12, 5))
        for idx, name in enumerate(feature_names):
            ax = fig.add_subplot(1, 3, idx + 1)
            other_idx = [i for i in range(3) if i != idx]
            grid_primary = np.linspace(X[:, idx].min(), X[:, idx].max(), 50)
            second = np.linspace(X[:, other_idx[0]].min(), X[:, other_idx[0]].max(), 50)
            mesh = np.meshgrid(grid_primary, second)
            grid = np.zeros((mesh[0].size, 3), dtype=np.float32)
            grid[:, idx] = mesh[0].ravel()
            grid[:, other_idx[0]] = mesh[1].ravel()
            grid[:, other_idx[1]] = float(np.median(X[:, other_idx[1]]))
            preds = model.predict(poly.transform(grid)).reshape(mesh[0].shape)
            im = ax.contourf(mesh[0], mesh[1], preds, levels=20, cmap="viridis")
            ax.set_xlabel(name)
            ax.set_ylabel(feature_names[other_idx[0]])
            fig.colorbar(im, ax=ax, label="MCS")
        fig.suptitle(f"Polynomial regression slices ({suffix})")
        fig.tight_layout()
        fig.savefig(out_dir / f"poly_contours_{suffix}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polynomial models for MCS")
    parser.add_argument("--data", type=Path, default=Path("features/all.parquet"))
    parser.add_argument("--sample", type=int, default=400000)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--grid", type=str, default="5,5,5", help="Grid counts for Newton interpolation (comma-separated, trimmed by feature count)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/polynomial"))
    parser.add_argument("--snr-step", type=float, default=0.2, help="Step size for SNR grid in LUT output")
    parser.add_argument("--cqi-step", type=float, default=1.0, help="Step size for CQI grid in LUT output")
    parser.add_argument("--ele-step", type=float, default=2.0, help="Step size for elevation grid in LUT output")
    parser.add_argument("--lut-dir", type=Path, default=Path("data"), help="Directory to write polynomial-based LUTs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data, args.sample)
    grid_counts = [int(v) for v in args.grid.split(",")]

    summary = {}
    artifacts: Dict[str, Dict[str, object]] = {}

    for key, features in FEATURE_SETS.items():
        X = np.column_stack([data[f].to_numpy().astype(np.float32, copy=False) for f in features])
        y = data["mcs"].to_numpy().astype(np.float32, copy=False)
        model, poly, metrics = fit_polynomial_regression(X, y, features, args.poly_degree)
        formula = polynomial_formula(poly, model, features)
        plot_surface(X, y, model, poly, features, args.output_dir, key)
        expr, stats = newton_interpolation(data, features, "mcs", grid_counts[: len(features)])
        artifacts[key] = {
            "features": features,
            "reg_model": model,
            "poly": poly,
            "expr": expr,
            "stats": stats,
        }
        summary[key] = {
            "features": features,
            "polynomial_regression": {
                "degree": args.poly_degree,
                "metrics": metrics,
                "formula": formula,
            },
            "newton_interpolation": {
                "expression": sp.simplify(expr),
                "domain": stats,
            },
        }

    args.lut_dir.mkdir(parents=True, exist_ok=True)
    for key, props in artifacts.items():
        features = props["features"]
        model: LinearRegression = props["reg_model"]
        poly: PolynomialFeatures = props["poly"]
        expr: sp.Expr = props["expr"]
        stats = props["stats"]
        if key == "snr_cqi":
            snr_vals = np.arange(stats["snr_min"], stats["snr_max"] + args.snr_step / 2, args.snr_step)
            cqi_vals = np.arange(stats["cqi_min"], stats["cqi_max"] + args.cqi_step / 2, args.cqi_step)
            grid_snr, grid_cqi = np.meshgrid(snr_vals, cqi_vals, indexing="ij")
            grid = np.column_stack([grid_snr.ravel(), grid_cqi.ravel()])
            reg_pred = model.predict(poly.transform(grid))
            lamb = sp.lambdify(sp.symbols("snr cqi"), expr, "numpy")
            newton_pred = lamb(grid[:, 0], grid[:, 1])
            reg_clamped = np.clip(np.rint(reg_pred), 0, 27).astype(int)
            newton_clamped = np.clip(np.rint(newton_pred), 0, 27).astype(int)
            table_poly = pl.DataFrame(
                {
                    "snr": grid[:, 0],
                    "cqi": grid[:, 1],
                    "mcs": reg_clamped,
                }
            )
            table_newton = pl.DataFrame(
                {
                    "snr": grid[:, 0],
                    "cqi": grid[:, 1],
                    "mcs": newton_clamped,
                }
            )
            table_poly.write_csv(args.lut_dir / "snr_cqi_poly_lut.csv")
            table_newton.write_csv(args.lut_dir / "snr_cqi_newton_lut.csv")
        else:
            snr_vals = np.arange(stats["snr_min"], stats["snr_max"] + args.snr_step / 2, args.snr_step)
            cqi_vals = np.arange(stats["cqi_min"], stats["cqi_max"] + args.cqi_step / 2, args.cqi_step)
            ele_vals = np.arange(stats["ele_angle_min"], stats["ele_angle_max"] + args.ele_step / 2, args.ele_step)
            grid_snr, grid_cqi, grid_ele = np.meshgrid(snr_vals, cqi_vals, ele_vals, indexing="ij")
            grid = np.column_stack([grid_snr.ravel(), grid_cqi.ravel(), grid_ele.ravel()])
            reg_pred = model.predict(poly.transform(grid))
            lamb = sp.lambdify(sp.symbols("snr cqi ele_angle"), expr, "numpy")
            newton_pred = lamb(grid[:, 0], grid[:, 1], grid[:, 2])
            reg_clamped = np.clip(np.rint(reg_pred), 0, 27).astype(int)
            newton_clamped = np.clip(np.rint(newton_pred), 0, 27).astype(int)
            table_poly = pl.DataFrame(
                {
                    "snr": grid[:, 0],
                    "cqi": grid[:, 1],
                    "ele_angle": grid[:, 2],
                    "mcs": reg_clamped,
                }
            )
            table_newton = pl.DataFrame(
                {
                    "snr": grid[:, 0],
                    "cqi": grid[:, 1],
                    "ele_angle": grid[:, 2],
                    "mcs": newton_clamped,
                }
            )
            table_poly.write_csv(args.lut_dir / "snr_cqi_ele_poly_lut.csv")
            table_newton.write_csv(args.lut_dir / "snr_cqi_ele_newton_lut.csv")

    # Convert sympy expressions to strings for JSON
    for key in summary:
        summary[key]["newton_interpolation"]["expression"] = str(summary[key]["newton_interpolation"]["expression"])

    output_path = args.output_dir / "polynomial_models.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
