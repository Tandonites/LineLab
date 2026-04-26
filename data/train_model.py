"""
train_model.py
================
Trains TWO XGBoost regressors:

  1. ridership_model.json       — predicts daily ridership per station
  2. peak_factor_model.json     — predicts peak-hour fraction per station
                                  (peak_hour_ridership / daily_ridership)

Both models are trained on raw/station_features.csv (output of
fetch_features.py).

The peak factor target requires hourly ridership data — set
--hourly-csv path to point at MTA hourly CSV. If missing, only
the daily model trains and a constant peak factor (~0.10) is used.

Outputs (in ./models/):
  ridership_model.json
  peak_factor_model.json     (only if --hourly-csv given)
  feature_columns.json
  training_report.json
  scatter_pred_vs_actual.png

Usage:
  python3 train_model.py
  python3 train_model.py --hourly-csv raw/MTA_Subway_Hourly_Ridership__*.csv
  python3 train_model.py --cv  # adds 5-fold cross-validation

Requirements:
  pip install xgboost scikit-learn matplotlib pandas numpy
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


RAW_DIR    = Path("./raw")
PROC_DIR   = Path("./processed")
MODELS_DIR = Path("./models")


NUMERIC_FEATURES = [
    "num_lines",
    "lat",
    "lon",
    "pop_density_tract",
    "population_500m",
    "median_income",
    "commuters_tract",
    "bus_stops_250m",
    "bus_stops_500m",
]
TARGET = "total_ridership"


# ─────────────────────────────────────────────────────────────
# Feature matrix
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, feature_cols: list = None) -> tuple:
    """Build X, y, feature_cols. If feature_cols passed, use that exact column order."""
    df = df.copy()
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "borough" in df.columns:
        df = pd.get_dummies(df, columns=["borough"], prefix="boro")

    if feature_cols is None:
        boro_cols = sorted([c for c in df.columns if c.startswith("boro_")])
        feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + boro_cols

    # Ensure all expected columns exist (zero-fill missing)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_cols].astype(float).values
    return X, feature_cols


# ─────────────────────────────────────────────────────────────
# Peak factor target — needs hourly ridership data
# ─────────────────────────────────────────────────────────────

def compute_peak_factors(hourly_csv: str, station_ids: pd.Series) -> dict:
    """
    For each station, compute peak_factor = max_hourly / avg_daily.

    Returns dict: {station_complex_id: peak_factor}
    """
    print(f"\nComputing peak factors from {hourly_csv}...")

    cap = 5_000_000  # cap rows for speed
    station_hour_totals = defaultdict(lambda: defaultdict(int))   # sid -> hour -> ridership
    station_day_count   = defaultdict(set)                        # sid -> set of dates

    with open(hourly_csv) as f:
        reader = pd.read_csv(f, chunksize=200_000)
        rows_seen = 0
        for chunk in reader:
            if rows_seen >= cap:
                break
            for _, row in chunk.iterrows():
                sid = str(row["station_complex_id"]).strip()
                ts  = str(row["transit_timestamp"])
                # extract hour from "09/30/2024 01:00:00 AM"
                m = re.match(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}):", ts)
                if not m:
                    continue
                date, hr_str = m.group(1), m.group(2)
                am_pm = "PM" if "PM" in ts else "AM"
                hr = int(hr_str)
                if am_pm == "PM" and hr != 12:
                    hr += 12
                if am_pm == "AM" and hr == 12:
                    hr = 0
                rid = int(float(row["ridership"] or 0))
                station_hour_totals[sid][hr] += rid
                station_day_count[sid].add(date)
            rows_seen += len(chunk)
            if rows_seen % 1_000_000 == 0:
                print(f"  {rows_seen:,} rows processed...")

    peak_factors = {}
    for sid in station_ids.astype(str):
        if sid not in station_hour_totals:
            continue
        hour_totals = station_hour_totals[sid]
        n_days = len(station_day_count[sid]) or 1
        # Average ridership per hour-of-day across all observed days
        hour_avgs = {h: total / n_days for h, total in hour_totals.items()}
        peak_hour_avg = max(hour_avgs.values())
        daily_avg = sum(hour_avgs.values())
        if daily_avg > 0:
            peak_factors[sid] = peak_hour_avg / daily_avg

    print(f"  Computed peak factors for {len(peak_factors):,} stations")
    if peak_factors:
        vals = list(peak_factors.values())
        print(f"  Range: {min(vals):.3f} – {max(vals):.3f}, median: {np.median(vals):.3f}")
    return peak_factors


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "learning_rate":    0.05,
    "max_depth":        4,
    "min_child_weight": 3,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "rmse",
    "seed":             42,
}


def train_one(X, y, feature_cols, name, log_target=True, test_size=0.2):
    """Train one XGBoost regressor, return (model, metrics_dict)."""
    y_t = np.log1p(y) if log_target else y

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_t, test_size=test_size, random_state=42
    )

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dte = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)

    print(f"\n=== Training: {name} (log_target={log_target}) ===")
    model = xgb.train(
        XGB_PARAMS, dtr, num_boost_round=1000,
        evals=[(dtr, "train"), (dte, "test")],
        early_stopping_rounds=30, verbose_eval=100,
    )

    pred_tr = model.predict(dtr)
    pred_te = model.predict(dte)
    if log_target:
        pred_tr = np.expm1(pred_tr); pred_te = np.expm1(pred_te)
        y_tr_real = np.expm1(y_tr); y_te_real = np.expm1(y_te)
    else:
        y_tr_real = y_tr; y_te_real = y_te

    metrics = {
        "model":          name,
        "log_target":     log_target,
        "n_samples":      int(len(X)),
        "n_features":     int(len(feature_cols)),
        "best_iteration": int(model.best_iteration),
        "train": {
            "mae":  float(mean_absolute_error(y_tr_real, pred_tr)),
            "rmse": float(np.sqrt(mean_squared_error(y_tr_real, pred_tr))),
            "r2":   float(r2_score(y_tr_real, pred_tr)),
        },
        "test": {
            "mae":  float(mean_absolute_error(y_te_real, pred_te)),
            "rmse": float(np.sqrt(mean_squared_error(y_te_real, pred_te))),
            "r2":   float(r2_score(y_te_real, pred_te)),
        },
    }

    print(f"  Train: MAE={metrics['train']['mae']:>12,.2f}  R²={metrics['train']['r2']:.4f}")
    print(f"  Test:  MAE={metrics['test']['mae']:>12,.2f}  R²={metrics['test']['r2']:.4f}")

    importance = model.get_score(importance_type="gain")
    metrics["feature_importance"] = sorted(
        [{"feature": k, "gain": float(v)} for k, v in importance.items()],
        key=lambda x: -x["gain"],
    )

    return model, metrics, (y_te_real, pred_te)


def cross_validate(X, y, params, log_target=True, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes, r2s = [], []
    y_t = np.log1p(y) if log_target else y
    for tr, te in kf.split(X):
        dtr = xgb.DMatrix(X[tr], label=y_t[tr])
        dte = xgb.DMatrix(X[te], label=y_t[te])
        m = xgb.train(params, dtr, num_boost_round=500,
                      evals=[(dte, "val")], verbose_eval=False,
                      early_stopping_rounds=20)
        pred = m.predict(dte)
        if log_target:
            pred = np.expm1(pred)
            actual = np.expm1(y_t[te])
        else:
            actual = y_t[te]
        maes.append(mean_absolute_error(actual, pred))
        r2s.append(r2_score(actual, pred))
    return {
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "r2_mean":  float(np.mean(r2s)),  "r2_std":  float(np.std(r2s)),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",   default=str(RAW_DIR / "station_features.csv"))
    p.add_argument("--hourly-csv", default=None,
                   help="MTA hourly ridership CSV — needed to train peak_factor_model")
    p.add_argument("--out-dir",    default=str(MODELS_DIR))
    p.add_argument("--test-size",  type=float, default=0.2)
    p.add_argument("--cv", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load station features ────────────────────────────────
    df = pd.read_csv(args.features)
    print(f"Loaded {len(df):,} stations")
    df = df[df[TARGET] > 0].reset_index(drop=True)
    print(f"After dropping zero-ridership: {len(df):,} stations")

    # ── Train daily ridership model ──────────────────────────
    X, feature_cols = build_feature_matrix(df)
    y_daily = df[TARGET].astype(float).values

    rid_model, rid_metrics, rid_eval = train_one(
        X, y_daily, feature_cols, "ridership_model", log_target=True,
        test_size=args.test_size,
    )

    if args.cv:
        cv = cross_validate(X, y_daily, XGB_PARAMS, log_target=True)
        rid_metrics["cv"] = cv
        print(f"\n  5-fold CV: MAE={cv['mae_mean']:,.0f} ± {cv['mae_std']:,.0f}  "
              f"R²={cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")

    # ── Train peak factor model (optional) ───────────────────
    pf_metrics = None
    pf_model = None
    if args.hourly_csv and Path(args.hourly_csv).exists():
        peak_factors = compute_peak_factors(args.hourly_csv, df["station_complex_id"])
        df["peak_factor"] = df["station_complex_id"].astype(str).map(peak_factors)
        df_pf = df.dropna(subset=["peak_factor"]).reset_index(drop=True)

        if len(df_pf) > 30:
            X_pf, _ = build_feature_matrix(df_pf, feature_cols=feature_cols)
            y_pf = df_pf["peak_factor"].astype(float).values
            pf_model, pf_metrics, _ = train_one(
                X_pf, y_pf, feature_cols, "peak_factor_model", log_target=False,
                test_size=args.test_size,
            )
        else:
            print(f"  [warn] Only {len(df_pf)} stations with peak factor data — skipping model")
    else:
        print("\n[info] No --hourly-csv provided. Inference will use a constant peak factor (0.10).")

    # ── Save artifacts ───────────────────────────────────────
    out = Path(args.out_dir)

    rid_model.save_model(out / "ridership_model.json")
    print(f"\nSaved -> {out / 'ridership_model.json'}")

    if pf_model is not None:
        pf_model.save_model(out / "peak_factor_model.json")
        print(f"Saved -> {out / 'peak_factor_model.json'}")

    with open(out / "feature_columns.json", "w") as f:
        json.dump({
            "feature_columns":  feature_cols,
            "log_target_daily": True,
            "log_target_peak":  False,
            "fallback_peak_factor": 0.10,
        }, f, indent=2)

    report = {
        "ridership_model":   rid_metrics,
        "peak_factor_model": pf_metrics,
    }
    with open(out / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Diagnostic plot
    y_te_real, pred_te = rid_eval
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te_real, pred_te, alpha=0.6, s=30, color="coral")
    lim = max(y_te_real.max(), pred_te.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
    ax.set_xlabel("Actual daily ridership")
    ax.set_ylabel("Predicted daily ridership")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"ridership_model — test R²={rid_metrics['test']['r2']:.3f}")
    plt.tight_layout()
    plt.savefig(out / "scatter_pred_vs_actual.png", dpi=120)
    print(f"Saved -> {out / 'scatter_pred_vs_actual.png'}")


if __name__ == "__main__":
    main()