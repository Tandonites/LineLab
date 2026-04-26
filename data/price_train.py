"""
price_train.py
==============
Train an XGBoost regressor to predict monthly operating cost.

Expected labels CSV:
  - month (YYYY-MM or parseable datetime)
  - monthly_operating_cost (target)
  - optional: line_group

Feature sources (already in this repo):
  - processed/ridership_hourly.csv
    - processed/ridership_monthly.csv (preferred for long monthly history)
  - processed/line_summary.csv

Outputs (default: ./models):
  - cost_model.json
  - cost_feature_columns.json
  - cost_training_report.json

Usage examples:
  python3 data/price_train.py \
      --labels-csv data/raw/monthly_operating_cost.csv

  python3 data/price_train.py \
      --labels-csv data/raw/monthly_operating_cost.csv \
      --target-col monthly_operating_cost

  python3 data/price_train.py \
      --labels-csv data/raw/monthly_operating_cost.csv \
      --external-features-csv data/raw/monthly_external_features.csv

  # Development only (no real labels):
  python3 data/price_train.py --bootstrap-target
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path(__file__).resolve().parent
PROC_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"

RIDERSHIP_HOURLY_CSV = PROC_DIR / "ridership_hourly.csv"
RIDERSHIP_MONTHLY_CSV = PROC_DIR / "ridership_monthly.csv"
LINE_SUMMARY_CSV = PROC_DIR / "line_summary.csv"
DEFAULT_LABELS_CSV = RAW_DIR / "monthly_operating_cost.csv"


def normalize_month(series: pd.Series) -> pd.Series:
    """Convert a series to YYYY-MM month strings."""
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().any():
        return dt.dt.to_period("M").astype(str)

    # Already string-like month values (e.g., 2024-09)
    s = series.astype(str).str.strip()
    s = s.str.replace(r"/", "-", regex=True)
    return s


def load_hourly_features(hourly_csv: Path) -> pd.DataFrame:
    """Build monthly x line_group features from station-hour ridership records."""
    df = pd.read_csv(hourly_csv)

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' in {hourly_csv}")
    if "line_group" not in df.columns:
        raise ValueError(f"Missing 'line_group' in {hourly_csv}")

    df = df.copy()
    df["month"] = normalize_month(df["timestamp"])

    num_cols = ["ridership", "transfers", "hour", "is_weekend"]
    for col in num_cols:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in {hourly_csv}")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Weighted temporal profile features
    df["is_peak_window"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(float)
    df["is_offpeak_window"] = ((df["hour"] <= 5) | (df["hour"] >= 22)).astype(float)

    grp = df.groupby(["month", "line_group"], dropna=False)

    out = grp.agg(
        ridership_sum=("ridership", "sum"),
        ridership_mean_hour=("ridership", "mean"),
        ridership_p90_hour=("ridership", lambda x: float(np.percentile(x, 90))),
        transfers_sum=("transfers", "sum"),
        transfers_mean_hour=("transfers", "mean"),
        weekend_share=("is_weekend", "mean"),
        peak_window_share=("is_peak_window", "mean"),
        offpeak_window_share=("is_offpeak_window", "mean"),
        active_hours=("hour", "nunique"),
        samples=("hour", "size"),
    ).reset_index()

    # Derived load profile features
    out["transfer_to_rider_ratio"] = out["transfers_sum"] / np.maximum(out["ridership_sum"], 1.0)
    out["peak_to_avg_ratio"] = out["ridership_p90_hour"] / np.maximum(out["ridership_mean_hour"], 1.0)

    return out


def load_line_group_features(line_summary_csv: Path) -> pd.DataFrame:
    """Aggregate static line-level summary metrics into line_group features."""
    line = pd.read_csv(line_summary_csv)
    req = {"line_group", "total_ridership", "total_transfers", "station_count"}
    missing = req - set(line.columns)
    if missing:
        raise ValueError(f"Missing columns in {line_summary_csv}: {sorted(missing)}")

    for c in ["total_ridership", "total_transfers", "station_count"]:
        line[c] = pd.to_numeric(line[c], errors="coerce").fillna(0.0)

    agg = (
        line.groupby("line_group", dropna=False)
        .agg(
            line_count=("line_group", "size"),
            station_count_total=("station_count", "sum"),
            baseline_ridership_total=("total_ridership", "sum"),
            baseline_transfers_total=("total_transfers", "sum"),
        )
        .reset_index()
    )

    agg["baseline_transfer_ratio"] = agg["baseline_transfers_total"] / np.maximum(
        agg["baseline_ridership_total"], 1.0
    )
    return agg


def build_feature_table(hourly_csv: Path, line_summary_csv: Path) -> pd.DataFrame:
    hourly = load_hourly_features(hourly_csv)
    line_group = load_line_group_features(line_summary_csv)

    feat = hourly.merge(line_group, on="line_group", how="left")

    # Monthly system context
    monthly_totals = (
        feat.groupby("month", as_index=False)
        .agg(
            monthly_system_ridership=("ridership_sum", "sum"),
            monthly_system_transfers=("transfers_sum", "sum"),
        )
    )
    feat = feat.merge(monthly_totals, on="month", how="left")
    feat["ridership_share_in_month"] = feat["ridership_sum"] / np.maximum(
        feat["monthly_system_ridership"], 1.0
    )

    # One-hot encode line_group for model flexibility
    feat = pd.get_dummies(feat, columns=["line_group"], prefix="lg")

    return feat


def load_monthly_ridership_features(monthly_csv: Path) -> pd.DataFrame:
    """Build month-level ridership features from monthly ridership CSV."""
    df = pd.read_csv(monthly_csv)
    if "month" not in df.columns:
        raise ValueError(f"Missing 'month' in {monthly_csv}")

    # Accept either monthly_ridership or ridership column names.
    if "monthly_ridership" in df.columns:
        rid_col = "monthly_ridership"
    elif "ridership" in df.columns:
        rid_col = "ridership"
    else:
        raise ValueError(f"Missing 'monthly_ridership' or 'ridership' in {monthly_csv}")

    tmp = df.copy()
    tmp["month"] = normalize_month(tmp["month"])
    tmp[rid_col] = pd.to_numeric(tmp[rid_col], errors="coerce").fillna(0.0)

    # Some sources can contain one row per agency per month; sum to month level.
    out = tmp.groupby("month", as_index=False)[rid_col].sum()
    out = out.rename(columns={rid_col: "monthly_ridership"}).sort_values("month").reset_index(drop=True)

    out["ridership_mom_change"] = out["monthly_ridership"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["ridership_3m_avg"] = out["monthly_ridership"].rolling(3, min_periods=1).mean()
    out["ridership_6m_avg"] = out["monthly_ridership"].rolling(6, min_periods=1).mean()

    # Month-of-year seasonality encoding
    dt = pd.to_datetime(out["month"] + "-01", errors="coerce")
    month_num = dt.dt.month.fillna(1).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    out["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    return out


def load_system_line_features(line_summary_csv: Path) -> dict[str, float]:
    """Build static system-level topology/service features from line summary."""
    line = pd.read_csv(line_summary_csv)
    req = {"total_ridership", "total_transfers", "station_count"}
    missing = req - set(line.columns)
    if missing:
        raise ValueError(f"Missing columns in {line_summary_csv}: {sorted(missing)}")

    for c in ["total_ridership", "total_transfers", "station_count"]:
        line[c] = pd.to_numeric(line[c], errors="coerce").fillna(0.0)

    baseline_ridership = float(line["total_ridership"].sum())
    baseline_transfers = float(line["total_transfers"].sum())
    station_count_total = float(line["station_count"].sum())
    line_count = float(len(line))

    return {
        "line_count": line_count,
        "station_count_total": station_count_total,
        "baseline_ridership_total": baseline_ridership,
        "baseline_transfers_total": baseline_transfers,
        "baseline_transfer_ratio": baseline_transfers / max(baseline_ridership, 1.0),
    }


def build_month_level_feature_table(monthly_csv: Path, line_summary_csv: Path) -> pd.DataFrame:
    """Build month-level features from long-history monthly ridership + line summary."""
    monthly = load_monthly_ridership_features(monthly_csv)
    static = load_system_line_features(line_summary_csv)

    out = monthly.copy()
    for k, v in static.items():
        out[k] = v

    # Proxy transfer pressure rises as ridership rises relative to baseline.
    out["monthly_system_transfers_proxy"] = out["monthly_ridership"] * static["baseline_transfer_ratio"]
    out["ridership_vs_baseline"] = out["monthly_ridership"] / max(static["baseline_ridership_total"], 1.0)
    return out


def load_labels(labels_csv: Path, target_col: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    if "month" not in labels.columns:
        raise ValueError(f"Missing 'month' column in {labels_csv}")
    if target_col not in labels.columns:
        raise ValueError(f"Missing '{target_col}' column in {labels_csv}")

    labels = labels.copy()
    labels["month"] = normalize_month(labels["month"])
    labels[target_col] = pd.to_numeric(labels[target_col], errors="coerce")
    labels = labels.dropna(subset=[target_col])

    if "line_group" in labels.columns:
        labels["line_group"] = labels["line_group"].astype(str)

    return labels


def load_external_monthly_features(external_csv: Path) -> pd.DataFrame:
    """
    Load optional month-level external regressors.
    Expected: a 'month' column plus numeric feature columns.
    """
    ext = pd.read_csv(external_csv)
    if "month" not in ext.columns:
        raise ValueError(f"Missing 'month' in {external_csv}")

    ext = ext.copy()
    ext["month"] = normalize_month(ext["month"])

    cols = [c for c in ext.columns if c != "month"]
    for c in cols:
        ext[c] = pd.to_numeric(ext[c], errors="coerce")

    # Prefix to avoid accidental name clashes with built-in features.
    rename_map = {c: f"ext_{c}" for c in cols}
    ext = ext.rename(columns=rename_map)
    ext = ext.dropna(subset=["month"]).drop_duplicates(subset=["month"]).reset_index(drop=True)
    return ext


def aggregate_features_to_month(features: pd.DataFrame) -> pd.DataFrame:
    """Collapse month x line_group features to month-level features."""
    # Keep one monthly total row plus summed line-group measures.
    cols_sum = [
        "ridership_sum",
        "transfers_sum",
        "samples",
        "line_count",
        "station_count_total",
        "baseline_ridership_total",
        "baseline_transfers_total",
    ]
    cols_mean = [
        "ridership_mean_hour",
        "ridership_p90_hour",
        "transfers_mean_hour",
        "weekend_share",
        "peak_window_share",
        "offpeak_window_share",
        "active_hours",
        "transfer_to_rider_ratio",
        "peak_to_avg_ratio",
        "baseline_transfer_ratio",
        "ridership_share_in_month",
        "monthly_system_ridership",
        "monthly_system_transfers",
    ]

    lg_cols = [c for c in features.columns if c.startswith("lg_")]

    agg_map: dict[str, str] = {}
    for c in cols_sum:
        if c in features.columns:
            agg_map[c] = "sum"
    for c in cols_mean + lg_cols:
        if c in features.columns:
            agg_map[c] = "mean"

    out = features.groupby("month", as_index=False).agg(agg_map)
    return out


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def add_target_lag_features(
    data: pd.DataFrame,
    target_col: str,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Add autoregressive target features:
      - target_lag_1
      - target_lag_3_avg
    """
    out = data.sort_values("month").copy()
    if group_col and group_col in out.columns:
        out["target_lag_1"] = out.groupby(group_col)[target_col].shift(1)
        out["target_lag_3_avg"] = (
            out.groupby(group_col)[target_col]
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        out["target_lag_1"] = out[target_col].shift(1)
        out["target_lag_3_avg"] = out[target_col].shift(1).rolling(3, min_periods=1).mean()
    return out


def train_time_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[xgb.Booster, dict]:
    """Train with time-based split (last 20% months as test)."""
    data = df.sort_values("month").reset_index(drop=True)

    months = sorted(data["month"].unique())
    if len(months) < 2:
        raise ValueError("Need at least 2 distinct months for time-split training")

    test_n = max(1, int(round(len(months) * 0.2)))
    test_months = set(months[-test_n:])

    train_df = data[~data["month"].isin(test_months)].copy()
    test_df = data[data["month"].isin(test_months)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Invalid split: empty train or test set")

    X_train = train_df[feature_cols].astype(float).values
    X_test = test_df[feature_cols].astype(float).values
    y_train = train_df[target_col].astype(float).values
    y_test = test_df[target_col].astype(float).values

    y_train_log = np.log1p(np.maximum(y_train, 0.0))
    y_test_log = np.log1p(np.maximum(y_test, 0.0))

    dtr = xgb.DMatrix(X_train, label=y_train_log, feature_names=feature_cols)
    dte = xgb.DMatrix(X_test, label=y_test_log, feature_names=feature_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtr,
        num_boost_round=1200,
        evals=[(dtr, "train"), (dte, "test")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    pred_train = np.expm1(model.predict(dtr))
    pred_test = np.expm1(model.predict(dte))

    # Naive baseline for test: previous month's actual cost when available,
    # otherwise fallback to train mean.
    if "target_lag_1" in test_df.columns:
        baseline_test = test_df["target_lag_1"].astype(float).fillna(float(np.mean(y_train))).values
    else:
        baseline_test = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)

    report = {
        "n_rows": int(len(data)),
        "n_features": int(len(feature_cols)),
        "months_total": len(months),
        "test_months": sorted(test_months),
        "best_iteration": int(model.best_iteration),
        "target_transform": "log1p",
        "train": {
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
            "r2": float(r2_score(y_train, pred_train)),
            "mape_pct": mape(y_train, pred_train),
        },
        "test": {
            "mae": float(mean_absolute_error(y_test, pred_test)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "r2": float(r2_score(y_test, pred_test)),
            "mape_pct": mape(y_test, pred_test),
        },
        "baseline_test": {
            "method": "lag_1_or_train_mean",
            "mae": float(mean_absolute_error(y_test, baseline_test)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_test))),
            "r2": float(r2_score(y_test, baseline_test)),
            "mape_pct": mape(y_test, baseline_test),
        },
    }

    # Positive values indicate model improvement vs baseline.
    b_mae = report["baseline_test"]["mae"]
    b_rmse = report["baseline_test"]["rmse"]
    report["vs_baseline"] = {
        "mae_improvement_pct": float(100.0 * (b_mae - report["test"]["mae"]) / max(b_mae, 1e-9)),
        "rmse_improvement_pct": float(100.0 * (b_rmse - report["test"]["rmse"]) / max(b_rmse, 1e-9)),
    }

    gain = model.get_score(importance_type="gain")
    report["feature_importance_gain"] = sorted(
        [{"feature": k, "gain": float(v)} for k, v in gain.items()],
        key=lambda x: -x["gain"],
    )

    return model, report


def make_bootstrap_labels(features: pd.DataFrame) -> pd.DataFrame:
    """
    Development-only pseudo labels when real cost labels are unavailable.
    This is useful for pipeline testing, not production.
    """
    base = 1_200_000.0
    labels = features.copy()

    # Works for both month-level and month x group tables
    ridership = pd.to_numeric(labels.get("ridership_sum", 0), errors="coerce").fillna(0)
    transfers = pd.to_numeric(labels.get("transfers_sum", 0), errors="coerce").fillna(0)
    stations = pd.to_numeric(labels.get("station_count_total", 10), errors="coerce").fillna(10)

    labels["monthly_operating_cost"] = (
        base
        + ridership * 0.35
        + transfers * 0.65
        + stations * 45_000
    )
    return labels[["month", "monthly_operating_cost"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train monthly operating cost model")
    parser.add_argument("--ridership-hourly-csv", default=str(RIDERSHIP_HOURLY_CSV))
    parser.add_argument(
        "--ridership-monthly-csv",
        default=str(RIDERSHIP_MONTHLY_CSV),
        help="Preferred month-level ridership feature source for long history",
    )
    parser.add_argument("--line-summary-csv", default=str(LINE_SUMMARY_CSV))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--target-col", default="monthly_operating_cost")
    parser.add_argument(
        "--external-features-csv",
        default=None,
        help="Optional month-level external regressors CSV (must include month column)",
    )
    parser.add_argument(
        "--disable-target-lags",
        action="store_true",
        help="Disable target autoregressive lag features",
    )
    parser.add_argument(
        "--bootstrap-target",
        action="store_true",
        help="Use pseudo labels if labels CSV is unavailable (for pipeline testing only)",
    )
    parser.add_argument("--out-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    ridership_path = Path(args.ridership_hourly_csv)
    ridership_monthly_path = Path(args.ridership_monthly_csv)
    line_summary_path = Path(args.line_summary_csv)
    labels_path = Path(args.labels_csv)
    external_features_path = Path(args.external_features_csv) if args.external_features_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not line_summary_path.exists():
        raise FileNotFoundError(f"Missing line summary CSV: {line_summary_path}")

    features_hourly = None
    if ridership_path.exists():
        features_hourly = build_feature_table(ridership_path, line_summary_path)

    month_features = None
    if ridership_monthly_path.exists():
        month_features = build_month_level_feature_table(ridership_monthly_path, line_summary_path)
    elif features_hourly is not None:
        month_features = aggregate_features_to_month(features_hourly)
    else:
        raise FileNotFoundError(
            "Missing ridership features. Provide either --ridership-monthly-csv or --ridership-hourly-csv"
        )

    # Optional external month-level regressors
    if external_features_path is not None:
        if not external_features_path.exists():
            raise FileNotFoundError(f"Missing external features CSV: {external_features_path}")
        ext = load_external_monthly_features(external_features_path)
        month_features = month_features.merge(ext, on="month", how="left")

    # Join strategy depends on label granularity
    if labels_path.exists():
        labels = load_labels(labels_path, args.target_col)
        if "line_group" in labels.columns:
            if features_hourly is None:
                raise FileNotFoundError(
                    "line_group labels require hourly features, but --ridership-hourly-csv was not found"
                )
            # Keep month x line_group granularity if labels are provided that way.
            # Need non-dummy line_group key for join; recover from one-hot fallback by rebuilding.
            hourly_raw = load_hourly_features(ridership_path)
            line_raw = load_line_group_features(line_summary_path)
            features_for_join = hourly_raw.merge(line_raw, on="line_group", how="left")
            labels["line_group"] = labels["line_group"].astype(str)
            data = features_for_join.merge(
                labels[["month", "line_group", args.target_col]],
                on=["month", "line_group"],
                how="inner",
            )
            # Encode line_group after join
            data = pd.get_dummies(data, columns=["line_group"], prefix="lg")
            granularity = "month_line_group"
        else:
            data = month_features.merge(
                labels[["month", args.target_col]],
                on="month",
                how="inner",
            )
            granularity = "month"
    else:
        if not args.bootstrap_target:
            raise FileNotFoundError(
                f"Labels CSV not found: {labels_path}\n"
                "Provide --labels-csv with real monthly costs, or use --bootstrap-target for dev-only training."
            )
        print("[warn] Using bootstrap pseudo labels; this is not a real cost model.")
        pseudo = make_bootstrap_labels(month_features)
        data = month_features.merge(pseudo, on="month", how="left")
        granularity = "month_bootstrap"

    # Add autoregressive target features for stronger month-level forecasting.
    if not args.disable_target_lags:
        group_col = "line_group" if "line_group" in data.columns else None
        data = add_target_lag_features(data, args.target_col, group_col=group_col)

    data = data.dropna(subset=[args.target_col]).copy()

    # Drop rows that cannot support lag features (typically earliest month per series)
    if not args.disable_target_lags:
        lag_cols = [c for c in ["target_lag_1", "target_lag_3_avg"] if c in data.columns]
        if lag_cols:
            data = data.dropna(subset=lag_cols).copy()

    excluded = {"month", "timestamp", args.target_col}
    feature_cols = [c for c in data.columns if c not in excluded]

    if len(data) < 20:
        print(
            f"[warn] Very small training set ({len(data)} rows). "
            "Model quality will be unstable; collect more months of labels."
        )

    model, report = train_time_split(data, feature_cols, args.target_col)
    report["granularity"] = granularity
    report["target_col"] = args.target_col
    report["feature_columns"] = feature_cols
    report["source_files"] = {
        "ridership_hourly_csv": str(ridership_path),
        "ridership_monthly_csv": str(ridership_monthly_path) if ridership_monthly_path.exists() else None,
        "line_summary_csv": str(line_summary_path),
        "labels_csv": str(labels_path) if labels_path.exists() else None,
        "external_features_csv": str(external_features_path) if external_features_path and external_features_path.exists() else None,
    }
    report["target_lags_enabled"] = not args.disable_target_lags

    model.save_model(out_dir / "cost_model.json")
    (out_dir / "cost_feature_columns.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_cols,
                "target_col": args.target_col,
                "target_transform": "log1p",
                "granularity": granularity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "cost_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nSaved artifacts:")
    print(f"  - {out_dir / 'cost_model.json'}")
    print(f"  - {out_dir / 'cost_feature_columns.json'}")
    print(f"  - {out_dir / 'cost_training_report.json'}")
    print("\nTest metrics:")
    print(
        f"  MAE={report['test']['mae']:,.2f}  "
        f"RMSE={report['test']['rmse']:,.2f}  "
        f"MAPE={report['test']['mape_pct']:.2f}%  "
        f"R2={report['test']['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
