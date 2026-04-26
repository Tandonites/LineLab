"""
cost_plot_export.py
===================
Generate presentation-ready PNG charts for the monthly cost model.

Outputs (default: data/models):
  - cost_actual_vs_pred_test.png
  - cost_monthly_trend_actual_pred.png
  - cost_feature_importance.png

Usage:
  python3 data/cost_plot_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import xgboost as xgb

from price_train import (
    add_target_lag_features,
    build_month_level_feature_table,
    load_labels,
    train_time_split,
)


DATA_DIR = Path(__file__).resolve().parent
PROC_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"

RIDERSHIP_MONTHLY_CSV = PROC_DIR / "ridership_monthly.csv"
LINE_SUMMARY_CSV = PROC_DIR / "line_summary.csv"
LABELS_CSV = RAW_DIR / "monthly_operating_cost.csv"


def _fmt_millions(x: float) -> str:
    return f"${x / 1_000_000:.0f}M"


def main() -> None:
    if not RIDERSHIP_MONTHLY_CSV.exists():
        raise FileNotFoundError(f"Missing {RIDERSHIP_MONTHLY_CSV}")
    if not LINE_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing {LINE_SUMMARY_CSV}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing {LABELS_CSV}")

    features = build_month_level_feature_table(RIDERSHIP_MONTHLY_CSV, LINE_SUMMARY_CSV)
    labels = load_labels(LABELS_CSV, "monthly_operating_cost")
    data = features.merge(labels[["month", "monthly_operating_cost"]], on="month", how="inner")

    # Match training pipeline: lag target features are part of the model input.
    data = add_target_lag_features(data, "monthly_operating_cost")
    data = data.dropna(subset=["monthly_operating_cost", "target_lag_1", "target_lag_3_avg"]).copy()

    # Use the exact saved feature order expected by the trained model.
    feat_meta_path = MODELS_DIR / "cost_feature_columns.json"
    if feat_meta_path.exists():
        feat_meta = json.loads(feat_meta_path.read_text(encoding="utf-8"))
        feature_cols = list(feat_meta.get("feature_columns", []))
    else:
        excluded = {"month", "monthly_operating_cost", "timestamp"}
        feature_cols = [c for c in data.columns if c not in excluded]

    # Zero-fill any missing columns so DMatrix schema always matches model schema.
    for c in feature_cols:
        if c not in data.columns:
            data[c] = 0.0

    # Re-run the same split logic used in training for consistent visuals.
    _, report = train_time_split(data, feature_cols, "monthly_operating_cost")
    test_months = set(report.get("test_months", []))

    # Load saved model for prediction outputs.
    model = xgb.Booster()
    model.load_model(str(MODELS_DIR / "cost_model.json"))

    plot_df = data.sort_values("month").reset_index(drop=True).copy()
    X = plot_df[feature_cols].astype(float).values
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    plot_df["pred"] = np.expm1(model.predict(dmat))
    plot_df["is_test"] = plot_df["month"].isin(test_months)

    # 1) Scatter: actual vs predicted (test set)
    test = plot_df[plot_df["is_test"]].copy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        test["monthly_operating_cost"],
        test["pred"],
        c="#2563eb",
        alpha=0.85,
        s=60,
        edgecolors="white",
        linewidths=0.6,
        label="Test months",
    )
    lo = min(test["monthly_operating_cost"].min(), test["pred"].min())
    hi = max(test["monthly_operating_cost"].max(), test["pred"].max())
    ax.plot([lo, hi], [lo, hi], color="#ef4444", linestyle="--", linewidth=1.8, label="Perfect fit")
    ax.set_title("Monthly Cost Model: Actual vs Predicted (Test)", fontsize=13, pad=12)
    ax.set_xlabel("Actual Monthly Cost")
    ax.set_ylabel("Predicted Monthly Cost")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "cost_actual_vs_pred_test.png", dpi=220)
    plt.close(fig)

    # 2) Trend line: actual vs predicted over months
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(plot_df["month"], plot_df["monthly_operating_cost"], color="#0f766e", linewidth=2.2, label="Actual")
    ax.plot(plot_df["month"], plot_df["pred"], color="#ea580c", linewidth=2.0, label="Predicted")

    # Highlight test window
    test_idx = np.where(plot_df["is_test"].values)[0]
    if len(test_idx) > 0:
        ax.axvspan(test_idx.min(), test_idx.max(), color="#f59e0b", alpha=0.12, label="Test window")

    ax.set_title("Monthly Operating Cost: Actual vs Predicted Over Time", fontsize=13, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Cost")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    # Use sparse x ticks for readability
    n = len(plot_df)
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([plot_df.loc[i, "month"] for i in ticks], rotation=45, ha="right")

    # Format y tick labels in millions
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _fmt_millions(y)))

    fig.tight_layout()
    fig.savefig(MODELS_DIR / "cost_monthly_trend_actual_pred.png", dpi=220)
    plt.close(fig)

    # 3) Feature importance bar chart
    imp = report.get("feature_importance_gain", [])
    imp_df = pd.DataFrame(imp)
    if not imp_df.empty:
        top = imp_df.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.barh(top["feature"], top["gain"], color="#7c3aed")
        ax.set_title("Cost Model Feature Importance (Gain)", fontsize=13, pad=12)
        ax.set_xlabel("Gain")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(MODELS_DIR / "cost_feature_importance.png", dpi=220)
        plt.close(fig)

    # Save a compact chart metadata file for slide notes.
    meta = {
        "test_metrics": report.get("test", {}),
        "train_metrics": report.get("train", {}),
        "test_months": report.get("test_months", []),
        "n_rows": report.get("n_rows"),
        "n_features": report.get("n_features"),
    }
    (MODELS_DIR / "cost_plot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved PNG charts:")
    print(f"  - {MODELS_DIR / 'cost_actual_vs_pred_test.png'}")
    print(f"  - {MODELS_DIR / 'cost_monthly_trend_actual_pred.png'}")
    print(f"  - {MODELS_DIR / 'cost_feature_importance.png'}")


if __name__ == "__main__":
    main()
