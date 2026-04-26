"""
ridership_pull.py
=================
Pull long-history monthly subway ridership from NY State Open Data and save
it as model features for monthly cost training.

Default source:
  - Dataset: xfre-bxip (MTA Monthly Ridership / Traffic Data)

Default filters:
  - agencies = NYCT,SIR

Output:
  data/processed/ridership_monthly.csv with columns:
    - month (YYYY-MM)
    - monthly_ridership

Usage:
  python3 data/ridership_pull.py
  python3 data/ridership_pull.py --agencies NYCT
  python3 data/ridership_pull.py --out data/processed/ridership_monthly.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parent
PROC_DIR = DATA_DIR / "processed"
DEFAULT_OUT = PROC_DIR / "ridership_monthly.csv"

DEFAULT_DOMAIN = "data.ny.gov"
DEFAULT_DATASET_ID = "xfre-bxip"
DEFAULT_AGENCIES = ["NYCT", "SIR"]


def _escape_sql(value: str) -> str:
    return value.replace("'", "''")


def build_where_clause(agencies: list[str]) -> str:
    if not agencies:
        return ""
    quoted = ",".join(f"'{_escape_sql(a)}'" for a in agencies)
    return f"agency in ({quoted})"


def fetch_monthly_ridership(
    domain: str,
    dataset_id: str,
    where_clause: str,
    app_token: str | None,
) -> pd.DataFrame:
    endpoint = f"https://{domain}/resource/{dataset_id}.json"

    params = {
        "$select": "month,sum(ridership) as monthly_ridership",
        "$group": "month",
        "$order": "month",
        "$limit": 50000,
    }
    if where_clause:
        params["$where"] = where_clause

    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    resp = requests.get(endpoint, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    rows = resp.json()
    if not rows:
        return pd.DataFrame(columns=["month", "monthly_ridership"])

    df = pd.DataFrame(rows)
    if "month" not in df.columns or "monthly_ridership" not in df.columns:
        raise ValueError("Unexpected response schema from Socrata API")

    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    df["monthly_ridership"] = pd.to_numeric(df["monthly_ridership"], errors="coerce")
    df = df.dropna(subset=["month", "monthly_ridership"]).sort_values("month")

    return df[["month", "monthly_ridership"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull monthly subway ridership history")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--agencies",
        default=",".join(DEFAULT_AGENCIES),
        help="Comma-separated agency codes, e.g. NYCT,SIR",
    )
    parser.add_argument("--app-token", default=None, help="Optional Socrata app token")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    args = parser.parse_args()

    agencies = [a.strip() for a in args.agencies.split(",") if a.strip()]
    where_clause = build_where_clause(agencies)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_monthly_ridership(
        domain=args.domain,
        dataset_id=args.dataset_id,
        where_clause=where_clause,
        app_token=args.app_token,
    )

    if df.empty:
        raise RuntimeError("No monthly ridership rows returned for selected filters")

    df.to_csv(out_path, index=False)

    print("Saved monthly ridership features:")
    print(f"  {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Month range: {df['month'].min()} -> {df['month'].max()}")
    print(f"Filters: {where_clause if where_clause else '[none]'}")


if __name__ == "__main__":
    main()
