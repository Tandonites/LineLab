"""
cost_pull.py
============
Pull monthly operating cost labels from NY State Open Data (Socrata)
using the MTA Statement of Operations dataset.

Default source:
  - Dataset: yg77-3tkj (MTA Statement of Operations: Beginning 2019)

Default filters (subway-focused labels):
  - scenario = Actual
  - expense_type = NREIMB
  - type = Total Expenses Before Non-Cash Liability Adjs.
  - agencies = NYCT,SIR

Output:
  data/raw/monthly_operating_cost.csv with columns:
    - month (YYYY-MM)
    - monthly_operating_cost

Usage:
  python3 data/cost_pull.py
  python3 data/cost_pull.py --agencies NYCT
  python3 data/cost_pull.py --scenario "Adopted Budget"
  python3 data/cost_pull.py --out data/raw/monthly_operating_cost.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
DEFAULT_OUT = RAW_DIR / "monthly_operating_cost.csv"

DEFAULT_DATASET_ID = "yg77-3tkj"
DEFAULT_DOMAIN = "data.ny.gov"

DEFAULT_SCENARIO = "Actual"
DEFAULT_EXPENSE_TYPE = "NREIMB"
DEFAULT_TYPE = "Total Expenses Before Non-Cash Liability Adjs."
DEFAULT_AGENCIES = ["NYCT", "SIR"]


def _escape_sql(value: str) -> str:
    return value.replace("'", "''")


def build_where_clause(
    scenario: str,
    expense_type: str,
    type_value: str,
    agencies: list[str],
) -> str:
    """Build a SoQL WHERE clause from filters."""
    clauses: list[str] = []

    if scenario:
        clauses.append(f"scenario='{_escape_sql(scenario)}'")
    if expense_type:
        clauses.append(f"expense_type='{_escape_sql(expense_type)}'")
    if type_value:
        clauses.append(f"type='{_escape_sql(type_value)}'")
    if agencies:
        quoted = ",".join(f"'{_escape_sql(a)}'" for a in agencies)
        clauses.append(f"agency in ({quoted})")

    return " AND ".join(clauses)


def fetch_monthly_costs(
    domain: str,
    dataset_id: str,
    where_clause: str,
    app_token: str | None,
) -> pd.DataFrame:
    """Fetch monthly aggregated operating cost from Socrata."""
    endpoint = f"https://{domain}/resource/{dataset_id}.json"

    params = {
        "$select": "month,sum(amount) as monthly_operating_cost",
        "$where": where_clause,
        "$group": "month",
        "$order": "month",
        "$limit": 50000,
    }

    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    resp = requests.get(endpoint, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    rows = resp.json()
    if not rows:
        return pd.DataFrame(columns=["month", "monthly_operating_cost"])

    df = pd.DataFrame(rows)
    if "month" not in df.columns or "monthly_operating_cost" not in df.columns:
        raise ValueError("Unexpected response schema from Socrata API")

    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
    df["monthly_operating_cost"] = pd.to_numeric(df["monthly_operating_cost"], errors="coerce")
    df = df.dropna(subset=["month", "monthly_operating_cost"]).sort_values("month")

    return df[["month", "monthly_operating_cost"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull monthly operating cost labels from NY Open Data")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="Socrata domain (default: data.ny.gov)")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Socrata dataset ID")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="Scenario filter")
    parser.add_argument("--expense-type", default=DEFAULT_EXPENSE_TYPE, help="Expense type filter")
    parser.add_argument("--type", dest="type_value", default=DEFAULT_TYPE, help="Type filter")
    parser.add_argument(
        "--agencies",
        default=",".join(DEFAULT_AGENCIES),
        help="Comma-separated agency codes, e.g. NYCT,SIR",
    )
    parser.add_argument("--app-token", default=None, help="Optional Socrata app token")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")

    args = parser.parse_args()

    agencies = [a.strip() for a in args.agencies.split(",") if a.strip()]
    where_clause = build_where_clause(
        scenario=args.scenario,
        expense_type=args.expense_type,
        type_value=args.type_value,
        agencies=agencies,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_monthly_costs(
        domain=args.domain,
        dataset_id=args.dataset_id,
        where_clause=where_clause,
        app_token=args.app_token,
    )

    if df.empty:
        raise RuntimeError(
            "No rows returned for the selected filters. "
            "Try different agencies/scenario/type values."
        )

    df.to_csv(out_path, index=False)

    print("Saved monthly operating cost labels:")
    print(f"  {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Month range: {df['month'].min()} -> {df['month'].max()}")
    print(f"Filters: {where_clause}")


if __name__ == "__main__":
    main()
