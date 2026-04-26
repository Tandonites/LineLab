"""
fetch_bus_data.py
=================
Download MTA bus ridership records from NYC Open Data (Socrata)
and save them as a CSV for bus_parser.py.

Output:
  ./raw/MTA_Bus_Hourly_Ridership.csv

Usage:
  python3 fetch_bus_data.py --resource-id YOUR_RESOURCE_ID

Examples:
  python3 fetch_bus_data.py --resource-id xxxx-xxxx
  python3 fetch_bus_data.py --resource-id xxxx-xxxx --where "transit_timestamp >= '2024-01-01T00:00:00'"

Notes:
  - If you have an app token, export SOCRATA_APP_TOKEN for better reliability.
  - This script paginates the dataset and writes one consolidated CSV.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import requests


RAW_DIR = Path("./raw")
DEFAULT_OUTPUT = RAW_DIR / "MTA_Bus_Hourly_Ridership.csv"


def build_url(domain: str, resource_id: str) -> str:
    return f"https://{domain}/resource/{resource_id}.csv"


def fetch_page(
    session: requests.Session,
    url: str,
    limit: int,
    offset: int,
    where: str | None,
) -> list[list[str]]:
    params: dict[str, str | int] = {
        "$limit": limit,
        "$offset": offset,
        "$order": ":id",
    }
    if where:
        params["$where"] = where

    resp = session.get(url, params=params, timeout=90)
    resp.raise_for_status()

    rows = list(csv.reader(io.StringIO(resp.text)))
    return rows


def download_bus_data(
    resource_id: str,
    output_path: Path,
    domain: str,
    page_size: int,
    where: str | None,
    app_token: str | None,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = build_url(domain, resource_id)
    offset = 0
    total_rows = 0
    wrote_header = False

    headers: dict[str, str] = {"Accept": "text/csv"}
    if app_token:
        headers["X-App-Token"] = app_token

    print(f"Fetching bus ridership from {url}")
    print(f"Page size: {page_size:,}")
    if where:
        print(f"Filter: {where}")

    session = requests.Session()
    session.headers.update(headers)

    with output_path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)

        while True:
            rows = fetch_page(
                session=session,
                url=url,
                limit=page_size,
                offset=offset,
                where=where,
            )

            if not rows:
                break

            header = rows[0]
            data = rows[1:]

            if not wrote_header:
                writer.writerow(header)
                wrote_header = True

            if not data:
                break

            writer.writerows(data)

            batch_count = len(data)
            total_rows += batch_count
            offset += batch_count

            print(f"  Downloaded {total_rows:,} rows...")

            if batch_count < page_size:
                break

    print(f"Done. Saved {total_rows:,} rows -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MTA bus ridership CSV from NYC Open Data")
    parser.add_argument(
        "--resource-id",
        default=os.getenv("MTA_BUS_RESOURCE_ID", ""),
        help="NYC Open Data resource id (example: abcd-1234). Can also be set via MTA_BUS_RESOURCE_ID.",
    )
    parser.add_argument(
        "--domain",
        default="data.ny.gov",
        help="Socrata domain (default: data.ny.gov)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50_000,
        help="Rows per page request (default: 50000)",
    )
    parser.add_argument(
        "--where",
        default=None,
        help="Optional SoQL where clause",
    )
    parser.add_argument(
        "--app-token",
        default=os.getenv("SOCRATA_APP_TOKEN", ""),
        help="Optional Socrata app token (or set SOCRATA_APP_TOKEN)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.resource_id:
        raise SystemExit(
            "Missing resource id. Provide --resource-id or set MTA_BUS_RESOURCE_ID."
        )

    try:
        download_bus_data(
            resource_id=args.resource_id,
            output_path=Path(args.output),
            domain=args.domain,
            page_size=args.page_size,
            where=args.where,
            app_token=args.app_token or None,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"Download failed: {exc}") from exc


if __name__ == "__main__":
    main()
