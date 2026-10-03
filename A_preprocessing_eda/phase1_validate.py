"""Independent checks for the Phase 1 cleaned supply-chain file.

Column lists are repeated here on purpose. This script does not import the
cleaning pipeline. It re-reads the raw CSV and the cleaned CSV and exits
non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ORDER_DATE_COL = "order date (DateOrders)"
ISO_FORMAT = "%Y-%m-%d %H:%M:%S"
WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

# Step 2 drops, plus the two exact-duplicate columns removed in Step 4.
FORBIDDEN_COLUMNS = [
    "Product Description",
    "Order Zipcode",
    "Product Status",
    "Product Image",
    "Customer Password",
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Category Id",
    "Department Id",
    "Product Card Id",
    "Product Category Id",
    "Order Item Cardprod Id",
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    "Sales per customer",
    "Product Price",
]

REQUIRED_COLUMNS = [
    "Benefit per order",
    "Late_delivery_risk",
    "Sales",
    "Order Item Quantity",
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Type",
    "Department Name",
    "Order Region",
    "Category Name",
    "Order State",
    "Order Country",
    "Order Status",
    "order date (DateOrders)",
    "order_year",
    "order_month",
    "order_weekday",
    "order_weekday_name",
    "Sales_log",
    "Benefit_signed_log",
    "Category Name_grouped",
    "Order State_grouped",
    "Order Country_grouped",
    "Customer City",
    "Customer Country",
    "Customer State",
    "Customer Zipcode",
    "Order City",
    "Product Name",
    "late",
    "loss",
    "Latitude",
    "Longitude",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Total",
]

# Longest prefix first so grouped dummies are not claimed by a shorter name.
ONE_HOT_PREFIXES = [
    "Category Name_grouped_",
    "Order State_grouped_",
    "Order Country_grouped_",
    "Customer Segment_",
    "Department Name_",
    "Shipping Mode_",
    "Order Region_",
    "Market_",
    "Type_",
]

UNSCALED_COLUMNS = [
    "Sales",
    "Benefit per order",
    "Order Item Quantity",
    "Days for shipment (scheduled)",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Total",
    "Latitude",
    "Longitude",
]

VALUE_COLUMNS = ["Benefit per order", "Sales", "Order Item Quantity"]


def parse_args() -> argparse.Namespace:
    """Read the raw and cleaned paths."""
    parser = argparse.ArgumentParser(description="Validate the Phase 1 cleaned file.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("dataset/DataCoSupplyChainDataset.csv"),
    )
    parser.add_argument(
        "--cleaned",
        type=Path,
        default=Path("A_preprocessing_eda/outputs/cleaned_supply_chain.csv"),
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    """Resolve a relative path from the repository root."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> int:
    """Run every check and return 0 only if all of them pass."""
    args = parse_args()
    raw_path = resolve_repo_path(args.raw)
    cleaned_path = resolve_repo_path(args.cleaned)
    raw = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    cleaned = pd.read_csv(cleaned_path, encoding="utf-8", low_memory=False)

    failures: list[str] = []
    passes = 0

    def check(ok: bool, name: str, detail: str = "") -> None:
        nonlocal passes
        if ok:
            passes += 1
            print(f"PASS {name}")
            return
        message = f"FAIL {name}" if not detail else f"FAIL {name}: {detail}"
        print(message)
        failures.append(message)

    check(
        len(cleaned) == len(raw),
        "row count equals raw row count",
        f"cleaned={len(cleaned)} raw={len(raw)}",
    )
    null_cells = int(cleaned.isna().sum().sum())
    check(null_cells == 0, "zero nulls in the cleaned file", f"{null_cells} null cells")

    for column in FORBIDDEN_COLUMNS:
        check(column not in cleaned.columns, f"dropped column absent: {column}")

    shipping_date_cols = [column for column in cleaned.columns if "shipping date" in column.lower()]
    check(
        not shipping_date_cols,
        "no column name contains 'shipping date'",
        ", ".join(shipping_date_cols),
    )

    missing_required = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    check(
        not missing_required,
        "required columns present",
        ", ".join(missing_required),
    )

    check(
        "Days for shipment (scheduled)" in cleaned.columns,
        "Days for shipment (scheduled) is present",
    )

    if {"late", "loss", "Late_delivery_risk", "Benefit per order"}.issubset(cleaned.columns):
        late = cleaned["late"]
        loss = cleaned["loss"]
        check(
            pd.api.types.is_integer_dtype(late) and set(late.unique()).issubset({0, 1}) and late.notna().all(),
            "late is an integer 0/1 column with no nulls",
            f"dtype={late.dtype}",
        )
        check(
            pd.api.types.is_integer_dtype(loss) and set(loss.unique()).issubset({0, 1}) and loss.notna().all(),
            "loss is an integer 0/1 column with no nulls",
            f"dtype={loss.dtype}",
        )
        loss_expected = (cleaned["Benefit per order"] < 0).astype("int64")
        check(
            loss.astype("int64").equals(loss_expected),
            "loss equals Benefit per order < 0 on every row",
        )
        check(
            late.astype("int64").equals(cleaned["Late_delivery_risk"].astype("int64")),
            "late equals Late_delivery_risk on every row",
        )
    else:
        check(False, "target columns exist", "late, loss, Late_delivery_risk, or Benefit per order is missing")

    for column in VALUE_COLUMNS:
        if column not in cleaned.columns:
            check(False, f"{column} equals the raw values", "column missing")
            continue
        equal, detail = _values_equal(cleaned[column], raw[column])
        check(equal, f"{column} equals the raw values", detail)

    if "Benefit per order" in cleaned.columns:
        raw_negative = int((raw["Benefit per order"] < 0).sum())
        cleaned_negative = int((cleaned["Benefit per order"] < 0).sum())
        check(
            raw_negative == cleaned_negative,
            "negative Benefit per order count equals the raw count",
            f"cleaned={cleaned_negative} raw={raw_negative}",
        )

    assigned = _assign_one_hot(cleaned.columns)
    for prefix, columns in assigned.items():
        source = prefix[:-1]
        if not columns:
            check(False, f"one-hot group {source}", "no columns found")
            continue
        frame = cleaned[columns]
        integer = all(pd.api.types.is_integer_dtype(frame[column]) for column in columns)
        values = set(np.unique(frame.to_numpy()))
        sums_ok = bool(frame.sum(axis=1).eq(1).all())
        check(
            integer and values.issubset({0, 1}) and sums_ok,
            f"one-hot group {source}: integer 0/1 and row sums equal 1",
            f"n={len(columns)} integer={integer} values={sorted(values)} sums_ok={sums_ok}",
        )

    if ORDER_DATE_COL in cleaned.columns and {"order_year", "order_month", "order_weekday"}.issubset(cleaned.columns):
        try:
            parsed = pd.to_datetime(cleaned[ORDER_DATE_COL], format=ISO_FORMAT, errors="raise")
        except (ValueError, TypeError) as exc:
            check(False, "order date parses as ISO", str(exc))
        else:
            check(
                parsed.dt.year.astype("int64").equals(cleaned["order_year"].astype("int64")),
                "order_year matches the parsed order date",
            )
            check(
                parsed.dt.month.astype("int64").equals(cleaned["order_month"].astype("int64")),
                "order_month matches the parsed order date",
            )
            check(
                parsed.dt.dayofweek.astype("int64").equals(cleaned["order_weekday"].astype("int64")),
                "order_weekday matches the parsed order date (Monday = 0)",
            )
            if "order_weekday_name" in cleaned.columns:
                expected_names = cleaned["order_weekday"].astype("int64").map(dict(enumerate(WEEKDAY_NAMES)))
                check(
                    cleaned["order_weekday_name"].equals(expected_names),
                    "order_weekday_name matches Monday = 0",
                )
    else:
        check(False, "order date columns exist", "missing parsed date or a derived column")

    if {"Sales", "Sales_log"}.issubset(cleaned.columns):
        expected_log = np.log1p(cleaned["Sales"].to_numpy(dtype=float))
        actual_log = cleaned["Sales_log"].to_numpy(dtype=float)
        log_diff = float(np.max(np.abs(expected_log - actual_log))) if len(cleaned) else 0.0
        check(log_diff <= 1e-8, "Sales_log equals log1p(Sales)", f"max abs diff {log_diff}")

    if {"Benefit per order", "Benefit_signed_log"}.issubset(cleaned.columns):
        benefit = cleaned["Benefit per order"].to_numpy(dtype=float)
        expected_signed = np.sign(benefit) * np.log1p(np.abs(benefit))
        actual_signed = cleaned["Benefit_signed_log"].to_numpy(dtype=float)
        signed_diff = float(np.max(np.abs(expected_signed - actual_signed))) if len(cleaned) else 0.0
        check(
            signed_diff <= 1e-8,
            "Benefit_signed_log equals sign(x) * log1p(abs(x))",
            f"max abs diff {signed_diff}",
        )

    duplicate_names = cleaned.columns[cleaned.columns.duplicated()].tolist()
    check(not duplicate_names, "no duplicate column names", ", ".join(map(str, duplicate_names)))

    raw_duplicate_rows = int(raw.duplicated().sum())
    cleaned_duplicate_rows = int(cleaned.duplicated().sum())
    check(
        len(cleaned) == len(raw),
        (
            "duplicate rows stated: "
            f"raw={raw_duplicate_rows}, cleaned={cleaned_duplicate_rows}. "
            "Row count equals the raw file, so no rows were deleted."
        ),
    )

    unscaled_problems: list[str] = []
    for column in UNSCALED_COLUMNS:
        if column not in cleaned.columns:
            unscaled_problems.append(f"{column} missing")
            continue
        mean = float(cleaned[column].mean())
        std = float(cleaned[column].std(ddof=0))
        if abs(mean) < 0.05 and abs(std - 1.0) < 0.05:
            unscaled_problems.append(f"{column} mean={mean:.4f} std={std:.4f}")
    check(
        not unscaled_problems,
        "measurement columns are not standardised (mean near 0 and std near 1)",
        "; ".join(unscaled_problems),
    )

    print(f"SUMMARY: {passes} passed, {len(failures)} failed")
    return 1 if failures else 0


def _values_equal(cleaned: pd.Series, raw: pd.Series) -> tuple[bool, str]:
    """Return whether two columns match exactly, with a short diff note."""
    if len(cleaned) != len(raw):
        return False, f"length cleaned={len(cleaned)} raw={len(raw)}"
    left = cleaned.to_numpy()
    right = raw.to_numpy()
    if np.array_equal(left, right):
        return True, ""
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        diff = float(np.nanmax(np.abs(left.astype(float) - right.astype(float))))
        return False, f"max abs diff {diff}"
    return False, "values differ"


def _assign_one_hot(columns: pd.Index) -> dict[str, list[str]]:
    """Assign dummy columns to one prefix each, longest prefix first."""
    prefixes = sorted(ONE_HOT_PREFIXES, key=len, reverse=True)
    assigned = {prefix: [] for prefix in ONE_HOT_PREFIXES}
    for column in columns:
        for prefix in prefixes:
            if column.startswith(prefix):
                assigned[prefix].append(column)
                break
    return assigned


if __name__ == "__main__":
    sys.exit(main())
