"""Phase 1 cleaning for the DataCo Smart Supply Chain dataset.

Run from anywhere:

    python A_preprocessing_eda/phase1_cleaning.py

The raw CSV is read and never modified. Outputs are written under
A_preprocessing_eda/outputs/. Paths in the log are relative to the repo root.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "A_preprocessing_eda" / "outputs"
PLOT_DIR = OUTPUT_DIR / "outlier_plots"
SCRIPT_REL = "A_preprocessing_eda/phase1_cleaning.py"
VALIDATOR_REL = "A_preprocessing_eda/phase1_validate.py"

RANDOM_STATE = 42
TOP_N = 10
FLOAT_TOL = 1e-9
SKEW_ABS_THRESHOLD = 1.0
EXPECTED_ROWS = 180_519
EXPECTED_COLS = 53
TOLERANCE_PP = 0.1
GITHUB_WARN_BYTES = 50 * 1024 * 1024

ORDER_DATE_COL = "order date (DateOrders)"
SHIP_DATE_COL = "shipping date (DateOrders)"
ORDER_DATE_FORMAT = "%m/%d/%Y %H:%M"
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

EMPTY_COLUMNS = [
    "Product Description",
    "Order Zipcode",
    "Product Status",
    "Product Image",
]
PERSONAL_COLUMNS = [
    "Customer Password",
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
]
ID_COLUMNS = [
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Category Id",
    "Department Id",
    "Product Card Id",
    "Product Category Id",
    "Order Item Cardprod Id",
]
LATE_LEAKAGE_COLUMNS = [
    "Delivery Status",
    "Days for shipping (real)",
    SHIP_DATE_COL,
]
PROFIT_LEAKAGE_COLUMNS = [
    "Order Profit Per Order",
    "Order Item Profit Ratio",
]
LOW_CARD_COLUMNS = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Type",
    "Department Name",
    "Order Region",
]
HIGH_CARD_COLUMNS = [
    "Category Name",
    "Order State",
    "Order Country",
]
TEXT_ONLY_COLUMNS = [
    "Order Status",
    "Product Name",
    "Customer City",
    "Customer State",
    "Customer Country",
    "Order City",
    "Customer Zipcode",
]
DERIVED_DATE_COLUMNS = [
    "order_year",
    "order_month",
    "order_weekday",
    "order_weekday_name",
]
PROTECTED_FROM_DEDUP = {
    "Benefit per order",
    "Late_delivery_risk",
    "Sales",
    "Order Item Quantity",
    "Days for shipment (scheduled)",
    "Order Item Total",
    "Order Item Product Price",
    "order_year",
    "order_month",
    "order_weekday",
}
BASELINE_COLUMNS = ["Sales", "Benefit per order", "Order Item Quantity"]
IDENTITY_PAIRS = [
    ("Benefit per order", "Order Profit Per Order"),
    ("Sales per customer", "Order Item Total"),
    ("Order Item Product Price", "Product Price"),
    ("Customer Id", "Order Customer Id"),
    ("Category Id", "Product Category Id"),
    ("Product Card Id", "Order Item Cardprod Id"),
]

LATE_LEAKAGE_RULES = {
    "Delivery Status": (
        "Known only after the delivery outcome is recorded, and it restates whether the order was late."
    ),
    "Days for shipping (real)": (
        "The actual days in transit are known only after delivery, and lateness is that value compared with the scheduled days."
    ),
    SHIP_DATE_COL: (
        "The ship timestamp is known only after dispatch, so it is not available at order time and leaks the late-delivery target."
    ),
}
PROFIT_LEAKAGE_RULES = {
    "Order Profit Per Order": (
        "It defines profit directly. Benefit per order is kept because it is the OLS target and the source of the loss target."
    ),
    "Order Item Profit Ratio": (
        "It expresses profit relative to the item value, so it defines the loss and profit targets."
    ),
}
ID_REASONS = {
    "Customer Id": "Customer key. It identifies a person and is not a descriptive feature.",
    "Order Customer Id": "Customer key stored on the order. It is not a descriptive feature.",
    "Order Id": "Order key. It is not a descriptive feature.",
    "Order Item Id": "Order-line key. It is not a descriptive feature.",
    "Category Id": "Category key. The category name is kept instead.",
    "Department Id": "Department key. The department name is kept instead.",
    "Product Card Id": "Product key. The product name is kept instead.",
    "Product Category Id": "Category key. The category name is kept instead.",
    "Order Item Cardprod Id": "Product key stored on the order line. The product name is kept instead.",
}

logger = logging.getLogger("phase1")


def parse_args() -> argparse.Namespace:
    """Read the raw-file path."""
    parser = argparse.ArgumentParser(description="Clean the DataCo supply chain file (Phase 1).")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("dataset/DataCoSupplyChainDataset.csv"),
        help="Raw CSV path, relative to the repo root unless absolute.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    """Resolve a relative path against the repository root."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: Path) -> str:
    """Return a repo-relative posix path when the file lives in the repo."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def fmt_int(value: int) -> str:
    """Format an integer with thousands separators."""
    return f"{int(value):,}"


def fmt_pct(numerator: int, denominator: int) -> str:
    """Format a percentage from a count and a total."""
    return f"{100.0 * numerator / denominator:.4f}%"


def fmt_float(value: float) -> str:
    """Format a float for the log tables."""
    return f"{float(value):.6f}"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a markdown table, escaping pipes in cells."""

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(item) for item in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def differing_rows(left: pd.Series, right: pd.Series, tol: float = FLOAT_TOL) -> tuple[int, float]:
    """Count rows that differ, and the maximum absolute difference.

    Nulls match only when both values are null. Non-null numeric values differ
    when their absolute gap is greater than `tol`.
    """
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    missing_mismatch = left_num.isna() ^ right_num.isna()
    gap = (left_num - right_num).abs()
    value_mismatch = gap.notna() & (gap > tol)
    n_diff = int(value_mismatch.sum() + missing_mismatch.sum())
    max_abs = float(gap.max()) if bool(gap.notna().any()) else 0.0
    return n_diff, max_abs


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV with the required encoding."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path, encoding="latin-1", low_memory=False)


def drop_listed(
    df: pd.DataFrame,
    columns: list[str],
    category: str,
    reasons: dict[str, str],
    dropped: list[dict[str, str]],
    discrepancies: list[str],
) -> pd.DataFrame:
    """Drop columns that exist. A missing listed column is a discrepancy, not a crash."""
    present: list[str] = []
    for column in columns:
        if column not in df.columns:
            discrepancies.append(
                f"Listed drop `{column}` ({category}) is not in the file, so it was not dropped."
            )
            continue
        dropped.append({"column": column, "category": category, "reason": reasons[column]})
        present.append(column)
    logger.info("Dropping %s columns (%s)", len(present), category)
    return df.drop(columns=present)


def parse_order_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the order date and add year, month, and weekday. Never uses the ship date."""
    if SHIP_DATE_COL in df.columns:
        raise RuntimeError("The shipping date is still present. Refusing to derive features from it.")
    if ORDER_DATE_COL not in df.columns:
        raise KeyError(f"Order date column {ORDER_DATE_COL!r} is missing.")
    parsed = pd.to_datetime(df[ORDER_DATE_COL], format=ORDER_DATE_FORMAT, errors="coerce")
    failed = int(parsed.isna().sum())
    if failed:
        raise ValueError(
            f"{failed} values in `{ORDER_DATE_COL}` failed to parse with format {ORDER_DATE_FORMAT}."
        )
    df = df.copy()
    df[ORDER_DATE_COL] = parsed
    df["order_year"] = parsed.dt.year.astype("int64")
    df["order_month"] = parsed.dt.month.astype("int64")
    df["order_weekday"] = parsed.dt.dayofweek.astype("int64")
    df["order_weekday_name"] = df["order_weekday"].map(dict(enumerate(WEEKDAY_NAMES)))
    if df["order_weekday_name"].isna().any():
        raise AssertionError("order_weekday_name has nulls.")
    return df


def top_n_labels(series: pd.Series, n: int) -> list[tuple[str, int]]:
    """Return the top n labels by frequency, with alphabetical tie breaks."""
    counts = series.astype(str).value_counts()
    ranked = sorted(((str(label), int(count)) for label, count in counts.items()), key=lambda item: (-item[1], item[0]))
    return ranked[:n]


def one_hot_frame(series: pd.Series, prefix: str) -> pd.DataFrame:
    """One-hot encode every level as int64, in alphabetical column order.

    All levels are retained. A reference level is not dropped here.
    """
    text = series.astype(str)
    levels = sorted(text.unique().tolist())
    dummies = pd.get_dummies(text, prefix=prefix, prefix_sep="_", dtype=int)
    expected = [f"{prefix}_{level}" for level in levels]
    missing = [column for column in expected if column not in dummies.columns]
    extra = [column for column in dummies.columns if column not in expected]
    if missing or extra:
        raise RuntimeError(f"One-hot names for `{prefix}` did not match the levels. missing={missing} extra={extra}")
    dummies = dummies.loc[:, expected]
    if not dummies.sum(axis=1).eq(1).all():
        raise AssertionError(f"One-hot columns for `{prefix}` do not sum to 1 on every row.")
    values = set(np.unique(dummies.to_numpy()))
    if not values.issubset({0, 1}):
        raise AssertionError(f"One-hot columns for `{prefix}` are not strictly 0/1: {values}")
    if any(not pd.api.types.is_integer_dtype(dummies[column]) for column in dummies.columns):
        raise AssertionError(f"One-hot columns for `{prefix}` are not integers.")
    return dummies


def choose_duplicate_drop(earlier: str, later: str) -> str | None:
    """Choose which exact-duplicate column to drop. None means keep both."""
    if later not in PROTECTED_FROM_DEDUP:
        return later
    if earlier not in PROTECTED_FROM_DEDUP:
        return earlier
    return None


def iqr_report(series: pd.Series) -> dict[str, float | int]:
    """Compute 1.5-IQR fences, outlier counts, and skewness."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outside = (series < lower) | (series > upper)
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "n_out": int(outside.sum()),
        "pct_out": 100.0 * float(outside.mean()),
        "skew": float(series.skew()),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
        "n_negative_out": int((outside & (series < 0)).sum()),
    }


def save_boxplot(series: pd.Series, title: str, ylabel: str, path: Path) -> None:
    """Save one box plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=series, ax=ax, color="#1f4e79", fliersize=2, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def save_histogram(series: pd.Series, title: str, xlabel: str, path: Path) -> None:
    """Save one histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(x=series, bins=40, ax=ax, color="#1f4e79", edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Orders")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_slug(column: str) -> str:
    """Stable file slug for a column name."""
    return column.replace(" ", "_")


def format_zip(series: pd.Series) -> pd.Series:
    """Cast an integral zip code to text and fill nulls with Unknown."""
    non_null = series.dropna()
    if len(non_null) and not np.all(np.equal(np.mod(non_null.to_numpy(dtype=float), 1), 0)):
        raise ValueError("Customer Zipcode has non-integral values. Refusing to invent a format.")
    as_int = series.round(0).astype("Int64").astype(str)
    return as_int.mask(series.isna(), "Unknown")


def dtype_class(dtype: object) -> str:
    """Collapse a dtype to the class checked on round-trip."""
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    return "other"


def save_and_check(df: pd.DataFrame, path: Path) -> None:
    """Write the CSV and prove a re-read matches the in-memory frame."""
    export = df.copy()
    export[ORDER_DATE_COL] = export[ORDER_DATE_COL].dt.strftime(ISO_FORMAT)
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    loaded = pd.read_csv(path, encoding="utf-8", low_memory=False)
    if list(loaded.columns) != list(export.columns):
        raise AssertionError("Re-read column names do not match the saved frame.")
    if loaded.shape != export.shape:
        raise AssertionError(f"Re-read shape {loaded.shape} does not match {export.shape}.")
    loaded[ORDER_DATE_COL] = pd.to_datetime(loaded[ORDER_DATE_COL], format=ISO_FORMAT)
    for column in df.columns:
        if dtype_class(df[column].dtype) != dtype_class(loaded[column].dtype):
            raise AssertionError(
                f"Dtype class changed for `{column}`: memory {df[column].dtype}, file {loaded[column].dtype}."
            )
    pd.testing.assert_frame_equal(
        loaded,
        df,
        check_dtype=False,
        check_exact=False,
        rtol=1e-7,
        atol=1e-9,
    )
    for column in BASELINE_COLUMNS:
        if not np.array_equal(loaded[column].to_numpy(), df[column].to_numpy()):
            raise AssertionError(f"`{column}` changed between memory and the re-read file.")


def reorder_columns(
    df: pd.DataFrame,
    original_order: list[str],
    transform_columns: list[str],
    grouped_columns: list[str],
    one_hot_columns: list[str],
) -> pd.DataFrame:
    """Place originals, dates, transforms, groupings, dummies, then targets."""
    ordered = original_order + DERIVED_DATE_COLUMNS + transform_columns + grouped_columns + one_hot_columns + ["late", "loss"]
    if len(ordered) != len(set(ordered)):
        raise AssertionError("Column order contains a duplicate name.")
    missing = [column for column in df.columns if column not in ordered]
    extra = [column for column in ordered if column not in df.columns]
    if missing or extra:
        raise AssertionError(f"Column order does not match the frame. missing={missing} extra={extra}")
    return df.loc[:, ordered]


def run_validator(raw_path: Path, cleaned_path: Path) -> tuple[str, int]:
    """Run the independent validator and return its output and exit code."""
    raw_arg = rel(raw_path) if raw_path.resolve().is_relative_to(REPO_ROOT) else str(raw_path)
    cleaned_arg = rel(cleaned_path)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / VALIDATOR_REL), "--raw", raw_arg, "--cleaned", cleaned_arg],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout
    if completed.stderr.strip():
        output = output.rstrip() + "\n" + completed.stderr
    return output, int(completed.returncode)


def describe_column(name: str, meta: dict[str, object]) -> tuple[str, str, str]:
    """Return role, allowed models, and a one-line note for one final column."""
    transforms: set[str] = meta["transforms"]  # type: ignore[assignment]
    one_hot: dict[str, str] = meta["one_hot"]  # type: ignore[assignment]
    grouped_sources: dict[str, str] = meta["grouped_sources"]  # type: ignore[assignment]
    encoded_originals: set[str] = meta["encoded_originals"]  # type: ignore[assignment]
    notes: dict[str, str] = meta["notes"]  # type: ignore[assignment]
    all_models = "late model, loss model, OLS"

    if name == "Benefit per order":
        return "target", "OLS", "Target for OLS, never a feature for loss or OLS."
    if name == "late":
        return "target", "late model", "Late-delivery target. Identical to `Late_delivery_risk`. Not a feature."
    if name == "loss":
        return "target", "loss model", "Loss target: 1 if `Benefit per order` < 0, else 0. Not a feature."
    if name == "Late_delivery_risk":
        return "target", "late model", "Late-delivery target. Identical to `late`. Not a feature."
    if name in transforms:
        return "transform", "none" if name == "Benefit_signed_log" else all_models, notes[name]
    if name in one_hot:
        source = one_hot[name]
        return (
            "one-hot",
            all_models,
            f"One-hot level of `{source}`. Drop one level from this group before a model with an intercept.",
        )
    if name in grouped_sources:
        return (
            "categorical (text)",
            "none",
            f"Grouped text of `{grouped_sources[name]}`. Models should use the one-hot columns, not this text.",
        )
    if name == "order_weekday_name":
        return "BI only", "none", "English weekday name for Power BI. Models should use `order_weekday`."
    if name == ORDER_DATE_COL:
        return (
            "derived date",
            "none",
            "Order timestamp, saved as ISO text `YYYY-MM-DD HH:MM:SS`. Use the integer date columns in models. Not derived from the shipping date.",
        )
    if name in DERIVED_DATE_COLUMNS:
        return "derived date", all_models, notes[name]
    if name in encoded_originals:
        return (
            "categorical (text)",
            "none",
            "Original category text kept for Power BI and EDA. Models should use the one-hot columns.",
        )
    if name in TEXT_ONLY_COLUMNS:
        return "retained as text", "none", notes.get(name, "Retained as text, not for modelling without a decision.")
    if name in notes:
        return "numeric feature", all_models, notes[name]
    return "numeric feature", all_models, "Unscaled numeric value retained from the raw file."


def render_log(state: dict[str, object]) -> str:
    """Render the viva cleaning log from values computed during the run."""
    dropped: list[dict[str, str]] = state["dropped"]  # type: ignore[assignment]
    added: list[dict[str, str]] = state["added"]  # type: ignore[assignment]
    checks: list[dict[str, str]] = state["checks"]  # type: ignore[assignment]
    outliers: list[dict[str, object]] = state["outliers"]  # type: ignore[assignment]
    encoding: list[dict[str, object]] = state["encoding"]  # type: ignore[assignment]
    dictionary: list[dict[str, str]] = state["dictionary"]  # type: ignore[assignment]
    pairs: list[dict[str, object]] = state["pairs"]  # type: ignore[assignment]
    correlations: list[dict[str, object]] = state["correlations"]  # type: ignore[assignment]
    discrepancies: list[str] = state["discrepancies"]  # type: ignore[assignment]
    assumptions: list[str] = state["assumptions"]  # type: ignore[assignment]
    n_rows = int(state["n_rows"])
    drop_counts = Counter(item["category"] for item in dropped)
    add_counts = Counter(item["type"].split(" (")[0] for item in added)

    def cat_count(label: str) -> int:
        return int(drop_counts.get(label, 0))

    summary = "\n".join(
        [
            (
                f"The raw file had {fmt_int(n_rows)} rows and {state['n_cols_raw']} columns. "
                f"The cleaned file has {fmt_int(n_rows)} rows and {state['n_cols_final']} columns. "
                "No rows were deleted."
            ),
            (
                "Columns dropped: "
                f"{cat_count('Empty or useless')} empty or useless, "
                f"{cat_count('Personal')} personal, "
                f"{cat_count('ID columns')} identifier, "
                f"{cat_count('Leakage for the late-delivery model')} late-delivery leakage, "
                f"{cat_count('Leakage for the loss and profit models')} profit leakage, "
                f"{cat_count('Exact duplicate')} exact duplicate "
                f"({len(dropped)} in total)."
            ),
            (
                "Columns added: "
                f"{int(add_counts.get('derived date', 0))} derived date, "
                f"{int(add_counts.get('transform', 0))} transform, "
                f"{int(add_counts.get('grouped text', 0))} grouped text, "
                f"{int(add_counts.get('one-hot', 0))} one-hot, "
                f"{int(add_counts.get('target', 0))} target "
                f"({len(added)} in total)."
            ),
            (
                f"Null cells fell from {fmt_int(int(state['null_cells_before']))} "
                f"to {fmt_int(int(state['null_cells_after']))}."
            ),
            (
                f"Late orders (`late` = 1): {state['late_pct']} "
                f"({fmt_int(int(state['late_count']))} of {fmt_int(n_rows)}). "
                f"Loss orders (`loss` = 1): {state['loss_pct']} "
                f"({fmt_int(int(state['loss_count']))} of {fmt_int(n_rows)})."
            ),
            (
                f"Exactly zero profit: {state['zero_pct']} "
                f"({fmt_int(int(state['zero_count']))} rows). All of them have `loss` = 0."
            ),
            (
                f"The cleaned file is unscaled, has no train/test split, and is "
                f"{float(state['file_size_mb']):.2f} MB "
                f"({fmt_int(int(state['file_size_bytes']))} bytes)."
                + (
                    " This is above 50 MB. GitHub warns at 50 MB and blocks at 100 MB. Git LFS may be needed. The file was not shrunk."
                    if bool(state["size_flag"])
                    else ""
                )
            ),
        ]
    )

    dtype_table = md_table(
        ["Column", "Dtype"],
        [[name, dtype] for name, dtype in state["dtypes"]],  # type: ignore[index]
    )
    null_table = md_table(
        ["Column", "Null count", "Null percent"],
        [[name, fmt_int(int(count)), f"{float(percent):.4f}%"] for name, count, percent in state["nulls"]],  # type: ignore[misc]
    )
    check_table = md_table(
        ["Check", "Expected", "Actual", "Result"],
        [[row["check"], row["expected"], row["actual"], row["result"]] for row in checks],
    )
    drop_table = md_table(
        ["Column", "Category", "Reason"],
        [[row["column"], row["category"], row["reason"]] for row in dropped],
    )
    add_table = md_table(
        ["Column", "Type", "How derived"],
        [[row["column"], row["type"], row["how"]] for row in added],
    )
    outlier_table = md_table(
        ["Column", "Q1", "Q3", "IQR", "Lower fence", "Upper fence", "Rows outside", "Percent outside", "Skewness"],
        [
            [
                row["column"],
                fmt_float(float(row["q1"])),
                fmt_float(float(row["q3"])),
                fmt_float(float(row["iqr"])),
                fmt_float(float(row["lower"])),
                fmt_float(float(row["upper"])),
                fmt_int(int(row["n_out"])),
                f"{float(row['pct_out']):.4f}%",
                fmt_float(float(row["skew"])),
            ]
            for row in outliers
        ],
    )
    pair_table = md_table(
        ["Left", "Right", "Differing rows", "Max absolute difference", "Action"],
        [
            [
                row["left"],
                row["right"],
                fmt_int(int(row["n_diff"])),
                fmt_float(float(row["max_abs"])),
                row["action"],
            ]
            for row in pairs
        ],
    )
    if correlations:
        corr_table = md_table(
            ["Column A", "Column B", "Pearson r", "Action"],
            [
                [row["left"], row["right"], fmt_float(float(row["r"])), "Retained. Not an exact duplicate."]
                for row in correlations
            ],
        )
    else:
        corr_table = "None."
    encoding_table = md_table(
        ["Column", "Method", "Levels", "Notes"],
        [[row["column"], row["method"], row["levels"], row["notes"]] for row in encoding],
    )
    dictionary_table = md_table(
        ["Column", "Dtype", "Role", "Downstream model", "Note"],
        [[row["name"], row["dtype"], row["role"], row["models"], row["note"]] for row in dictionary],
    )
    leakage_table = md_table(
        ["Column", "Leaks for", "Why", "In the cleaned file"],
        [
            [
                "Delivery Status",
                "late model",
                LATE_LEAKAGE_RULES["Delivery Status"],
                "No. Dropped.",
            ],
            [
                "Days for shipping (real)",
                "late model",
                LATE_LEAKAGE_RULES["Days for shipping (real)"],
                "No. Dropped.",
            ],
            [
                SHIP_DATE_COL,
                "late model",
                LATE_LEAKAGE_RULES[SHIP_DATE_COL],
                "No. Dropped.",
            ],
            [
                "Order Profit Per Order",
                "loss model, OLS",
                PROFIT_LEAKAGE_RULES["Order Profit Per Order"],
                "No. Dropped.",
            ],
            [
                "Order Item Profit Ratio",
                "loss model, OLS",
                PROFIT_LEAKAGE_RULES["Order Item Profit Ratio"],
                "No. Dropped.",
            ],
            [
                "Benefit per order",
                "loss model, OLS (if used as a feature)",
                "It is the profit target and the source of `loss`. Using it as a feature would leak both targets.",
                "Yes. Kept as the OLS target. Not a feature for loss or OLS.",
            ],
        ],
    )
    plot_lines = "\n".join(f"- `{path}`" for path in state["plot_files"])  # type: ignore[union-attr]
    transform_lines = "\n".join(f"- {line}" for line in state["transform_lines"])  # type: ignore[union-attr]
    encoding_detail = "\n\n".join(str(block) for block in state["encoding_blocks"])  # type: ignore[union-attr]
    one_hot_lines = "\n".join(f"- `{name}`" for name in state["one_hot_columns"])  # type: ignore[union-attr]
    discrepancy_text = "None." if not discrepancies else "\n".join(f"- {item}" for item in discrepancies)
    assumption_text = "\n".join(f"- {item}" for item in assumptions)
    group_lines = "\n".join(f"- {line}" for line in state["group_checks"])  # type: ignore[union-attr]
    validation = str(state["validation_text"]).rstrip()
    size_note = (
        f"Final file size is {float(state['file_size_mb']):.2f} MB "
        f"({fmt_int(int(state['file_size_bytes']))} bytes). "
    )
    if bool(state["size_flag"]):
        size_note += (
            "This is above the 50 MB GitHub warning threshold and was not shrunk or truncated. "
            "GitHub blocks files at 100 MB. Git LFS may be needed."
        )
    else:
        size_note += "This is under the 50 MB GitHub warning threshold."

    return f"""# Phase 1 cleaning log: DataCo Smart Supply Chain

- Run date: {state['run_date']}
- Script: `{SCRIPT_REL}`
- Raw file: `{state['raw_rel']}`
- Python: {state['python_version']}
- pandas: {state['pandas_version']}
- numpy: {state['numpy_version']}

## Summary

{summary}

## Step-by-step log

### Step 1. Load and inspect

The raw file was loaded with `encoding="latin-1"` and was not modified. Shape: {fmt_int(n_rows)} rows by {state['n_cols_raw']} columns. Deep memory use: {float(state['memory_mb']):.2f} MB. Duplicate rows: {fmt_int(int(state['duplicate_rows_raw']))}. Columns with no nulls: {state['n_cols_without_nulls']} of {state['n_cols_raw']}.

Dtype table:

{dtype_table}

Nulls (columns with at least one null):

{null_table}

Expected facts:

{check_table}

### Step 2. Drop columns

Columns were dropped only from explicit lists. A listed column that is absent is recorded under Discrepancies and does not stop the run. `{fmt_int(n_rows)}` rows remained. Column count after this step: {state['n_cols_after_drop']}.

`Days for shipment (scheduled)` was kept. It is the scheduled transit time known when the order is placed, so it is not leakage.

Leakage rules:

- `Delivery Status`: {LATE_LEAKAGE_RULES['Delivery Status']}
- `Days for shipping (real)`: {LATE_LEAKAGE_RULES['Days for shipping (real)']}
- `{SHIP_DATE_COL}`: {LATE_LEAKAGE_RULES[SHIP_DATE_COL]}
- `Order Profit Per Order`: {PROFIT_LEAKAGE_RULES['Order Profit Per Order']}
- `Order Item Profit Ratio`: {PROFIT_LEAKAGE_RULES['Order Item Profit Ratio']}

`Benefit per order` was not dropped. It is the continuous profit target and the source of `loss`.

Identifier columns dropped: {", ".join(f"`{column}`" for column in ID_COLUMNS)}. A column was treated as a pure identifier when it is an entity key (customer, order, order line, category, department, or product card). `Latitude` and `Longitude` were kept because they are measurements.

The full drop list, including later exact-duplicate drops, is in the dropped-columns table.

### Step 3. Parse the order date

`{ORDER_DATE_COL}` was parsed with `pd.to_datetime` and format `{ORDER_DATE_FORMAT}`. Failed parses: {fmt_int(int(state['date_parse_failures']))}. Nothing was derived from `{SHIP_DATE_COL}`, which had already been dropped.

Derived columns: `order_year` (calendar year), `order_month` (1-12), `order_weekday` (Monday = 0, Sunday = 6), `order_weekday_name` (fixed English names from that integer, not from the machine locale). The original column is kept as a datetime and written as ISO text `YYYY-MM-DD HH:MM:SS` so Power BI can parse it.

Actual order-date minimum: {state['date_min']}. Actual order-date maximum: {state['date_max']}. Column count after this step: {state['n_cols_after_dates']}.

### Step 4. Near-duplicates and remaining nulls

Pairs were compared by value with absolute tolerance {FLOAT_TOL:.0e}. Identity of `Benefit per order` and `Order Profit Per Order` was measured on the raw file before that column was dropped.

{pair_table}

Systematic scan of the remaining numeric columns: exact duplicates were dropped unless the column is a target, `Benefit per order`, `Sales`, `Order Item Quantity`, `Days for shipment (scheduled)`, `Order Item Total`, `Order Item Product Price`, or a derived date column. When both names were unprotected, the later column in the current frame was dropped. Correlated but non-identical pairs were not dropped.

### High-correlation pairs retained (for C to handle, for example via VIF)

Pearson correlation was computed on the numeric columns that remained after the exact-duplicate drops. Pairs with absolute correlation above 0.95 are listed. Exact duplicates are not repeated here.

{corr_table}

Null handling: {state['null_handling']} Assertion after this step: null cells = {fmt_int(int(state['null_cells_after']))}. Column count after this step: {state['n_cols_after_dups']}.

### Step 5. Outlier review

Quartiles use `Series.quantile` (linear interpolation). Fences are Q1 - 1.5 IQR and Q3 + 1.5 IQR. Skewness is `Series.skew` (bias-adjusted). No row was deleted or capped. Original columns were not overwritten.

{outlier_table}

`Benefit per order` outliers that are also negative (loss rows): {fmt_int(int(state['benefit_negative_outliers']))} of {fmt_int(int(state['benefit_outlier_rows']))} outlier rows. Those rows stay in the file because they are the loss class.

Observed ranges: {state['ranges']}.

Transforms:

{transform_lines}

A plain log cannot be used on `Benefit per order` because the column contains negative values and zeros. `Benefit_signed_log` keeps the sign, maps 0 to 0, and compresses the magnitude with `log1p`.

Plot files:

{plot_lines}

Column count after this step: {state['n_cols_after_outliers']}.

### Step 6. Encode categoricals

Original text columns are kept and encoded columns are added in the same file. Power BI and EDA need readable categories, and the models need numeric columns. Encoded names use the source column as a prefix so they are not confused with the original text. `pd.get_dummies(..., prefix=<column name>, prefix_sep="_", dtype=int)` was used. `drop_first` was not used. Dummy values are integers 0/1, and dummy columns within each group are ordered alphabetically so the file does not depend on row order.

Low-cardinality one-hot columns: {", ".join(f"`{column}`" for column in LOW_CARD_COLUMNS)}. `Order Region` is included because the late-delivery model uses it.

High-cardinality columns use `TOP_N = {TOP_N}`. The kept values are the most frequent, with ties broken alphabetically. Other values are mapped to `Other`. The grouped text is saved as `<column>_grouped` and then one-hot encoded.

{encoding_detail}

One-hot row-sum checks (each group sums to 1 on every row, values are 0/1, dtype is integer):

{group_lines}

New one-hot columns:

{one_hot_lines}

Columns left as text and not encoded: {", ".join(f"`{column}`" for column in TEXT_ONLY_COLUMNS)}. They are marked in the column dictionary as retained text, not for modelling without a decision. Column count after this step: {state['n_cols_after_encode']}.

### Step 7. Add targets

`late` was copied from `Late_delivery_risk` and both columns were kept. Rows where they differ: {fmt_int(int(state['late_mismatch']))}. `loss` is 1 when `Benefit per order` < 0 and 0 otherwise. Both are integers in {{0, 1}} with no nulls.

Class counts:

- `late` = 1: {fmt_int(int(state['late_count']))} ({state['late_pct']}). `late` = 0: {fmt_int(int(state['late_zero']))} ({state['late_zero_pct']}).
- `loss` = 1: {fmt_int(int(state['loss_count']))} ({state['loss_pct']}). `loss` = 0: {fmt_int(int(state['loss_zero']))} ({state['loss_zero_pct']}).
- `Benefit per order` = 0: {fmt_int(int(state['zero_count']))} ({state['zero_pct']}). Rows with zero profit and `loss` = 0: {fmt_int(int(state['zero_as_not_loss']))}. Zero-profit orders are not loss.

Column count after this step: {state['n_cols_final']}.

### Step 8. Save outputs

Columns were ordered as surviving original columns, derived date columns, transforms, grouped text, one-hot columns, then `late` and `loss`. The file was written with `index=False` and `encoding="utf-8"`, then read back. Shape, dtype class, and values matched the in-memory frame (`assert_frame_equal`, float tolerance rtol 1e-7 and atol 1e-9). Integer columns were still integers. `Sales`, `Benefit per order`, and `Order Item Quantity` matched the pre-save values exactly on that re-read.

{size_note}

Duplicate rows in the cleaned file: {fmt_int(int(state['duplicate_rows_clean']))}. Duplicate rows in the raw file: {fmt_int(int(state['duplicate_rows_raw']))}. The cleaned row count equals the raw row count. Any increase in duplicate rows is the result of removing identifier columns, not of deleting or copying records.

Seed `{RANDOM_STATE}` was set with `numpy.random.seed`. No step in this phase draws random numbers.

## Dropped columns

{drop_table}

## Added columns

{add_table}

## Outliers

{outlier_table}

`Benefit per order` negative outliers: {fmt_int(int(state['benefit_negative_outliers']))}.

Plot files:

{plot_lines}

## Encoding

{encoding_table}

## Column dictionary

{dictionary_table}

## Leakage summary

{leakage_table}

`Days for shipment (scheduled)` is not in this table because it is known at order time and remains available to every model.

## Verification results

Output of `{VALIDATOR_REL}` (exit code {state['validation_code']}):

```
{validation}
```

## Discrepancies

{discrepancy_text}

## Assumptions and open decisions

{assumption_text}

## Limitations

The data come from a single company (DataCo). Orders in this file run from {state['date_min']} to {state['date_max']}. Loss is defined only as `Benefit per order` < 0, so an order with profit of exactly zero is not a loss. Columns that leak the late-delivery outcome, and columns that define profit other than the target itself, were removed. `Benefit per order` remains because it is the OLS target. The cleaned file is not scaled and is not split into train and test.
"""


def render_handoff(state: dict[str, object]) -> str:
    """Fill the group-chat hand-off from the same computed state as the log."""
    dropped: list[dict[str, str]] = state["dropped"]  # type: ignore[assignment]
    added: list[dict[str, str]] = state["added"]  # type: ignore[assignment]
    drop_counts = Counter(item["category"] for item in dropped)
    add_counts = Counter(item["type"].split(" (")[0] for item in added)
    drop_bits = ", ".join(f"{label} {count}" for label, count in drop_counts.items())
    add_bits = ", ".join(f"{label} {count}" for label, count in add_counts.items())
    outlier_bits = "; ".join(
        f"{row['column']} {fmt_int(int(row['n_out']))} ({float(row['pct_out']):.4f}%)"
        for row in state["outliers"]  # type: ignore[union-attr]
    )
    corr_bits = (
        "; ".join(
            f"`{row['left']}` and `{row['right']}` (r = {float(row['r']):.6f})"
            for row in state["correlations"]  # type: ignore[union-attr]
        )
        if state["correlations"]
        else "none"
    )
    text_only = ", ".join(f"`{column}`" for column in TEXT_ONLY_COLUMNS)
    size_line = f"{float(state['file_size_mb']):.2f} MB ({fmt_int(int(state['file_size_bytes']))} bytes)"
    if bool(state["size_flag"]):
        size_line += " Above 50 MB: GitHub warns at 50 MB and blocks at 100 MB. Git LFS may be needed. The file was not shrunk"
    lines = [
        "HAND-OFF: Phase 1, Data cleaning and preprocessing (A)",
        "1. Done: Loaded the raw DataCo file with encoding latin-1, dropped empty, personal, identifier, and leakage columns, and parsed the order date only. Exact duplicate columns were removed after a value check. Outliers in Sales, Benefit per order, and Order Item Quantity were reviewed and left in place. Skew transforms were added as new columns where the rule required them. Low-cardinality categories, including Order Region, were one-hot encoded, and Category Name, Order State, and Order Country were grouped at top 10 and then one-hot encoded. Targets late and loss were added. The file is unscaled and there is no train/test split.",
        (
            f"2. Key numbers: {fmt_int(int(state['n_rows']))} x {state['n_cols_raw']} before; "
            f"{fmt_int(int(state['n_rows']))} x {state['n_cols_final']} after. "
            f"Dropped {len(dropped)} columns ({drop_bits}). "
            f"Added {len(added)} columns ({add_bits}). "
            f"Null cells {fmt_int(int(state['null_cells_before']))} before, {fmt_int(int(state['null_cells_after']))} after. "
            f"Late class {state['late_pct']}. Loss class {state['loss_pct']}. Zero-profit {state['zero_pct']} (classed as not loss). "
            f"Outliers: {outlier_bits}. "
            f"Benefit per order negative outliers: {fmt_int(int(state['benefit_negative_outliers']))}. "
            f"Final file size {size_line}."
        ),
        (
            "3. Decisions: Benefit per order stays; it is the OLS target and the source of loss, and it is never a feature for loss or OLS. "
            "Late_delivery_risk and late are both kept and are identical. "
            "Original category text is kept beside the integer dummies, and no reference level was dropped. "
            f"TOP_N = {TOP_N}, with frequency ties broken alphabetically. "
            "Sales per customer matched Order Item Total on every compared row, so Sales per customer was dropped and Order Item Total was kept. "
            "Product Price matched Order Item Product Price on every compared row, so Product Price was dropped. "
            "Order Status is still in the file as text and is flagged: it may contain post-order states, and it must not be used until that is decided. "
            f"Sales received Sales_log because absolute skewness was above {SKEW_ABS_THRESHOLD:.1f}. "
            "Benefit per order received Benefit_signed_log (sign times log1p of the absolute value) because a plain log is undefined for losses. "
            f"Order Item Quantity was not log-transformed: absolute skewness {float(state['quantity_skew']):.6f} is not above {SKEW_ABS_THRESHOLD:.1f}. "
            "No rows were removed and nothing was scaled."
        ),
        (
            "4. Files: `A_preprocessing_eda/outputs/cleaned_supply_chain.csv`; "
            "`A_preprocessing_eda/outputs/cleaning_log.md`; "
            "`A_preprocessing_eda/outputs/outlier_plots/`."
        ),
        (
            "5. Before you start: The file is unscaled. Scale only after the train/test split, and fit the scaler on the training fold. "
            "Drop one dummy level per one-hot group before a model with an intercept. "
            "`Benefit per order`, `loss`, `Late_delivery_risk`, and `late` are targets, not features. "
            f"`Benefit_signed_log` is a transform of the profit target, so do not use it as a regressor. "
            f"Text-only columns, not for modelling without a decision: {text_only}. "
            f"The grouped text columns (`Category Name_grouped`, `Order State_grouped`, `Order Country_grouped`) are labels; use their one-hot columns in models. "
            f"Collinear pairs left in the file: {corr_bits}. "
            "`Order Status` is unresolved. "
            f"`random_state = {RANDOM_STATE}` is the project seed; this phase did not split the data."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the Phase 1 pipeline and write the cleaned file, log, and hand-off."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(RANDOM_STATE)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    args = parse_args()
    raw_path = resolve_repo_path(args.raw)
    logger.info("Loading %s", rel(raw_path))
    df = load_raw(raw_path)
    n_rows, n_cols_raw = df.shape
    raw_columns = list(df.columns)
    baselines = {column: df[column].copy() for column in BASELINE_COLUMNS}

    null_counts = df.isna().sum()
    null_columns = [column for column in df.columns if int(null_counts[column]) > 0]
    dtypes = [(column, str(df[column].dtype)) for column in df.columns]
    null_rows = [
        (column, int(null_counts[column]), 100.0 * int(null_counts[column]) / n_rows) for column in null_columns
    ]
    memory_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    duplicate_rows_raw = int(df.duplicated().sum())

    pair_stats: dict[tuple[str, str], tuple[int, float]] = {}
    for left, right in IDENTITY_PAIRS:
        if left in df.columns and right in df.columns:
            pair_stats[(left, right)] = differing_rows(df[left], df[right])

    discrepancies: list[str] = []
    checks: list[dict[str, str]] = []

    def add_check(name: str, expected: str, actual: str, ok: bool, discrepancy: str | None = None) -> None:
        checks.append({"check": name, "expected": expected, "actual": actual, "result": "PASS" if ok else "DISCREPANCY"})
        if not ok and discrepancy:
            discrepancies.append(discrepancy)

    add_check("Rows", fmt_int(EXPECTED_ROWS), fmt_int(n_rows), n_rows == EXPECTED_ROWS,
              f"Row count is {fmt_int(n_rows)}, not {fmt_int(EXPECTED_ROWS)}.")
    add_check("Columns", str(EXPECTED_COLS), str(n_cols_raw), n_cols_raw == EXPECTED_COLS,
              f"Column count is {n_cols_raw}, not {EXPECTED_COLS}.")
    add_check("Duplicate rows", "0", fmt_int(duplicate_rows_raw), duplicate_rows_raw == 0,
              f"Duplicate rows are {fmt_int(duplicate_rows_raw)}, not 0.")

    late_count_raw = int((df["Late_delivery_risk"] == 1).sum())
    loss_count_raw = int((df["Benefit per order"] < 0).sum())
    zero_count_raw = int((df["Benefit per order"] == 0).sum())
    late_pct_value = 100.0 * late_count_raw / n_rows
    loss_pct_value = 100.0 * loss_count_raw / n_rows
    zero_pct_value = 100.0 * zero_count_raw / n_rows
    add_check(
        "Late_delivery_risk = 1",
        "54.8% (tolerance 0.1 percentage points)",
        f"{late_pct_value:.4f}% ({fmt_int(late_count_raw)} rows; difference {abs(late_pct_value - 54.8):.4f} pp)",
        abs(late_pct_value - 54.8) <= TOLERANCE_PP,
        f"Late rate is {late_pct_value:.4f}%, outside 0.1 percentage points of 54.8%.",
    )
    add_check(
        "Benefit per order < 0",
        "18.7% (tolerance 0.1 percentage points)",
        f"{loss_pct_value:.4f}% ({fmt_int(loss_count_raw)} rows; difference {abs(loss_pct_value - 18.7):.4f} pp)",
        abs(loss_pct_value - 18.7) <= TOLERANCE_PP,
        f"Loss rate is {loss_pct_value:.4f}%, outside 0.1 percentage points of 18.7%.",
    )
    add_check(
        "Benefit per order = 0",
        "0.7% (tolerance 0.1 percentage points)",
        f"{zero_pct_value:.4f}% ({fmt_int(zero_count_raw)} rows; difference {abs(zero_pct_value - 0.7):.4f} pp)",
        abs(zero_pct_value - 0.7) <= TOLERANCE_PP,
        f"Zero-profit rate is {zero_pct_value:.4f}%, outside 0.1 percentage points of 0.7%.",
    )

    expected_nulls = {
        "Product Description": n_rows,
        "Order Zipcode": 155_679,
        "Customer Zipcode": 3,
        "Customer Lname": 8,
    }
    for column, expected in expected_nulls.items():
        actual = int(df[column].isna().sum()) if column in df.columns else -1
        if column == "Product Description":
            expected_label = f"100% ({fmt_int(n_rows)} rows)"
        else:
            expected_label = fmt_int(expected)
        add_check(
            f"Nulls in {column}",
            expected_label,
            fmt_int(actual),
            actual == expected,
            f"Nulls in `{column}` are {fmt_int(actual)}, not {expected_label}.",
        )

    dropped: list[dict[str, str]] = []
    empty_reasons = {
        "Product Description": "Entirely null in this file, so it has no information.",
        "Order Zipcode": "Almost entirely null in this file, so it is not a usable feature.",
        "Product Status": (
            f"Listed as empty or useless. Distinct values in this file: {int(df['Product Status'].nunique(dropna=False))}."
            if "Product Status" in df.columns
            else "Listed as empty or useless."
        ),
        "Product Image": "Image URL, not used by the models or the hypothesis tests.",
    }
    personal_reasons = {
        column: "Personal data. Removed so the cleaned file can be shared inside the group."
        for column in PERSONAL_COLUMNS
    }
    id_reasons = dict(ID_REASONS)
    for left, right in (
        ("Customer Id", "Order Customer Id"),
        ("Category Id", "Product Category Id"),
        ("Product Card Id", "Order Item Cardprod Id"),
    ):
        stats = pair_stats.get((left, right))
        if stats is not None and stats[0] == 0:
            id_reasons[left] += f" Matches `{right}` on every row (maximum absolute difference {stats[1]:.6f})."
            id_reasons[right] += f" Matches `{left}` on every row (maximum absolute difference {stats[1]:.6f})."

    benefit_pair = pair_stats.get(("Benefit per order", "Order Profit Per Order"))
    profit_reasons = dict(PROFIT_LEAKAGE_RULES)
    if benefit_pair is not None:
        profit_reasons["Order Profit Per Order"] += (
            f" Before the drop, differing rows versus `Benefit per order`: {fmt_int(benefit_pair[0])}; "
            f"maximum absolute difference {benefit_pair[1]:.6f}."
        )

    df = drop_listed(df, EMPTY_COLUMNS, "Empty or useless", empty_reasons, dropped, discrepancies)
    df = drop_listed(df, PERSONAL_COLUMNS, "Personal", personal_reasons, dropped, discrepancies)
    df = drop_listed(df, ID_COLUMNS, "ID columns", id_reasons, dropped, discrepancies)
    df = drop_listed(df, LATE_LEAKAGE_COLUMNS, "Leakage for the late-delivery model", LATE_LEAKAGE_RULES, dropped, discrepancies)
    df = drop_listed(df, PROFIT_LEAKAGE_COLUMNS, "Leakage for the loss and profit models", profit_reasons, dropped, discrepancies)
    if len(df) != n_rows:
        raise AssertionError("Rows were lost during column drops.")
    n_cols_after_drop = df.shape[1]
    logger.info("After drops: %s columns", n_cols_after_drop)

    df = parse_order_date(df)
    date_min = df[ORDER_DATE_COL].min()
    date_max = df[ORDER_DATE_COL].max()
    if not isinstance(date_min, pd.Timestamp) or not isinstance(date_max, pd.Timestamp):
        raise AssertionError("Order date minimum or maximum is missing.")
    add_check(
        "Order date minimum",
        "2015-01-01",
        date_min.strftime("%Y-%m-%d %H:%M:%S"),
        date_min.date() == datetime(2015, 1, 1).date(),
        f"Order date minimum is {date_min}, not 1 Jan 2015.",
    )
    add_check(
        "Order date maximum",
        "2017-09-09",
        date_max.strftime("%Y-%m-%d %H:%M:%S"),
        date_max.date() == datetime(2017, 9, 9).date(),
        (
            f"The project brief expected order dates from 1 Jan 2015 to 9 Sep 2017. "
            f"The minimum is {date_min.strftime('%Y-%m-%d %H:%M:%S')}. "
            f"The maximum is {date_max.strftime('%Y-%m-%d %H:%M:%S')}, which is after 9 Sep 2017. "
            "No rows were removed to force the expected window."
        ),
    )
    n_cols_after_dates = df.shape[1]

    pairs: list[dict[str, object]] = []
    benefit_n, benefit_max = pair_stats.get(("Benefit per order", "Order Profit Per Order"), (-1, float("nan")))
    pairs.append(
        {
            "left": "Benefit per order",
            "right": "Order Profit Per Order",
            "n_diff": benefit_n,
            "max_abs": benefit_max,
            "action": "Identical before the Step 2 drop. `Order Profit Per Order` was dropped as leakage. `Benefit per order` was kept.",
        }
    )
    sales_pair = pair_stats.get(("Sales per customer", "Order Item Total"))
    if sales_pair is None:
        discrepancies.append("`Sales per customer` or `Order Item Total` was missing, so the pair was not compared.")
        sales_n, sales_max = -1, float("nan")
        sales_action = "Pair could not be compared."
    else:
        sales_n, sales_max = sales_pair
        if sales_n == 0 and "Sales per customer" in df.columns:
            df = df.drop(columns=["Sales per customer"])
            dropped.append(
                {
                    "column": "Sales per customer",
                    "category": "Exact duplicate",
                    "reason": (
                        "Identical to `Order Item Total` "
                        f"({fmt_int(sales_n)} differing rows, maximum absolute difference {sales_max:.6f}, tolerance {FLOAT_TOL:.0e}). "
                        "`Order Item Total` was kept."
                    ),
                }
            )
            sales_action = "Identical. Dropped `Sales per customer` and kept `Order Item Total`."
        elif sales_n == 0:
            sales_action = "Identical, and `Sales per customer` was already absent."
        else:
            sales_action = "Not identical. Both columns kept."
            discrepancies.append(
                f"`Sales per customer` and `Order Item Total` differ on {fmt_int(sales_n)} rows "
                f"(maximum absolute difference {sales_max:.6f}). Both were kept."
            )
    pairs.append(
        {
            "left": "Sales per customer",
            "right": "Order Item Total",
            "n_diff": sales_n,
            "max_abs": sales_max,
            "action": sales_action,
        }
    )

    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    already_dropping: set[str] = set()
    for index, earlier in enumerate(numeric_columns):
        if earlier in already_dropping:
            continue
        for later in numeric_columns[index + 1 :]:
            if later in already_dropping:
                continue
            n_diff, max_abs = differing_rows(df[earlier], df[later])
            if n_diff != 0:
                continue
            drop_name = choose_duplicate_drop(earlier, later)
            action = (
                f"Identical. Dropped `{drop_name}` and kept the other column."
                if drop_name is not None
                else "Identical, but both columns are protected, so both were kept."
            )
            pairs.append(
                {
                    "left": earlier,
                    "right": later,
                    "n_diff": n_diff,
                    "max_abs": max_abs,
                    "action": action,
                }
            )
            if drop_name is None:
                assumptions_hold = (
                    f"`{earlier}` and `{later}` are exact duplicates and both are protected, so neither was dropped."
                )
                discrepancies.append(assumptions_hold)
                continue
            keep_name = later if drop_name == earlier else earlier
            already_dropping.add(drop_name)
            dropped.append(
                {
                    "column": drop_name,
                    "category": "Exact duplicate",
                    "reason": (
                        f"Exact duplicate of `{keep_name}` "
                        f"({fmt_int(n_diff)} differing rows, maximum absolute difference {max_abs:.6f}, tolerance {FLOAT_TOL:.0e}). "
                        f"`{keep_name}` was kept because it appears earlier or is protected."
                    ),
                }
            )
    if already_dropping:
        df = df.drop(columns=sorted(already_dropping, key=lambda column: numeric_columns.index(column)))
        logger.info("Dropped exact duplicates: %s", sorted(already_dropping))

    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    corr = df[numeric_columns].corr(method="pearson")
    correlations: list[dict[str, object]] = []
    for index, left in enumerate(numeric_columns):
        for right in numeric_columns[index + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            if abs(float(value)) > 0.95:
                correlations.append({"left": left, "right": right, "r": float(value)})
    correlations.sort(key=lambda row: (-abs(float(row["r"])), str(row["left"]), str(row["right"])))

    null_notes: list[str] = []
    zip_filled = 0
    if "Customer Zipcode" in df.columns:
        zip_filled = int(df["Customer Zipcode"].isna().sum())
        df["Customer Zipcode"] = format_zip(df["Customer Zipcode"])
        null_notes.append(
            f"`Customer Zipcode`: {fmt_int(zip_filled)} missing values filled with `Unknown`. "
            "Non-missing values are integral floats and were stored as digit strings without a decimal point."
        )
    other_nulls = [column for column in df.columns if int(df[column].isna().sum()) > 0]
    for column in other_nulls:
        n_missing = int(df[column].isna().sum())
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = float(df[column].median())
            df[column] = df[column].fillna(fill_value)
            null_notes.append(
                f"`{column}`: {fmt_int(n_missing)} missing values filled with the column median ({fill_value:.6f}). Rows were not dropped."
            )
        else:
            df[column] = df[column].fillna("Unknown")
            null_notes.append(
                f"`{column}`: {fmt_int(n_missing)} missing values filled with `Unknown`. Rows were not dropped."
            )
    null_cells_after = int(df.isna().sum().sum())
    if null_cells_after != 0:
        raise AssertionError(f"Nulls remain after Step 4: {null_cells_after}.")
    n_cols_after_dups = df.shape[1]
    original_order = [column for column in raw_columns if column in df.columns]

    review_columns = ["Sales", "Benefit per order", "Order Item Quantity"]
    outliers: list[dict[str, object]] = []
    for column in review_columns:
        report = iqr_report(df[column])
        report["column"] = column
        outliers.append(report)
    if (df["Sales"] < 0).any():
        raise AssertionError("Sales has negative values, so log1p was not applied.")
    df["Sales_log"] = np.log1p(df["Sales"].to_numpy(dtype=float))
    benefit_values = df["Benefit per order"].to_numpy(dtype=float)
    df["Benefit_signed_log"] = np.sign(benefit_values) * np.log1p(np.abs(benefit_values))
    quantity_skew = float(df["Order Item Quantity"].skew())
    quantity_log_skew = float(pd.Series(np.log1p(df["Order Item Quantity"].to_numpy(dtype=float))).skew())
    transform_columns = ["Sales_log", "Benefit_signed_log"]
    quantity_logged = abs(quantity_skew) > SKEW_ABS_THRESHOLD
    if quantity_logged:
        df["Order_Item_Quantity_log"] = np.log1p(df["Order Item Quantity"].to_numpy(dtype=float))
        transform_columns.append("Order_Item_Quantity_log")

    def skew_line(source: str, transformed: pd.Series, added_name: str) -> str:
        before = float(df[source].skew())
        after = float(transformed.skew())
        helped = "It reduced absolute skewness." if abs(after) < abs(before) else "It did not reduce absolute skewness."
        return (
            f"`{added_name}` from `{source}`: skewness {before:.6f} before, {after:.6f} after. {helped}"
        )

    transform_lines = [
        skew_line("Sales", df["Sales_log"], "Sales_log"),
        skew_line("Benefit per order", df["Benefit_signed_log"], "Benefit_signed_log"),
    ]
    if quantity_logged:
        transform_lines.append(skew_line("Order Item Quantity", df["Order_Item_Quantity_log"], "Order_Item_Quantity_log"))
    else:
        transform_lines.append(
            f"`Order Item Quantity` skewness is {quantity_skew:.6f}. "
            f"A log column is added only when absolute skewness is above {SKEW_ABS_THRESHOLD:.1f}, so no log column was added. "
            f"`log1p` of this column would have skewness {quantity_log_skew:.6f}, which was computed only to record the decision."
        )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for old_plot in PLOT_DIR.glob("*.png"):
        old_plot.unlink()
    plot_files: list[str] = []
    plot_plan = [
        ("Sales", "Sales", "Sales_log", "Sales_log"),
        ("Benefit per order", "Benefit per order", "Benefit_signed_log", "Benefit_signed_log"),
        ("Order Item Quantity", "Order Item Quantity", "Order_Item_Quantity_log" if quantity_logged else None, "Order_Item_Quantity_log"),
    ]
    for column, ylabel, transform_name, transform_label in plot_plan:
        box_name = f"box_{plot_slug(column)}.png"
        box_path = PLOT_DIR / box_name
        save_boxplot(df[column], f"{column}: box plot (1.5 IQR fences)", ylabel, box_path)
        plot_files.append(rel(box_path))
        if transform_name is not None:
            before_name = f"hist_{plot_slug(column)}_before.png"
            after_name = f"hist_{plot_slug(column)}_after.png"
            save_histogram(df[column], f"{column} before transform", ylabel, PLOT_DIR / before_name)
            save_histogram(df[transform_name], f"{column} after transform", transform_label, PLOT_DIR / after_name)
            plot_files.extend([rel(PLOT_DIR / before_name), rel(PLOT_DIR / after_name)])
    n_cols_after_outliers = df.shape[1]

    added: list[dict[str, str]] = [
        {"column": "order_year", "type": "derived date (int64)", "how": "Calendar year of `order date (DateOrders)`."},
        {"column": "order_month", "type": "derived date (int64)", "how": "Calendar month of `order date (DateOrders)`, numbered 1 to 12."},
        {"column": "order_weekday", "type": "derived date (int64)", "how": "Weekday of `order date (DateOrders)`. Monday is 0 and Sunday is 6."},
        {
            "column": "order_weekday_name",
            "type": "derived date (text)",
            "how": "English weekday name mapped from `order_weekday` with a fixed Monday-first list.",
        },
        {
            "column": "Sales_log",
            "type": "transform (float64)",
            "how": "`log1p(Sales)`. `Sales` was asserted non-negative. The original column is unchanged.",
        },
        {
            "column": "Benefit_signed_log",
            "type": "transform (float64)",
            "how": "`sign(Benefit per order) * log1p(abs(Benefit per order))`. The original column is unchanged.",
        },
    ]
    if quantity_logged:
        added.append(
            {
                "column": "Order_Item_Quantity_log",
                "type": "transform (float64)",
                "how": "`log1p(Order Item Quantity)`, added because absolute skewness exceeded the threshold. The original column is unchanged.",
            }
        )

    encoding: list[dict[str, object]] = []
    encoding_blocks: list[str] = []
    one_hot_columns: list[str] = []
    one_hot_sources: dict[str, str] = {}
    grouped_columns: list[str] = []
    grouped_sources: dict[str, str] = {}
    group_checks: list[str] = []

    def attach_dummies(source_series: pd.Series, prefix: str, source_label: str) -> None:
        dummies = one_hot_frame(source_series, prefix)
        overlap = set(dummies.columns) & set(df.columns)
        if overlap:
            raise AssertionError(f"One-hot names collide with existing columns: {sorted(overlap)}")
        for column in dummies.columns:
            level = column[len(prefix) + 1 :]
            one_hot_columns.append(column)
            one_hot_sources[column] = source_label
            added.append(
                {
                    "column": column,
                    "type": "one-hot (int64)",
                    "how": f"Integer 0/1 indicator for `{source_label}` = {level!r}. Every level of this group is kept.",
                }
            )
        group_checks.append(
            f"`{source_label}`: {len(dummies.columns)} dummy columns, integer dtype, values in {{0, 1}}, row sums equal 1 on all {fmt_int(len(df))} rows."
        )
        df_columns[prefix] = dummies

    df_columns: dict[str, pd.DataFrame] = {}
    for column in LOW_CARD_COLUMNS:
        counts = df[column].astype(str).value_counts()
        ranked = sorted(((str(label), int(count)) for label, count in counts.items()), key=lambda item: (-item[1], item[0]))
        names = ", ".join(f"{label!r} ({fmt_int(count)})" for label, count in ranked)
        encoding.append(
            {
                "column": column,
                "method": "one-hot, all levels, integer 0/1",
                "levels": str(len(ranked)),
                "notes": f"All levels kept, in frequency order: {names}.",
            }
        )
        encoding_blocks.append(f"`{column}`: {len(ranked)} unique values, all one-hot encoded. {names}.")
        attach_dummies(df[column], column, column)

    for column in HIGH_CARD_COLUMNS:
        ranked_all = sorted(
            ((str(label), int(count)) for label, count in df[column].astype(str).value_counts().items()),
            key=lambda item: (-item[1], item[0]),
        )
        unique_n = len(ranked_all)
        if unique_n <= TOP_N:
            kept = ranked_all
            grouped = df[column].astype(str)
            other_rows = 0
        else:
            kept = ranked_all[:TOP_N]
            keep_set = {label for label, _count in kept}
            text = df[column].astype(str)
            if "Other" in set(text.unique()):
                discrepancies.append(
                    f"`{column}` already contains the label `Other`, so grouped rare levels are not distinguishable from that original label."
                )
            grouped = text.where(text.isin(keep_set), other="Other")
            other_rows = n_rows - sum(count for _label, count in kept)
        covered = sum(count for _label, count in kept)
        coverage = 100.0 * covered / n_rows
        grouped_name = f"{column}_grouped"
        df[grouped_name] = grouped
        grouped_columns.append(grouped_name)
        grouped_sources[grouped_name] = column
        kept_text = ", ".join(f"{label!r} ({fmt_int(count)})" for label, count in kept)
        level_count = int(grouped.nunique())
        other_note = (
            f" Remaining rows mapped to `Other`: {fmt_int(other_rows)} ({fmt_pct(other_rows, n_rows)})."
            if unique_n > TOP_N
            else " No `Other` bucket was needed."
        )
        encoding.append(
            {
                "column": column,
                "method": f"top-{TOP_N} grouping, then one-hot of `{grouped_name}`, integer 0/1",
                "levels": str(level_count),
                "notes": (
                    f"{fmt_int(unique_n)} unique values; {len(kept)} kept; "
                    f"kept rows cover {coverage:.4f}%.{other_note}"
                ),
            }
        )
        encoding_blocks.append(
            f"`{column}`: {fmt_int(unique_n)} unique values; {len(kept)} kept; "
            f"coverage {coverage:.4f}% ({fmt_int(covered)} of {fmt_int(n_rows)} rows). "
            f"Kept values: {kept_text}.{other_note}"
        )
        added.append(
            {
                "column": grouped_name,
                "type": "grouped text",
                "how": (
                    f"Top {min(TOP_N, unique_n)} values of `{column}` by row count, ties broken alphabetically. "
                    + ("Other values mapped to `Other`." if unique_n > TOP_N else "Every original value was kept.")
                ),
            }
        )
        attach_dummies(df[grouped_name], grouped_name, grouped_name)

    dummy_frames = [frame for frame in df_columns.values()]
    if dummy_frames:
        df = pd.concat([df, *dummy_frames], axis=1)
    n_cols_after_encode = df.shape[1]

    whitespace_hits: list[str] = []
    for column in LOW_CARD_COLUMNS + HIGH_CARD_COLUMNS + TEXT_ONLY_COLUMNS:
        if column not in df.columns or column == "Customer Zipcode":
            continue
        padded = sorted({value for value in df[column].astype(str).unique() if value != value.strip()})
        if padded:
            shown = ", ".join(repr(value) for value in padded)
            whitespace_hits.append(f"`{column}`: {shown}")

    df["late"] = df["Late_delivery_risk"].astype("int64")
    df["loss"] = (df["Benefit per order"] < 0).astype("int64")
    if not set(df["late"].unique()).issubset({0, 1}) or not set(df["loss"].unique()).issubset({0, 1}):
        raise AssertionError("late or loss is not strictly 0/1.")
    if df["late"].isna().any() or df["loss"].isna().any():
        raise AssertionError("late or loss contains nulls.")
    late_mismatch = int((df["late"] != df["Late_delivery_risk"]).sum())
    if late_mismatch != 0:
        raise AssertionError("late is not identical to Late_delivery_risk.")
    zero_count = int((df["Benefit per order"] == 0).sum())
    zero_as_not_loss = int(((df["Benefit per order"] == 0) & (df["loss"] == 0)).sum())
    if zero_as_not_loss != zero_count:
        raise AssertionError("A zero-profit row was coded as loss.")
    late_count = int((df["late"] == 1).sum())
    late_zero = int((df["late"] == 0).sum())
    loss_count = int((df["loss"] == 1).sum())
    loss_zero = int((df["loss"] == 0).sum())
    added.extend(
        [
            {
                "column": "late",
                "type": "target (int64)",
                "how": "Copied from `Late_delivery_risk`. The original column is also kept.",
            },
            {
                "column": "loss",
                "type": "target (int64)",
                "how": "1 if `Benefit per order` < 0, else 0. Zero-profit orders are 0.",
            },
        ]
    )

    df = reorder_columns(df, original_order, transform_columns, grouped_columns, one_hot_columns)
    if len(df) != n_rows:
        raise AssertionError("Row count changed before save.")
    for column, baseline in baselines.items():
        if not np.array_equal(df[column].to_numpy(), baseline.to_numpy()):
            raise AssertionError(f"`{column}` no longer matches the raw values.")
    if int(df.isna().sum().sum()) != 0:
        raise AssertionError("Nulls appeared before save.")

    cleaned_path = OUTPUT_DIR / "cleaned_supply_chain.csv"
    logger.info("Writing %s", rel(cleaned_path))
    save_and_check(df, cleaned_path)
    file_size_bytes = cleaned_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    duplicate_rows_clean = int(df.duplicated().sum())

    export_for_types = df.copy()
    export_for_types[ORDER_DATE_COL] = export_for_types[ORDER_DATE_COL].dt.strftime(ISO_FORMAT)
    notes = {
        "Days for shipment (scheduled)": "Known at order time. This is scheduled days, not the real shipping delay, so it is not leakage.",
        "Order Item Total": "Unscaled. Kept because it matched `Sales per customer` on every row; that column was dropped.",
        "Order Item Product Price": "Unscaled. Kept because it matched `Product Price` on every row; that column was dropped.",
        "Sales": "Unscaled item sales. `Sales_log` is the skew transform. This column is not an exact copy of `Order Item Total`.",
        "Latitude": "Geographic coordinate from the raw file. Unscaled.",
        "Longitude": "Geographic coordinate from the raw file. Unscaled.",
        "Order Item Quantity": (
            f"Unscaled quantity. Absolute skewness {quantity_skew:.6f} did not exceed {SKEW_ABS_THRESHOLD:.1f}, so no log column was added."
            if not quantity_logged
            else "Unscaled quantity. A log1p copy is in `Order_Item_Quantity_log`."
        ),
        "Order Item Discount": "Unscaled discount amount from the raw file.",
        "Order Item Discount Rate": "Unscaled discount rate from the raw file. Not standardised.",
        "Customer Zipcode": (
            f"Cast to text. {fmt_int(zip_filled)} missing values filled with Unknown. "
            "Retained as text, not for modelling without a decision."
        ),
        "Order Status": "Retained as text, not for modelling without a decision. It may carry post-order information such as cancellation.",
        "order_year": "Calendar year of the order date. Not derived from the shipping date.",
        "order_month": "Calendar month of the order date, 1 to 12. Not derived from the shipping date.",
        "order_weekday": "Weekday of the order date, Monday = 0. Not derived from the shipping date.",
        "Sales_log": "`log1p` of `Sales`. The original `Sales` column is unchanged and unscaled.",
        "Benefit_signed_log": "Signed log of `Benefit per order` for skew review. Not a feature for loss or OLS, because it is a transform of the profit target.",
        "Order_Item_Quantity_log": "`log1p` of `Order Item Quantity`. The original column is unchanged.",
    }
    for column in TEXT_ONLY_COLUMNS:
        notes.setdefault(column, "Retained as text, not for modelling without a decision.")
    meta = {
        "transforms": set(transform_columns),
        "one_hot": one_hot_sources,
        "grouped_sources": grouped_sources,
        "encoded_originals": set(LOW_CARD_COLUMNS + HIGH_CARD_COLUMNS),
        "notes": notes,
    }
    dictionary: list[dict[str, str]] = []
    for column in export_for_types.columns:
        role, models, note = describe_column(column, meta)
        dictionary.append(
            {
                "name": column,
                "dtype": str(export_for_types[column].dtype),
                "role": role,
                "models": models,
                "note": note,
            }
        )
    if len(dictionary) != export_for_types.shape[1]:
        raise AssertionError("Column dictionary does not cover every final column.")

    ranges = "; ".join(
        f"`{row['column']}` {fmt_float(float(row['minimum']))} to {fmt_float(float(row['maximum']))}" for row in outliers
    )
    benefit_outlier = next(row for row in outliers if row["column"] == "Benefit per order")
    assumptions = [
        f"`random_state = {RANDOM_STATE}` is set, and no step samples or shuffles rows. The pipeline is deterministic.",
        "No rows are deleted, including negative-profit rows. Nothing is scaled or split in this phase.",
        "Original category text is kept beside the one-hot columns so Power BI can read the labels. Models should use the 0/1 columns. No reference level is dropped here.",
        f"`TOP_N = {TOP_N}`. Ties in frequency are broken by sorting the label alphabetically. Coverage of the kept labels is in the encoding table. `Order State` coverage is low, so most state rows fall in `Other`; the constant was not changed.",
        "Near-duplicates: `Sales per customer` was dropped in favour of `Order Item Total` because the two matched within tolerance. `Product Price` was dropped in favour of `Order Item Product Price` by the same exact-match rule, keeping the earlier protected column.",
        "`Order Status` is retained as text and is not encoded. Levels such as cancellation, suspected fraud, and payment review can be known only after the order is placed. Whether any model may use it is an open decision. It was not dropped.",
        f"A log column for `Order Item Quantity` is added only if absolute skewness exceeds {SKEW_ABS_THRESHOLD:.1f}. The decision and both skewness figures are in Step 5.",
        "One-hot columns inside each group are sorted alphabetically. This does not drop a level.",
        "Weekday names come from a fixed English list with Monday = 0, so they do not depend on the machine locale.",
        f"`Customer Zipcode` nulls ({fmt_int(zip_filled)}) were filled with `Unknown` and the rest were written as text. The column was not one-hot encoded.",
        "Category labels were not stripped. Source values with leading or trailing spaces were kept so the original text stays faithful to the file.",
        (
            "Labels whose stored text is not equal to their stripped text: " + "; ".join(whitespace_hits)
            if whitespace_hits
            else "No encoded or retained text column had labels that differed from their stripped form."
        ),
        "`Customer Country` and the other text-only columns were not encoded because they are outside the specified one-hot and top-N lists, even where cardinality is low.",
        "The order date was parsed as month/day/year hour:minute because that format accepted every value. A day-first format does not.",
        (
            f"Cleaned duplicate rows are {fmt_int(duplicate_rows_clean)}, the same as the raw file ({fmt_int(duplicate_rows_raw)}). "
            "No row was deleted. Dropping the identifier columns did not create duplicate rows in this file."
            if duplicate_rows_clean == duplicate_rows_raw
            else (
                f"Cleaned duplicate rows are {fmt_int(duplicate_rows_clean)} and raw duplicate rows are {fmt_int(duplicate_rows_raw)}. "
                "The increase comes from removing identifier columns. No row was deleted."
            )
        ),
    ]

    logger.info("Running validator")
    validation_text, validation_code = run_validator(raw_path, cleaned_path)
    if validation_code != 0:
        discrepancies.append(f"`phase1_validate.py` exited with code {validation_code}. See Verification results.")

    state: dict[str, object] = {
        "run_date": date.today().isoformat(),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "raw_rel": rel(raw_path),
        "n_rows": n_rows,
        "n_cols_raw": n_cols_raw,
        "n_cols_after_drop": n_cols_after_drop,
        "n_cols_after_dates": n_cols_after_dates,
        "n_cols_after_dups": n_cols_after_dups,
        "n_cols_after_outliers": n_cols_after_outliers,
        "n_cols_after_encode": n_cols_after_encode,
        "n_cols_final": df.shape[1],
        "null_cells_before": int(null_counts.sum()),
        "null_cells_after": null_cells_after,
        "n_cols_without_nulls": int((null_counts == 0).sum()),
        "memory_mb": memory_mb,
        "duplicate_rows_raw": duplicate_rows_raw,
        "duplicate_rows_clean": duplicate_rows_clean,
        "dtypes": dtypes,
        "nulls": null_rows,
        "checks": checks,
        "dropped": dropped,
        "added": added,
        "outliers": outliers,
        "pairs": pairs,
        "correlations": correlations,
        "encoding": encoding,
        "encoding_blocks": encoding_blocks,
        "one_hot_columns": one_hot_columns,
        "group_checks": group_checks,
        "plot_files": plot_files,
        "transform_lines": transform_lines,
        "dictionary": dictionary,
        "discrepancies": discrepancies,
        "assumptions": assumptions,
        "date_min": date_min.strftime("%Y-%m-%d %H:%M:%S"),
        "date_max": date_max.strftime("%Y-%m-%d %H:%M:%S"),
        "date_parse_failures": 0,
        "null_handling": " ".join(null_notes) if null_notes else "No nulls remained.",
        "ranges": ranges,
        "benefit_negative_outliers": int(benefit_outlier["n_negative_out"]),
        "benefit_outlier_rows": int(benefit_outlier["n_out"]),
        "late_count": late_count,
        "late_zero": late_zero,
        "late_pct": fmt_pct(late_count, n_rows),
        "late_zero_pct": fmt_pct(late_zero, n_rows),
        "late_mismatch": late_mismatch,
        "loss_count": loss_count,
        "loss_zero": loss_zero,
        "loss_pct": fmt_pct(loss_count, n_rows),
        "loss_zero_pct": fmt_pct(loss_zero, n_rows),
        "zero_count": zero_count,
        "zero_pct": fmt_pct(zero_count, n_rows),
        "zero_as_not_loss": zero_as_not_loss,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "size_flag": file_size_bytes > GITHUB_WARN_BYTES,
        "validation_text": validation_text,
        "validation_code": validation_code,
        "quantity_skew": quantity_skew,
    }
    log_path = OUTPUT_DIR / "cleaning_log.md"
    handoff_path = OUTPUT_DIR / "handoff_phase1.md"
    log_path.write_text(render_log(state), encoding="utf-8")
    handoff_path.write_text(render_handoff(state), encoding="utf-8")
    logger.info("Wrote %s", rel(log_path))
    logger.info("Wrote %s", rel(handoff_path))
    logger.info("File size %.2f MB; validator exit %s", file_size_mb, validation_code)
    return validation_code


if __name__ == "__main__":
    sys.exit(main())
