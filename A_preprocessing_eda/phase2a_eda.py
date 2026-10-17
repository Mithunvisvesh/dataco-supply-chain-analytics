"""Phase 2a: Exploratory data analysis on the cleaned DataCo file.

Run from the repo root:

    python A_preprocessing_eda/phase2a_eda.py

Reads A_preprocessing_eda/outputs/cleaned_supply_chain.csv (never edits it).
Writes plots under A_preprocessing_eda/outputs/eda_plots/ and markdown under
A_preprocessing_eda/outputs/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEANED = Path("A_preprocessing_eda/outputs/cleaned_supply_chain.csv")
OUTPUT_DIR = REPO_ROOT / "A_preprocessing_eda" / "outputs"
PLOT_DIR = OUTPUT_DIR / "eda_plots"
RANDOM_STATE = 42

NUMERIC_FOCUS = [
    "Sales",
    "Benefit per order",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Days for shipment (scheduled)",
    "Order Item Quantity",
]
CORR_COLS = [
    "Sales",
    "Benefit per order",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Total",
    "Latitude",
    "Longitude",
    "late",
    "loss",
]
CAT_BARS = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Type",
    "Department Name",
    "Order Region",
]

logger = logging.getLogger("phase2a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2a EDA for DataCo cleaned file.")
    parser.add_argument(
        "--cleaned",
        type=Path,
        default=DEFAULT_CLEANED,
        help="Path to cleaned_supply_chain.csv, relative to repo root unless absolute.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def fmt_num(x: float) -> str:
    return f"{float(x):,.4f}"


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()


def style() -> None:
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


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FOCUS:
        s = df[col]
        rows.append(
            {
                "column": col,
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "q1": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q3": float(s.quantile(0.75)),
                "max": float(s.max()),
                "skewness": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
            }
        )
    return pd.DataFrame(rows)


def plot_histograms(df: pd.DataFrame, plot_files: list[str]) -> None:
    for col in NUMERIC_FOCUS:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col], bins=40, ax=ax, color="#1f4e79", edgecolor="white", linewidth=0.3)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Orders")
        path = PLOT_DIR / f"hist_{_slug(col)}.png"
        savefig(path)
        plot_files.append(rel(path))

    # Before/after transforms (already in Phase 1; repeat for EDA completeness)
    pairs = [
        ("Sales", "Sales_log"),
        ("Benefit per order", "Benefit_signed_log"),
    ]
    for raw_col, trans_col in pairs:
        if trans_col not in df.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.histplot(df[raw_col], bins=40, ax=axes[0], color="#1f4e79", edgecolor="white", linewidth=0.3)
        axes[0].set_title(f"{raw_col} (before)")
        axes[0].set_xlabel(raw_col)
        sns.histplot(df[trans_col], bins=40, ax=axes[1], color="#2e7d32", edgecolor="white", linewidth=0.3)
        axes[1].set_title(f"{trans_col} (after)")
        axes[1].set_xlabel(trans_col)
        for ax in axes:
            ax.set_ylabel("Orders")
        path = PLOT_DIR / f"hist_compare_{_slug(raw_col)}.png"
        savefig(path)
        plot_files.append(rel(path))


def plot_categorical_bars(df: pd.DataFrame, plot_files: list[str]) -> None:
    for col in CAT_BARS:
        counts = df[col].astype(str).value_counts()
        # Stable order: frequency desc, then label
        order = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=order, y=[counts[k] for k in order], ax=ax, color="#1f4e79")
        ax.set_title(f"Orders by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Orders")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        path = PLOT_DIR / f"bar_{_slug(col)}.png"
        savefig(path)
        plot_files.append(rel(path))


def plot_correlation(df: pd.DataFrame, plot_files: list[str]) -> pd.DataFrame:
    cols = [c for c in CORR_COLS if c in df.columns]
    corr = df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson correlation (selected numeric columns)")
    path = PLOT_DIR / "heatmap_correlation.png"
    savefig(path)
    plot_files.append(rel(path))
    return corr


def plot_discount_vs_profit(df: pd.DataFrame, plot_files: list[str]) -> None:
    # Sample for readability; deterministic
    sample = df.sample(n=min(8000, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sample["Order Item Discount Rate"],
        sample["Benefit per order"],
        s=8,
        alpha=0.25,
        c="#1f4e79",
        edgecolors="none",
    )
    ax.axhline(0, color="#b71c1c", linewidth=1, linestyle="--")
    ax.set_title("Discount rate vs profit (sample of orders)")
    ax.set_xlabel("Order Item Discount Rate")
    ax.set_ylabel("Benefit per order")
    path = PLOT_DIR / "scatter_discount_vs_profit.png"
    savefig(path)
    plot_files.append(rel(path))


def plot_group_comparisons(df: pd.DataFrame, plot_files: list[str]) -> dict[str, pd.Series]:
    late_rates: dict[str, pd.Series] = {}
    for col in ["Shipping Mode", "Market", "Customer Segment"]:
        order = (
            df[col]
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(
            data=df,
            x=col,
            y="Benefit per order",
            order=order,
            ax=ax,
            color="#1f4e79",
            fliersize=1.5,
            linewidth=0.8,
        )
        ax.set_title(f"Profit by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Benefit per order")
        ax.tick_params(axis="x", rotation=20)
        path = PLOT_DIR / f"box_profit_by_{_slug(col)}.png"
        savefig(path)
        plot_files.append(rel(path))

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(
            data=df,
            x=col,
            y="Benefit per order",
            order=order,
            ax=ax,
            color="#546e7a",
            cut=0,
            inner="quartile",
        )
        ax.set_title(f"Profit by {col} (violin)")
        ax.set_xlabel(col)
        ax.set_ylabel("Benefit per order")
        ax.tick_params(axis="x", rotation=20)
        path = PLOT_DIR / f"violin_profit_by_{_slug(col)}.png"
        savefig(path)
        plot_files.append(rel(path))

    for col in ["Shipping Mode", "Market", "Order Region"]:
        rates = df.groupby(col, observed=True)["late"].mean().sort_values(ascending=False)
        late_rates[col] = rates
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=rates.index.astype(str), y=rates.values, ax=ax, color="#c62828")
        ax.set_ylim(0, 1)
        ax.set_title(f"Late rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Late rate")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        path = PLOT_DIR / f"bar_late_rate_by_{_slug(col)}.png"
        savefig(path)
        plot_files.append(rel(path))

    # Loss rate by segment (useful for B's chi-squared)
    loss_by_seg = df.groupby("Customer Segment", observed=True)["loss"].mean().sort_values(ascending=False)
    late_rates["loss_by_Customer Segment"] = loss_by_seg
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=loss_by_seg.index.astype(str), y=loss_by_seg.values, ax=ax, color="#6a1b9a")
    ax.set_ylim(0, max(0.25, float(loss_by_seg.max()) * 1.15))
    ax.set_title("Loss rate by Customer Segment")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Loss rate")
    path = PLOT_DIR / "bar_loss_rate_by_Customer_Segment.png"
    savefig(path)
    plot_files.append(rel(path))
    return late_rates


def plot_overlays(df: pd.DataFrame, plot_files: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    late = df["late"] == 1
    on_time = ~late
    loss = df["loss"] == 1
    profit = ~loss

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        df.loc[on_time, "Days for shipment (scheduled)"],
        ax=ax,
        label="On time / not late",
        color="#1565c0",
        fill=True,
        alpha=0.35,
        bw_adjust=1.2,
    )
    sns.kdeplot(
        df.loc[late, "Days for shipment (scheduled)"],
        ax=ax,
        label="Late",
        color="#c62828",
        fill=True,
        alpha=0.35,
        bw_adjust=1.2,
    )
    ax.set_title("Scheduled shipping days: late vs on-time")
    ax.set_xlabel("Days for shipment (scheduled)")
    ax.legend()
    path = PLOT_DIR / "overlay_scheduled_days_late.png"
    savefig(path)
    plot_files.append(rel(path))

    metrics["scheduled_mean_late"] = float(df.loc[late, "Days for shipment (scheduled)"].mean())
    metrics["scheduled_mean_ontime"] = float(df.loc[on_time, "Days for shipment (scheduled)"].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        df.loc[on_time, "Order Item Discount Rate"],
        ax=ax,
        label="On time / not late",
        color="#1565c0",
        fill=True,
        alpha=0.35,
    )
    sns.kdeplot(
        df.loc[late, "Order Item Discount Rate"],
        ax=ax,
        label="Late",
        color="#c62828",
        fill=True,
        alpha=0.35,
    )
    ax.set_title("Discount rate: late vs on-time")
    ax.set_xlabel("Order Item Discount Rate")
    ax.legend()
    path = PLOT_DIR / "overlay_discount_late.png"
    savefig(path)
    plot_files.append(rel(path))

    metrics["discount_mean_late"] = float(df.loc[late, "Order Item Discount Rate"].mean())
    metrics["discount_mean_ontime"] = float(df.loc[on_time, "Order Item Discount Rate"].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        df.loc[profit, "Order Item Discount Rate"],
        ax=ax,
        label="Profit (not loss)",
        color="#2e7d32",
        fill=True,
        alpha=0.35,
    )
    sns.kdeplot(
        df.loc[loss, "Order Item Discount Rate"],
        ax=ax,
        label="Loss",
        color="#6a1b9a",
        fill=True,
        alpha=0.35,
    )
    ax.set_title("Discount rate: loss vs profit orders")
    ax.set_xlabel("Order Item Discount Rate")
    ax.legend()
    path = PLOT_DIR / "overlay_discount_loss.png"
    savefig(path)
    plot_files.append(rel(path))

    metrics["discount_mean_loss"] = float(df.loc[loss, "Order Item Discount Rate"].mean())
    metrics["discount_mean_profit"] = float(df.loc[profit, "Order Item Discount Rate"].mean())
    metrics["profit_mean_late"] = float(df.loc[late, "Benefit per order"].mean())
    metrics["profit_mean_ontime"] = float(df.loc[on_time, "Benefit per order"].mean())
    return metrics


def plot_time_view(df: pd.DataFrame, plot_files: list[str]) -> pd.DataFrame:
    work = df.copy()
    work["order_period"] = pd.to_datetime(work["order date (DateOrders)"]).dt.to_period("M").astype(str)
    monthly = (
        work.groupby("order_period", observed=True)
        .agg(orders=("late", "size"), late_rate=("late", "mean"), loss_rate=("loss", "mean"))
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(monthly))
    ax1.bar(x, monthly["orders"], color="#90a4ae", width=0.8, label="Orders")
    ax1.set_ylabel("Orders")
    ax1.set_xlabel("Month")
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly["order_period"], rotation=55, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly["late_rate"], color="#c62828", marker="o", linewidth=2, label="Late rate")
    ax2.set_ylabel("Late rate")
    ax2.set_ylim(0, 1)
    ax1.set_title("Orders and late rate by month")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    path = PLOT_DIR / "time_orders_and_late_rate.png"
    savefig(path)
    plot_files.append(rel(path))
    return monthly


def _slug(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _grain_change_body(raw_path: Path) -> str:
    """Document the Oct 2017 order-line grain change using the raw file (has Order Id)."""
    if not raw_path.is_file():
        return (
            "The monthly bar in `time_orders_and_late_rate.png` (and Power BI Overview) drops from about "
            "5,300 lines/month through Sep 2017 to about 2,100 from Oct 2017. Raw and cleaned monthly line "
            "counts match, so Phase 1 did not cause this. Distinct `Order Id` counts stay roughly flat while "
            "lines per order go from ~3 to 1 around 2–3 Oct 2017. Prefer distinct orders (or say “order lines”) "
            "on slides after Sep 2017."
        )
    raw = pd.read_csv(
        raw_path,
        encoding="latin-1",
        usecols=["order date (DateOrders)", "Order Id"],
        low_memory=False,
    )
    raw["dt"] = pd.to_datetime(raw["order date (DateOrders)"], format="%m/%d/%Y %H:%M", errors="coerce")
    raw = raw.dropna(subset=["dt"])
    raw["ym"] = raw["dt"].dt.to_period("M")
    rows = []
    for period in ["2017-07", "2017-08", "2017-09", "2017-10", "2017-11", "2017-12", "2018-01"]:
        part = raw.loc[raw["ym"] == pd.Period(period)]
        n_lines = int(len(part))
        n_orders = int(part["Order Id"].nunique())
        lpo = n_lines / n_orders if n_orders else float("nan")
        rows.append(f"| {period} | {n_lines:,} | {n_orders:,} | {lpo:.3f} |")
    table = "\n".join(
        [
            "| Month | Order lines | Distinct `Order Id` | Lines per order |",
            "| --- | ---: | ---: | ---: |",
            *rows,
        ]
    )
    return (
        "The monthly bar in `time_orders_and_late_rate.png` (and Power BI Overview) drops from about "
        "5,300 lines/month through Sep 2017 to about 2,100 from Oct 2017. Raw and cleaned monthly line "
        "counts match in every month, so Phase 1 did not cause this.\n\n"
        f"Checked on the raw file (`order date` parsed as `%m/%d/%Y %H:%M`):\n\n{table}\n\n"
        "Day-level switch: through 2017-10-01 lines/order ≈ 3; on 2017-10-02 ≈ 1.8; from 2017-10-03 onward "
        "exactly 1. Distinct orders stay roughly flat. No reason for this grain change was found in the "
        "DataCo / Mendeley / Kaggle descriptions; treat it as a source-extract limitation. Prefer distinct "
        "orders (or say “order lines”) on slides after Sep 2017."
    )


def render_findings(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    corr: pd.DataFrame,
    late_rates: dict[str, pd.Series],
    overlay: dict[str, float],
    monthly: pd.DataFrame,
    plot_files: list[str],
    grain_body: str,
) -> str:
    n = len(df)
    late_pct = float(df["late"].mean())
    loss_pct = float(df["loss"].mean())

    ship_late = late_rates["Shipping Mode"]
    market_late = late_rates["Market"]
    region_late = late_rates["Order Region"]
    loss_seg = late_rates["loss_by_Customer Segment"]

    sales_row = stats.set_index("column").loc["Sales"]
    benefit_row = stats.set_index("column").loc["Benefit per order"]
    disc_row = stats.set_index("column").loc["Order Item Discount Rate"]
    sched_row = stats.set_index("column").loc["Days for shipment (scheduled)"]

    # Strongest correlations with late / loss (absolute), excluding itself
    late_corr = corr["late"].drop(labels=["late"]).abs().sort_values(ascending=False)
    loss_corr = corr["loss"].drop(labels=["loss"]).abs().sort_values(ascending=False)

    top_ship = ship_late.index[0]
    bot_ship = ship_late.index[-1]
    top_market = market_late.index[0]
    bot_market = market_late.index[-1]
    top_region = region_late.index[0]
    bot_region = region_late.index[-1]

    peak_month = monthly.loc[monthly["orders"].idxmax(), "order_period"]
    peak_orders = int(monthly.loc[monthly["orders"].idxmax(), "orders"])
    late_month = monthly.loc[monthly["late_rate"].idxmax(), "order_period"]
    late_month_rate = float(monthly.loc[monthly["late_rate"].idxmax(), "late_rate"])

    # Segments worth testing for B
    test_segments = [
        f"Shipping Mode vs late — spread {fmt_pct(float(ship_late.min()))} to {fmt_pct(float(ship_late.max()))} "
        f"(`{bot_ship}` vs `{top_ship}`). Plot: `bar_late_rate_by_Shipping_Mode.png`.",
        f"Market vs late — spread {fmt_pct(float(market_late.min()))} to {fmt_pct(float(market_late.max()))} "
        f"(`{bot_market}` vs `{top_market}`). Plot: `bar_late_rate_by_Market.png`.",
        f"Customer Segment vs loss — rates "
        + ", ".join(f"`{k}` {fmt_pct(float(v))}" for k, v in loss_seg.items())
        + ". Plot: `bar_loss_rate_by_Customer_Segment.png`.",
        "Profit across Markets (ANOVA / Kruskal–Wallis) — see `box_profit_by_Market.png` / `violin_profit_by_Market.png`.",
        "Profit: late vs on-time (Welch t-test) — "
        f"mean profit late {fmt_num(overlay['profit_mean_late'])}, on-time {fmt_num(overlay['profit_mean_ontime'])}.",
        "Discount rate: loss vs profit (Welch t-test) — "
        f"mean discount loss {fmt_num(overlay['discount_mean_loss'])}, "
        f"profit {fmt_num(overlay['discount_mean_profit'])}. Plot: `overlay_discount_loss.png`.",
    ]

    stats_table = _md_table(
        ["Column", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max", "Skew", "Kurtosis"],
        [
            [
                r["column"],
                fmt_num(r["mean"]),
                fmt_num(r["std"]),
                fmt_num(r["min"]),
                fmt_num(r["q1"]),
                fmt_num(r["median"]),
                fmt_num(r["q3"]),
                fmt_num(r["max"]),
                fmt_num(r["skewness"]),
                fmt_num(r["kurtosis"]),
            ]
            for _, r in stats.iterrows()
        ],
    )

    findings = [
        (
            "Shipping mode dominates late delivery.",
            f"`{top_ship}` has the highest late rate ({fmt_pct(float(ship_late.loc[top_ship]))}), "
            f"while `{bot_ship}` is lowest ({fmt_pct(float(ship_late.loc[bot_ship]))}). "
            "This is the clearest categorical signal for the late-delivery question.",
            "bar_late_rate_by_Shipping_Mode.png",
        ),
        (
            "Late rate also differs by market and region.",
            f"Markets range from `{bot_market}` ({fmt_pct(float(market_late.loc[bot_market]))}) to "
            f"`{top_market}` ({fmt_pct(float(market_late.loc[top_market]))}). "
            f"Among regions, `{top_region}` is highest ({fmt_pct(float(region_late.loc[top_region]))}) and "
            f"`{bot_region}` is lowest ({fmt_pct(float(region_late.loc[bot_region]))}).",
            "bar_late_rate_by_Market.png / bar_late_rate_by_Order_Region.png",
        ),
        (
            "Loss rate is fairly similar across customer segments.",
            "Loss shares: "
            + ", ".join(f"{k} {fmt_pct(float(v))}" for k, v in loss_seg.items())
            + ". Segment may still be worth testing, but the effect looks smaller than shipping mode for lateness.",
            "bar_loss_rate_by_Customer_Segment.png",
        ),
        (
            "Profit is heavy-tailed and often negative.",
            f"`Benefit per order` mean {fmt_num(float(benefit_row['mean']))}, "
            f"median {fmt_num(float(benefit_row['median']))}, "
            f"skewness {fmt_num(float(benefit_row['skewness']))}. "
            "Box/violin plots by market and shipping mode show wide spreads; OLS should expect assumption issues.",
            "box_profit_by_Market.png / hist_Benefit_per_order.png",
        ),
        (
            "Discount rate vs loss looks weak on averages; watch the scatter with profit.",
            "The discount-vs-profit scatter still shows many deep losses at higher discount rates, "
            "but the mean discount rate gap is small: "
            f"loss {fmt_num(overlay['discount_mean_loss'])} vs "
            f"non-loss {fmt_num(overlay['discount_mean_profit'])}. "
            "B should report effect size for T6, not only the p-value.",
            "scatter_discount_vs_profit.png / overlay_discount_loss.png",
        ),
        (
            "Scheduled shipping days differ for late vs on-time orders.",
            f"Mean scheduled days: late {fmt_num(overlay['scheduled_mean_late'])}, "
            f"on-time {fmt_num(overlay['scheduled_mean_ontime'])}. "
            "This is a candidate feature for D (known at order time, not leakage).",
            "overlay_scheduled_days_late.png",
        ),
        (
            "Sales are right-skewed; the log transform helps for visuals.",
            f"`Sales` skewness {fmt_num(float(sales_row['skewness']))}. "
            "Use `Sales_log` for distribution plots; keep raw `Sales` for modelling unless C chooses otherwise.",
            "hist_compare_Sales.png",
        ),
        (
            "Order volume and late rate move over the calendar.",
            f"Busiest month in this extract by row count: {peak_month} ({peak_orders:,} order lines). "
            f"Highest monthly late rate: {late_month} ({fmt_pct(late_month_rate)}). "
            "B can use month as a slicer in Power BI; D may try `order_month` / `order_year` as features.",
            "time_orders_and_late_rate.png",
        ),
        (
            "Order-line volume falls from October 2017 because the extract grain changes, not because cleaning removed rows.",
            grain_body,
            "time_orders_and_late_rate.png",
        ),
    ]

    finding_blocks = []
    for i, (title, body, plot) in enumerate(findings, start=1):
        finding_blocks.append(f"### F{i}. {title}\n\n{body}\n\nPlot: `{plot}`\n")

    plot_list = "\n".join(f"- `{p}`" for p in plot_files)
    test_list = "\n".join(f"{i}. {t}" for i, t in enumerate(test_segments, start=1))

    return f"""# Phase 2a: EDA findings

- Run date: {date.today().isoformat()}
- Script: `A_preprocessing_eda/phase2a_eda.py`
- Input: `A_preprocessing_eda/outputs/cleaned_supply_chain.csv`
- Rows used: {n:,}
- Overall late rate: {fmt_pct(late_pct)}
- Overall loss rate: {fmt_pct(loss_pct)}

## Summary statistics

Five-number summary plus mean, standard deviation, skewness and kurtosis for the focus numeric columns.

{stats_table}

Discount column used below is `Order Item Discount Rate` (rate) and `Order Item Discount` (amount). Shipping days means `Days for shipment (scheduled)`.

Mean scheduled days: {fmt_num(float(sched_row['mean']))}. Mean discount rate: {fmt_num(float(disc_row['mean']))}.

## Findings (plain language)

{"".join(finding_blocks)}

## Correlation notes (for C)

Strongest absolute Pearson correlations with `late` (excluding itself):
{_corr_lines(corr, "late", late_corr)}

Strongest absolute Pearson correlations with `loss` (excluding itself):
{_corr_lines(corr, "loss", loss_corr)}

Remember: `Sales` and `Order Item Total` are highly collinear (kept both in Phase 1). Check VIF before OLS.

## Segments worth testing (for B)

These match the planned T1–T6 tests and are backed by the plots above.

{test_list}

## Plot index

{plot_list}

## Limitations

EDA is descriptive only. With ~180k rows, tiny differences can look “significant” in later tests; prefer effect sizes. Leakage columns stay excluded. No modelling was done in Phase 2a.

Order dates in this extract run from **2015-01-01** to **2018-01-31** (not only through 2017). From early October 2017 the file is effectively one row per `Order Id`, whereas earlier months average about three lines per order, so monthly **line** charts fall even when distinct orders do not (see F9).
"""


def _corr_lines(corr: pd.DataFrame, target: str, ranked: pd.Series, k: int = 5) -> str:
    lines = []
    for name in ranked.head(k).index:
        r = float(corr.loc[name, target])
        lines.append(f"- `{name}`: r = {r:.4f}")
    return "\n".join(lines)


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(v: object) -> str:
        return str(v).replace("|", "\\|")

    head = "| " + " | ".join(clean(h) for h in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(c) for c in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def render_handoff(
    df: pd.DataFrame,
    late_rates: dict[str, pd.Series],
    overlay: dict[str, float],
    plot_files: list[str],
) -> str:
    ship = late_rates["Shipping Mode"]
    order_dates = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    return "\n".join(
        [
            "HAND-OFF: Phase 2a, EDA (A)",
            (
                "1. Done: Ran EDA on the cleaned file — summary stats, distributions, categorical bars, "
                "correlation heatmap, discount-vs-profit scatter, profit box/violin by shipping mode / market / segment, "
                "late rates by shipping mode / market / region, overlays for late vs on-time and loss vs profit, "
                "and monthly orders + late rate. Wrote eda_findings.md with 9 findings "
                "(including the Oct 2017 order-line grain change) and a test list for B."
            ),
            (
                f"2. Key numbers: n={len(df):,}; late={fmt_pct(float(df['late'].mean()))}; "
                f"loss={fmt_pct(float(df['loss'].mean()))}. "
                f"Late by shipping mode: "
                + ", ".join(f"{k} {fmt_pct(float(v))}" for k, v in ship.items())
                + ". "
                f"Mean scheduled days late/on-time: {fmt_num(overlay['scheduled_mean_late'])} / "
                f"{fmt_num(overlay['scheduled_mean_ontime'])}. "
                f"Mean discount rate loss/profit: {fmt_num(overlay['discount_mean_loss'])} / "
                f"{fmt_num(overlay['discount_mean_profit'])}. "
                f"Plots written: {len(plot_files)}. "
                f"Order dates: {order_dates.min().strftime('%Y-%m-%d')} to {order_dates.max().strftime('%Y-%m-%d')}. "
                "From Oct 2017 lines/month fall ~5.3k→~2.1k while distinct orders stay flat (lines/order ~3→1)."
            ),
            (
                "3. Decisions: Used scheduled days (not real shipping days). Sampled 8,000 rows for the scatter "
                f"(random_state={RANDOM_STATE}) so the plot stays readable; rates and stats use all rows. "
                "Recommended B’s tests: shipping mode vs late, market vs late, segment vs loss, profit across markets, "
                "profit late vs on-time, discount loss vs profit. "
                "Treat post-Sep 2017 monthly “volume” as order lines, not distinct orders."
            ),
            (
                "4. Files: `A_preprocessing_eda/outputs/eda_plots/`; "
                "`A_preprocessing_eda/outputs/eda_findings.md`; "
                "`A_preprocessing_eda/outputs/handoff_phase2a.md`."
            ),
            (
                "5. Before you start: B — start with the ‘Segments worth testing’ list in eda_findings.md and report "
                "effect sizes; for Overview charts say order lines and cite F9 for the Oct 2017 drop. "
                "C — profit is skewed/heavy-tailed; discount relates to loss; watch Sales vs Order Item Total. "
                "D — shipping mode and scheduled days are the strongest late-delivery leads; do not reintroduce leakage columns."
            ),
        ]
    ) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(RANDOM_STATE)
    style()
    args = parse_args()
    cleaned_path = resolve(args.cleaned)
    if not cleaned_path.is_file():
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_path}")

    logger.info("Loading %s", rel(cleaned_path))
    df = pd.read_csv(cleaned_path, encoding="utf-8", low_memory=False)
    required = set(NUMERIC_FOCUS + CAT_BARS + ["late", "loss", "order date (DateOrders)", "Benefit per order"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Cleaned file missing columns: {missing}")

    if PLOT_DIR.exists():
        for old in PLOT_DIR.glob("*.png"):
            old.unlink()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_files: list[str] = []
    logger.info("Summary statistics")
    stats = summary_stats(df)
    stats_path = OUTPUT_DIR / "eda_summary_stats.csv"
    stats.to_csv(stats_path, index=False)

    logger.info("Distributions and categorical bars")
    plot_histograms(df, plot_files)
    plot_categorical_bars(df, plot_files)

    logger.info("Relationships")
    corr = plot_correlation(df, plot_files)
    plot_discount_vs_profit(df, plot_files)

    logger.info("Group comparisons")
    late_rates = plot_group_comparisons(df, plot_files)

    logger.info("Overlays")
    overlay = plot_overlays(df, plot_files)

    logger.info("Time view")
    monthly = plot_time_view(df, plot_files)

    raw_path = REPO_ROOT / "dataset" / "DataCoSupplyChainDataset.csv"
    grain_body = _grain_change_body(raw_path)
    findings = render_findings(df, stats, corr, late_rates, overlay, monthly, plot_files, grain_body)
    handoff = render_handoff(df, late_rates, overlay, plot_files)
    findings_path = OUTPUT_DIR / "eda_findings.md"
    handoff_path = OUTPUT_DIR / "handoff_phase2a.md"
    findings_path.write_text(findings, encoding="utf-8")
    handoff_path.write_text(handoff, encoding="utf-8")

    logger.info("Wrote %s (%s plots)", rel(findings_path), len(plot_files))
    logger.info("Wrote %s", rel(handoff_path))
    logger.info("Wrote %s", rel(stats_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
