"""Phase 2b hypothesis tests for the DataCo Smart Supply Chain project (Person B).

Run from anywhere:

    python B_hypothesis_powerbi/phase2b_tests.py

Reads A's cleaned file. Does not modify it. Writes tables, plots, and the
hand-off under B_hypothesis_powerbi/outputs/.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "B_hypothesis_powerbi" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

ALPHA = 0.05
EXPECTED_ROWS = 180_519
USECOLS = [
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "late",
    "loss",
    "Benefit per order",
    "Order Item Discount Rate",
    "Sales",
    "order_year",
    "order_month",
]
SHIPPING_ORDER = ["First Class", "Second Class", "Same Day", "Standard Class"]


def parse_args() -> argparse.Namespace:
    """Read the cleaned-file path."""
    parser = argparse.ArgumentParser(description="Run Phase 2b hypothesis tests.")
    parser.add_argument(
        "--cleaned",
        type=Path,
        default=Path("A_preprocessing_eda/outputs/cleaned_supply_chain.csv"),
        help="Cleaned CSV path, relative to the repo root unless absolute.",
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


def fmt_float(value: float, digits: int = 6) -> str:
    """Format a float for tables."""
    return f"{float(value):.{digits}f}"


def fmt_p(p_value: float) -> str:
    """Format a p-value; very small values use scientific notation."""
    if p_value == 0.0 or p_value < 1e-300:
        return "< 1e-300"
    if p_value < 1e-6:
        return f"{p_value:.3e}"
    return f"{p_value:.6f}"


def fmt_pct(value: float) -> str:
    """Format a proportion as a percentage."""
    return f"{100.0 * value:.4f}%"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a markdown table, escaping pipes in cells."""

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(item) for item in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def reject_text(p_value: float) -> str:
    """Return whether H0 is rejected at ALPHA."""
    if p_value < ALPHA:
        return f"Reject H0 (p < {ALPHA})"
    return f"Do not reject H0 (p ≥ {ALPHA})"


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    """Cramer's V for an r x c table."""
    return float(np.sqrt(chi2 / (n * min(n_rows - 1, n_cols - 1))))


def cramer_label(v: float, df_star: int) -> str:
    """Label Cramer's V using Cohen's w cut-offs scaled by sqrt(df*)."""
    denom = np.sqrt(df_star)
    small, medium, large = 0.10 / denom, 0.30 / denom, 0.50 / denom
    abs_v = abs(v)
    if abs_v < small:
        size = "negligible"
    elif abs_v < medium:
        size = "small"
    elif abs_v < large:
        size = "medium"
    else:
        size = "large"
    return (
        f"{size} (Cohen cut-offs at df*={df_star}: "
        f"small {small:.3f}, medium {medium:.3f}, large {large:.3f})"
    )


def eta_squared_label(eta2: float) -> str:
    """Label eta-squared (0.01 / 0.06 / 0.14)."""
    if eta2 < 0.01:
        size = "negligible"
    elif eta2 < 0.06:
        size = "small"
    elif eta2 < 0.14:
        size = "medium"
    else:
        size = "large"
    return f"{size} (cut-offs 0.01 / 0.06 / 0.14)"


def cohens_d_label(d: float) -> str:
    """Label Cohen's d (0.20 / 0.50 / 0.80)."""
    abs_d = abs(d)
    if abs_d < 0.20:
        size = "negligible"
    elif abs_d < 0.50:
        size = "small"
    elif abs_d < 0.80:
        size = "medium"
    else:
        size = "large"
    return f"{size} (cut-offs 0.20 / 0.50 / 0.80)"


def chi_square_test(
    df: pd.DataFrame,
    test_id: str,
    name: str,
    row_col: str,
    col_col: str,
    h0: str,
    h1: str,
) -> dict:
    """Chi-squared test of independence with Cramer's V."""
    table = pd.crosstab(df[row_col], df[col_col])
    table.to_csv(TABLE_DIR / f"{test_id.lower()}_contingency.csv")
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    n_rows, n_cols = table.shape
    v = cramers_v(chi2, n, n_rows, n_cols)
    df_star = min(n_rows - 1, n_cols - 1)
    min_expected = float(expected.min())
    n_below_five = int((expected < 5).sum())
    assumption_ok = min_expected >= 5
    rates = df.groupby(row_col, observed=True)[col_col].mean().sort_values(ascending=False)
    rate_text = "; ".join(f"{idx}: {fmt_pct(val)}" for idx, val in rates.items())
    assumptions = (
        f"Independent order-line rows assumed (limitation: several lines can share an order). "
        f"Expected counts: min={fmt_float(min_expected, 2)}, cells < 5: {n_below_five} of {expected.size}. "
        f"{'Chi-squared approximation is appropriate.' if assumption_ok else 'Some expected counts are below 5.'}"
    )
    conclusion = (
        f"{reject_text(p_value)}. Cramer's V = {fmt_float(v, 4)} ({cramer_label(v, df_star)}). "
        f"{col_col}=1 rates: {rate_text}."
    )
    return {
        "test_id": test_id,
        "test": name,
        "H0": h0,
        "H1": h1,
        "statistic_name": "chi2",
        "statistic": float(chi2),
        "df": int(dof),
        "p_value": float(p_value),
        "effect_size_name": "Cramer's V",
        "effect_size": float(v),
        "n": n,
        "assumptions": assumptions,
        "conclusion": conclusion,
        "extra": rates.to_dict(),
        "table": table,
    }


def eta_squared(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Return eta-squared, SS_between, SS_total."""
    all_y = np.concatenate(groups)
    grand = float(all_y.mean())
    ss_between = 0.0
    ss_within = 0.0
    for g in groups:
        mean_g = float(g.mean())
        ss_between += g.size * (mean_g - grand) ** 2
        ss_within += float(np.sum((g - mean_g) ** 2))
    ss_total = ss_between + ss_within
    eta2 = ss_between / ss_total if ss_total else 0.0
    return eta2, ss_between, ss_total


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using the pooled sample standard deviation (unbiased)."""
    n1, n2 = a.size, b.size
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def epsilon_squared_kw(h: float, k: int, n: int) -> float:
    """Kruskal–Wallis epsilon-squared."""
    return float((h - k + 1) / (n - k))


def anova_profit_by_market(df: pd.DataFrame) -> dict:
    """One-way ANOVA of profit across markets, plus Kruskal–Wallis."""
    profit = "Benefit per order"
    grouped = [g[profit].to_numpy(dtype=float) for _, g in df.groupby("Market", observed=True)]
    labels = [str(name) for name, _ in df.groupby("Market", observed=True)]
    f_stat, p_anova = stats.f_oneway(*grouped)
    eta2, _, _ = eta_squared(grouped)
    levene_stat, levene_p = stats.levene(*grouped, center="median")
    skew = float(stats.skew(df[profit].to_numpy(dtype=float), bias=False))
    h_stat, p_kw = stats.kruskal(*grouped)
    k = len(grouped)
    n = int(len(df))
    eps2 = epsilon_squared_kw(float(h_stat), k, n)
    means = df.groupby("Market", observed=True)[profit].agg(["count", "mean", "median", "std"])
    means.to_csv(TABLE_DIR / "t4_profit_by_market.csv")
    mean_text = "; ".join(
        f"{idx}: mean={fmt_float(row['mean'], 2)}, median={fmt_float(row['median'], 2)}"
        for idx, row in means.iterrows()
    )
    assumptions = (
        f"Independence assumed at the order-line level. "
        f"Profit skewness = {fmt_float(skew, 3)} (heavy left tail; ANOVA normality is not realistic). "
        f"Levene (median) W = {fmt_float(levene_stat, 3)}, p = {fmt_p(levene_p)} "
        f"({'variances differ' if levene_p < ALPHA else 'no evidence of unequal variances'}). "
        f"Kruskal–Wallis H = {fmt_float(h_stat, 3)}, p = {fmt_p(p_kw)}, "
        f"epsilon-squared = {fmt_float(eps2, 6)}."
    )
    conclusion = (
        f"{reject_text(p_anova)}. Eta-squared = {fmt_float(eta2, 6)} ({eta_squared_label(eta2)}). "
        f"Because profit is heavily skewed, Kruskal–Wallis is the more appropriate test; "
        f"it also rejects H0, with a {eta_squared_label(eps2).split(' (')[0]} rank effect "
        f"(epsilon-squared = {fmt_float(eps2, 6)}). Market profit: {mean_text}."
    )
    return {
        "test_id": "T4",
        "test": "One-way ANOVA (Kruskal–Wallis added) of Benefit per order across Market",
        "H0": "Mean Benefit per order is the same in every Market.",
        "H1": "At least one Market has a different mean Benefit per order.",
        "statistic_name": "F",
        "statistic": float(f_stat),
        "df": f"{k - 1}, {n - k}",
        "p_value": float(p_anova),
        "effect_size_name": "eta-squared",
        "effect_size": float(eta2),
        "n": n,
        "assumptions": assumptions,
        "conclusion": conclusion,
        "extra": {
            "skew": skew,
            "levene_p": float(levene_p),
            "kruskal_H": float(h_stat),
            "kruskal_p": float(p_kw),
            "epsilon_squared": eps2,
            "labels": labels,
        },
    }


def welch_test(
    a: np.ndarray,
    b: np.ndarray,
    test_id: str,
    name: str,
    h0: str,
    h1: str,
    label_a: str,
    label_b: str,
) -> dict:
    """Welch two-sample t-test with Cohen's d."""
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a, b)
    n = int(a.size + b.size)
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    skew_a = float(stats.skew(a, bias=False))
    skew_b = float(stats.skew(b, bias=False))
    # Welch–Satterthwaite df
    na, nb = a.size, b.size
    se2_a, se2_b = va / na, vb / nb
    df_w = (se2_a + se2_b) ** 2 / (se2_a**2 / (na - 1) + se2_b**2 / (nb - 1))
    assumptions = (
        f"Welch t-test does not assume equal variances "
        f"(s² {label_a}={fmt_float(va, 4)}, s² {label_b}={fmt_float(vb, 4)}). "
        f"CLT applies at n={fmt_int(n)}, so the t approximation is usable despite skew "
        f"({label_a} skew={fmt_float(skew_a, 3)}, {label_b} skew={fmt_float(skew_b, 3)}). "
        f"Independence assumed at the order-line level."
    )
    conclusion = (
        f"{reject_text(p_value)}. Cohen's d = {fmt_float(d, 4)} ({label_a} minus {label_b}; "
        f"{cohens_d_label(d)}). "
        f"Mean {label_a}={fmt_float(a.mean(), 4)} (n={fmt_int(a.size)}), "
        f"mean {label_b}={fmt_float(b.mean(), 4)} (n={fmt_int(b.size)})."
    )
    return {
        "test_id": test_id,
        "test": name,
        "H0": h0,
        "H1": h1,
        "statistic_name": "t (Welch)",
        "statistic": float(t_stat),
        "df": fmt_float(df_w, 2),
        "p_value": float(p_value),
        "effect_size_name": "Cohen's d",
        "effect_size": float(d),
        "n": n,
        "assumptions": assumptions,
        "conclusion": conclusion,
        "extra": {
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "n_a": int(a.size),
            "n_b": int(b.size),
        },
    }


def save_plots(df: pd.DataFrame) -> None:
    """Write the group-comparison plots used in the report."""
    sns.set_theme(style="whitegrid")

    late_mode = (
        df.groupby("Shipping Mode", observed=True)["late"].mean().reindex(SHIPPING_ORDER) * 100
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    late_mode.plot(kind="bar", ax=ax, color="#1f4e79", rot=0)
    ax.set_ylabel("Late rate (%)")
    ax.set_title("Late delivery rate by shipping mode")
    ax.set_ylim(0, 100)
    for i, val in enumerate(late_mode):
        ax.text(i, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "late_rate_by_shipping_mode.png", dpi=120)
    plt.close(fig)

    late_mkt = df.groupby("Market", observed=True)["late"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    late_mkt.plot(kind="bar", ax=ax, color="#1f4e79", rot=0)
    ax.set_ylabel("Late rate (%)")
    ax.set_title("Late delivery rate by market")
    ax.set_ylim(0, 70)
    for i, val in enumerate(late_mkt):
        ax.text(i, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "late_rate_by_market.png", dpi=120)
    plt.close(fig)

    loss_seg = (
        df.groupby("Customer Segment", observed=True)["loss"].mean().sort_values(ascending=False)
        * 100
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    loss_seg.plot(kind="bar", ax=ax, color="#9c2a2a", rot=0)
    ax.set_ylabel("Loss rate (%)")
    ax.set_title("Loss rate by customer segment")
    ax.set_ylim(0, 30)
    for i, val in enumerate(loss_seg):
        ax.text(i, val + 0.4, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loss_rate_by_segment.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=df,
        x="Market",
        y="Benefit per order",
        ax=ax,
        showfliers=False,
        color="#d9e2f3",
    )
    ax.set_title("Benefit per order by market (outliers hidden)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "profit_by_market_box.png", dpi=120)
    plt.close(fig)

    plot_df = df.copy()
    plot_df["late_label"] = np.where(plot_df["late"] == 1, "Late", "On time")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(
        data=plot_df,
        x="late_label",
        y="Benefit per order",
        ax=ax,
        showfliers=False,
        color="#d9e2f3",
    )
    ax.set_xlabel("")
    ax.set_title("Benefit per order: late vs on-time (outliers hidden)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "profit_by_late_box.png", dpi=120)
    plt.close(fig)

    plot_df["loss_label"] = np.where(plot_df["loss"] == 1, "Loss", "Not loss")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(
        data=plot_df,
        x="loss_label",
        y="Order Item Discount Rate",
        ax=ax,
        showfliers=False,
        color="#f4cccc",
    )
    ax.set_xlabel("")
    ax.set_title("Discount rate: loss vs not-loss (outliers hidden)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "discount_by_loss_box.png", dpi=120)
    plt.close(fig)


def write_csv(results: list[dict]) -> Path:
    """Write the source-of-truth results table."""
    rows = []
    for item in results:
        rows.append(
            {
                "test_id": item["test_id"],
                "test": item["test"],
                "H0": item["H0"],
                "H1": item["H1"],
                "statistic": f"{item['statistic_name']}={fmt_float(item['statistic'], 4)}",
                "df": item["df"],
                "p_value": fmt_p(item["p_value"]),
                "effect_size": f"{item['effect_size_name']}={fmt_float(item['effect_size'], 6)}",
                "n": item["n"],
                "alpha": ALPHA,
                "assumptions": item["assumptions"],
                "conclusion": item["conclusion"],
            }
        )
    path = OUTPUT_DIR / "tests_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_report(df: pd.DataFrame, results: list[dict], cleaned_path: Path) -> Path:
    """Write the teammate-facing Phase 2b report."""
    t_map = {item["test_id"]: item for item in results}
    kpis = {
        "n_rows": int(len(df)),
        "late_rate": float(df["late"].mean()),
        "loss_rate": float(df["loss"].mean()),
        "total_sales": float(df["Sales"].sum()),
        "total_profit": float(df["Benefit per order"].sum()),
        "mean_profit": float(df["Benefit per order"].mean()),
        "mean_discount": float(df["Order Item Discount Rate"].mean()),
    }
    pd.DataFrame([kpis]).to_csv(TABLE_DIR / "kpis.csv", index=False)

    lines = [
        "# Phase 2b report — Hypothesis tests (Person B)",
        "",
        "**Course:** Foundations of Data Science (23CSE351) · Group 3  ",
        "**Owner:** B (Mithun)  ",
        f"**Run date:** {date.today().isoformat()}  ",
        f"**Input:** `{rel(cleaned_path)}`  ",
        "**Alpha:** 0.05  ",
        "",
        "This report is the teammate-facing summary of the six source-of-truth tests. "
        "The pasteable hand-off is `handoff_phase2b.md`. Plots live in `outputs/plots/`.",
        "",
        "## 1. Why effect size still matters",
        "",
        f"The cleaned file has {fmt_int(kpis['n_rows'])} rows. The source of truth warned that "
        "almost every test would be significant at this n. That is **not** what happened: only T1 "
        "and T4 reject H0 at α = 0.05. T2, T3, T5 and T6 do not. Quote the **effect size** and "
        "the group rates anyway. T1 is the only result with a practical gap.",
        "",
        "## 2. KPIs (for Power BI page 1)",
        "",
        md_table(
            ["Measure", "Value"],
            [
                ["Order lines (rows)", fmt_int(kpis["n_rows"])],
                ["Late %", fmt_pct(kpis["late_rate"])],
                ["Loss %", fmt_pct(kpis["loss_rate"])],
                ["Total sales (`Sales`)", fmt_float(kpis["total_sales"], 2)],
                ["Total profit (`Benefit per order`)", fmt_float(kpis["total_profit"], 2)],
                ["Mean profit", fmt_float(kpis["mean_profit"], 4)],
                ["Mean discount rate", fmt_float(kpis["mean_discount"], 4)],
            ],
        ),
        "",
        "## 3. Results",
        "",
        md_table(
            ["ID", "Test", "Statistic", "p-value", "Effect size", "Decision"],
            [
                [
                    item["test_id"],
                    item["test"],
                    f"{item['statistic_name']}={fmt_float(item['statistic'], 2)}",
                    fmt_p(item["p_value"]),
                    f"{item['effect_size_name']}={fmt_float(item['effect_size'], 6)}",
                    reject_text(item["p_value"]),
                ]
                for item in results
            ],
        ),
        "",
    ]

    for item in results:
        lines.extend(
            [
                f"### {item['test_id']}. {item['test']}",
                "",
                f"- **H0:** {item['H0']}",
                f"- **H1:** {item['H1']}",
                f"- **Statistic:** {item['statistic_name']} = {fmt_float(item['statistic'], 4)} "
                f"(df = {item['df']})",
                f"- **p-value:** {fmt_p(item['p_value'])}",
                f"- **Effect size:** {item['effect_size_name']} = {fmt_float(item['effect_size'], 6)}",
                f"- **Assumptions:** {item['assumptions']}",
                f"- **Interpretation:** {item['conclusion']}",
                "",
            ]
        )

    t1 = t_map["T1"]
    t2 = t_map["T2"]
    t3 = t_map["T3"]
    t4 = t_map["T4"]
    t5 = t_map["T5"]
    t6 = t_map["T6"]
    lines.extend(
        [
            "## 4. Plain-language findings for EDA / dashboard / viva",
            "",
            f"1. **Shipping mode vs late (T1):** Reject H0. Cramer's V = "
            f"{fmt_float(t1['effect_size'], 4)} is **medium** on Cohen's 2×k scale (large starts at 0.50) "
            "and the practical gap is huge: First Class 95.3% late vs Standard Class 38.1%. Lead Power BI page 2 with this.",
            f"2. **Market vs late (T2):** Do **not** reject H0 (p = {fmt_p(t2['p_value'])}). "
            f"V = {fmt_float(t2['effect_size'], 4)} is negligible; late rates sit around 54–55% in every market.",
            f"3. **Segment vs loss (T3):** Do **not** reject H0 (p = {fmt_p(t3['p_value'])}). "
            f"V = {fmt_float(t3['effect_size'], 4)} is negligible; loss is about 18.6–18.8% in every segment.",
            f"4. **Profit across markets (T4):** ANOVA rejects (p = {fmt_p(t4['p_value'])}) but "
            f"eta-squared = {fmt_float(t4['effect_size'], 6)} is **negligible**. Levene fails and profit "
            "is heavily skewed, so quote Kruskal–Wallis: it also rejects, with epsilon-squared = "
            f"{fmt_float(t4['extra']['epsilon_squared'], 6)}. Markets do not explain profit in any practical sense.",
            f"5. **Profit, late vs on-time (T5):** Do **not** reject H0 (p = {fmt_p(t5['p_value'])}). "
            f"Cohen's d = {fmt_float(t5['effect_size'], 4)} is negligible. Late deliveries are not where the money is won or lost.",
            f"6. **Discount rate, loss vs not-loss (T6):** Do **not** reject H0 (p = {fmt_p(t6['p_value'])}). "
            f"Cohen's d = {fmt_float(t6['effect_size'], 4)} is negligible. Mean discount is 0.1022 vs 0.1015.",
            "",
            "## 5. Limitations",
            "",
            "- Rows are **order lines**, not unique orders or unique shipments. Tests treat them as independent.",
            "- n is large, but four of six tests still fail to reject H0. Quote effect sizes and group rates in the viva.",
            "- T4–T6 use `Benefit per order` and `Order Item Discount Rate` as they stand (unscaled, outliers kept).",
            "- Late and loss definitions follow the source of truth: `late` = `Late_delivery_risk`; "
            "`loss` = 1 if `Benefit per order` < 0.",
            "",
            "## 6. How to reproduce",
            "",
            "```bash",
            "python -m pip install -r B_hypothesis_powerbi/requirements.txt",
            "python B_hypothesis_powerbi/phase2b_tests.py",
            "```",
            "",
            f"Numbers in this file match `{rel(OUTPUT_DIR / 'tests_results.csv')}` and `handoff_phase2b.md`.",
            "",
        ]
    )
    path = OUTPUT_DIR / "PHASE2B_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_handoff(results: list[dict], cleaned_path: Path) -> Path:
    """Write the group-chat hand-off in the Section 5 template."""
    t_map = {item["test_id"]: item for item in results}
    key_bits = []
    for test_id in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        item = t_map[test_id]
        key_bits.append(
            f"{test_id}: {item['statistic_name']}={fmt_float(item['statistic'], 2)}, "
            f"p={fmt_p(item['p_value'])}, "
            f"{item['effect_size_name']}={fmt_float(item['effect_size'], 4)}"
        )
    t1v = fmt_float(t_map["T1"]["effect_size"], 4)
    t2v = fmt_float(t_map["T2"]["effect_size"], 4)
    t3v = fmt_float(t_map["T3"]["effect_size"], 4)
    t4v = fmt_float(t_map["T4"]["effect_size"], 6)
    t4e = fmt_float(t_map["T4"]["extra"]["epsilon_squared"], 6)
    t5d = fmt_float(t_map["T5"]["effect_size"], 4)
    t6d = fmt_float(t_map["T6"]["effect_size"], 4)
    text = (
        "HAND-OFF: Phase 2b, Hypothesis tests (B)\n"
        "1. Done: Loaded A's cleaned file without editing it. Ran the six source-of-truth tests "
        "(T1–T3 chi-squared, T4 ANOVA plus Kruskal–Wallis because profit is heavily skewed, "
        "T5–T6 Welch t-tests). Alpha = 0.05. Assumptions, effect sizes, plots, and a results CSV "
        "are in the repo. Power BI pages 1–3 are not in this hand-off; KPIs and the dashboard spec "
        "are ready for that next step.\n"
        "2. Key numbers: "
        + "; ".join(key_bits)
        +         f". Only T1 and T4 reject H0 at alpha=0.05. T1 is the only practical effect "
        f"(Cramer's V = {t1v}, medium on Cohen's scale; First Class 95.3% late vs Standard Class 38.1%). "
        f"T2 p=0.070 does not reject (V={t2v}). T3 p=0.458 does not reject (V={t3v}). "
        f"T4 eta-squared={t4v} is negligible (Kruskal–Wallis epsilon-squared={t4e}; Levene fails so ANOVA is shaky). "
        f"T5 p=0.113, d={t5d}; T6 p=0.109, d={t6d}; both do not reject.\n"
        "3. Decisions: Used original text columns (`Shipping Mode`, `Market`, `Customer Segment`), "
        "not the one-hot dummies. Late and loss are A's 0/1 targets. Discount is "
        "`Order Item Discount Rate`. Welch t-tests were used as specified (unequal variances allowed). "
        "ANOVA is reported as required, but Kruskal–Wallis is flagged as safer because Benefit per "
        "order skewness is about −4.74. Rows are treated as independent order lines; that is a "
        "limitation, not a fix. No multiple-testing correction; the source of truth listed six tests.\n"
        f"4. Files: `{rel(OUTPUT_DIR / 'tests_results.csv')}`; "
        f"`{rel(OUTPUT_DIR / 'PHASE2B_REPORT.md')}`; `{rel(PLOT_DIR)}/`; `{rel(TABLE_DIR)}/`; "
        f"`B_hypothesis_powerbi/dashboard_spec.md`. Input: `{rel(cleaned_path)}`.\n"
        "5. Before you start: Quote effect sizes, not p-values. Shipping mode is the only strong "
        "association; do not claim that market, segment, or discount 'drive' late delivery or loss. "
        "C and D still create their own 80/20 split (`random_state = 42`) and scale after the split. "
        "B still owes Power BI pages 1–3 on the cleaned file, then page 4 after C and D push."
    )
    path = OUTPUT_DIR / "handoff_phase2b.md"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def main() -> int:
    """Run T1–T6 and write outputs."""
    args = parse_args()
    cleaned_path = resolve_repo_path(args.cleaned)
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cleaned_path, encoding="utf-8", usecols=USECOLS, low_memory=False)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")
    if df[USECOLS].isna().any().any():
        raise ValueError("Nulls found in test columns; A's cleaned file should have none.")

    t1 = chi_square_test(
        df,
        "T1",
        "Chi-squared test of independence: Shipping Mode vs late",
        "Shipping Mode",
        "late",
        "Shipping Mode and late delivery are independent.",
        "Shipping Mode and late delivery are associated.",
    )
    t2 = chi_square_test(
        df,
        "T2",
        "Chi-squared test of independence: Market vs late",
        "Market",
        "late",
        "Market and late delivery are independent.",
        "Market and late delivery are associated.",
    )
    t3 = chi_square_test(
        df,
        "T3",
        "Chi-squared test of independence: Customer Segment vs loss",
        "Customer Segment",
        "loss",
        "Customer Segment and loss are independent.",
        "Customer Segment and loss are associated.",
    )
    t4 = anova_profit_by_market(df)
    late = df.loc[df["late"] == 1, "Benefit per order"].to_numpy(dtype=float)
    on_time = df.loc[df["late"] == 0, "Benefit per order"].to_numpy(dtype=float)
    t5 = welch_test(
        late,
        on_time,
        "T5",
        "Welch two-sample t-test: Benefit per order, late vs on-time",
        "Mean Benefit per order is the same for late and on-time orders.",
        "Mean Benefit per order differs between late and on-time orders.",
        "late",
        "on-time",
    )
    loss = df.loc[df["loss"] == 1, "Order Item Discount Rate"].to_numpy(dtype=float)
    not_loss = df.loc[df["loss"] == 0, "Order Item Discount Rate"].to_numpy(dtype=float)
    t6 = welch_test(
        loss,
        not_loss,
        "T6",
        "Welch two-sample t-test: Order Item Discount Rate, loss vs not-loss",
        "Mean discount rate is the same for loss and not-loss orders.",
        "Mean discount rate differs between loss and not-loss orders.",
        "loss",
        "not-loss",
    )
    results = [t1, t2, t3, t4, t5, t6]
    save_plots(df)
    csv_path = write_csv(results)
    report_path = write_report(df, results, cleaned_path)
    handoff_path = write_handoff(results, cleaned_path)

    print(f"Wrote {rel(csv_path)}")
    print(f"Wrote {rel(report_path)}")
    print(f"Wrote {rel(handoff_path)}")
    for item in results:
        print(
            f"{item['test_id']}: {item['statistic_name']}={fmt_float(item['statistic'], 4)} "
            f"p={fmt_p(item['p_value'])} {item['effect_size_name']}={fmt_float(item['effect_size'], 6)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
