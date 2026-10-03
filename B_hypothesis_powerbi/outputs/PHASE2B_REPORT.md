# Phase 2b report — Hypothesis tests (Person B)

**Course:** Foundations of Data Science (23CSE351) · Group 3  
**Owner:** B (Mithun)  
**Run date:** 2026-10-03  
**Input:** `A_preprocessing_eda/outputs/cleaned_supply_chain.csv`  
**Alpha:** 0.05  

This report is the teammate-facing summary of the six source-of-truth tests. The pasteable hand-off is `handoff_phase2b.md`. Plots live in `outputs/plots/`.

## 1. Why effect size still matters

The cleaned file has 180,519 rows. The source of truth warned that almost every test would be significant at this n. That is **not** what happened: only T1 and T4 reject H0 at α = 0.05. T2, T3, T5 and T6 do not. Quote the **effect size** and the group rates anyway. T1 is the only result with a practical gap.

## 2. KPIs (for Power BI page 1)

| Measure | Value |
| --- | --- |
| Order lines (rows) | 180,519 |
| Late % | 54.8291% |
| Loss % | 18.7149% |
| Total sales (`Sales`) | 36784735.01 |
| Total profit (`Benefit per order`) | 3966902.97 |
| Mean profit | 21.9750 |
| Mean discount rate | 0.1017 |

## 3. Results

| ID | Test | Statistic | p-value | Effect size | Decision |
| --- | --- | --- | --- | --- | --- |
| T1 | Chi-squared test of independence: Shipping Mode vs late | chi2=37716.04 | < 1e-300 | Cramer's V=0.457090 | Reject H0 (p < 0.05) |
| T2 | Chi-squared test of independence: Market vs late | chi2=8.65 | 0.070448 | Cramer's V=0.006923 | Do not reject H0 (p ≥ 0.05) |
| T3 | Chi-squared test of independence: Customer Segment vs loss | chi2=1.56 | 0.458255 | Cramer's V=0.002940 | Do not reject H0 (p ≥ 0.05) |
| T4 | One-way ANOVA (Kruskal–Wallis added) of Benefit per order across Market | F=3.34 | 0.009626 | eta-squared=0.000074 | Reject H0 (p < 0.05) |
| T5 | Welch two-sample t-test: Benefit per order, late vs on-time | t (Welch)=-1.59 | 0.112539 | Cohen's d=-0.007489 | Do not reject H0 (p ≥ 0.05) |
| T6 | Welch two-sample t-test: Order Item Discount Rate, loss vs not-loss | t (Welch)=1.60 | 0.108513 | Cohen's d=0.009682 | Do not reject H0 (p ≥ 0.05) |

### T1. Chi-squared test of independence: Shipping Mode vs late

- **H0:** Shipping Mode and late delivery are independent.
- **H1:** Shipping Mode and late delivery are associated.
- **Statistic:** chi2 = 37716.0425 (df = 3)
- **p-value:** < 1e-300
- **Effect size:** Cramer's V = 0.457090
- **Assumptions:** Independent order-line rows assumed (limitation: several lines can share an order). Expected counts: min=4398.29, cells < 5: 0 of 8. Chi-squared approximation is appropriate.
- **Interpretation:** Reject H0 (p < 0.05). Cramer's V = 0.4571 (medium (Cohen cut-offs at df*=1: small 0.100, medium 0.300, large 0.500)). late=1 rates: First Class: 95.3225%; Second Class: 76.6328%; Same Day: 45.7430%; Standard Class: 38.0717%.

### T2. Chi-squared test of independence: Market vs late

- **H0:** Market and late delivery are independent.
- **H1:** Market and late delivery are associated.
- **Statistic:** chi2 = 8.6507 (df = 4)
- **p-value:** 0.070448
- **Effect size:** Cramer's V = 0.006923
- **Assumptions:** Independent order-line rows assumed (limitation: several lines can share an order). Expected counts: min=5246.14, cells < 5: 0 of 10. Chi-squared approximation is appropriate.
- **Interpretation:** Do not reject H0 (p ≥ 0.05). Cramer's V = 0.0069 (negligible (Cohen cut-offs at df*=1: small 0.100, medium 0.300, large 0.500)). late=1 rates: Europe: 55.2078%; Pacific Asia: 55.0460%; USCA: 54.8006%; Africa: 54.5893%; LATAM: 54.3552%.

### T3. Chi-squared test of independence: Customer Segment vs loss

- **H0:** Customer Segment and loss are independent.
- **H1:** Customer Segment and loss are associated.
- **Statistic:** chi2 = 1.5607 (df = 2)
- **p-value:** 0.458255
- **Effect size:** Cramer's V = 0.002940
- **Assumptions:** Independent order-line rows assumed (limitation: several lines can share an order). Expected counts: min=6031.07, cells < 5: 0 of 6. Chi-squared approximation is appropriate.
- **Interpretation:** Do not reject H0 (p ≥ 0.05). Cramer's V = 0.0029 (negligible (Cohen cut-offs at df*=1: small 0.100, medium 0.300, large 0.500)). loss=1 rates: Home Office: 18.8481%; Corporate: 18.8249%; Consumer: 18.6046%.

### T4. One-way ANOVA (Kruskal–Wallis added) of Benefit per order across Market

- **H0:** Mean Benefit per order is the same in every Market.
- **H1:** At least one Market has a different mean Benefit per order.
- **Statistic:** F = 3.3412 (df = 4, 180514)
- **p-value:** 0.009626
- **Effect size:** eta-squared = 0.000074
- **Assumptions:** Independence assumed at the order-line level. Profit skewness = -4.742 (heavy left tail; ANOVA normality is not realistic). Levene (median) W = 30.776, p = 1.184e-25 (variances differ). Kruskal–Wallis H = 85.339, p = 1.285e-17, epsilon-squared = 0.000451.
- **Interpretation:** Reject H0 (p < 0.05). Eta-squared = 0.000074 (negligible (cut-offs 0.01 / 0.06 / 0.14)). Because profit is heavily skewed, Kruskal–Wallis is the more appropriate test; it also rejects H0, with a negligible rank effect (epsilon-squared = 0.000451). Market profit: Africa: mean=21.70, median=31.49; Europe: mean=23.27, median=32.78; LATAM: mean=21.77, median=32.02; Pacific Asia: mean=20.79, median=29.18; USCA: mean=21.87, median=31.49.

### T5. Welch two-sample t-test: Benefit per order, late vs on-time

- **H0:** Mean Benefit per order is the same for late and on-time orders.
- **H1:** Mean Benefit per order differs between late and on-time orders.
- **Statistic:** t (Welch) = -1.5869 (df = 175327.64)
- **p-value:** 0.112539
- **Effect size:** Cohen's d = -0.007489
- **Assumptions:** Welch t-test does not assume equal variances (s² late=11123.4471, s² on-time=10642.6568). CLT applies at n=180,519, so the t approximation is usable despite skew (late skew=-5.030, on-time skew=-4.366). Independence assumed at the order-line level.
- **Interpretation:** Do not reject H0 (p ≥ 0.05). Cohen's d = -0.0075 (late minus on-time; negligible (cut-offs 0.20 / 0.50 / 0.80)). Mean late=21.6217 (n=98,977), mean on-time=22.4038 (n=81,542).

### T6. Welch two-sample t-test: Order Item Discount Rate, loss vs not-loss

- **H0:** Mean discount rate is the same for loss and not-loss orders.
- **H1:** Mean discount rate differs between loss and not-loss orders.
- **Statistic:** t (Welch) = 1.6049 (df = 50531.20)
- **p-value:** 0.108513
- **Effect size:** Cohen's d = 0.009682
- **Assumptions:** Welch t-test does not assume equal variances (s² loss=0.0050, s² not-loss=0.0050). CLT applies at n=180,519, so the t approximation is usable despite skew (loss skew=0.325, not-loss skew=0.345). Independence assumed at the order-line level.
- **Interpretation:** Do not reject H0 (p ≥ 0.05). Cohen's d = 0.0097 (loss minus not-loss; negligible (cut-offs 0.20 / 0.50 / 0.80)). Mean loss=0.1022 (n=33,784), mean not-loss=0.1015 (n=146,735).

## 4. Plain-language findings for EDA / dashboard / viva

1. **Shipping mode vs late (T1):** Reject H0. Cramer's V = 0.4571 is **medium** on Cohen's 2×k scale (large starts at 0.50) and the practical gap is huge: First Class 95.3% late vs Standard Class 38.1%. Lead Power BI page 2 with this.
2. **Market vs late (T2):** Do **not** reject H0 (p = 0.070448). V = 0.0069 is negligible; late rates sit around 54–55% in every market.
3. **Segment vs loss (T3):** Do **not** reject H0 (p = 0.458255). V = 0.0029 is negligible; loss is about 18.6–18.8% in every segment.
4. **Profit across markets (T4):** ANOVA rejects (p = 0.009626) but eta-squared = 0.000074 is **negligible**. Levene fails and profit is heavily skewed, so quote Kruskal–Wallis: it also rejects, with epsilon-squared = 0.000451. Markets do not explain profit in any practical sense.
5. **Profit, late vs on-time (T5):** Do **not** reject H0 (p = 0.112539). Cohen's d = -0.0075 is negligible. Late deliveries are not where the money is won or lost.
6. **Discount rate, loss vs not-loss (T6):** Do **not** reject H0 (p = 0.108513). Cohen's d = 0.0097 is negligible. Mean discount is 0.1022 vs 0.1015.

## 5. Limitations

- Rows are **order lines**, not unique orders or unique shipments. Tests treat them as independent.
- n is large, but four of six tests still fail to reject H0. Quote effect sizes and group rates in the viva.
- T4–T6 use `Benefit per order` and `Order Item Discount Rate` as they stand (unscaled, outliers kept).
- Late and loss definitions follow the source of truth: `late` = `Late_delivery_risk`; `loss` = 1 if `Benefit per order` < 0.

## 6. How to reproduce

```bash
python -m pip install -r B_hypothesis_powerbi/requirements.txt
python B_hypothesis_powerbi/phase2b_tests.py
```

Numbers in this file match `B_hypothesis_powerbi/outputs/tests_results.csv` and `handoff_phase2b.md`.
