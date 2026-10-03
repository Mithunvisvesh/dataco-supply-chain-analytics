# Phase 1 report — Data cleaning and preprocessing (Person A)

**Course:** Foundations of Data Science (23CSE351) · Group 3  
**Owner:** A  
**Status:** Complete (validated)  
**Run date:** 2026-10-03  
**Input:** `dataset/DataCoSupplyChainDataset.csv`  
**Output:** `A_preprocessing_eda/outputs/cleaned_supply_chain.csv`

This report is the teammate-facing summary of Phase 1. Use it before you start hypothesis tests, OLS, loss classification, the late-delivery model, or Power BI. The detailed viva document remains `cleaning_log.md`.

---

## 1. What was delivered

| File | Purpose |
| --- | --- |
| `A_preprocessing_eda/outputs/cleaned_supply_chain.csv` | Shared analysis table (unscaled, no train/test split) |
| `A_preprocessing_eda/outputs/cleaning_log.md` | Full step log, column dictionary, leakage table, validation |
| `A_preprocessing_eda/outputs/handoff_phase1.md` | Short pasteable hand-off |
| `A_preprocessing_eda/outputs/outlier_plots/` | Box plots and before/after histograms |
| `A_preprocessing_eda/phase1_cleaning.py` | Reproducible pipeline |
| `A_preprocessing_eda/phase1_validate.py` | Independent checks (56 passed, 0 failed) |

---

## 2. Key numbers (from the data)

| Metric | Value |
| --- | --- |
| Rows × columns (raw) | 180,519 × 53 |
| Rows × columns (cleaned) | 180,519 × 122 |
| Rows deleted | 0 |
| Null cells (raw → cleaned) | 336,209 → 0 |
| Columns dropped | 25 (empty 4, personal 5, ID 9, late leakage 3, profit leakage 2, exact duplicate 2) |
| Columns added | 94 (date 4, transform 2, grouped text 3, one-hot 83, targets 2) |
| `late` = 1 | 98,977 (54.8291%) |
| `late` = 0 | 81,542 (45.1709%) |
| `loss` = 1 | 33,784 (18.7149%) |
| `loss` = 0 | 146,735 (81.2851%) |
| Exact zero profit | 1,177 (0.6520%), all coded as **not loss** |
| Order date range | 2015-01-01 00:00:00 to 2018-01-31 23:38:00 |
| Cleaned file size | ~93.65 MB |

### Outliers reviewed (not removed)

| Column | Outside 1.5×IQR | Percent | Skewness |
| --- | --- | --- | --- |
| `Sales` | 488 | 0.2703% | 2.884249 |
| `Benefit per order` | 18,942 | 10.4931% | −4.741834 |
| `Order Item Quantity` | 0 | 0.0000% | 0.880252 |

Of the `Benefit per order` outliers, 13,727 are negative (loss rows). They stay in the file.

---

## 3. What was done (pipeline)

1. Load with `encoding="latin-1"`; verify shape, nulls, class balances.
2. Drop empty/useless, personal, identifier, and leakage columns (lists in the cleaning log).
3. Parse `order date (DateOrders)` only; derive `order_year`, `order_month`, `order_weekday`, `order_weekday_name`. Nothing from shipping date.
4. Drop exact duplicates when values match (`Sales per customer` → keep `Order Item Total`; `Product Price` → keep `Order Item Product Price`). Fill `Customer Zipcode` nulls with `"Unknown"`.
5. Document outliers; add `Sales_log` and `Benefit_signed_log`. No row deletion or capping. No scaling.
6. One-hot encode low-cardinality categoricals (including `Order Region`). Top-10 group then one-hot for `Category Name`, `Order State`, `Order Country`. Keep original text columns.
7. Add targets `late` (= `Late_delivery_risk`) and `loss` (= `Benefit per order` < 0).
8. Save UTF-8 CSV; re-read and assert; run `phase1_validate.py`.

Seed: `random_state = 42` (no random sampling in this phase).

---

## 4. Decisions teammates must respect

| Decision | Implication |
| --- | --- |
| File is unscaled | Fit scalers on the **training** fold only |
| No train/test split here | Create your own 80/20 stratified split with `random_state = 42` |
| One-hot includes all levels | Drop one reference level per group before intercept models |
| `Benefit per order` kept | OLS target only — never a feature for loss or OLS |
| `Benefit_signed_log` | Transform of the profit target — do not use as a regressor for loss/OLS |
| `late` and `Late_delivery_risk` both kept | Identical; both are targets |
| `Order Status` retained as text | Open decision — may leak post-order info; do not use silently |
| `TOP_N = 10` | `Order State` kept levels cover only ~19.4% of rows; most map to `Other` |
| Collinear pair retained | `Sales` and `Order Item Total` (Pearson r ≈ 0.9897) — C should check VIF |
| Trailing spaces in some labels | e.g. `'Health and Beauty '`, some `Order Region` values — dummies use exact text |

### Leakage removed (do not bring them back)

- Late model: `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)`
- Loss / OLS features: `Order Profit Per Order`, `Order Item Profit Ratio`
- Kept on purpose: `Days for shipment (scheduled)` (known at order time)

### Text columns not encoded (not for modelling without a decision)

`Order Status`, `Product Name`, `Customer City`, `Customer State`, `Customer Country`, `Order City`, `Customer Zipcode`

Use the one-hot / grouped one-hot columns for models. Original category text is for Power BI and EDA.

---

## 5. Who starts next

| Person | Work | First step |
| --- | --- | --- |
| B | Hypothesis tests + Power BI | Load cleaned CSV; run T1–T6 from the source of truth; build dashboard pages 1–3 |
| C | OLS + loss classification | Exclude profit leakage and targets-as-features; check VIF on collinear sales columns; 80/20 stratified split |
| D | Late-delivery model | Confirm leakage columns are absent; use scheduled days + shipping mode / market / region / segment etc.; expect modest accuracy |
| A | Phase 2a EDA | Distributions, correlations, group plots, `eda_findings.md` |

---

## 6. How to reproduce Phase 1

```bash
python -m pip install -r A_preprocessing_eda/requirements.txt
python A_preprocessing_eda/phase1_cleaning.py
python A_preprocessing_eda/phase1_validate.py
```

Expect: validator summary `56 passed, 0 failed`. Running the cleaner twice produces a byte-identical cleaned CSV.

---

## 7. Discrepancies logged

- Project brief expected order dates ending **9 Sep 2017**. Actual maximum is **2018-01-31 23:38:00**. No rows were removed to force the window.
- Zero-profit share is **0.6520%** vs brief **0.7%** (within the 0.1 percentage-point tolerance → check marked PASS).

---

## 8. Limitations

Single company; orders span 2015–early 2018 in this extract; loss defined strictly as `Benefit per order` < 0; leakage columns excluded; cleaned file is large (~94 MB).

---

*Generated to accompany the Phase 1 pipeline. Numbers match `cleaning_log.md` and `handoff_phase1.md`.*
