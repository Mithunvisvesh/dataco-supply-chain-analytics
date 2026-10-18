# DataCo Supply Chain Analytics — Verification Report

**Run date:** 2026-10-17 | **Python:** 3.13.2 | **Platform:** Windows  
**Verdict: ✅ ALL PHASES PASS**

---

## Environment

| Package | Required | Installed | Status |
|---------|----------|-----------|--------|
| pandas | 2.3.3 | 2.3.3 | ✅ |
| numpy | 2.3.5 | 2.3.3 | ✅ Compatible |
| matplotlib | 3.10.6 | 3.10.6 | ✅ |
| seaborn | 0.13.2 | 0.13.2 | ✅ |
| scipy | 1.16.2 | 1.16.2 | ✅ |
| statsmodels | 0.15.0 | 0.15.0 | ✅ |
| scikit-learn | 1.7.2 | 1.7.2 | ✅ |
| imbalanced-learn | 0.14.0 | 0.14.0 | ✅ |

> **NumPy version note:** All four phase requirements files pin `numpy==2.3.5`. The local Python environment inspected during this audit uses NumPy 2.3.3. Available evidence does not conclusively establish which NumPy version was used to generate the committed Phase 2B and Phase 3B outputs. No claim of output incompatibility or numerical impact is made without further evidence.

---

## Phase 1 — Cleaning & Preprocessing (Oscar)

**Script:** `A_preprocessing_eda/phase1_cleaning.py` → Exit code: **0** ✅  
**Validator:** `A_preprocessing_eda/phase1_validate.py` → Exit code: **0** ✅ — **56/56 checks PASSED**

### Cleaned File Stats

| Metric | Value |
|--------|-------|
| Raw shape | 180,519 rows × 53 columns |
| Cleaned shape | 180,519 rows × 122 columns |
| File size | 93.65 MB |
| Nulls | 0 |
| Rows deleted | 0 |
| Late class (`late = 1`) | 54.83% |
| Loss class (`loss = 1`) | 18.71% |

### Validator — All 56 Checks PASSED

| Check Category | Result |
|----------------|--------|
| Row count preserved (180,519) | ✅ PASS |
| Zero nulls in cleaned file | ✅ PASS |
| All 25 empty / personal / ID / leakage / exact-duplicate columns dropped | ✅ PASS |
| Required columns present | ✅ PASS |
| `late` and `loss` binary flags correct | ✅ PASS |
| `loss` equals `Benefit per order < 0` on every row | ✅ PASS |
| `late` equals `Late_delivery_risk` on every row | ✅ PASS |
| `Benefit per order`, `Sales`, `Order Item Quantity` match raw | ✅ PASS |
| One-hot groups valid (row sums = 1, values 0/1) | ✅ PASS |
| Date features (`order_year`, `order_month`, `order_weekday`) correct | ✅ PASS |
| `Sales_log`, `Benefit_signed_log` transforms correct | ✅ PASS |
| No duplicate column names | ✅ PASS |
| No duplicate rows | ✅ PASS |
| Columns NOT pre-standardised | ✅ PASS |

### Outputs Produced

| File | Status |
|------|--------|
| `A_preprocessing_eda/outputs/cleaned_supply_chain.csv` | ✅ |
| `A_preprocessing_eda/outputs/cleaning_log.md` | ✅ |
| `A_preprocessing_eda/outputs/handoff_phase1.md` | ✅ |

---

## Phase 2a — EDA (Oscar)

**Script:** `A_preprocessing_eda/phase2a_eda.py` → Exit code: **0** ✅

### Outputs Produced

| File | Status |
|------|--------|
| `A_preprocessing_eda/outputs/eda_findings.md` (9 findings) | ✅ |
| `A_preprocessing_eda/outputs/handoff_phase2a.md` | ✅ |
| `A_preprocessing_eda/outputs/eda_summary_stats.csv` | ✅ |
| `A_preprocessing_eda/outputs/eda_plots/` — 30 plots | ✅ |

---

## Phase 2b — Hypothesis Tests (Mithun)

**Script:** `B_hypothesis_powerbi/phase2b_tests.py` → Exit code: **0** ✅

### Test Results (T1–T6), α = 0.05

| Test | Description | Statistic | p-value | Effect Size | Decision |
|------|-------------|-----------|---------|-------------|----------|
| **T1** | Chi-sq: Shipping Mode vs `late` | χ²=37,716.04, df=3 | < 1e-300 | Cramer's V = **0.4571** (medium) | **Reject H₀** |
| **T2** | Chi-sq: Market vs `late` | χ²=8.65, df=4 | 0.0704 | Cramer's V = 0.0069 (negligible) | Do not reject H₀ |
| **T3** | Chi-sq: Customer Segment vs `loss` | χ²=1.56, df=2 | 0.4583 | Cramer's V = 0.0029 (negligible) | Do not reject H₀ |
| **T4** | ANOVA: `Benefit per order` across Market | F=3.34, df=(4,180514) | 0.0096 | η²=0.000074 (negligible) | **Reject H₀** (negligible practical effect) |
| **T5** | Welch t-test: profit, late vs on-time | t=−1.587, df=175,327 | 0.1125 | Cohen's d = −0.0075 (negligible) | Do not reject H₀ |
| **T6** | Welch t-test: discount rate, loss vs not-loss | t=1.605, df=50,531 | 0.1085 | Cohen's d = 0.0097 (negligible) | Do not reject H₀ |

### Key Finding

**Shipping Mode is the only factor with a meaningful association with late delivery** (Cramer's V = 0.457).

| Shipping Mode | Late Rate |
|---------------|-----------|
| First Class | **95.32%** |
| Second Class | **76.63%** |
| Same Day | 45.74% |
| Standard Class | 38.07% |

All other tested associations are statistically non-significant or have negligible effect sizes.

### Outputs Produced

| File | Status |
|------|--------|
| `B_hypothesis_powerbi/outputs/tests_results.csv` | ✅ |
| `B_hypothesis_powerbi/outputs/PHASE2B_REPORT.md` | ✅ |
| `B_hypothesis_powerbi/outputs/handoff_phase2b.md` | ✅ |
| `B_hypothesis_powerbi/outputs/plots/` | ✅ |
| `B_hypothesis_powerbi/outputs/tables/` | ✅ |

---

## Phase 3a — OLS + Loss Classification (Hisana)

**Script:** `C_loss_ols/c_ols_loss.py` → Exit code: **0** ✅  
**Script:** `C_loss_ols/c_signal_check.py` → Exit code: **0** ✅

### OLS on `Benefit per order`

| Model | R² Train | Adj. R² Train | R² Test | RMSE Test |
|-------|----------|---------------|---------|-----------|
| OLS raw (classical SE) | 0.0165 | 0.0163 | **0.0200** | \$102.79 |
| OLS raw (HC3 robust SE) | 0.0165 | 0.0163 | **0.0200** | \$102.79 |
| OLS signed-log response | 0.0146 | 0.0144 | 0.0163 | \$104.20 |
| Mean-only baseline | 0.0000 | 0.0000 | ~0.0000 | \$103.84 |

### OLS Assumption Tests

| Model | Residual Skew | Excess Kurtosis | Jarque-Bera p | Breusch-Pagan p | Durbin-Watson | Max VIF |
|-------|--------------|----------------|---------------|-----------------|---------------|---------|
| Raw | −5.63 | 87.77 | 0.0 ❌ | 0.0 ❌ | 1.994 ✅ | 3.091 ✅ |
| Signed-log | −1.58 | 1.03 | 0.0 ❌ | 0.0 ❌ | 1.994 ✅ | 3.091 ✅ |

- ❌ Normality (Jarque-Bera) and homoscedasticity (Breusch-Pagan) fail on both scales → HC3 robust SEs used.
- ✅ No multicollinearity (max VIF 3.09 < 5).
- ✅ No autocorrelation (Durbin-Watson ≈ 2).
- 10 of 27 coefficients significant at p < 0.05 (HC3).

### Loss Classification (test set, 36,104 orders)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Always-predict-profit (dummy) | 0.8128 | 0.000 | 0.000 | 0.000 | 0.500 |
| LR baseline | 0.8128 | 0.000 | 0.000 | 0.000 | 0.503 |
| LR class_weight=balanced | 0.5112 | 0.188 | 0.486 | 0.271 | 0.503 |
| LR random oversampling | 0.5033 | 0.189 | 0.504 | 0.275 | 0.505 |
| LR SMOTE | 0.4941 | 0.189 | 0.519 | 0.277 | **0.504** |

### Signal Check (Gradient-Boosting Ceiling Probe)

| Check | Value | Interpretation |
|-------|-------|----------------|
| Loss ROC-AUC (GB, all columns incl. product name) | 0.505 | No better than random |
| Profit R² (GB) | 0.011 | Weaker than OLS |
| Same product, fixed price, discount ≤ 5% — loss rate | 18.9% | Unexplained variance in backend costs |

**Conclusion:** Loss cannot be forecast from order-time data. Missing information is backend operational cost (fulfilment, returns). Do not describe any variable as a "driver" of loss.

### Outputs Produced

| File | Status |
|------|--------|
| `C_loss_ols/outputs/PHASE3A_REPORT.md` | ✅ |
| `C_loss_ols/outputs/ols_summary.txt` | ✅ |
| `C_loss_ols/outputs/ols_coefficients_raw.csv` | ✅ |
| `C_loss_ols/outputs/ols_coefficients_raw_HC3.csv` | ✅ |
| `C_loss_ols/outputs/ols_coefficients_signedlog.csv` | ✅ |
| `C_loss_ols/outputs/ols_model_comparison.csv` | ✅ |
| `C_loss_ols/outputs/ols_assumption_tests.csv` | ✅ |
| `C_loss_ols/outputs/ols_vif.csv` | ✅ |
| `C_loss_ols/outputs/loss_logit_summary.txt` | ✅ |
| `C_loss_ols/outputs/loss_model_comparison.csv` | ✅ |
| `C_loss_ols/outputs/loss_odds_ratios.csv` | ✅ |
| `C_loss_ols/outputs/loss_predictions.csv` | ✅ |
| `C_loss_ols/outputs/ols_predictions_test.csv` | ✅ |
| `C_loss_ols/outputs/signal_check.csv` / `signal_check.txt` | ✅ |
| `C_loss_ols/outputs/ols_plots/` | ✅ |
| `C_loss_ols/outputs/handoff_phase3a.md` | ✅ |

---

## Phase 3b — Late-Delivery Model (Preetham)

**Script:** `D_late_model_report/late_model.py` → Exit code: **0** ✅

### Data Integrity Checks

| Check | Result |
|-------|--------|
| No missing values | ✅ |
| Target (`Late_delivery_risk`) is binary | ✅ |
| Shipping Mode / scheduled-day mapping verified (no leakage) | ✅ |
| No forbidden leakage features in model | ✅ |

### Dataset Split

| Set | Rows | Late rate |
|-----|------|-----------|
| Full dataset | 180,519 | 54.83% |
| Training (80%, stratified) | 144,415 | 54.83% |
| Test (20%, stratified) | 36,104 | 54.83% |

### Test-Set Results

| Metric | Value |
|--------|-------|
| Accuracy | **0.6968** |
| Precision | 0.8437 |
| Recall | 0.5485 |
| F1-score | 0.6649 |
| **ROC-AUC** | **0.7409** |

### Confusion Matrix

| | Predicted: On-time (0) | Predicted: Late (1) |
|-|----------------------|-------------------|
| **Actual: On-time (0)** | 14,297 ✅ | 2,011 ❌ |
| **Actual: Late (1)** | 8,937 ❌ | 10,859 ✅ |

### Top Odds Ratios

| Feature | Odds Ratio | Interpretation |
|---------|-----------|----------------|
| Shipping Mode: First Class | **34.22** | 34× more likely to be late vs Standard Class |
| Shipping Mode: Second Class | **5.39** | 5.4× more likely to be late vs Standard Class |
| Shipping Mode: Same Day | 1.38 | Slightly more likely to be late |
| Type: TRANSFER | 0.64 | Less likely to be late |
| Order Region: Canada | 0.76 | Less likely to be late |

Shipping Mode dominates the model — consistent with the T1 hypothesis test finding (Cramer's V = 0.457). These are predictive associations, not causal effects.

### Outputs Produced

| File | Status |
|------|--------|
| `D_late_model_report/outputs/late_model_results.csv` | ✅ |
| `D_late_model_report/outputs/late_predictions.csv` | ✅ |
| `D_late_model_report/outputs/late_odds_ratios.csv` | ✅ |
| `D_late_model_report/outputs/late_model_summary.txt` | ✅ |
| `D_late_model_report/outputs/plots/confusion_matrix.png` | ✅ |
| `D_late_model_report/outputs/plots/roc_curve.png` | ✅ |

---

## Summary

| Phase | Script(s) | Exit Code | Verdict |
|-------|-----------|-----------|---------|
| Phase 1 — Cleaning | `phase1_cleaning.py` | 0 | ✅ PASS |
| Phase 1 — Validation | `phase1_validate.py` | 0 | ✅ PASS — 56/56 checks |
| Phase 2a — EDA | `phase2a_eda.py` | 0 | ✅ PASS — 30 plots |
| Phase 2b — Hypothesis Tests | `phase2b_tests.py` | 0 | ✅ PASS — T1–T6 |
| Phase 3a — OLS + Loss | `c_ols_loss.py` | 0 | ✅ PASS |
| Phase 3a — Signal Check | `c_signal_check.py` | 0 | ✅ PASS |
| Phase 3b — Late Model | `late_model.py` | 0 | ✅ PASS |

### Reproducibility

All numerical outputs from the fresh run exactly match the committed outputs:

| Metric | Committed | Fresh Run |
|--------|-----------|-----------|
| Phase 1 validator checks | 56/56 | 56/56 |
| T1 Cramer's V | 0.457090 | 0.457090 |
| OLS R² (test) | 0.0200 | 0.0200 |
| OLS RMSE (test) | \$102.79 | \$102.79 |
| Signal check GB AUC | 0.505 | 0.505 |
| Late model ROC-AUC | 0.7409 | 0.7409 |
| Late model accuracy | 0.6968 | 0.6968 |

---

## Bug Fixed

### `D_late_model_report/late_model.py` — UnicodeEncodeError on Windows

**Severity:** High — script crashed before producing any output on default Windows (cp1252) terminals.

**Root cause:** `print()` calls containing Unicode characters (`✓` U+2713, `—` U+2014) are not encodable in cp1252.

**Fix applied:**
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
```
Added immediately after the `import sys` / `from pathlib import Path` lines. The script now runs without any environment variable workarounds.
