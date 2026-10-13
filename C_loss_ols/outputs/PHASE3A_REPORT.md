# Phase 3a report: OLS on profit and loss classification

Block C (Hisana) · Foundations of Data Science (23CSE351) · Group 4

Scripts: `C_loss_ols/c_ols_loss.py` (models), `C_loss_ols/c_signal_check.py` (signal check).
Input: `A_preprocessing_eda/outputs/cleaned_supply_chain.csv` (read only).

## 1. Headline finding

**Order-time information (price, quantity, discount, category, shipping mode, market, segment, payment type, month, weekday) is not enough to explain or forecast whether an order loses money.** OLS explains about 2% of the variance in `Benefit per order`, and every loss classifier has ROC-AUC of about 0.50. This is the result of the study, not a modelling failure: a flexible gradient-boosting probe using every order-time column, including product name, did no better (section 5).

Plain-language reading for the viva: the dataset records what was ordered, at what price and discount, and how it shipped. It does not record what the order cost the company to fulfil (product cost, handling, returns, carrier charges). Profit is revenue minus those costs, so without them the sign of the margin cannot be forecast from the order alone.

## 2. Setup

| Item | Value |
| --- | --- |
| Rows | 180,519 order lines; 144,415 train, 36,104 test |
| Split | 80/20, stratified on `loss`, `random_state = 42`; the same split is used for OLS and the classifiers |
| Scaling | `StandardScaler` fitted on the training set only |
| Resampling | Applied to the training set only |
| OLS target | `Benefit per order` (raw), and a signed-log version (`sign(x) * log1p(abs(x))`) because plain log is undefined for losses |
| Classification target | `loss = 1` if `Benefit per order` < 0 (18.72% of test rows) |
| Features (27) | Product Price, Quantity, Discount Rate, order month, order weekday; Shipping Mode, Market, Customer Segment, Type, Category (grouped) as one-hot with the most frequent level dropped (Standard Class, LATAM, Consumer, DEBIT, Cleats) |

Excluded features and why:

| Excluded | Reason |
| --- | --- |
| `Benefit per order`, `Benefit_signed_log`, `loss` | Targets or transforms of the target (`Benefit_signed_log` is used only as a response variable) |
| `Late_delivery_risk`, `late` | Known only after delivery |
| `Order Status` | Post-order state; CANCELED and SUSPECTED_FRAUD orders are never late |
| `Sales`, `Order Item Total`, `Order Item Discount` | Algebraically fixed by price, quantity and discount rate (Sales vs Order Item Total r = 0.99) |
| `Days for shipment (scheduled)` | One-to-one with Shipping Mode (perfect collinearity) |
| `Order Region`, `Department Name` | Nested inside Market and Category |
| Text columns | Not encoded; high cardinality |

## 3. OLS on Benefit per order

| Model | R² train (target scale) | R² test (target scale) | RMSE test (dollars) |
| --- | --- | --- | --- |
| OLS raw (classical or HC3 SE; same coefficients) | 0.0165 | 0.0200 | 102.79 |
| OLS signed-log response | 0.0146 (signed-log) | 0.0163 (signed-log) | 104.20 (back-transformed) |
| Mean-only baseline | 0 | -0.0000 | 103.84 |

The model beats the mean-only baseline by about $1 of RMSE. Adjusted R² (train) is 0.0163.

The signed-log model's R² values are calculated on the signed-log response, while its RMSE is calculated after predictions are transformed back to dollars. They are therefore not directly comparable with the raw-dollar R² values. On the shared dollar-scale comparison, the raw OLS model has the lower test RMSE (102.79 vs 104.20), so the signed-log refit improves residual shape but not dollar-scale predictive performance.

Assumption checks (full output in `ols_summary.txt`, `ols_assumption_tests.csv`, `ols_plots/`):

| Check | Raw | Signed-log | Reading |
| --- | --- | --- | --- |
| Residual skew | -5.63 | -1.58 | Strong left tail from large losses |
| Excess kurtosis | 87.8 | 1.03 | Very heavy tails on the raw scale |
| Jarque-Bera p | ~0 | ~0 | Residuals are not normal |
| Breusch-Pagan p | ~0 | ~0 | Variance is not constant |
| Durbin-Watson | 1.99 | 1.99 | No first-order autocorrelation |
| Max VIF | 3.09 | 3.09 | No multicollinearity problem |

Because normality and constant variance fail, the raw-scale coefficients are reported with HC3 robust standard errors. With 144k rows the point estimates are still unbiased; what the failures affect is the classical standard errors and prediction intervals. The signed-log fit fixes most of the skew and kurtosis but does not add predictive power.

Statistically significant coefficients (HC3, p < 0.05; 10 of 27), with 95% CIs in `ols_coefficients_raw_HC3.csv`:

| Term | Coefficient | Reading |
| --- | --- | --- |
| Order Item Quantity | +5.72 per unit | Larger lines earn more dollars in absolute terms |
| Order Item Discount Rate | -25.16 per 1.00 (about -2.52 per 10 points) | Deeper discounts lower profit per order |
| Order Item Product Price | +0.10 per $1 | Higher-priced items earn slightly more |
| Category: Cardio Equipment | +6.70 vs Cleats | |
| Category: Shop By Sport | -6.28 vs Cleats | |
| Category: Electronics | -5.32 vs Cleats | |
| Other significant | Indoor/Outdoor Games, Women's Apparel, Other (about -2 to -3), order month (+0.21 per month) | Small |

"Significant" does not mean "useful": with 144k rows even tiny effects clear p < 0.05, and together the 27 terms explain 2% of the variance.

## 4. Loss classification

Test set: 36,104 orders, 6,757 losses (18.72%).

| Model (threshold 0.5 unless stated) | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Always predict "profit" | 0.8128 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic, no imbalance handling | 0.8128 | 0.000 | 0.000 | 0.000 | 0.503 |
| Logistic, class_weight = balanced | 0.5112 | 0.188 | 0.486 | 0.271 | 0.503 |
| Logistic, random oversampling | 0.5033 | 0.189 | 0.504 | 0.275 | 0.505 |
| Logistic, SMOTE | 0.4942 | 0.189 | 0.519 | 0.277 | 0.504 |
| Balanced, F1-max threshold 0.465 | 0.1872 | 0.187 | 1.000 | 0.315 | 0.503 |
| Balanced, recall >= 0.5 threshold 0.4996 | 0.4979 | 0.188 | 0.508 | 0.275 | 0.503 |

How to read it:

1. **The accuracy trap.** "Always profit" scores 81.3% accuracy and catches no losses. The unweighted logistic model does the same. Accuracy alone is not a valid score here.
2. **Rebalancing changes the trade-off, not the skill.** The balanced, oversampled and SMOTE models catch about half the losses, but only about 19% of the orders they flag are real losses. That precision equals the base loss rate (18.7%), so flagging is no better than picking orders at random.
3. **The F1-maximising threshold is degenerate.** It flags every order as a loss (recall 1.0, precision 0.187, F1 0.315). The highest F1 in the table therefore comes from a rule that does nothing. Thresholds were chosen from out-of-fold training predictions only; the test set was not used to tune them.
4. **Odds ratios** (`loss_odds_ratios.csv`): only one of 27 terms is significant, Category "Shop By Sport" (OR 1.080, 95% CI 1.013 to 1.151, p = 0.018, vs Cleats). With 27 tests, one p of 0.018 is within what chance produces, so it should not be presented as a driver.

## 5. Evidence that the limit is in the data, not the model

From `c_signal_check.py` (output in `signal_check.txt`):

| Check | Result |
| --- | --- |
| Gradient boosting, all order-time columns incl. Product Name, test set | Loss ROC-AUC 0.505; Benefit R² 0.011 (OLS: 0.020) |
| Loss rate across 72 products with at least 200 lines | Ranges 13.3% to 24.0%; sd 0.0216 vs 0.0194 expected from chance alone, so products do not differ in loss risk beyond sampling noise |
| One product held fixed ("Perfect Fitness Perfect Rip Deck", quantity 1, discount at most 5%) | 1,342 order lines at a single unit price: 254 losses and 1,088 profits (18.9% loss). Benefit per order runs from -$43.95 (5th percentile) through $15.55 (median) to $28.51 (95th percentile) |

The third row is the clearest: the same item, at the same price, same quantity and similar discount, loses money on about one order in five. Whatever decides which orders lose money is not in the columns we have.

Agreement with Phase 2b (Mithun): loss versus customer segment is not significant (T3: p = 0.458, Cramer's V = 0.0029), and mean discount rate does not differ between loss and profit orders (T6: p = 0.109, d = 0.0097). The two are compatible: T6 asks whether the average discount rate differs between loss and profit orders (it does not), while the OLS asks how the dollar size of profit moves with discount (a small negative slope). Neither says discount decides whether an order loses money.

## 6. Interpretation and how to word it

**Supported by the data:** order-time attributes carry almost no information about the sign or size of an order's margin.

**Interpretation (consistent with the evidence, but not tested, because the dataset has no cost columns):** the missing ingredient is backend operational cost, for example product acquisition cost, warehousing and handling, carrier charges, returns and fraud write-offs. We cannot prove this from the data; it is a plausible explanation for a margin that varies on identical items and is invisible in the order record.

Suggested wording for the report and for ma'am:

> "Loss-making orders could not be forecast from order-time variables: OLS explained about 2% of profit variance and the best loss classifiers had ROC-AUC of about 0.50, no better than chance. A flexible gradient-boosting probe confirmed this, and orders for the same product at the same price, quantity and discount still ended in loss about 19% of the time. This indicates that margin outcomes depend on cost-side information not present in the dataset, such as product and fulfilment costs. Predicting loss would require that backend operational cost data."

Avoid:

- Do not say the model "predicts" or "detects" loss orders.
- Do not call Shop By Sport or the discount rate "drivers" of loss.
- Do not quote the F1 of 0.315 as a result.
- Do not state the missing-cost explanation as a proven fact; say "indicates" or "suggests".

## 7. Limitations

- Rows are order lines, not independent orders, and several lines can share an order and customer. Standard errors may be somewhat too small; this does not change the conclusion that the effects are tiny.
- The split is random, not time-based, so it does not test forecasting into the future. A time-based check is a reasonable extension.
- Customer-level and product-level information was reduced to grouped categories; the probe with full product detail did not change the result.
- The signed-log refit changes the scale; its dollar RMSE comes from back-transforming and is slightly biased towards the median.
- Loss is defined as `Benefit per order` < 0; zero-profit orders (0.65%) are counted as not-loss.

## 8. Files

| File | Content |
| --- | --- |
| `c_ols_loss.py` | OLS and loss-classification pipeline |
| `c_signal_check.py` | Ceiling probe and same-product check |
| `outputs/ols_summary.txt` | Full statsmodels summaries, CIs, hold-out comparison |
| `outputs/ols_model_comparison.csv`, `ols_assumption_tests.csv`, `ols_vif.csv` | OLS tables |
| `outputs/ols_coefficients_raw.csv`, `_raw_HC3.csv`, `_signedlog.csv` | Coefficients with 95% CI |
| `outputs/ols_predictions_test.csv` | Test actual vs predicted (for Power BI page 4) |
| `outputs/loss_model_comparison.csv`, `loss_odds_ratios.csv`, `loss_logit_summary.txt` | Classification tables |
| `outputs/loss_predictions.csv` | One row per cleaned row: `row_id`, `split`, `loss_actual`, `loss_prob`, two threshold predictions |
| `outputs/signal_check.txt`, `signal_check.csv` | Section 5 evidence |
| `outputs/ols_plots/` | Assumption plots, coefficient plot, actual vs predicted, PR curve, confusion matrices |
