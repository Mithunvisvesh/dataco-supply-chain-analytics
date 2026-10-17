# DataCo Supply Chain Analytics

Foundations of Data Science (23CSE351) · Group 4

Shared repository for the DataCo Smart Supply Chain project. We study what drives late deliveries and loss-making orders, test those drivers statistically, fit OLS and classification models taught in the course, and present results in Power BI.

## Project questions

1. Which factors are associated with late delivery and with loss-making orders (chi-squared, ANOVA, t-tests)?
2. How well can profit be explained with OLS regression (coefficients, uncertainty, assumption checks)?
3. How well can we classify loss vs profit orders, and late vs on-time deliveries, once leaking columns are removed?

## Shared definitions

| Term | Definition |
| --- | --- |
| Late-delivery target | `Late_delivery_risk` / `late` (0/1) |
| Loss target | `loss = 1` if `Benefit per order` < 0, else 0 (zero profit = not loss) |
| Profit target (OLS) | `Benefit per order` (continuous) |
| Split and seed | 80/20 stratified for classifiers, `random_state = 42` |
| Cleaned file | `A_preprocessing_eda/outputs/cleaned_supply_chain.csv` — produced by Oscar; nobody else edits it |

## Repository layout

```text
dataset/
  DataCoSupplyChainDataset.csv          # raw extract (latin-1), do not edit
A_preprocessing_eda/
  phase1_cleaning.ipynb                 # Phase 1 walkthrough notebook
  phase1_cleaning.py                    # reproducible Phase 1 pipeline script
  phase1_validate.py                    # independent checks (must exit 0)
  phase2a_eda.py                        # Phase 2a EDA script
  requirements.txt
  outputs/
    cleaned_supply_chain.csv
    cleaning_log.md                     # viva document for preprocessing
    handoff_phase1.md                   # short group-chat hand-off
    PHASE1_REPORT.md                    # teammate-facing Phase 1 report
    outlier_plots/
    eda_plots/                          # Phase 2a plots
    eda_findings.md
    eda_summary_stats.csv
    handoff_phase2a.md
B_hypothesis_powerbi/
  phase2b_tests.py                      # T1–T6 hypothesis tests
  dashboard_spec.md                     # Power BI page layout
  outputs/
    tests_results.csv
    PHASE2B_REPORT.md
    handoff_phase2b.md
    plots/
C_loss_ols/
  c_ols_loss.py                         # Phase 3a: OLS + loss classification
  c_signal_check.py                     # Phase 3a: ceiling probe + same-product check
  requirements.txt
  outputs/
    PHASE3A_REPORT.md                   # findings and viva wording
    handoff_phase3a.md
    ols_*.csv / ols_summary.txt         # OLS tables, coefficients, assumption tests
    loss_*.csv / loss_logit_summary.txt # classification tables and odds ratios
    loss_predictions.csv                # one row per cleaned row (for Power BI)
    ols_predictions_test.csv
    signal_check.txt
    ols_plots/
D_late_model_report/                    # Phase 3b late-delivery model and saved outputs
  late_model.py                         # Phase 3b: leakage-controlled late-delivery model
  requirements.txt                      # pinned direct dependencies for Phase 3b
  handoff_phase3b.md                    # results and model-use hand-off
  outputs/
    late_model_results.csv
    late_predictions.csv
    late_odds_ratios.csv
    late_model_summary.txt
    plots/
README.md
```

## Current status

| Owner | Block | Status |
| --- | --- | --- |
| Oscar | Phase 1 — cleaning and preprocessing | Done |
| Oscar | Phase 2a — EDA | Done |
| Mithun | Phase 2b — hypothesis tests | Done |
| Mithun | Phase 4 — Power BI dashboard | Done; all 4 pages built and committed (`dataco_supply_chain_dashboard.pbix`) |
| Hisana | Phase 3a — OLS + loss classification | Done |
| Preetham | Phase 3b — late-delivery model | Done; model outputs and hand-off are included |

## Phase 1 (Oscar) — start here

**Use this file:** `A_preprocessing_eda/outputs/cleaned_supply_chain.csv`

| Item | Value |
| --- | --- |
| Raw shape | 180,519 rows × 53 columns |
| Cleaned shape | 180,519 rows × 122 columns |
| Rows deleted | 0 |
| Nulls in cleaned file | 0 |
| Scaled? | No — scale inside your own train/test step |
| Train/test split? | No — create your own with `random_state = 42` |
| Late class (`late` = 1) | 54.8291% |
| Loss class (`loss` = 1) | 18.7149% |
| Zero-profit (not loss) | 0.6520% |

Full write-up for teammates: [`A_preprocessing_eda/outputs/PHASE1_REPORT.md`](A_preprocessing_eda/outputs/PHASE1_REPORT.md)

Detailed viva log: [`A_preprocessing_eda/outputs/cleaning_log.md`](A_preprocessing_eda/outputs/cleaning_log.md)

Short hand-off: [`A_preprocessing_eda/outputs/handoff_phase1.md`](A_preprocessing_eda/outputs/handoff_phase1.md)

## Phase 2a (Oscar) — EDA

| Item | Path |
| --- | --- |
| Findings (9 plain-language points + tests for B) | [`A_preprocessing_eda/outputs/eda_findings.md`](A_preprocessing_eda/outputs/eda_findings.md) |
| Plots | [`A_preprocessing_eda/outputs/eda_plots/`](A_preprocessing_eda/outputs/eda_plots/) |
| Summary stats CSV | [`A_preprocessing_eda/outputs/eda_summary_stats.csv`](A_preprocessing_eda/outputs/eda_summary_stats.csv) |
| Hand-off | [`A_preprocessing_eda/outputs/handoff_phase2a.md`](A_preprocessing_eda/outputs/handoff_phase2a.md) |
| Script | [`A_preprocessing_eda/phase2a_eda.py`](A_preprocessing_eda/phase2a_eda.py) |

```bash
python A_preprocessing_eda/phase2a_eda.py
```

### Rules every teammate must follow

- Do not edit the cleaned CSV. Request changes through Oscar.
- Do not use leakage columns: `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)`, `Order Profit Per Order`, `Order Item Profit Ratio`.
- `Benefit per order` is the OLS target only — never a feature for loss or OLS.
- `Benefit_signed_log` is a transform of `Benefit per order` (correlation ≈ 0.94 with `loss`): use it only as a response, never as a feature.
- `late`, `loss`, and `Late_delivery_risk` are targets, not features.
- Scale numeric features after the split, fitting on the training fold only.
- Drop one dummy level per one-hot group before a model with an intercept.
- Keep `Order Status` out of models until the group decides. It is post-order information, and CANCELED and SUSPECTED_FRAUD orders are 100% not-late, so it leaks the late target.
- `Days for shipment (scheduled)` is one-to-one with `Shipping Mode` (0 = Same Day, 1 = First, 2 = Second, 4 = Standard): use one, not both.
- `Sales`, `Order Item Total` and `Order Item Discount` are algebraically tied to price, quantity and discount rate (Sales vs Order Item Total r ≈ 0.99): do not put them in the same linear model.

### Re-run Phase 1 locally

```bash
python -m pip install -r A_preprocessing_eda/requirements.txt
python A_preprocessing_eda/phase1_cleaning.py
python A_preprocessing_eda/phase1_validate.py
```

Or open and run `A_preprocessing_eda/phase1_cleaning.ipynb` (from the repo root or that folder).

Raw path default: `dataset/DataCoSupplyChainDataset.csv` (`encoding="latin-1"`).

## Phase 2b (Mithun) — hypothesis tests

```bash
python -m pip install -r B_hypothesis_powerbi/requirements.txt
python B_hypothesis_powerbi/phase2b_tests.py
```

Results: [`B_hypothesis_powerbi/outputs/PHASE2B_REPORT.md`](B_hypothesis_powerbi/outputs/PHASE2B_REPORT.md) · [`tests_results.csv`](B_hypothesis_powerbi/outputs/tests_results.csv) · [`handoff_phase2b.md`](B_hypothesis_powerbi/outputs/handoff_phase2b.md)

## Phase 3a (Hisana) — OLS and loss classification

```bash
python -m pip install -r C_loss_ols/requirements.txt
python C_loss_ols/c_ols_loss.py
python C_loss_ols/c_signal_check.py
```

**Main finding:** order-time variables do not explain or forecast loss-making orders. This is a result about the data, not a modelling failure.

| Result (test set, 36,104 orders) | Value |
| --- | --- |
| OLS on `Benefit per order`, R² (train / test) | 0.0165 / 0.0200 |
| OLS test RMSE vs mean-only baseline | $102.79 vs $103.84 |
| OLS assumptions | Fail on the raw scale (residual skew -5.63, Breusch-Pagan and Jarque-Bera p ≈ 0); HC3 robust SEs and a signed-log refit reported |
| "Always predict profit" accuracy | 0.8128 (recall 0 — accuracy alone is misleading) |
| Loss models (balanced, oversampling, SMOTE) | precision ≈ 0.19 (the base rate), recall ≈ 0.49–0.52, ROC-AUC ≈ 0.50 |
| Gradient-boosting probe, all order-time columns incl. product | loss AUC 0.505, profit R² 0.011 |
| Same product, price, quantity 1, discount ≤ 5% (1,342 lines) | 18.9% still end in loss |

Interpretation (not tested, because the dataset has no cost columns): the missing information is likely backend operational cost (product, fulfilment, returns), so loss cannot be forecast from the order record alone. Do not describe any variable as a "driver" of loss, and do not quote the F1 of 0.315 (it comes from flagging every order as a loss).

Full write-up: [`C_loss_ols/outputs/PHASE3A_REPORT.md`](C_loss_ols/outputs/PHASE3A_REPORT.md) · Hand-off: [`C_loss_ols/outputs/handoff_phase3a.md`](C_loss_ols/outputs/handoff_phase3a.md)

## Phase 3b (Preetham) — late-delivery model

Run these commands from the **repository root**. The script reads
`A_preprocessing_eda/outputs/cleaned_supply_chain.csv` and overwrites only
files under `D_late_model_report/outputs/`.

```bash
python -m pip install -r D_late_model_report/requirements.txt
python D_late_model_report/late_model.py
```

Saved outputs: [`late_model_results.csv`](D_late_model_report/outputs/late_model_results.csv),
[`late_predictions.csv`](D_late_model_report/outputs/late_predictions.csv),
[`late_odds_ratios.csv`](D_late_model_report/outputs/late_odds_ratios.csv),
[`late_model_summary.txt`](D_late_model_report/outputs/late_model_summary.txt), and the
confusion-matrix and ROC plots. The implementation and saved outputs are complete; a separate
Phase 3b narrative report and presentation slides are not included in this repository.

The held-out test results are 69.68% accuracy and ROC-AUC 0.7409. These are predictive
associations, not causal effects; shipping mode is the strongest feature after removing
delivery-time leakage.

## Reproducible environment

Each analytical phase has its own pinned `requirements.txt`. The requirements were checked on
Windows with Python 3.13.2. Phase 3a requires `imbalanced-learn` for its oversampling and SMOTE
comparisons; version 0.14.0 is pinned because it supports scikit-learn 1.7. Phase 1 and Phase 2
requirements retain their existing NumPy 2.3.5 pin; the audit environment used NumPy 2.3.3 for
read-only validation, so a full regeneration should use the pinned phase requirements rather than
mixing packages from an existing environment.

## Contributors

Group 4 · Foundations of Data Science (23CSE351).

## Roles

| Member | Owns |
| --- | --- |
| Oscar | Preprocessing, cleaning log, EDA |
| Mithun | Hypothesis tests, Power BI dashboard |
| Hisana | OLS on profit, loss classification |
| Preetham | Late-delivery model, report, slides |

## Note on large files

`dataset/DataCoSupplyChainDataset.csv` (~91 MB) and `cleaned_supply_chain.csv` (~94 MB) are under GitHub’s 100 MB hard limit. GitHub may warn above 50 MB. If clone/push becomes awkward, switch those two files to Git LFS.

Git LFS is installed locally but these CSVs are currently ordinary tracked Git blobs, not LFS
objects. Migrating existing history would rewrite shared history, so this repository intentionally
leaves them unchanged. Agree on a coordinated migration before making that change.

## License / data

Course project using the public DataCo Smart Supply Chain dataset. For educational use in 23CSE351 only.
