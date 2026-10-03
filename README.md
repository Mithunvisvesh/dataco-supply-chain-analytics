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
  requirements.txt
  outputs/
    cleaned_supply_chain.csv
    cleaning_log.md                     # viva document for preprocessing
    handoff_phase1.md                   # short group-chat hand-off
    PHASE1_REPORT.md                    # teammate-facing Phase 1 report
    outlier_plots/
B_hypothesis_powerbi/
  phase2b_tests.py                      # T1–T6 hypothesis tests
  dashboard_spec.md                     # Power BI page layout
  outputs/
    tests_results.csv
    PHASE2B_REPORT.md
    handoff_phase2b.md
    plots/
C_ols_loss/                             # (Hisana) OLS + loss classification — to add
D_late_model_report/                    # (Preetham) late model + report/slides — to add
README.md
```

## Current status

| Owner | Block | Status |
| --- | --- | --- |
| Oscar | Phase 1 — cleaning and preprocessing | Done |
| Oscar | Phase 2a — EDA | Not started |
| Mithun | Phase 2b — hypothesis tests | Done |
| Mithun | Phase 4 — Power BI dashboard | Pages 1–3 not started; page 4 waits on Hisana and Preetham |
| Hisana | Phase 3a — OLS + loss classification | Cleaned file ready |
| Preetham | Phase 3b — late-delivery model | Cleaned file ready |

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

### Rules every teammate must follow

- Do not edit the cleaned CSV. Request changes through Oscar.
- Do not use leakage columns: `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)`, `Order Profit Per Order`, `Order Item Profit Ratio`.
- `Benefit per order` is the OLS target only — never a feature for loss or OLS.
- `late`, `loss`, and `Late_delivery_risk` are targets, not features.
- Scale numeric features after the split, fitting on the training fold only.
- Drop one dummy level per one-hot group before a model with an intercept.
- Keep `Order Status` out of models until the group decides (may be post-order).

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

## License / data

Course project using the public DataCo Smart Supply Chain dataset. For educational use in 23CSE351 only.
