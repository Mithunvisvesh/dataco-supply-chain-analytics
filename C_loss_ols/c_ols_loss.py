"""
Phase 3a (Block C): OLS regression on 'Benefit per order' + loss classification.

Run from anywhere:  python C_loss_ols/c_ols_loss.py
Reads   : A_preprocessing_eda/outputs/cleaned_supply_chain.csv  (never modified)
Writes  : C_loss_ols/outputs/*

Design rules (Source of Truth, Sections 3 and 5):
  * 80/20 stratified split on `loss`, random_state = 42, SAME rows for OLS and classifiers.
  * Scaling is fitted on the training set only. Resampling touches training data only.
  * No profit-leaking columns, no post-order columns, no targets used as features.
"""
import warnings
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, accuracy_score, mean_squared_error, r2_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "A_preprocessing_eda" / "outputs" / "cleaned_supply_chain.csv"
OUT = ROOT / "C_loss_ols" / "outputs"
PLOTS = OUT / "ols_plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Load + feature selection
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA, low_memory=False)
assert df.shape == (180519, 122), df.shape
Y_PROFIT = "Benefit per order"

NUMERIC = ["Order Item Product Price", "Order Item Quantity",
           "Order Item Discount Rate", "order_month", "order_weekday"]
# categorical group -> one-hot prefix in the cleaned file. The reference level
# (dropped) is the most frequent level, so coefficients read as "vs the usual case".
GROUPS = {
    "Shipping Mode": "Shipping Mode_",
    "Market": "Market_",
    "Customer Segment": "Customer Segment_",
    "Type": "Type_",
    "Category": "Category Name_grouped_",
}

dummy_cols, reference = [], {}
for g, pre in GROUPS.items():
    cols = [c for c in df.columns if c.startswith(pre)]
    ref = df[cols].sum().idxmax()
    reference[g] = ref.replace(pre, "")
    dummy_cols += [c for c in cols if c != ref]

FEATURES = NUMERIC + dummy_cols
EXCLUDED = {
    "Benefit per order, Benefit_signed_log, loss": "target itself / transform of the target",
    "Order Profit Per Order, Order Item Profit Ratio": "already dropped by A (define profit)",
    "Late_delivery_risk, late": "known only after delivery; not an order-time feature",
    "Order Status": "post-order state (A flagged it as unresolved)",
    "Sales, Order Item Total, Order Item Discount": "Sales = price x quantity, discount = rate x sales, "
                                                    "Order Item Total = Sales - discount; r(Sales, Total)=0.99",
    "Days for shipment (scheduled)": "one-to-one with Shipping Mode (perfect collinearity)",
    "Order Region, Department Name": "nested in Market / Category (VIF and interpretability)",
    "Text columns (Product Name, cities, zipcode, ...)": "not encoded, high cardinality",
}

X_all = df[FEATURES].astype(float)
y_loss = df["loss"].values
y_profit = df[Y_PROFIT].values
y_slog = df["Benefit_signed_log"].values  # allowed ONLY as a response variable, never a regressor

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                  random_state=SEED, stratify=y_loss)
Xtr, Xte = X_all.iloc[idx_tr], X_all.iloc[idx_te]

# ----------------------------------------------------------------------------
# 2. OLS on Benefit per order
# ----------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def inv_slog(z):
    """Inverse of sign(x) * log1p(|x|)."""
    return np.sign(z) * np.expm1(np.abs(z))


Xtr_c, Xte_c = sm.add_constant(Xtr, has_constant="add"), sm.add_constant(Xte, has_constant="add")

# VIF on the training design matrix
vif = pd.DataFrame({"feature": Xtr.columns,
                    "VIF": [variance_inflation_factor(Xtr_c.values, i + 1) for i in range(Xtr.shape[1])]})
vif = vif.sort_values("VIF", ascending=False)
vif.to_csv(OUT / "ols_vif.csv", index=False)


def diagnostics(res, name):
    resid, fitted = res.resid, res.fittedvalues
    bp = het_breuschpagan(resid, res.model.exog)
    jb = stats.jarque_bera(resid)
    return {"model": name, "resid_skew": float(stats.skew(resid)),
            "resid_excess_kurtosis": float(stats.kurtosis(resid)),
            "jarque_bera_p": float(jb.pvalue), "breusch_pagan_p": float(bp[1]),
            "durbin_watson": float(durbin_watson(resid)), "max_VIF": float(vif.VIF.max())}


def assumption_plots(res, name, fname):
    resid, fitted = res.resid, res.fittedvalues
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    s = np.random.RandomState(SEED).choice(len(resid), 20000, replace=False)
    ax[0].scatter(fitted.iloc[s], resid.iloc[s], s=4, alpha=.3)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set(title=f"Residuals vs fitted ({name})", xlabel="Fitted", ylabel="Residual")
    sm.qqplot(resid, line="s", ax=ax[1], markersize=2)
    ax[1].set_title(f"Normal Q-Q ({name})")
    ax[2].hist(resid, bins=120, color="steelblue")
    ax[2].set(title=f"Residual histogram ({name})", xlabel="Residual")
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=130)
    plt.close()


# Model 1: raw target
ols_raw = sm.OLS(y_profit[idx_tr], Xtr_c).fit()
# Model 1b: same fit, heteroskedasticity-robust (HC3) standard errors
ols_raw_hc3 = sm.OLS(y_profit[idx_tr], Xtr_c).fit(cov_type="HC3")
# Model 2: signed-log response (log is undefined for losses, so sign*log1p|x|)
ols_slog = sm.OLS(y_slog[idx_tr], Xtr_c).fit()

pred_raw = ols_raw.predict(Xte_c)
pred_slog_z = ols_slog.predict(Xte_c)
pred_slog = inv_slog(pred_slog_z)

rows = [
    {"model": "OLS raw (classical SE)", "response_scale": "dollars",
     "R2_train": ols_raw.rsquared, "adj_R2_train": ols_raw.rsquared_adj,
     "R2_test": r2_score(y_profit[idx_te], pred_raw), "RMSE_test_dollars": rmse(y_profit[idx_te], pred_raw)},
    {"model": "OLS raw (HC3 robust SE)", "response_scale": "dollars",
     "R2_train": ols_raw_hc3.rsquared, "adj_R2_train": ols_raw_hc3.rsquared_adj,
     "R2_test": r2_score(y_profit[idx_te], pred_raw), "RMSE_test_dollars": rmse(y_profit[idx_te], pred_raw)},
    {"model": "OLS signed-log response", "response_scale": "signed log (RMSE back-transformed to dollars)",
     "R2_train": ols_slog.rsquared, "adj_R2_train": ols_slog.rsquared_adj,
     "R2_test": r2_score(y_slog[idx_te], pred_slog_z), "RMSE_test_dollars": rmse(y_profit[idx_te], pred_slog)},
    {"model": "Mean-only baseline", "response_scale": "dollars",
     "R2_train": 0.0, "adj_R2_train": 0.0,
     "R2_test": r2_score(y_profit[idx_te], np.full(len(idx_te), y_profit[idx_tr].mean())),
     "RMSE_test_dollars": rmse(y_profit[idx_te], np.full(len(idx_te), y_profit[idx_tr].mean()))},
]
ols_cmp = pd.DataFrame(rows)
ols_cmp.to_csv(OUT / "ols_model_comparison.csv", index=False)
diag = pd.DataFrame([diagnostics(ols_raw, "raw"), diagnostics(ols_slog, "signed_log")])
diag.to_csv(OUT / "ols_assumption_tests.csv", index=False)

assumption_plots(ols_raw, "raw", "ols_raw_assumptions.png")
assumption_plots(ols_slog, "signed log", "ols_signedlog_assumptions.png")

# coefficient tables with 95% CI
def coef_table(res):
    ci = res.conf_int(0.05)
    t = pd.DataFrame({"coef": res.params, "ci_low": ci[0], "ci_high": ci[1],
                      "std_err": res.bse, "p_value": res.pvalues})
    return t


coef_table(ols_raw).to_csv(OUT / "ols_coefficients_raw.csv")
coef_table(ols_raw_hc3).to_csv(OUT / "ols_coefficients_raw_HC3.csv")
coef_table(ols_slog).to_csv(OUT / "ols_coefficients_signedlog.csv")

# coefficient plot (raw, robust CI) - ignoring the constant
ct = coef_table(ols_raw_hc3).drop("const").sort_values("coef")
fig, ax = plt.subplots(figsize=(8, 9))
ax.errorbar(ct.coef, range(len(ct)), xerr=[ct.coef - ct.ci_low, ct.ci_high - ct.coef], fmt="o", ms=4, capsize=2)
ax.axvline(0, color="red", lw=1)
ax.set_yticks(range(len(ct)))
ax.set_yticklabels(ct.index, fontsize=7)
ax.set_title("OLS coefficients on Benefit per order (95% CI, HC3 robust)")
ax.set_xlabel("Change in Benefit per order ($) per unit / vs reference level")
plt.tight_layout()
plt.savefig(PLOTS / "ols_coefficients.png", dpi=130)
plt.close()

# actual vs predicted (test)
fig, ax = plt.subplots(figsize=(6, 6))
s = np.random.RandomState(SEED).choice(len(idx_te), 8000, replace=False)
ax.scatter(y_profit[idx_te][s], pred_raw.values[s], s=4, alpha=.3)
lim = [y_profit.min(), y_profit.max()]
ax.plot(lim, lim, color="red", lw=1)
ax.set(title="OLS raw: actual vs predicted (test sample)", xlabel="Actual Benefit per order",
       ylabel="Predicted")
plt.tight_layout()
plt.savefig(PLOTS / "ols_actual_vs_predicted.png", dpi=130)
plt.close()

pd.DataFrame({"row_id": idx_te, "actual": y_profit[idx_te], "pred_ols_raw": pred_raw.values,
              "pred_ols_signedlog": pred_slog}).to_csv(OUT / "ols_predictions_test.csv", index=False)

with open(OUT / "ols_summary.txt", "w", encoding="utf-8") as f:
    f.write("OLS on Benefit per order - Phase 3a (C)\n")
    f.write(f"Train n = {len(idx_tr)}, test n = {len(idx_te)}, split = 80/20 stratified on loss, random_state = {SEED}\n")
    f.write("Reference levels (dropped dummies): " + str(reference) + "\n\n")
    f.write("Features used (" + str(len(FEATURES)) + "):\n  " + "\n  ".join(FEATURES) + "\n\n")
    f.write("Excluded and why:\n")
    for k, v in EXCLUDED.items():
        f.write(f"  - {k}: {v}\n")
    f.write("\n" + "=" * 78 + "\nMODEL 1: raw Benefit per order (classical SE)\n" + "=" * 78 + "\n")
    f.write(str(ols_raw.summary()) + "\n\n")
    f.write("95% CI table (classical):\n" + coef_table(ols_raw).round(5).to_string() + "\n\n")
    f.write("=" * 78 + "\nMODEL 1b: raw, HC3 robust standard errors\n" + "=" * 78 + "\n")
    f.write(coef_table(ols_raw_hc3).round(5).to_string() + "\n\n")
    f.write("=" * 78 + "\nMODEL 2: signed-log response\n" + "=" * 78 + "\n")
    f.write(str(ols_slog.summary()) + "\n\n")
    f.write("=" * 78 + "\nHold-out comparison\n" + "=" * 78 + "\n")
    f.write(ols_cmp.round(5).to_string(index=False) + "\n\n")
    f.write("Assumption tests:\n" + diag.round(5).to_string(index=False) + "\n\n")
    f.write("VIF (top 10):\n" + vif.head(10).round(3).to_string(index=False) + "\n")

# ----------------------------------------------------------------------------
# 3. Loss classification
# ----------------------------------------------------------------------------
ytr, yte = y_loss[idx_tr], y_loss[idx_te]
scaler = StandardScaler().fit(Xtr[NUMERIC])  # train only
def prep(X):
    Z = X.copy()
    Z[NUMERIC] = scaler.transform(X[NUMERIC])
    return Z.values
Ztr, Zte = prep(Xtr), prep(Xte)


def metrics(name, y, p, thr, split="test"):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"model": name, "threshold": round(float(thr), 4), "split": split,
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "F1": f1_score(y, pred),
            "ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def fit_lr(X, y, **kw):
    return LogisticRegression(max_iter=1000, random_state=SEED, **kw).fit(X, y)


models = {}
models["LR baseline (no imbalance handling)"] = (fit_lr(Ztr, ytr), None)
models["LR class_weight=balanced"] = (fit_lr(Ztr, ytr, class_weight="balanced"), None)
Xo, yo = RandomOverSampler(random_state=SEED).fit_resample(Ztr, ytr)
models["LR random oversampling"] = (fit_lr(Xo, yo), None)
Xs, ys = SMOTE(random_state=SEED).fit_resample(Ztr, ytr)
models["LR SMOTE"] = (fit_lr(Xs, ys), None)

results, probs = [], {}
base_rate = ytr.mean()

# reference: always predict "profit"
always0 = np.zeros_like(yte)
results.append({"model": "Always predict profit (dummy)", "threshold": np.nan, "split": "test",
                "accuracy": accuracy_score(yte, always0), "precision": 0.0, "recall": 0.0, "F1": 0.0,
                "ROC_AUC": 0.5, "PR_AUC": float(yte.mean()),
                "TN": int((yte == 0).sum()), "FP": 0, "FN": int((yte == 1).sum()), "TP": 0})

for name, (m, _) in models.items():
    p = m.predict_proba(Zte)[:, 1]
    probs[name] = p
    # default threshold 0.5 for the unweighted baseline; for rebalanced models 0.5 is on the balanced scale
    results.append(metrics(name, yte, p, 0.5))

# ---- threshold choice: out-of-fold probabilities on TRAIN only (test never used to tune) ----
chosen = "LR class_weight=balanced"
oof = cross_val_predict(LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
                        Ztr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(ytr, oof)
f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
thr_f1 = float(thr[np.argmax(f1s)])
# Alternative operating point that is not degenerate: highest-precision threshold that still catches >= 50% of losses
ok = np.where(rec[:-1] >= 0.5)[0]
thr_r50 = float(thr[ok[-1]]) if len(ok) else thr_f1
p_chosen = probs[chosen]
results.append(metrics(chosen + " @ train-CV F1-max threshold", yte, p_chosen, thr_f1))
results.append(metrics(chosen + " @ train-CV recall>=0.5 threshold", yte, p_chosen, thr_r50))

cmp_df = pd.DataFrame(results)
cmp_df.round(5).to_csv(OUT / "loss_model_comparison.csv", index=False)

# ---- PR curve (test) ----
fig, ax = plt.subplots(figsize=(7, 5.5))
for name, p in probs.items():
    pr, rc, _ = precision_recall_curve(yte, p)
    ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(yte, p):.3f})")
ax.axhline(yte.mean(), color="grey", ls="--", label=f"No-skill = loss rate {yte.mean():.3f}")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve, loss class (test set)", ylim=(0, 1))
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(PLOTS / "loss_pr_curve.png", dpi=130)
plt.close()

# ---- confusion matrices ----
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, name in zip(ax, models):
    thr_use = 0.5
    cm = confusion_matrix(yte, (probs[name] >= thr_use).astype(int))
    a.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    a.set(title=name.replace("LR ", ""), xticks=[0, 1], yticks=[0, 1],
          xticklabels=["pred profit", "pred loss"], yticklabels=["actual profit", "actual loss"])
    a.title.set_fontsize(8)
plt.tight_layout()
plt.savefig(PLOTS / "loss_confusion_matrices.png", dpi=130)
plt.close()

# ---- odds ratios: statsmodels Logit on the training data (scaled numerics -> OR per 1 SD) ----
logit = sm.Logit(ytr, sm.add_constant(pd.DataFrame(Ztr, columns=FEATURES))).fit(disp=0)
ci = np.exp(logit.conf_int())
orr = pd.DataFrame({"odds_ratio": np.exp(logit.params), "ci_low": ci[0], "ci_high": ci[1],
                    "p_value": logit.pvalues}).drop("const")
orr["abs_log_or"] = np.abs(np.log(orr.odds_ratio))
orr = orr.sort_values("abs_log_or", ascending=False).drop(columns="abs_log_or")
orr.round(5).to_csv(OUT / "loss_odds_ratios.csv")
with open(OUT / "loss_logit_summary.txt", "w", encoding="utf-8") as f:
    f.write(str(logit.summary()))

# ---- predictions file: all rows, chosen model ----
final = models[chosen][0]
p_all = final.predict_proba(prep(X_all))[:, 1]
split = np.full(len(df), "train", dtype=object)
split[idx_te] = "test"
pd.DataFrame({"row_id": np.arange(len(df)), "split": split, "loss_actual": y_loss,
              "loss_prob": p_all.round(5),
              "loss_pred_f1max": (p_all >= thr_f1).astype(int),
              "loss_pred_recall50": (p_all >= thr_r50).astype(int)}).to_csv(OUT / "loss_predictions.csv", index=False)

# ----------------------------------------------------------------------------
# 4. Console digest (copy numbers from files for the hand-off)
# ----------------------------------------------------------------------------
pd.set_option("display.width", 220, "display.max_columns", 30)
print("Reference levels:", reference)
print("\nOLS comparison:\n", ols_cmp.round(4).to_string(index=False))
print("\nOLS assumption tests:\n", diag.round(4).to_string(index=False))
print("\nTop VIF:\n", vif.head(5).round(2).to_string(index=False))
print("\nLoss comparison:\n", cmp_df.round(4).drop(columns=["split"]).to_string(index=False))
print(f"\nTrain loss rate {ytr.mean():.4f}, test loss rate {yte.mean():.4f}; thr_f1={thr_f1:.4f}, thr_r50={thr_r50:.4f}")
print("\nTop odds ratios:\n", orr.head(8).round(4).to_string())
print("\nSignificant (p<.05) coefficients in raw OLS (HC3):",
      int((ols_raw_hc3.pvalues.drop('const') < .05).sum()), "of", len(FEATURES))