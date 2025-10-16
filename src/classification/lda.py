from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional
from sklearn.metrics import (
    precision_recall_curve, average_precision_score
)

CSV_PATH = "../../assets/huggingface_with_fairness.csv"
FEATURES = ["performance_score", "co2_eq_emissions", "likes", "downloads", "size"]
TARGET = "is_fair"
FEATURE_COLS = ["performance_score", "co2_eq_emissions", "likes", "downloads", "size"]

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=FEATURES + [TARGET]).copy()

for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=FEATURES)

y = df[TARGET].astype(bool).values
X = df[FEATURES].copy()


class Log1pSome(BaseEstimator, TransformerMixin):
    def __init__(self, cols_log):
        self.cols_log = cols_log
        self.cols_passthrough = None

    def fit(self, X, y=None):
        self.cols_passthrough = [c for c in X.columns if c not in self.cols_log]
        return self

    def transform(self, X):
        Z = X.copy()
        for c in self.cols_log:
            Z[c] = np.log1p(np.clip(Z[c].values, a_min=0, a_max=None))
        return Z


cols_log = ["co2_eq_emissions", "likes", "downloads", "size"]

pre = Pipeline([
    ("log", Log1pSome(cols_log)),
    ("scaler", StandardScaler())
])

Xp = pre.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.25, random_state=42, stratify=y)


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay

X_df = df[FEATURES].copy()
y_sr = df[TARGET].astype(bool).copy()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_df, y_sr, test_size=0.25, random_state=42, stratify=y_sr
)

pipe_lda = Pipeline([
    ("pre", pre),
    ("lda", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", store_covariance=True))
])

pipe_lda.fit(X_tr, y_tr)


def eval_pipe(name, pipe, Xtr, ytr, Xte, yte):
    print(f"\n=== {name} ===")
    yhat_tr = pipe.predict(Xtr)
    yhat_te = pipe.predict(Xte)
    print("[Train] Confusion matrix:\n", confusion_matrix(ytr, yhat_tr))
    print(classification_report(ytr, yhat_tr, digits=3))
    print("[Test]  Confusion matrix:\n", confusion_matrix(yte, yhat_te))
    print(classification_report(yte, yhat_te, digits=3))
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(Xte)[:, 1]
        print(f"[Test] ROC-AUC: {roc_auc_score(yte, proba):.3f}")


eval_pipe("LDA (eigen+shrinkage)", pipe_lda, X_tr, y_tr, X_te, y_te)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lda_cv = cross_val_score(pipe_lda, X_df, y_sr, cv=cv, scoring="accuracy")
print(f"LDA CV accuracy: {lda_cv.mean():.3f} ± {lda_cv.std():.3f}")


def save_lda_results_pipeline(
    pipe,
    Xtr: pd.DataFrame, ytr: pd.Series,
    Xte: pd.DataFrame, yte: pd.Series,
    *,
    outdir: str = "../../assets/lda_results_eigen",
    model_name: str = "LDA-eigen",
    FEATURES: Optional[list] = None,
    X_all: Optional[pd.DataFrame] = None,
    y_all: Optional[pd.Series] = None,
    make_pr_curve: bool = True,
    make_thresh_curve: bool = True,
    make_conf_heatmap: bool = True,
    make_loadings_plot: bool = True
):
    os.makedirs(outdir, exist_ok=True)
    outdir = Path(outdir)

    preproc = pipe.named_steps["pre"]
    lda     = pipe.named_steps["lda"]

    if FEATURES is None:
        FEATURES = list(Xtr.columns)

    Xtr_pre = preproc.transform(Xtr)
    Xte_pre = preproc.transform(Xte)

    try:
        Z_tr = lda.transform(Xtr_pre)
        Z_te = lda.transform(Xte_pre)
        proj_cols = [f"LD{i+1}" for i in range(Z_tr.shape[1])]
    except NotImplementedError:
        Z_tr = lda.decision_function(Xtr_pre).reshape(-1, 1)
        Z_te = lda.decision_function(Xte_pre).reshape(-1, 1)
        proj_cols = ["score"]

    pd.DataFrame(Z_tr, columns=proj_cols, index=Xtr.index).assign(y=ytr.values) \
      .to_csv(outdir / f"{model_name}_proj_train.csv")
    pd.DataFrame(Z_te, columns=proj_cols, index=Xte.index).assign(y=yte.values) \
      .to_csv(outdir / f"{model_name}_proj_test.csv")

    means_ = pd.DataFrame(lda.means_, columns=FEATURES)
    means_["class"] = lda.classes_
    means_.to_csv(outdir / f"{model_name}_class_means.csv", index=False)

    pd.Series(lda.priors_, index=lda.classes_, name="prior") \
      .to_csv(outdir / f"{model_name}_class_priors.csv")

    from numpy.linalg import inv
    mu = lda.means_
    priors = lda.priors_
    Sigma = lda.covariance_
    Sigma_inv = inv(Sigma)
    Ak = (Sigma_inv @ mu.T).T
    bk = np.array([-0.5 * mu[k].T @ Sigma_inv @ mu[k] + np.log(priors[k]) for k in range(mu.shape[0])])

    Ak_df = pd.DataFrame(Ak, columns=FEATURES, index=lda.classes_)
    Ak_df["b_k"] = bk
    Ak_df.index.name = "class"
    Ak_df.to_csv(outdir / f"{model_name}_discriminant_params.csv")

    if hasattr(lda, "scalings_") and lda.scalings_ is not None:
        pd.DataFrame(lda.scalings_, index=FEATURES) \
          .to_csv(outdir / f"{model_name}_scalings.csv")

    yhat_tr = pipe.predict(Xtr)
    yhat_te = pipe.predict(Xte)

    cm_train = pd.DataFrame(
        confusion_matrix(ytr, yhat_tr),
        index=[f"true_{c}" for c in lda.classes_],
        columns=[f"pred_{c}" for c in lda.classes_]
    )
    cm_test = pd.DataFrame(
        confusion_matrix(yte, yhat_te),
        index=[f"true_{c}" for c in lda.classes_],
        columns=[f"pred_{c}" for c in lda.classes_]
    )
    cm_train.to_csv(outdir / f"{model_name}_confusion_train.csv")
    cm_test.to_csv(outdir / f"{model_name}_confusion_test.csv")

    rep_tr = classification_report(ytr, yhat_tr, output_dict=True)
    rep_te = classification_report(yte, yhat_te, output_dict=True)
    pd.DataFrame(rep_tr).to_csv(outdir / f"{model_name}_report_train.csv")
    pd.DataFrame(rep_te).to_csv(outdir / f"{model_name}_report_test.csv")

    pred_train = pd.DataFrame({"y_true": ytr.values, "y_pred": yhat_tr}, index=Xtr.index)
    pred_test  = pd.DataFrame({"y_true": yte.values, "y_pred": yhat_te}, index=Xte.index)

    if hasattr(pipe, "predict_proba"):
        proba_tr = pipe.predict_proba(Xtr)
        proba_te = pipe.predict_proba(Xte)
        class_labels = lda.classes_
        proba_tr_df = pd.DataFrame(proba_tr, index=Xtr.index, columns=[f"p_{c}" for c in class_labels])
        proba_te_df = pd.DataFrame(proba_te, index=Xte.index, columns=[f"p_{c}" for c in class_labels])
        pred_train = pd.concat([pred_train, proba_tr_df], axis=1)
        pred_test  = pd.concat([pred_test,  proba_te_df], axis=1)

    pred_train.to_csv(outdir / f"{model_name}_pred_train.csv")
    pred_test.to_csv(outdir / f"{model_name}_pred_test.csv")

    metrics = {}
    is_binary = (len(lda.classes_) == 2)
    if is_binary and hasattr(pipe, "predict_proba"):
        p_pos = pred_test.filter(like="p_").iloc[:, -1].values
        if "p_True" in pred_test.columns:
            p_pos = pred_test["p_True"].values

        try:
            metrics["roc_auc_test"] = float(roc_auc_score(yte, p_pos))
            RocCurveDisplay.from_predictions(yte, p_pos)
            plt.title(f"{model_name} ROC (test)")
            plt.tight_layout()
            plt.savefig(outdir / f"{model_name}_roc_test.png", dpi=150)
            plt.close()
        except Exception:
            pass

        if make_pr_curve:
            prec, rec, _ = precision_recall_curve(yte, p_pos)
            ap = average_precision_score(yte, p_pos)
            plt.figure(figsize=(4.8, 3.8))
            plt.plot(rec, prec)
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(f"Precision–Recall (AP={ap:.3f})")
            plt.tight_layout()
            plt.savefig(outdir / f"{model_name}_pr_test.png", dpi=150)
            plt.close()
            metrics["average_precision_test"] = float(ap)

        if make_thresh_curve:
            def sens_spec_at_threshold(y, p, thr):
                yhat = (p >= thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
                sens = tp / (tp + fn) if tp + fn > 0 else np.nan
                spec = tn / (tn + fp) if tn + fp > 0 else np.nan
                return sens, spec

            ths = np.linspace(0, 1, 201)
            sens, spec = [], []
            for t in ths:
                s1, s2 = sens_spec_at_threshold(yte.values if hasattr(yte, "values") else yte, p_pos, t)
                sens.append(s1); spec.append(s2)

            plt.figure(figsize=(5.2, 3.8))
            plt.plot(ths, sens, label="Sensitivity")
            plt.plot(ths, spec, label="Specificity")
            plt.xlabel("Threshold"); plt.ylabel("Rate")
            plt.title("Sensitivity / Specificity vs Threshold")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(outdir / f"{model_name}_sens_spec_vs_threshold.png", dpi=150)
            plt.close()

    if (X_all is not None) and (y_all is not None):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_cv = cross_val_score(pipe, X_all, y_all, cv=cv, scoring="accuracy")
        metrics["cv_acc_mean"] = float(acc_cv.mean())
        metrics["cv_acc_std"]  = float(acc_cv.std())

    with open(outdir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(6, 4))
    for cls in np.unique(yte):
        m = (yte.values if hasattr(yte, "values") else yte) == cls
        plt.hist(Z_te[m, 0], bins=30, alpha=0.6, density=True, label=f"class={cls}")
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.title(f"{model_name} — discriminant scores (test)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / f"{model_name}_scores_hist_test.png", dpi=150)
    plt.close()

    if Z_te.shape[1] >= 2:
        plt.figure(figsize=(5.5, 5))
        for cls in np.unique(yte):
            m = (yte.values if hasattr(yte, "values") else yte) == cls
            plt.scatter(Z_te[m, 0], Z_te[m, 1], s=12, alpha=0.7, label=f"class={cls}")
        plt.axhline(0, color="k", lw=0.5); plt.axvline(0, color="k", lw=0.5)
        plt.xlabel("LD1"); plt.ylabel("LD2")
        plt.title(f"{model_name} — LD space (test)")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / f"{model_name}_scatter_LD_test.png", dpi=150)
        plt.close()

    if make_conf_heatmap:
        cm = cm_test.values
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(cm)
        ax.set_xticks(range(cm_test.shape[1])); ax.set_xticklabels(cm_test.columns)
        ax.set_yticks(range(cm_test.shape[0])); ax.set_yticklabels(cm_test.index)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        ax.set_title(f"{model_name} — Confusion Matrix (test)")
        plt.tight_layout()
        plt.savefig(outdir / f"{model_name}_confusion_test_heatmap.png", dpi=150)
        plt.close()

    if make_loadings_plot:
        feat_cols = [c for c in Ak_df.columns if c != "b_k"]
        abs_mean = Ak_df[feat_cols].abs().mean(axis=0).sort_values(ascending=True)
        plt.figure(figsize=(6.0, 4.8))
        abs_mean.plot(kind="barh")
        plt.title(f"{model_name} — |loadings| on LD (avg across classes)")
        plt.tight_layout()
        plt.savefig(outdir / f"{model_name}_loadings_abs_bar.png", dpi=150)
        plt.close()

    print(f"[OK] Guardado en: {outdir.resolve()}")


save_lda_results_pipeline(pipe_lda, X_tr, y_tr, X_te, y_te,
                          outdir="../../assets/lda_results_eigen", model_name="LDA-eigen")

