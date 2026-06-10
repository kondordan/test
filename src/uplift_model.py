"""Uplift modelling pipeline.

We frame the problem as a *causal uplift* task:

- treatment  = the mailing carried a discount (vs. no discount),
- outcome    = the client clicked the e-mail (click_rate is the target metric),
- uplift(x)  = P(click | discount, x) - P(click | no discount, x).

The model therefore predicts, per e-mail/client, how much a discount changes the
probability of a click -- i.e. "how people interact with discounts".

Anti-overfitting practices applied here:
- strict time-based train/test split (and an overall cutoff at 01.08.2026),
- leakage-free features (see features.py),
- regularised, shallow base learners with early stopping,
- stratified cross-validation to report metric stability (confidence interval),
- evaluation with proper *uplift* metrics (Qini AUC, AUUC, uplift@k) rather than
  plain accuracy, plus a bootstrap CI on the held-out Qini.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])


def make_base_learner(kind: str = "hgb"):
    """Regularised base classifiers (deliberately conservative to avoid overfit)."""
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            min_samples_leaf=50,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )
    if kind == "logreg":
        return LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_STATE)
    raise ValueError(kind)


def make_uplift_model(name: str, base_kind: str):
    """Factory for the three classic meta-learners from scikit-uplift."""
    from sklift.models import ClassTransformation, SoloModel, TwoModels

    if name == "SoloModel":          # S-learner
        return SoloModel(make_base_learner(base_kind))
    if name == "TwoModels":          # T-learner
        return TwoModels(
            make_base_learner(base_kind),
            make_base_learner(base_kind),
            method="vanilla",
        )
    if name == "ClassTransformation":  # Class transformation (needs ~balanced A/B)
        return ClassTransformation(make_base_learner(base_kind))
    raise ValueError(name)


@dataclass
class EvalResult:
    qini: float
    auuc: float
    uplift_at_30: float

    def as_dict(self):
        return {"qini_auc": self.qini, "auuc": self.auuc, "uplift_at_30": self.uplift_at_30}


def evaluate_uplift(y_true, uplift, treatment, k: float = 0.3) -> EvalResult:
    return EvalResult(
        qini=qini_auc_score(y_true, uplift, treatment),
        auuc=uplift_auc_score(y_true, uplift, treatment),
        uplift_at_30=uplift_at_k(y_true, uplift, treatment, strategy="overall", k=k),
    )


def cross_val_uplift(model_name, base_kind, preprocessor, X, treatment, target,
                     n_splits: int = 5):
    """Stratified CV (on the joint treatment*outcome label) reporting Qini mean/std.

    This gives an honest, overfitting-aware estimate of generalisation: we report
    mean +/- std and a normal-approx 95% CI across folds.
    """
    strat = (treatment.astype(str) + "_" + target.astype(str)).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    qinis, auucs, upk = [], [], []
    Xv = X.reset_index(drop=True)
    tv = treatment.reset_index(drop=True)
    yv = target.reset_index(drop=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xv, strat), 1):
        pre = build_preprocessor(*preprocessor)
        X_tr = pre.fit_transform(Xv.iloc[tr_idx])
        X_va = pre.transform(Xv.iloc[va_idx])

        model = make_uplift_model(model_name, base_kind)
        model.fit(X_tr, yv.iloc[tr_idx].to_numpy(), tv.iloc[tr_idx].to_numpy())
        up = model.predict(X_va)

        res = evaluate_uplift(yv.iloc[va_idx].to_numpy(), up, tv.iloc[va_idx].to_numpy())
        qinis.append(res.qini)
        auucs.append(res.auuc)
        upk.append(res.uplift_at_30)
        logger.debug("    fold %d/%d: Qini=%.4f AUUC=%.4f uplift@30%%=%.4f",
                     fold, n_splits, res.qini, res.auuc, res.uplift_at_30)

    qinis = np.array(qinis)
    return {
        "qini_mean": float(qinis.mean()),
        "qini_std": float(qinis.std(ddof=1)),
        "qini_ci95": (
            float(qinis.mean() - 1.96 * qinis.std(ddof=1) / np.sqrt(n_splits)),
            float(qinis.mean() + 1.96 * qinis.std(ddof=1) / np.sqrt(n_splits)),
        ),
        "auuc_mean": float(np.mean(auucs)),
        "uplift_at_30_mean": float(np.mean(upk)),
        "fold_qini": qinis.tolist(),
    }


def bootstrap_qini_ci(y_true, uplift, treatment, n_boot: int = 500, seed: int = 42):
    """Bootstrap 95% CI of the held-out Qini AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # need both treatment arms present in the resample
        if treatment[idx].min() == treatment[idx].max():
            continue
        try:
            scores.append(qini_auc_score(y_true[idx], uplift[idx], treatment[idx]))
        except Exception:
            continue
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)), float(scores.mean())


def response_auc(preprocessor, X_train, y_train, X_test, y_test) -> float:
    """Plain click-probability ROC-AUC -- a sanity 'accuracy' of response prediction.

    (Uplift quality is judged by Qini/AUUC, not by this, but it is a useful
    secondary diagnostic that the features carry signal about clicks.)"""
    pre = build_preprocessor(*preprocessor)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    clf = make_base_learner("hgb")
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y_test, proba))
