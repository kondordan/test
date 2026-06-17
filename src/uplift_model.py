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
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Cap one-hot width per categorical column. High-cardinality fields (e.g. a real
# `mailing_name` with thousands of campaign codes) would otherwise blow up the
# DENSE design matrix to tens of GB and OOM-kill the process. Keeping the most
# frequent categories and grouping the long, rare tail into a single
# "infrequent" bucket bounds memory AND curbs overfitting on rare values, so it
# does not reduce predictive quality. Override via the env var if ever needed.
MAX_OHE_CATEGORIES = int(os.environ.get("UPLIFT_MAX_OHE_CATEGORIES", "50"))


def _to_float32(a):
    """Module-level (picklable) caster so the fitted preprocessor can be saved
    with joblib. Keeps the design matrix float32 to halve memory."""
    return np.asarray(a, dtype=np.float32)


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("cast", FunctionTransformer(_to_float32, feature_names_out="one-to-one")),
    ])
    # max_categories bounds the encoded width; "infrequent_if_exist" also maps
    # unseen categories (at holdout/scoring time) to the infrequent bucket
    # instead of erroring.
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              max_categories=MAX_OHE_CATEGORIES,
                              sparse_output=False, dtype=np.float32)),
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
    treatment = np.asarray(treatment)
    # Uplift metrics need BOTH arms present; otherwise sklift raises. Degrade
    # gracefully to NaN instead of crashing the whole pipeline.
    uniq = np.unique(treatment)
    if not (0 in uniq and 1 in uniq):
        logger.warning("evaluate_uplift: only one treatment arm present "
                       "(values=%s) -> uplift metrics undefined (NaN).", uniq.tolist())
        return EvalResult(qini=float("nan"), auuc=float("nan"),
                          uplift_at_30=float("nan"))
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


def bootstrap_qini_ci(y_true, uplift, treatment, n_boot: int = 200,
                      seed: int = 42, max_sample: int = 100_000):
    """Bootstrap 95% CI of the held-out Qini AUC.

    Each iteration resamples at most ``max_sample`` rows so the cost stays bounded
    on large holdouts (Qini is O(n log n) per iteration)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    uplift = np.asarray(uplift)
    treatment = np.asarray(treatment)
    n = len(y_true)
    draw = min(n, max_sample)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, draw)
        # need both treatment arms present in the resample
        if treatment[idx].min() == treatment[idx].max():
            continue
        try:
            scores.append(qini_auc_score(y_true[idx], uplift[idx], treatment[idx]))
        except Exception:
            continue
    scores = np.array(scores)
    if scores.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)), float(scores.mean())


def choose_uplift_threshold(y_true, uplift, treatment):
    """Pick the OPTIMAL predicted-uplift cutoff above which to send.

    Clients are ranked by predicted uplift (high -> low). We accumulate the
    Qini-style incremental clicks (treated responders minus control responders,
    rescaled by group sizes) and take the cutoff at the PEAK of that curve:
    beyond this point, adding lower-uplift clients reduces total incremental
    clicks. The predicted-uplift value at the peak is the threshold.

    Returns (threshold, info). Falls back to 0.0 if a treatment arm is missing.
    """
    y = np.asarray(y_true, dtype=float)
    u = np.asarray(uplift, dtype=float)
    t = np.asarray(treatment)
    uniq = np.unique(t)
    if not (0 in uniq and 1 in uniq) or len(u) == 0:
        logger.warning("choose_uplift_threshold: one treatment arm missing -> "
                       "defaulting threshold to 0.0")
        return 0.0, {"targeted_fraction": float("nan"),
                     "max_incremental_clicks": float("nan")}

    order = np.argsort(-u, kind="mergesort")  # high uplift first
    us, ts, ys = u[order], t[order], y[order]
    treat = (ts == 1).astype(float)
    ctrl = (ts == 0).astype(float)
    cum_tr = np.cumsum(ys * treat)            # cumulative treated responders
    cum_tn = np.cumsum(treat)                 # cumulative treated count
    cum_cr = np.cumsum(ys * ctrl)             # cumulative control responders
    cum_cn = np.cumsum(ctrl)                  # cumulative control count
    # Qini-style cumulative incremental responses if we target the top-k clients.
    inc = cum_tr - cum_cr * (cum_tn / np.maximum(cum_cn, 1.0))
    best = int(np.argmax(inc))
    threshold = float(us[best])
    info = {
        "targeted_fraction": (best + 1) / len(u),
        "max_incremental_clicks": float(inc[best]),
    }
    return threshold, info


def train_ensemble(model_name, base_kind, preprocessor_cols, X_train, t_train,
                   y_train, n_models: int = 3, frac: float = 0.8, seed: int = 42):
    """Train ``n_models`` of the chosen meta-learner, each on a DIFFERENT random
    subsample of the training data (with its own fitted preprocessor).

    Because the pipeline samples large files, individual draws vary; training an
    ensemble over several subsamples and averaging their uplift reduces this
    sampling variance and curbs overfitting (bagging-style).
    """
    rng = np.random.default_rng(seed)
    X_train = X_train.reset_index(drop=True)
    t_train = t_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    n = len(X_train)
    size = max(1, int(frac * n))

    members = []
    for m in range(n_models):
        idx = rng.choice(n, size=size, replace=False)
        pre = build_preprocessor(*preprocessor_cols)
        Xm = pre.fit_transform(X_train.iloc[idx])
        model = make_uplift_model(model_name, base_kind)
        model.fit(Xm, y_train.iloc[idx].to_numpy(), t_train.iloc[idx].to_numpy())
        members.append({"preprocessor": pre, "model": model})
        logger.info("  ensemble model %d/%d trained on %d rows (seed-draw %d)",
                    m + 1, n_models, size, m)
    return members


def predict_ensemble(members, X) -> np.ndarray:
    """Average uplift prediction across ensemble members (each applies its own
    fitted preprocessor)."""
    preds = []
    for mem in members:
        Xt = mem["preprocessor"].transform(X)
        preds.append(mem["model"].predict(Xt))
    return np.mean(preds, axis=0)


def save_ensemble(path, members, model_name, base_kind, numeric_cols,
                  categorical_cols, extra: dict | None = None) -> None:
    """Persist the trained ensemble (+ metadata) so it can be reloaded and run
    independently of training (see predict.py)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "members": members,
        "model_name": model_name,
        "base_kind": base_kind,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "extra": extra or {},
    }
    joblib.dump(payload, path, compress=3)
    logger.info("saved ensemble (%d models) -> %s", len(members), path)


def load_ensemble(path):
    payload = joblib.load(path)
    logger.info("loaded ensemble: %d models (%s+%s) from %s",
                len(payload["members"]), payload["model_name"],
                payload["base_kind"], path)
    return payload


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
