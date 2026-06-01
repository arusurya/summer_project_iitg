"""
src/model.py
────────────────────────────────────────────────────────────────────────────
Model inference, SHAP explanation, and prediction utilities.

These functions are called by:
  - 05_churn_model.ipynb   (training + evaluation)
  - dashboard/app.py       (real-time member lookup)

Nothing in this file trains a model from scratch — training lives in the
notebook.  This file is purely about loading saved artefacts and producing
predictions + explanations that are identical everywhere.

Usage
-----
from src.model import (
    load_model,
    load_explainer,
    load_meta,
    predict_churn_proba,
    predict_risk_bucket,
    get_shap_explanation,
    get_top_shap_drivers,
    score_members,
)
"""

import json
import warnings
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Default paths (relative to project root) ──────────────────────────────
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_PROC_DIR  = Path(__file__).resolve().parent.parent / "data" / "processed"


# ═══════════════════════════════════════════════════════════════════════════
# 0. Artefact Loaders  (called once at app startup, then cached)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR):
    """
    Load the trained XGBoost churn model from disk.

    Parameters
    ----------
    model_dir : path to the models/ folder

    Returns
    -------
    XGBClassifier
    """
    path = Path(model_dir) / "churn_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            "Run 05_churn_model.ipynb first to train and save the model."
        )
    return joblib.load(path)


def load_scaler(model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR):
    """
    Load the StandardScaler fitted on the training feature matrix.

    Used by the logistic regression baseline — not required for XGBoost
    inference but kept available for consistency checks.

    Parameters
    ----------
    model_dir : path to the models/ folder

    Returns
    -------
    StandardScaler
    """
    path = Path(model_dir) / "scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}.\n"
            "Run 05_churn_model.ipynb first."
        )
    return joblib.load(path)


def load_explainer(model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR):
    """
    Load the pre-fitted SHAP TreeExplainer.

    Computing SHAP values from scratch on every dashboard request is
    slow (~2 min).  The explainer is saved once during training so the
    dashboard can call it instantly.

    Parameters
    ----------
    model_dir : path to the models/ folder

    Returns
    -------
    shap.TreeExplainer
    """
    path = Path(model_dir) / "shap_explainer.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP explainer not found at {path}.\n"
            "Run 05_churn_model.ipynb first."
        )
    return joblib.load(path)


def load_meta(model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR) -> dict:
    """
    Load model metadata (feature columns, threshold, CV scores, etc.).

    Returns
    -------
    dict with keys:
        feature_cols   : ordered list of feature column names
        threshold      : optimal classification threshold
        xgb_cv_auc     : XGBoost cross-validation AUC
        lr_cv_auc      : Logistic regression cross-validation AUC
        churn_rate     : overall churn rate in training data
        n_members      : number of members in training data
        risk_buckets   : ordered list of risk bucket labels
    """
    path = Path(model_dir) / "model_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {path}.\n"
            "Run 05_churn_model.ipynb first."
        )
    with open(path) as f:
        return json.load(f)


def load_label_encoders(proc_dir: Union[str, Path] = _DEFAULT_PROC_DIR) -> dict:
    """
    Load the label encoders fitted on demographic columns in notebook 04.

    Returns
    -------
    dict mapping column_name → fitted LabelEncoder
    """
    path = Path(proc_dir) / "04_label_encoders.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Label encoders not found at {path}.\n"
            "Run 04_feature_engineering.ipynb first."
        )
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Core Prediction
# ═══════════════════════════════════════════════════════════════════════════

def predict_churn_proba(
    X: pd.DataFrame,
    model=None,
    feature_cols: Optional[list] = None,
    model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR,
) -> np.ndarray:
    """
    Return churn probability for every row in X.

    Parameters
    ----------
    X            : feature DataFrame — must contain all columns in feature_cols
    model        : pre-loaded XGBClassifier.  If None, loaded from disk.
    feature_cols : ordered list of feature column names.
                   If None, loaded from model_meta.json.
    model_dir    : path to models/ folder (used only if model is None)

    Returns
    -------
    np.ndarray of shape (n_members,) with probabilities in [0, 1]
    """
    if model is None:
        model = load_model(model_dir)

    if feature_cols is None:
        feature_cols = load_meta(model_dir)["feature_cols"]

    # Align columns — fill any missing ones with 0 (safe default)
    missing = set(feature_cols) - set(X.columns)
    if missing:
        warnings.warn(
            f"predict_churn_proba: {len(missing)} feature columns missing "
            f"from X — filling with 0.  Missing: {sorted(missing)[:5]}..."
        )
        for col in missing:
            X = X.copy()
            X[col] = 0

    X_ordered = X[feature_cols].copy()

    # Replace inf / -inf silently (can arise from ratio features on edge cases)
    X_ordered = X_ordered.replace([np.inf, -np.inf], 0).fillna(0)

    return model.predict_proba(X_ordered)[:, 1]


def predict_risk_bucket(proba: Union[float, np.ndarray]) -> Union[str, list]:
    """
    Map a churn probability (or array) to a human-readable risk bucket.

    Thresholds match the bucket definitions in notebooks 05–07:
        >= 0.75  →  Critical (>75%)
        >= 0.50  →  High (50-75%)
        >= 0.25  →  Medium (25-50%)
        <  0.25  →  Low (<25%)

    Parameters
    ----------
    proba : float or np.ndarray

    Returns
    -------
    str  (if scalar input) or list of str
    """
    def _bucket(p: float) -> str:
        if p >= 0.75:
            return "Critical (>75%)"
        elif p >= 0.50:
            return "High (50-75%)"
        elif p >= 0.25:
            return "Medium (25-50%)"
        else:
            return "Low (<25%)"

    if np.isscalar(proba):
        return _bucket(float(proba))
    return [_bucket(float(p)) for p in proba]


def predict_binary(
    proba: Union[float, np.ndarray],
    threshold: Optional[float] = None,
    model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR,
) -> Union[int, np.ndarray]:
    """
    Apply the business-optimal threshold to convert probabilities to 0/1 labels.

    The threshold is loaded from model_meta.json (set in notebook 05 to
    achieve recall >= 0.70 at maximum precision).

    Parameters
    ----------
    proba     : float or np.ndarray of churn probabilities
    threshold : override value.  If None, use value from model_meta.json.
    model_dir : path to models/ folder

    Returns
    -------
    int (if scalar) or np.ndarray of int
    """
    if threshold is None:
        threshold = load_meta(model_dir)["threshold"]

    if np.isscalar(proba):
        return int(float(proba) >= threshold)
    return (np.array(proba) >= threshold).astype(int)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SHAP Explanations
# ═══════════════════════════════════════════════════════════════════════════

def get_shap_explanation(
    X_row: pd.DataFrame,
    explainer=None,
    feature_cols: Optional[list] = None,
    model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR,
) -> dict:
    """
    Compute SHAP values for a single member row and return structured output.

    This powers the 'Why is this member at risk?' section of the
    Streamlit dashboard.

    Parameters
    ----------
    X_row        : single-row DataFrame with feature columns
    explainer    : pre-loaded shap.TreeExplainer.  If None, loaded from disk.
    feature_cols : ordered feature column list.  If None, loaded from meta.
    model_dir    : path to models/ folder

    Returns
    -------
    dict with keys:
        base_value    : float  — model's average prediction (log-odds)
        shap_values   : np.ndarray  — SHAP value per feature
        feature_names : list of str
        feature_values: list of float  — actual input values for this member
        prediction    : float  — final predicted probability
    """
    import shap  # lazy import — not required for non-SHAP usage

    if explainer is None:
        explainer = load_explainer(model_dir)

    if feature_cols is None:
        feature_cols = load_meta(model_dir)["feature_cols"]

    # Align and clean
    missing = set(feature_cols) - set(X_row.columns)
    for col in missing:
        X_row = X_row.copy()
        X_row[col] = 0

    X_clean = X_row[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    sv = explainer.shap_values(X_clean)
    if isinstance(sv, list):
        sv = sv[1]   # binary classification: class-1 SHAP values

    # Convert log-odds base value to probability
    base_log_odds = explainer.expected_value
    if isinstance(base_log_odds, (list, np.ndarray)):
        base_log_odds = base_log_odds[1]
    base_prob = float(1 / (1 + np.exp(-float(base_log_odds))))

    # Final prediction probability for this member
    final_prob = float(1 / (1 + np.exp(
        -(float(base_log_odds) + float(sv[0].sum()))
    )))

    return {
        "base_value"    : base_prob,
        "shap_values"   : sv[0],
        "feature_names" : feature_cols,
        "feature_values": X_clean.iloc[0].tolist(),
        "prediction"    : final_prob,
    }


def get_top_shap_drivers(
    shap_explanation: dict,
    n: int = 10,
    direction: str = "both",
) -> pd.DataFrame:
    """
    Return the top-N features driving a single member's churn prediction.

    Parameters
    ----------
    shap_explanation : dict returned by get_shap_explanation()
    n                : number of features to return
    direction        : 'positive'  — only features that increase churn risk
                       'negative'  — only features that decrease churn risk
                       'both'      — top N by absolute SHAP value (default)

    Returns
    -------
    pd.DataFrame with columns:
        feature       : feature name
        shap_value    : SHAP contribution (positive = increases churn risk)
        feature_value : actual value for this member
        direction     : 'increases_risk' or 'decreases_risk'
        abs_impact    : absolute SHAP value (for sorting)
    """
    sv   = np.array(shap_explanation["shap_values"])
    fns  = shap_explanation["feature_names"]
    fvs  = shap_explanation["feature_values"]

    df = pd.DataFrame({
        "feature"      : fns,
        "shap_value"   : sv,
        "feature_value": fvs,
    })
    df["abs_impact"] = df["shap_value"].abs()
    df["direction"]  = df["shap_value"].apply(
        lambda v: "increases_risk" if v > 0 else "decreases_risk"
    )

    if direction == "positive":
        df = df[df["shap_value"] > 0]
    elif direction == "negative":
        df = df[df["shap_value"] < 0]

    return (
        df.sort_values("abs_impact", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def build_shap_waterfall_data(
    shap_explanation: dict,
    n: int = 12,
) -> pd.DataFrame:
    """
    Prepare data for a horizontal waterfall bar chart in Streamlit.

    Each row represents one feature's contribution to the prediction.
    Rows are ordered from most negative (blue / decreases risk) at the
    bottom to most positive (red / increases risk) at the top.

    Parameters
    ----------
    shap_explanation : dict from get_shap_explanation()
    n                : max features to show

    Returns
    -------
    pd.DataFrame with columns:
        feature, shap_value, feature_value, color, label
    """
    drivers = get_top_shap_drivers(shap_explanation, n=n, direction="both")

    drivers["color"] = drivers["shap_value"].apply(
        lambda v: "#C0392B" if v > 0 else "#2471A3"
    )
    # Human-readable label: "feature_name = 0.42"
    drivers["label"] = drivers.apply(
        lambda r: f'{r["feature"].replace("_", " ")} = {r["feature_value"]:.3g}',
        axis=1,
    )
    return drivers.sort_values("shap_value")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Batch Scoring (used by dashboard overview pages)
# ═══════════════════════════════════════════════════════════════════════════

def score_members(
    features_df: pd.DataFrame,
    model=None,
    explainer=None,
    feature_cols: Optional[list] = None,
    model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR,
    include_shap: bool = False,
) -> pd.DataFrame:
    """
    Score all members in a feature DataFrame and return enriched predictions.

    Parameters
    ----------
    features_df  : member feature matrix (one row per member, must include KEY)
    model        : pre-loaded XGBClassifier.  If None, loaded from disk.
    explainer    : pre-loaded SHAP explainer.  If None, loaded from disk.
    feature_cols : ordered feature column list.  If None, loaded from meta.
    model_dir    : path to models/ folder
    include_shap : if True, compute SHAP values for every member and attach
                   as columns shap_{feature_name}.  Slow for large datasets.

    Returns
    -------
    pd.DataFrame with original KEY column plus:
        churn_prob      : float — churn probability in [0, 1]
        churn_pct       : float — churn_prob as percentage (e.g. 73.4)
        predicted_label : int   — 0 or 1 using optimal threshold
        risk_bucket     : str   — human-readable risk tier
        priority_score  : float — churn_prob × behavioral_clv (if available)
    """
    if model is None:
        model = load_model(model_dir)

    if feature_cols is None:
        feature_cols = load_meta(model_dir)["feature_cols"]

    key_col = "loyalty_number"
    key     = features_df[key_col].values if key_col in features_df.columns else None

    proba   = predict_churn_proba(features_df, model, feature_cols, model_dir)
    labels  = predict_binary(proba, model_dir=model_dir)
    buckets = predict_risk_bucket(proba)

    out = pd.DataFrame({
        key_col         : key if key is not None else range(len(proba)),
        "churn_prob"    : proba,
        "churn_pct"     : np.round(proba * 100, 1),
        "predicted_label": labels,
        "risk_bucket"   : buckets,
    })

    # Priority score: churn_prob × behavioral_clv (if bclv is available)
    if "behavioral_clv" in features_df.columns:
        out["priority_score"] = proba * features_df["behavioral_clv"].values
    elif "clv" in features_df.columns:
        out["priority_score"] = proba * features_df["clv"].values
    else:
        out["priority_score"] = proba  # fallback: rank by probability alone

    out["priority_rank"] = out["priority_score"].rank(ascending=False).astype(int)

    # Optional: attach SHAP values as columns
    if include_shap:
        if explainer is None:
            explainer = load_explainer(model_dir)
        import shap
        X_clean = (
            features_df[feature_cols]
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        sv = explainer.shap_values(X_clean)
        if isinstance(sv, list):
            sv = sv[1]
        shap_df = pd.DataFrame(sv, columns=[f"shap_{c}" for c in feature_cols])
        out = pd.concat([out.reset_index(drop=True),
                         shap_df.reset_index(drop=True)], axis=1)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. Behavioral CLV
# ═══════════════════════════════════════════════════════════════════════════

def compute_behavioral_clv(
    members_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    clv_col: str = "clv",
    months_since_col: str = "months_since_last_flight",
    key_col: str = "loyalty_number",
    k: float = 0.15,
) -> pd.DataFrame:
    """
    Compute behavioral CLV for every member.

    behavioral_clv = CLV × (1 − churn_prob) × recency_decay_weight

    This combines three signals:
      - Historical value (CLV)
      - Forward risk     (1 − churn_prob)
      - Engagement freshness (recency decay using hyperbolic function)

    The gap between CLV rank and behavioral_clv rank identifies members
    who are overvalued by the existing CLV metric — the central business
    finding of this project.

    Parameters
    ----------
    members_df     : member profile table with clv_col
    predictions_df : output of score_members() — must contain churn_prob
    clv_col        : name of the CLV column in members_df
    months_since_col: name of recency column in members_df or features_df
    key_col        : join key
    k              : hyperbolic decay parameter (default 0.15, same as nb04)

    Returns
    -------
    pd.DataFrame with columns:
        key_col, clv, behavioral_clv, clv_rank, bclv_rank, rank_gap
    """
    from src.features import recency_weight as _rw

    df = predictions_df[[key_col, "churn_prob"]].merge(
        members_df[[key_col, clv_col]], on=key_col, how="left"
    )

    # Attach recency if available
    if months_since_col in members_df.columns:
        df = df.merge(members_df[[key_col, months_since_col]], on=key_col, how="left")
    elif months_since_col in predictions_df.columns:
        df = df.merge(predictions_df[[key_col, months_since_col]], on=key_col, how="left")
    else:
        df[months_since_col] = 3  # conservative default: 3 months

    df[months_since_col] = df[months_since_col].fillna(3)

    df["recency_decay"]  = df[months_since_col].apply(lambda m: _rw(m, k=k))
    df["behavioral_clv"] = (
        df[clv_col].fillna(0)
        * (1 - df["churn_prob"])
        * df["recency_decay"]
    )

    n = len(df)
    df["clv_rank"]  = df[clv_col].rank(ascending=False)
    df["bclv_rank"] = df["behavioral_clv"].rank(ascending=False)
    df["rank_gap"]  = df["clv_rank"] - df["bclv_rank"]
    # Positive rank_gap = overvalued by CLV
    # Negative rank_gap = undervalued by CLV

    return df[[key_col, clv_col, "behavioral_clv",
               "clv_rank", "bclv_rank", "rank_gap"]]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Member Lookup (used by dashboard page 1)
# ═══════════════════════════════════════════════════════════════════════════

def lookup_member(
    member_id: Union[str, int],
    features_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    playbook_df: pd.DataFrame,
    model=None,
    explainer=None,
    feature_cols: Optional[list] = None,
    model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR,
) -> dict:
    """
    Full profile for a single member — used by the Streamlit member lookup page.

    Combines predictions, SHAP explanation, segment label, and assigned nudge
    into one structured dict that the dashboard can render directly.

    Parameters
    ----------
    member_id    : loyalty_number to look up
    features_df  : full feature matrix (one row per member)
    segments_df  : output of notebook 06 (06_segments.csv)
    playbook_df  : output of notebook 07 (07_retention_playbook.csv)
    model        : pre-loaded XGBClassifier (or None to load from disk)
    explainer    : pre-loaded SHAP explainer (or None to load from disk)
    feature_cols : feature column list (or None to load from meta)
    model_dir    : path to models/ folder

    Returns
    -------
    dict with keys:
        found           : bool — False if member_id not in features_df
        member_id       : str
        churn_prob      : float
        risk_bucket     : str
        segment         : str   — cluster_label from segmentation
        rfm_segment     : str
        clv             : float
        behavioral_clv  : float
        nudge           : dict  — full nudge record from playbook
        shap_drivers    : pd.DataFrame — top 10 SHAP drivers
        shap_waterfall  : pd.DataFrame — waterfall chart data
        explanation     : dict  — raw SHAP explanation
    """
    key_col = "loyalty_number"

    if model is None:
        model = load_model(model_dir)
    if feature_cols is None:
        feature_cols = load_meta(model_dir)["feature_cols"]

    # ── Find member ────────────────────────────────────────────────────────
    member_features = features_df[features_df[key_col] == member_id]
    if member_features.empty:
        return {"found": False, "member_id": member_id}

    # ── Churn prediction ───────────────────────────────────────────────────
    proba  = float(predict_churn_proba(member_features, model, feature_cols, model_dir)[0])
    bucket = predict_risk_bucket(proba)

    # ── SHAP explanation ───────────────────────────────────────────────────
    if explainer is None:
        try:
            explainer = load_explainer(model_dir)
        except FileNotFoundError:
            explainer = None

    shap_exp      = None
    shap_drivers  = pd.DataFrame()
    shap_waterfall= pd.DataFrame()

    if explainer is not None:
        shap_exp       = get_shap_explanation(member_features, explainer,
                                               feature_cols, model_dir)
        shap_drivers   = get_top_shap_drivers(shap_exp, n=10)
        shap_waterfall = build_shap_waterfall_data(shap_exp, n=12)

    # ── Segment info ───────────────────────────────────────────────────────
    seg_row = segments_df[segments_df[key_col] == member_id]
    segment      = seg_row["cluster_label"].values[0]  if not seg_row.empty and "cluster_label"  in seg_row.columns else "Unknown"
    rfm_segment  = seg_row["rfm_segment"].values[0]    if not seg_row.empty and "rfm_segment"     in seg_row.columns else "Unknown"
    clv_val      = float(seg_row["clv"].values[0])     if not seg_row.empty and "clv"             in seg_row.columns else 0.0
    bclv_val     = float(seg_row["behavioral_clv"].values[0]) if not seg_row.empty and "behavioral_clv" in seg_row.columns else 0.0

    # ── Nudge from playbook ────────────────────────────────────────────────
    nudge_row = playbook_df[playbook_df[key_col] == member_id]
    nudge_cols = [
        "nudge_type", "mechanism", "channel", "timing",
        "subject_line", "message_frame", "be_principle",
        "success_kpi", "expected_lift", "priority_tier",
    ]
    nudge = {}
    if not nudge_row.empty:
        for col in nudge_cols:
            if col in nudge_row.columns:
                nudge[col] = nudge_row[col].values[0]

    return {
        "found"         : True,
        "member_id"     : member_id,
        "churn_prob"    : proba,
        "churn_pct"     : round(proba * 100, 1),
        "risk_bucket"   : bucket,
        "segment"       : segment,
        "rfm_segment"   : rfm_segment,
        "clv"           : clv_val,
        "behavioral_clv": bclv_val,
        "nudge"         : nudge,
        "shap_drivers"  : shap_drivers,
        "shap_waterfall": shap_waterfall,
        "explanation"   : shap_exp,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Model Performance Summary  (used by dashboard overview)
# ═══════════════════════════════════════════════════════════════════════════

def get_model_summary(
    model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR,
) -> dict:
    """
    Return a human-readable performance summary for the dashboard header.

    Parameters
    ----------
    model_dir : path to models/ folder

    Returns
    -------
    dict with display-ready strings and floats:
        xgb_auc, lr_auc, threshold, churn_rate, n_members,
        xgb_auc_pct, threshold_pct, churn_rate_pct
    """
    meta = load_meta(model_dir)
    return {
        "xgb_auc"       : meta.get("xgb_cv_auc", 0.0),
        "lr_auc"        : meta.get("lr_cv_auc",  0.0),
        "threshold"     : meta.get("threshold",  0.5),
        "churn_rate"    : meta.get("churn_rate", 0.0),
        "n_members"     : meta.get("n_members",  0),
        # Display-ready percentage strings
        "xgb_auc_pct"   : f'{meta.get("xgb_cv_auc", 0.0)*100:.1f}%',
        "threshold_pct" : f'{meta.get("threshold",  0.5)*100:.0f}%',
        "churn_rate_pct": f'{meta.get("churn_rate", 0.0)*100:.1f}%',
    }
