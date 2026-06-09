"""
src/segments.py
────────────────────────────────────────────────────────────────────────────
Segmentation utilities: RFM scoring, K-Means cluster labelling,
behavioral CLV gap analysis, and nudge assignment.

Called by:
  - 06_customer_segmentation.ipynb  (batch segmentation)
  - 07_retention_strategy.ipynb     (nudge assignment)
  - dashboard/app.py                (segment map + nudge engine pages)

Usage
-----
from src.segments import (
    build_rfm_scores,
    label_rfm_segment,
    load_kmeans,
    predict_cluster,
    label_cluster,
    assign_nudge,
    build_segment_summary,
    build_priority_list,
    SEGMENT_COLORS,
    RISK_BUCKET_ORDER,
    NUDGE_DEFINITIONS,
)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_PROC_DIR  = Path(__file__).resolve().parent.parent / "data" / "processed"

# ── Constants shared across notebooks and dashboard ───────────────────────
KEY    = "loyalty_number"
TARGET = "churned"
CLV_COL = "clv"

RISK_BUCKET_ORDER = [
    "Critical (>75%)",
    "High (50-75%)",
    "Medium (25-50%)",
    "Low (<25%)",
]

SEGMENT_COLORS = {
    "Silent Quitters"      : "#C0392B",
    "At-Risk High-Value"   : "#E74C3C",
    "Declining Actives"    : "#E67E22",
    "Loyal Champions"      : "#1A8A4A",
    "Stable Infrequent"    : "#2471A3",
    "Low-Value Lapsers"    : "#8E44AD",
    "Moderate Risk"        : "#566573",
    # RFM labels
    "Champions"            : "#1A8A4A",
    "Loyal"                : "#2471A3",
    "Recent Low-Freq"      : "#16A085",
    "Needs Attention"      : "#E67E22",
    "At-Risk High-Value"   : "#E74C3C",
    "Promising New"        : "#8E44AD",
    "Hibernating"          : "#566573",
    "Lost"                 : "#C0392B",
}

RISK_COLORS = {
    "Critical (>75%)" : "#C0392B",
    "High (50-75%)"   : "#E67E22",
    "Medium (25-50%)" : "#F1C40F",
    "Low (<25%)"      : "#2471A3",
}

# ── Nudge catalogue ────────────────────────────────────────────────────────
# Central definition so notebooks and dashboard always show identical copy.
NUDGE_DEFINITIONS: dict[str, dict] = {
    "Loss-Framed Points Alert": {
        "mechanism"    : "Loss aversion (Kahneman & Tversky, 1979)",
        "channel"      : "Email + App Push",
        "timing"       : "Immediate — within 24 hours of risk flag",
        "priority_tier": "TIER 1 — act within 48 hours",
        "success_kpi"  : "Flight booked within 60 days of send",
        "expected_lift": "2-3x vs gain-framed equivalent",
        "be_principle" : (
            "Frame inaction as losing something already owned, "
            "not missing a gain. Loss framing consistently "
            "outperforms gain framing by 2x in travel contexts."
        ),
    },
    "Personal Relationship Outreach": {
        "mechanism"    : "Endowment effect + personalised attention (Thaler, 1980)",
        "channel"      : "Email (personalised) + Phone call if CLV > $2,000",
        "timing"       : "Within 48 hours of risk flag",
        "priority_tier": "TIER 1 — personal contact required",
        "success_kpi"  : "Status retained + 1 flight booked within 90 days",
        "expected_lift": "35-50% retention rate for personally contacted members",
        "be_principle" : (
            "Pre-give the reward (status extension) to activate "
            "loss aversion of losing it. Members who receive "
            "something feel compelled to reciprocate."
        ),
    },
    "Commitment Device — Quarterly Flight Goal": {
        "mechanism"    : "Commitment & consistency (Cialdini, 1984)",
        "channel"      : "Email + In-app goal-setter",
        "timing"       : "First week of each quarter (Jan, Apr, Jul, Oct)",
        "priority_tier": "TIER 2 — batch send quarterly",
        "success_kpi"  : "Goal set + minimum 1 additional flight vs prior quarter",
        "expected_lift": "25-40% increase in quarterly flights among goal-setters",
        "be_principle" : (
            "Small public commitments create consistency pressure. "
            "Members who state a goal are 3x more likely to "
            "follow through than those who receive a standard offer."
        ),
    },
    "Social Proof + Tier Preview": {
        "mechanism"    : "Social proof + anticipated reward",
        "channel"      : "App notification + monthly email digest",
        "timing"       : "Monthly, timed to post-flight (within 48h of landing)",
        "priority_tier": "TIER 3 — automated monthly",
        "success_kpi"  : "Maintained flight frequency + tier upgrade within 6 months",
        "expected_lift": "15-20% increase in booking rate post-notification",
        "be_principle" : (
            "Social comparison reinforces positive behaviour. "
            "Tier preview creates anticipatory reward — "
            "members begin to feel they already have the next tier."
        ),
    },
    "Points Redemption Nudge": {
        "mechanism"    : "Endowment effect — unredeemed points feel already owned",
        "channel"      : "Email",
        "timing"       : "60 days before points expiry or quarterly",
        "priority_tier": "TIER 3 — automated trigger on expiry window",
        "success_kpi"  : "Points redemption event within 30 days",
        "expected_lift": "40% redemption rate vs 12% baseline",
        "be_principle" : (
            "Points feel owned, so spending them feels safe. "
            "Redemption re-engages members who have drifted "
            "without cancelling — the silent churn cohort."
        ),
    },
    "Lightweight Reactivation Offer": {
        "mechanism"    : "Sunk cost + minimal friction reactivation",
        "channel"      : "Email only (low cost)",
        "timing"       : "Monthly batch — do not over-contact",
        "priority_tier": "TIER 4 — low priority",
        "success_kpi"  : "1 flight booked within 90 days",
        "expected_lift": "8-12% reactivation rate",
        "be_principle" : (
            "Sunk cost framing (time invested in membership) "
            "reduces the psychological cost of re-engaging. "
            "Minimal friction (one click) removes behavioural barriers."
        ),
    },
    "Seasonal Travel Prompt": {
        "mechanism"    : "Peak-end rule — make the next interaction a positive peak",
        "channel"      : "Email",
        "timing"       : "Pre-peak season (November for winter, April for summer)",
        "priority_tier": "TIER 3 — automated seasonal",
        "success_kpi"  : "Email open + flight search within 7 days",
        "expected_lift": "20-30% open rate; 6-10% conversion to search",
        "be_principle" : (
            "The last emotional memory before a new booking cycle "
            "disproportionately shapes future behaviour (peak-end rule). "
            "A personalised, timely message becomes that peak."
        ),
    },
    "General Engagement Email": {
        "mechanism"    : "Awareness baseline",
        "channel"      : "Email",
        "timing"       : "Monthly newsletter",
        "priority_tier": "TIER 4 — low priority",
        "success_kpi"  : "Email open rate",
        "expected_lift": "Baseline — no specific behavioural lift expected",
        "be_principle" : "Maintains top-of-mind awareness with no specific mechanism.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. RFM Segmentation
# ═══════════════════════════════════════════════════════════════════════════

def build_rfm_scores(
    df: pd.DataFrame,
    flights_col: str = "total_flights_historical",
    clv_col: str = CLV_COL,
    months_since_col: str = "months_since_last_flight",
    pts_acc_col: str = "total_pts_accumulated",
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Compute R, F, M scores (1–4) and concatenated RFM string per member.

    Recency  : months_since_last_flight  (lower = better → score reversed)
    Frequency: total historical flights
    Monetary : CLV if available, else total points accumulated

    Parameters
    ----------
    df              : member-level DataFrame (one row per member)
    flights_col     : frequency column name
    clv_col         : monetary column name (preferred)
    months_since_col: recency column name
    pts_acc_col     : fallback monetary column
    n_bins          : number of quartile bins (default 4)

    Returns
    -------
    pd.DataFrame with added columns:
        R, F, M         : int scores 1–n_bins
        R_raw, F_raw, M_raw : original values used for scoring
        RFM_score       : concatenated string e.g. "443"
        RFM_total       : int sum of R + F + M
    """
    out = df.copy()

    # ── Resolve column names ──────────────────────────────────────────────
    r_col = next((c for c in [months_since_col, "months_since_last_flight"]
                  if c in df.columns), None)
    f_col = next((c for c in [flights_col, "total_flights_historical",
                               "total_flights_sum", "total_flights"]
                  if c in df.columns), None)
    m_col = next((c for c in [clv_col, "clv", pts_acc_col,
                               "total_pts_accumulated"]
                  if c in df.columns), None)

    if not all([r_col, f_col, m_col]):
        missing = [n for n, c in [("recency", r_col),
                                   ("frequency", f_col),
                                   ("monetary", m_col)] if c is None]
        raise ValueError(f"build_rfm_scores: cannot resolve columns for {missing}")

    out["R_raw"] = out[r_col]
    out["F_raw"] = out[f_col]
    out["M_raw"] = out[m_col]

    # ── Recency: lower months_since = better = higher score ───────────────
    out["R"] = pd.qcut(
        out["R_raw"], q=n_bins, labels=list(range(n_bins, 0, -1)),
        duplicates="drop",
    ).astype(int)

    # ── Frequency: higher flights = better = higher score ─────────────────
    out["F"] = pd.qcut(
        out["F_raw"].rank(method="first"), q=n_bins,
        labels=list(range(1, n_bins + 1)),
        duplicates="drop",
    ).astype(int)

    # ── Monetary: higher CLV = better = higher score ───────────────────────
    out["M"] = pd.qcut(
        out["M_raw"].rank(method="first"), q=n_bins,
        labels=list(range(1, n_bins + 1)),
        duplicates="drop",
    ).astype(int)

    out["RFM_score"] = (
        out["R"].astype(str)
        + out["F"].astype(str)
        + out["M"].astype(str)
    )
    out["RFM_total"] = out["R"] + out["F"] + out["M"]

    return out


def label_rfm_segment(row: pd.Series) -> str:
    """
    Map a single member's R, F, M scores to an actionable segment label.

    Called via df.apply(label_rfm_segment, axis=1).

    Labels (in priority order):
        Champions         : high recency + high frequency + high monetary
        Loyal             : high recency + high frequency
        Recent Low-Freq   : high recency + low frequency (new or re-activated)
        At-Risk High-Value: low recency + high total score (valuable but drifting)
        Needs Attention   : medium recency + high frequency (starting to slip)
        Promising New     : high recency + very low frequency (early stage)
        Lost              : low recency + low overall score
        Hibernating       : default for everything else
    """
    r, f, m   = int(row["R"]), int(row["F"]), int(row["M"])
    total     = r + f + m
    max_score = 4  # n_bins

    if r >= max_score and f >= max_score - 1 and m >= max_score - 1:
        return "Champions"
    if r >= max_score - 1 and f >= max_score - 1:
        return "Loyal"
    if r == max_score and f <= 2:
        return "Recent Low-Freq"
    if r == 1 and total >= max_score * 2:
        return "At-Risk High-Value"
    if r == 1 and total <= max_score + 2:
        return "Lost"
    if r == 2 and f >= max_score - 1:
        return "Needs Attention"
    if r >= max_score - 1 and f == 1:
        return "Promising New"
    return "Hibernating"


# ═══════════════════════════════════════════════════════════════════════════
# 2. K-Means Cluster Labelling
# ═══════════════════════════════════════════════════════════════════════════

def load_kmeans(model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR):
    """
    Load the fitted K-Means model saved by notebook 06.

    Returns
    -------
    sklearn.cluster.KMeans
    """
    path = Path(model_dir) / "kmeans_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"K-Means model not found at {path}.\n"
            "Run 06_customer_segmentation.ipynb first."
        )
    return joblib.load(path)


def predict_cluster(
    X: pd.DataFrame,
    cluster_feature_cols: list[str],
    kmeans=None,
    scaler=None,
    model_dir: Union[str, Path] = _DEFAULT_MODEL_DIR,
) -> np.ndarray:
    """
    Assign K-Means cluster labels to a feature matrix.

    Parameters
    ----------
    X                    : feature DataFrame
    cluster_feature_cols : ordered list of columns used during K-Means training
    kmeans               : pre-loaded KMeans model (or None to load from disk)
    scaler               : pre-fitted StandardScaler for the cluster features
                           (or None — features will be z-scored on the fly)
    model_dir            : path to models/ folder

    Returns
    -------
    np.ndarray of int cluster IDs, shape (n_members,)
    """
    from sklearn.preprocessing import StandardScaler

    if kmeans is None:
        kmeans = load_kmeans(model_dir)

    # Align columns
    missing = set(cluster_feature_cols) - set(X.columns)
    X_aligned = X.copy()
    for col in missing:
        X_aligned[col] = 0

    X_vals = X_aligned[cluster_feature_cols].fillna(0).replace(
        [np.inf, -np.inf], 0
    )

    if scaler is None:
        X_scaled = StandardScaler().fit_transform(X_vals)
    else:
        X_scaled = scaler.transform(X_vals)

    return kmeans.predict(X_scaled)


def label_cluster(cluster_stats: pd.Series, global_churn: float,
                  global_clv: float) -> str:
    """
    Map a cluster's aggregate statistics to a human-readable segment label.

    Parameters
    ----------
    cluster_stats : pd.Series with keys:
        churn_rate, avg_clv, avg_recency, avg_hyp_score, avg_trajectory
    global_churn  : overall churn rate across all members
    global_clv    : overall mean CLV across all members

    Returns
    -------
    str  — segment label from the SEGMENT_COLORS keys
    """
    cr   = float(cluster_stats.get("churn_rate",   0))
    clv  = float(cluster_stats.get("avg_clv",      0))
    hyp  = float(cluster_stats.get("avg_hyp_score",0))
    traj = float(cluster_stats.get("avg_trajectory",0))

    if cr > global_churn * 1.5 and clv >= global_clv:
        return "Silent Quitters"
    if cr > global_churn * 1.5 and clv < global_clv:
        return "Low-Value Lapsers"
    if cr < global_churn * 0.5 and hyp > 0:
        return "Loyal Champions"
    if cr < global_churn and traj < 0:
        return "Stable Infrequent"
    if traj < 0 and cr > global_churn:
        return "Declining Actives"
    return "Moderate Risk"


def build_cluster_label_map(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    churn_col: str = TARGET,
    clv_col: str = CLV_COL,
    hyp_col: str = "hyperbolic_flight_score",
    traj_col: str = "flight_trajectory",
) -> dict[int, str]:
    """
    Compute per-cluster statistics and return a cluster_id → label mapping.

    Parameters
    ----------
    df          : member DataFrame with cluster assignments and features
    cluster_col : column containing integer cluster IDs

    Returns
    -------
    dict  {cluster_id: label_string}
    """
    global_churn = float(df[churn_col].mean()) if churn_col in df.columns else 0.2
    global_clv   = float(df[clv_col].mean())   if clv_col   in df.columns else 1.0

    agg: dict = {churn_col: "mean"}
    if clv_col  in df.columns: agg[clv_col]  = "mean"
    if hyp_col  in df.columns: agg[hyp_col]  = "mean"
    if traj_col in df.columns: agg[traj_col] = "mean"

    stats = (
        df.groupby(cluster_col).agg(agg)
        .rename(columns={
            churn_col: "churn_rate",
            clv_col  : "avg_clv",
            hyp_col  : "avg_hyp_score",
            traj_col : "avg_trajectory",
        })
    )

    return {
        int(cid): label_cluster(row, global_churn, global_clv)
        for cid, row in stats.iterrows()
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Nudge Assignment
# ═══════════════════════════════════════════════════════════════════════════

def assign_nudge(row: pd.Series) -> dict:
    """
    Assign a fully specified retention intervention to a single member.

    Reads seven behavioural signals per member and returns the nudge dict
    from NUDGE_DEFINITIONS, augmented with a personalised subject_line
    and message_frame.

    Parameters
    ----------
    row : pd.Series with fields:
        cluster_label, risk_bucket, net_unredeemed_points,
        months_since_last_flight, flight_trajectory,
        tenure_months, loss_aversion_score, flights_last_3m,
        clv (optional), behavioral_clv (optional)

    Returns
    -------
    dict  — nudge record ready for display in the dashboard or export
    """
    seg     = str(row.get("cluster_label",              "")).strip()
    risk    = str(row.get("risk_bucket",                "")).strip()
    pts     = float(row.get("net_unredeemed_points",    0) or 0)
    recency = float(row.get("months_since_last_flight", 6) or 6)
    traj    = float(row.get("flight_trajectory",        0) or 0)
    tenure  = float(row.get("tenure_months",            12) or 12)
    loss_av = float(row.get("loss_aversion_score",      0) or 0)
    f3m     = float(row.get("flights_last_3m",          0) or 0)
    clv_val = float(row.get(CLV_COL,                    0) or 0)

    pts_str  = f"{pts:,.0f}" if pts > 0 else "your accumulated"
    high_clv = clv_val > 2000

    # ── 1. Silent Quitters — high value, high risk ────────────────────────
    if "Silent" in seg and risk in ("Critical (>75%)", "High (50-75%)"):
        nudge_key = "Loss-Framed Points Alert"
        subject   = f"Your {pts_str} points expire in 90 days"
        message   = (
            f"You have earned {pts_str} points — enough for a round trip. "
            f"Members who let their points expire lose an average of $340 in "
            f"travel value. Book one flight this month to keep everything you "
            f"have earned."
        )

    # ── 2. At-Risk High-Value — personal contact ──────────────────────────
    elif "At-Risk" in seg and risk == "Critical (>75%)":
        nudge_key = "Personal Relationship Outreach"
        subject   = "A personal note about your membership"
        message   = (
            f"As one of our most valued members over {tenure:.0f} months, "
            f"we noticed your recent travel has changed. We have reserved a "
            f"complimentary status extension for you — no flights needed this "
            f"quarter. We want to earn your next trip."
        )

    # ── 3. Declining Actives — commitment device ──────────────────────────
    elif "Declining" in seg and traj < 0 and risk in (
        "High (50-75%)", "Medium (25-50%)"
    ):
        nudge_key    = "Commitment Device — Quarterly Flight Goal"
        quarter_goal = max(int(f3m) + 2, 3)
        subject      = "Set your flight goal for this quarter"
        message      = (
            f"You flew {f3m:.0f} times last quarter. Members who set a "
            f"quarterly goal fly {quarter_goal} or more times and maintain "
            f"their status 3x more often. Tap to set your goal — it takes "
            f"10 seconds."
        )

    # ── 4. Loyal Champions — social proof ────────────────────────────────
    elif "Champion" in seg or "Loyal" in seg:
        nudge_key = "Social Proof + Tier Preview"
        subject   = "You flew more than 91% of members this month"
        message   = (
            f"You are in the top 10% of active members this quarter. "
            f"At your current pace, you will reach the next tier in "
            f"{max(1, 12 - int(tenure % 12))} months. "
            f"Here is what unlocks when you get there."
        )

    # ── 5. High loss aversion, low risk — redemption nudge ───────────────
    elif loss_av > 0.6 and risk == "Low (<25%)":
        nudge_key = "Points Redemption Nudge"
        subject   = f"Your {pts_str} points are waiting for you"
        message   = (
            f"You have {pts_str} points — here are 3 ways to use them before "
            f"they expire. Members who redeem at least once per year are 2x "
            f"more likely to stay active long-term."
        )

    # ── 6. Low-Value Lapsers / Lost ───────────────────────────────────────
    elif "Lapser" in seg or "Lost" in seg:
        nudge_key = "Lightweight Reactivation Offer"
        subject   = "We saved your membership — one click to reactivate"
        message   = (
            f"You joined {tenure:.0f} months ago and we have not seen you "
            f"lately. Your account and points are still here. One flight this "
            f"season keeps your membership active — and we will add 500 bonus "
            f"points to welcome you back."
        )

    # ── 7. Seasonal / Stable Infrequent ───────────────────────────────────
    elif "Infrequent" in seg or recency > 4:
        nudge_key = "Seasonal Travel Prompt"
        subject   = "Your next trip is closer than you think"
        message   = (
            f"Based on when you have flown before, this is typically your "
            f"travel season. We have matched routes to your history — here "
            f"are 3 options with your points applied."
        )

    # ── Default ───────────────────────────────────────────────────────────
    else:
        nudge_key = "General Engagement Email"
        subject   = "What is new with your membership"
        message   = (
            "Standard newsletter with personalised points balance and "
            "route suggestions."
        )

    # Merge base definition with personalised copy
    nudge = {**NUDGE_DEFINITIONS.get(nudge_key, {})}
    nudge["nudge_type"]    = nudge_key
    nudge["subject_line"]  = subject
    nudge["message_frame"] = message

    return nudge


def apply_nudges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply assign_nudge() to every row and return an enriched DataFrame.

    Parameters
    ----------
    df : member DataFrame with segmentation and feature columns

    Returns
    -------
    pd.DataFrame — original columns plus all nudge fields as new columns
    """
    nudge_records = df.apply(assign_nudge, axis=1)
    nudge_df      = pd.DataFrame(nudge_records.tolist())
    return pd.concat(
        [df.reset_index(drop=True), nudge_df.reset_index(drop=True)],
        axis=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Segment Summary Tables  (used by dashboard overview pages)
# ═══════════════════════════════════════════════════════════════════════════

def build_segment_summary(
    df: pd.DataFrame,
    segment_col: str = "cluster_label",
    churn_col: str = TARGET,
    clv_col: str = CLV_COL,
    bclv_col: str = "behavioral_clv",
    churn_prob_col: str = "churn_prob",
) -> pd.DataFrame:
    """
    Aggregate segment-level statistics for dashboard display.

    Parameters
    ----------
    df           : member DataFrame with segment, churn, and CLV columns
    segment_col  : column name for segment labels
    churn_col    : actual churn label (0/1)
    clv_col      : existing CLV column
    bclv_col     : behavioral CLV column
    churn_prob_col: predicted churn probability column

    Returns
    -------
    pd.DataFrame with one row per segment, sorted by avg churn probability:
        segment, n_members, churn_rate, avg_churn_prob,
        avg_clv, avg_behavioral_clv, clv_gap,
        n_critical, pct_of_total, color
    """
    agg: dict = {
        KEY          : "count",
        churn_prob_col: "mean",
    }
    if churn_col   in df.columns: agg[churn_col]   = "mean"
    if clv_col     in df.columns: agg[clv_col]     = "mean"
    if bclv_col    in df.columns: agg[bclv_col]    = "mean"

    # Count Critical-risk members per segment
    if "risk_bucket" in df.columns:
        df = df.copy()
        df["_is_critical"] = (df["risk_bucket"] == "Critical (>75%)").astype(int)
        agg["_is_critical"] = "sum"

    summary = (
        df.groupby(segment_col).agg(agg)
        .reset_index()
        .rename(columns={
            KEY           : "n_members",
            churn_col     : "churn_rate",
            churn_prob_col: "avg_churn_prob",
            clv_col       : "avg_clv",
            bclv_col      : "avg_behavioral_clv",
            "_is_critical": "n_critical",
        })
        .sort_values("avg_churn_prob", ascending=False)
    )

    if "avg_clv" in summary.columns and "avg_behavioral_clv" in summary.columns:
        summary["clv_gap"] = summary["avg_clv"] - summary["avg_behavioral_clv"]

    summary["pct_of_total"] = summary["n_members"] / summary["n_members"].sum()
    summary["color"]        = summary[segment_col].map(SEGMENT_COLORS).fillna("#566573")

    return summary.reset_index(drop=True)


def build_priority_list(
    df: pd.DataFrame,
    n: Optional[int] = None,
    min_churn_prob: float = 0.0,
    segments: Optional[list[str]] = None,
    risk_buckets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return members sorted by priority score with filters applied.

    Used by the dashboard's Nudge Engine page to generate outreach lists.

    Parameters
    ----------
    df             : member DataFrame from 07_retention_playbook.csv
    n              : max rows to return (None = all)
    min_churn_prob : filter: only members with churn_prob >= this value
    segments       : filter: only members in these cluster_label values
    risk_buckets   : filter: only members in these risk_bucket values

    Returns
    -------
    pd.DataFrame sorted by priority_score descending, containing:
        loyalty_number, cluster_label, risk_bucket, churn_prob, churn_pct,
        clv, behavioral_clv, priority_score, nudge_type, channel,
        timing, subject_line, priority_tier
    """
    out = df.copy()

    if min_churn_prob > 0:
        out = out[out["churn_prob"] >= min_churn_prob]

    if segments:
        out = out[out["cluster_label"].isin(segments)]

    if risk_buckets:
        out = out[out["risk_bucket"].isin(risk_buckets)]

    # Ensure priority score exists
    if "priority_score" not in out.columns:
        clv_col_use = "behavioral_clv" if "behavioral_clv" in out.columns else CLV_COL
        out["priority_score"] = out["churn_prob"] * out.get(clv_col_use, pd.Series(1, index=out.index))

    out = out.sort_values("priority_score", ascending=False)

    display_cols = [
        c for c in [
            KEY, "cluster_label", "rfm_segment", "risk_bucket",
            "churn_prob", "churn_pct", CLV_COL, "behavioral_clv",
            "priority_score", "nudge_type", "channel",
            "timing", "subject_line", "priority_tier",
        ] if c in out.columns
    ]
    out = out[display_cols]

    if n is not None:
        out = out.head(n)

    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ROI Estimates  (used by dashboard and report)
# ═══════════════════════════════════════════════════════════════════════════

# Conservative retention lift assumptions per nudge (literature-backed)
NUDGE_ROI_ASSUMPTIONS: dict[str, dict] = {
    "Loss-Framed Points Alert"                : {"lift": 0.25, "cost_per_contact": 0.50},
    "Personal Relationship Outreach"          : {"lift": 0.40, "cost_per_contact": 8.00},
    "Commitment Device — Quarterly Flight Goal": {"lift": 0.30, "cost_per_contact": 0.30},
    "Social Proof + Tier Preview"             : {"lift": 0.15, "cost_per_contact": 0.10},
    "Points Redemption Nudge"                 : {"lift": 0.20, "cost_per_contact": 0.50},
    "Lightweight Reactivation Offer"          : {"lift": 0.10, "cost_per_contact": 0.30},
    "Seasonal Travel Prompt"                  : {"lift": 0.12, "cost_per_contact": 0.50},
    "General Engagement Email"                : {"lift": 0.05, "cost_per_contact": 0.20},
}


def build_roi_table(playbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute estimated ROI for each nudge type using conservative assumptions.

    Parameters
    ----------
    playbook_df : output of apply_nudges() — must contain:
                  nudge_type, behavioral_clv (or clv), loyalty_number

    Returns
    -------
    pd.DataFrame sorted by roi_estimate descending:
        nudge_type, n_members, total_bclv, retention_lift,
        clv_saved_est, campaign_cost_est, roi_estimate
    """
    clv_use = "behavioral_clv" if "behavioral_clv" in playbook_df.columns else CLV_COL

    summary = (
        playbook_df.groupby("nudge_type")
        .agg(n_members=(KEY, "count"), total_bclv=(clv_use, "sum"))
        .reset_index()
    )

    rows = []
    for _, row in summary.iterrows():
        nudge = row["nudge_type"]
        assume = NUDGE_ROI_ASSUMPTIONS.get(nudge, {"lift": 0.05, "cost_per_contact": 0.50})
        lift        = assume["lift"]
        cost_unit   = assume["cost_per_contact"]
        n           = int(row["n_members"])
        bclv        = float(row["total_bclv"])
        saved       = bclv * lift
        cost        = n * cost_unit
        roi         = saved / max(cost, 1)
        rows.append({
            "nudge_type"        : nudge,
            "n_members"         : n,
            "total_bclv"        : round(bclv, 2),
            "retention_lift"    : lift,
            "clv_saved_est"     : round(saved, 2),
            "campaign_cost_est" : round(cost, 2),
            "roi_estimate"      : round(roi, 1),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("roi_estimate", ascending=False)
        .reset_index(drop=True)
    )
