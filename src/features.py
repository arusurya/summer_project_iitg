
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Column name constants ─────────────────────────────────────────────────
KEY          = "loyalty_number"
YEAR_COL     = "year"
MONTH_COL    = "month"
FLIGHTS_COL  = "total_flights"
DISTANCE_COL = "distance"
POINTS_ACC   = "points_accumulated"
POINTS_RED   = "points_redeemed"
CLV_COL      = "clv"
CARD_COL     = "loyalty_card"

PREDICTION_CUTOFF = pd.Timestamp("2016-12-01")

# ── Master feature column list (populated as functions run) ───────────────
# Import this list in notebooks and the dashboard to guarantee consistency.
FEATURE_COLS: list[str] = []


# ═══════════════════════════════════════════════════════════════════════════
# 0. Utilities
# ═══════════════════════════════════════════════════════════════════════════

def resolve_col(df: pd.DataFrame, candidates: list[str], fallback: str) -> str:
    """Return the first candidate column name that exists in df, else fallback."""
    for c in candidates:
        if c in df.columns:
            return c
    return fallback


def enforce_leakage_boundary(
    activity: pd.DataFrame,
    cutoff: pd.Timestamp = PREDICTION_CUTOFF,
) -> pd.DataFrame:
    """
    Drop all activity rows after the prediction cutoff.

    This is the single most important call in the pipeline — every feature
    must be computed on the returned DataFrame, never on the raw one.

    Parameters
    ----------
    activity : pd.DataFrame
        Monthly activity table with a 'period' datetime column.
    cutoff : pd.Timestamp
        Leakage boundary.  Default = PREDICTION_CUTOFF.

    Returns
    -------
    pd.DataFrame
        Activity rows with period <= cutoff only.
    """
    if "period" not in activity.columns:
        activity = activity.copy()
        activity["period"] = pd.to_datetime(
            activity[YEAR_COL].astype(str) + "-"
            + activity[MONTH_COL].astype(str).str.zfill(2)
        )
    return activity[activity["period"] <= cutoff].copy()


def recency_weight(months_ago: float, k: float = 0.15) -> float:
    """
    Hyperbolic discount factor.

    w(t) = 1 / (1 + k * t)

    At t=0  → 1.00 (current month, full weight)
    At t=6  → 0.53 (6 months ago, half weight)
    At t=12 → 0.36 (12 months ago, one-third weight)

    Theory: Laibson (1997) — humans discount the future hyperbolically,
    not exponentially, giving disproportionate weight to recent events.
    """
    return 1.0 / (1.0 + k * float(months_ago))


def _register(col: str) -> None:
    """Append col to FEATURE_COLS if not already present."""
    if col not in FEATURE_COLS:
        FEATURE_COLS.append(col)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Raw Activity Features  (Family 1 — baseline)
# ═══════════════════════════════════════════════════════════════════════════

def build_raw_activity_features(
    activity: pd.DataFrame,
    cohort: list,
) -> pd.DataFrame:
    """
    Aggregate raw flight, distance, and points totals per member.

    These are the features every conventional model would use.
    We include them so we can demonstrate that behavioural features
    add explanatory power on top of these baselines.

    Parameters
    ----------
    activity : pd.DataFrame
        Leakage-safe activity table (already filtered to <= cutoff).
    cohort : list
        Member IDs to include (from churn labels).

    Returns
    -------
    pd.DataFrame
        One row per member with raw aggregate features.
    """
    pts_acc_col = resolve_col(activity, ["total_points_accumulated", POINTS_ACC], POINTS_ACC)
    pts_red_col = resolve_col(activity, ["total_points_redeemed",    POINTS_RED], POINTS_RED)
    dist_col    = resolve_col(activity, [DISTANCE_COL, "distance_flown", "km_flown"], DISTANCE_COL)

    act = activity[activity[KEY].isin(cohort)].copy()

    # ── Aggregate stats ───────────────────────────────────────────────────
    agg_dict: dict = {}
    for col, fns in [
        (FLIGHTS_COL,  ["sum", "mean", "max", "std"]),
        (pts_acc_col,  ["sum", "mean"]),
        (pts_red_col,  ["sum", "mean"]),
        (dist_col,     ["sum", "mean"]),
    ]:
        if col in act.columns:
            agg_dict[col] = fns

    raw = act.groupby(KEY).agg(agg_dict)
    raw.columns = ["_".join(c).strip() for c in raw.columns]
    raw = raw.reset_index()

    # ── Months with at least 1 flight ─────────────────────────────────────
    months_active = (
        act[act[FLIGHTS_COL] > 0]
        .groupby(KEY).size()
        .reset_index(name="months_with_flights")
    )

    # ── Tenure ────────────────────────────────────────────────────────────
    tenure = act.groupby(KEY)["period"].agg(
        first_month="min", last_month="max"
    ).reset_index()
    tenure["tenure_months"] = (
        (tenure["last_month"].dt.year  - tenure["first_month"].dt.year)  * 12
        + (tenure["last_month"].dt.month - tenure["first_month"].dt.month)
        + 1
    )

    # ── Combine ───────────────────────────────────────────────────────────
    out = (
        pd.DataFrame({KEY: cohort})
        .merge(raw,           on=KEY, how="left")
        .merge(months_active, on=KEY, how="left")
        .merge(tenure[[KEY, "tenure_months"]], on=KEY, how="left")
    )
    out["months_with_flights"] = out["months_with_flights"].fillna(0)
    out["activity_rate"] = (
        out["months_with_flights"] / out["tenure_months"].replace(0, np.nan)
    ).fillna(0)

    new_cols = [c for c in out.columns if c != KEY]
    for c in new_cols:
        _register(c)

    return out.fillna(0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Hyperbolic Discounting  (Family 2a — behavioural economics)
# ═══════════════════════════════════════════════════════════════════════════

def build_hyperbolic_features(
    activity: pd.DataFrame,
    cutoff: pd.Timestamp = PREDICTION_CUTOFF,
) -> pd.DataFrame:
    """
    Recency-weighted engagement scores using a hyperbolic decay function.

    Behavioural basis: Laibson (1997) — people weight recent activity
    disproportionately more than distant activity.  A member who flew
    10 times last month matters far more than one who flew 10 times
    two years ago.

    Features produced
    -----------------
    hyperbolic_flight_score   : recency-weighted total flights (12m window)
    hyperbolic_distance_score : recency-weighted total distance (if available)
    flights_last_3m           : raw flight count, last 3 months
    flights_last_6m           : raw flight count, last 6 months
    flights_last_12m          : raw flight count, last 12 months
    recency_ratio             : flights_last_3m / flights_last_12m
                                < 0.25 → declining trajectory

    Parameters
    ----------
    activity : pd.DataFrame
        Leakage-safe monthly activity table.
    cutoff : pd.Timestamp
        Prediction boundary.

    Returns
    -------
    pd.DataFrame  (one row per KEY)
    """
    dist_col = resolve_col(activity, [DISTANCE_COL, "distance_flown", "km_flown"], DISTANCE_COL)

    act_12m = activity[
        activity["period"] >= cutoff - pd.DateOffset(months=11)
    ].copy()

    act_12m["months_ago"] = (
        (cutoff.year  - act_12m["period"].dt.year)  * 12
        + (cutoff.month - act_12m["period"].dt.month)
    )
    act_12m["rw"]              = act_12m["months_ago"].apply(recency_weight)
    act_12m["weighted_flights"] = act_12m[FLIGHTS_COL] * act_12m["rw"]

    hyp_agg: dict = {"weighted_flights": "sum"}
    if dist_col in act_12m.columns:
        act_12m["weighted_distance"] = act_12m[dist_col] * act_12m["rw"]
        hyp_agg["weighted_distance"] = "sum"

    out = (
        act_12m.groupby(KEY).agg(hyp_agg).reset_index()
        .rename(columns={
            "weighted_flights" : "hyperbolic_flight_score",
            "weighted_distance": "hyperbolic_distance_score",
        })
    )

    # Rolling windows
    for window, label in [(3, "last_3m"), (6, "last_6m"), (12, "last_12m")]:
        wd = activity[activity["period"] > cutoff - pd.DateOffset(months=window)]
        wa = (
            wd.groupby(KEY)[FLIGHTS_COL].sum().reset_index()
            .rename(columns={FLIGHTS_COL: f"flights_{label}"})
        )
        out = out.merge(wa, on=KEY, how="left")

    out["flights_last_3m"]  = out.get("flights_last_3m",  pd.Series(0, index=out.index)).fillna(0)
    out["flights_last_6m"]  = out.get("flights_last_6m",  pd.Series(0, index=out.index)).fillna(0)
    out["flights_last_12m"] = out.get("flights_last_12m", pd.Series(0, index=out.index)).fillna(0)

    out["recency_ratio"] = (
        out["flights_last_3m"] / out["flights_last_12m"].replace(0, np.nan)
    ).fillna(0)

    for c in [col for col in out.columns if col != KEY]:
        out[c] = out[c].fillna(0)
        _register(c)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. Loss Aversion  (Family 2b — behavioural economics)
# ═══════════════════════════════════════════════════════════════════════════

def build_loss_aversion_features(activity: pd.DataFrame,
                                  cutoff: pd.Timestamp = PREDICTION_CUTOFF) -> pd.DataFrame:
    """
    Points hoarding and redemption trend features.

    Behavioural basis: Kahneman & Tversky (1979) — people are roughly
    2x more sensitive to losses than equivalent gains.  Members who
    accumulate but never redeem are afraid to 'spend' something they
    feel they already own — a classic loss-aversion signal.

    Features produced
    -----------------
    total_pts_accumulated    : lifetime points earned (up to cutoff)
    total_pts_redeemed       : lifetime points spent
    net_unredeemed_points    : accumulated − redeemed (clipped ≥ 0)
    redemption_ratio         : redeemed / accumulated  (0–1)
    pts_acc_last_6m          : points accumulated in last 6 months
    pts_red_last_6m          : points redeemed   in last 6 months
    recent_redemption_ratio  : redemption ratio for last 6 months only
    redemption_trend         : recent_ratio − overall_ratio
                               negative → hoarding more recently (risk signal)
    loss_aversion_score      : composite  (normalised unredeemed) × (1 − ratio)
                               high = lots of unspent points AND low engagement

    Parameters
    ----------
    activity  : leakage-safe monthly activity table
    cutoff    : prediction boundary

    Returns
    -------
    pd.DataFrame  (one row per KEY)
    """
    pts_acc_col = resolve_col(activity, ["total_points_accumulated", POINTS_ACC], POINTS_ACC)
    pts_red_col = resolve_col(activity, ["total_points_redeemed",    POINTS_RED], POINTS_RED)

    # Lifetime totals
    totals = (
        activity.groupby(KEY)
        .agg({pts_acc_col: "sum", pts_red_col: "sum"})
        .reset_index()
        .rename(columns={pts_acc_col: "total_pts_accumulated",
                         pts_red_col: "total_pts_redeemed"})
    )
    totals["net_unredeemed_points"] = (
        totals["total_pts_accumulated"] - totals["total_pts_redeemed"]
    ).clip(lower=0)
    totals["redemption_ratio"] = (
        totals["total_pts_redeemed"] / totals["total_pts_accumulated"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # Recent 6-month totals
    recent = (
        activity[activity["period"] > cutoff - pd.DateOffset(months=6)]
        .groupby(KEY)
        .agg({pts_acc_col: "sum", pts_red_col: "sum"})
        .reset_index()
        .rename(columns={pts_acc_col: "pts_acc_last_6m",
                         pts_red_col: "pts_red_last_6m"})
    )
    recent["recent_redemption_ratio"] = (
        recent["pts_red_last_6m"] / recent["pts_acc_last_6m"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    out = totals.merge(recent, on=KEY, how="left").fillna(0)
    out["redemption_trend"] = out["recent_redemption_ratio"] - out["redemption_ratio"]

    # Loss aversion composite score
    scaler = MinMaxScaler()
    out["_pts_norm"]          = scaler.fit_transform(out[["net_unredeemed_points"]])
    out["loss_aversion_score"] = out["_pts_norm"] * (1 - out["redemption_ratio"])
    out.drop(columns=["_pts_norm"], inplace=True)

    loss_cols = [
        "total_pts_accumulated", "total_pts_redeemed", "net_unredeemed_points",
        "redemption_ratio", "pts_acc_last_6m", "pts_red_last_6m",
        "recent_redemption_ratio", "redemption_trend", "loss_aversion_score",
    ]
    for c in loss_cols:
        _register(c)

    return out[[KEY] + loss_cols]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Status Quo Bias  (Family 2c — behavioural economics)
# ═══════════════════════════════════════════════════════════════════════════

def build_status_quo_features(
    activity: pd.DataFrame,
    members: pd.DataFrame,
    cutoff: pd.Timestamp = PREDICTION_CUTOFF,
) -> pd.DataFrame:
    """
    Tier rank and flight trajectory — operationalising status quo bias.

    Behavioural basis: Samuelson & Zeckhauser (1988) — people accept
    the current state as the default.  A member stuck in the same tier
    for years with declining flights is not loyal; they are inert.
    Inertia predicts disengagement better than tier level alone.

    Features produced
    -----------------
    tier_rank            : numeric rank of loyalty card tier
    flights_first_6m_avg : avg monthly flights in first 6 recorded months
    flights_last_6m_avg  : avg monthly flights in last 6 months before cutoff
    flight_trajectory    : last_6m_avg − first_6m_avg
                           negative = flying less now than at start (inertia signal)

    Parameters
    ----------
    activity : leakage-safe monthly activity table
    members  : member profile table containing CARD_COL
    cutoff   : prediction boundary

    Returns
    -------
    pd.DataFrame  (one row per KEY, empty if CARD_COL not found)
    """
    if CARD_COL not in members.columns:
        return pd.DataFrame({KEY: members[KEY].unique()})

    TIER_ORDER = {
        # Canadian airline tiers (adjust if your data uses different names)
        "star": 1, "nova": 2, "aurora": 3,
        "blue": 1, "silver": 2, "gold": 3, "platinum": 4,
        "bronze": 1, "standard": 1,
    }

    tier_df = members[[KEY, CARD_COL]].copy()
    tier_df["tier_rank"] = (
        tier_df[CARD_COL].str.lower().str.strip().map(TIER_ORDER)
    )
    # Fallback: label-encode if names not in map
    if tier_df["tier_rank"].isnull().any():
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        tier_df["tier_rank"] = le.fit_transform(
            tier_df[CARD_COL].fillna("unknown")
        )

    first_6m = (
        activity.sort_values("period").groupby(KEY).head(6)
        .groupby(KEY)[FLIGHTS_COL].mean().reset_index()
        .rename(columns={FLIGHTS_COL: "flights_first_6m_avg"})
    )
    last_6m = (
        activity[activity["period"] > cutoff - pd.DateOffset(months=6)]
        .groupby(KEY)[FLIGHTS_COL].mean().reset_index()
        .rename(columns={FLIGHTS_COL: "flights_last_6m_avg"})
    )

    traj = first_6m.merge(last_6m, on=KEY, how="outer").fillna(0)
    traj["flight_trajectory"] = traj["flights_last_6m_avg"] - traj["flights_first_6m_avg"]

    out = tier_df.merge(traj, on=KEY, how="left").fillna(0)

    for c in ["tier_rank", "flights_first_6m_avg", "flights_last_6m_avg", "flight_trajectory"]:
        _register(c)

    return out[[KEY, "tier_rank", "flights_first_6m_avg",
                "flights_last_6m_avg", "flight_trajectory"]]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Consistency  (Family 2d — commitment & regularity)
# ═══════════════════════════════════════════════════════════════════════════

def build_consistency_features(
    activity: pd.DataFrame,
    cutoff: pd.Timestamp = PREDICTION_CUTOFF,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Flight regularity, longest active streak, and months since last flight.

    Behavioural basis: Cialdini (1984) — members who fly on a consistent
    schedule have made an implicit commitment; disrupting it requires
    deliberate action.  Erratic flyers (high CV) disengage more easily.

    Features produced
    -----------------
    flight_cv                 : coefficient of variation of monthly flights
                                0 = perfectly consistent, high = erratic
    longest_active_streak     : longest run of consecutive active months
    months_since_last_flight  : months between last flight and cutoff

    Parameters
    ----------
    activity : leakage-safe monthly activity table
    cutoff   : prediction boundary
    verbose  : print progress (streak computation is the slowest step)

    Returns
    -------
    pd.DataFrame  (one row per KEY)
    """
    # Coefficient of variation
    cv = (
        activity.groupby(KEY)[FLIGHTS_COL]
        .agg(mean_f="mean", std_f="std").reset_index()
    )
    cv["flight_cv"] = (cv["std_f"] / cv["mean_f"].replace(0, np.nan)).fillna(0)

    # Longest consecutive active streak
    def _longest_streak(grp: pd.DataFrame) -> int:
        active = sorted(
            grp[grp[FLIGHTS_COL] > 0]["period"].dt.to_period("M").unique()
        )
        if not active:
            return 0
        max_s = s = 1
        for i in range(1, len(active)):
            if (active[i] - active[i - 1]).n == 1:
                s += 1
                max_s = max(max_s, s)
            else:
                s = 1
        return max_s

    if verbose:
        print("Computing longest active streaks (~30s)...")

    streak = (
        activity.groupby(KEY)
        .apply(_longest_streak)
        .reset_index()
        .rename(columns={0: "longest_active_streak"})
    )

    # Months since last flight
    last = (
        activity[activity[FLIGHTS_COL] > 0]
        .groupby(KEY)["period"].max().reset_index()
        .rename(columns={"period": "last_flight_date"})
    )
    last["months_since_last_flight"] = (
        (cutoff.year  - last["last_flight_date"].dt.year)  * 12
        + (cutoff.month - last["last_flight_date"].dt.month)
    ).clip(lower=0)

    out = (
        cv[[KEY, "flight_cv"]]
        .merge(streak,  on=KEY, how="outer")
        .merge(last[[KEY, "months_since_last_flight"]], on=KEY, how="outer")
        .fillna(0)
    )

    for c in ["flight_cv", "longest_active_streak", "months_since_last_flight"]:
        _register(c)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6. Fixed-Effects Deviation  (Family 3a — econometric)
# ═══════════════════════════════════════════════════════════════════════════

def build_fixed_effects_features(
    activity: pd.DataFrame,
    members: pd.DataFrame,
) -> pd.DataFrame:
    """
    Member deviation from their tier-year cohort average.

    Econometric basis: fixed-effects regression controls for unobserved
    group-level factors.  A member flying 2x/month in a cohort averaging 5
    is underperforming.  The same member in a cohort averaging 1 is
    outperforming.  Absolute counts mislead; deviation from group
    expectation is the true signal.

    Features produced
    -----------------
    mean_fe_deviation : member's mean monthly deviation from cohort avg
    std_fe_deviation  : volatility of that deviation

    Parameters
    ----------
    activity : leakage-safe monthly activity table
    members  : member profile table containing CARD_COL + YEAR_COL

    Returns
    -------
    pd.DataFrame  (one row per KEY, empty DataFrame if CARD_COL missing)
    """
    if CARD_COL not in members.columns:
        return pd.DataFrame({KEY: activity[KEY].unique()})

    tier_map      = members[[KEY, CARD_COL]].drop_duplicates(KEY)
    act_tier      = activity.merge(tier_map, on=KEY, how="left")
    cohort_avg    = (
        act_tier.groupby([CARD_COL, YEAR_COL])[FLIGHTS_COL]
        .mean().reset_index()
        .rename(columns={FLIGHTS_COL: "cohort_avg_flights"})
    )
    act_base      = act_tier.merge(cohort_avg, on=[CARD_COL, YEAR_COL], how="left")
    act_base["fe_deviation"] = (
        act_base[FLIGHTS_COL] - act_base["cohort_avg_flights"].fillna(0)
    )
    out = (
        act_base.groupby(KEY)["fe_deviation"]
        .agg(mean_fe_deviation="mean", std_fe_deviation="std")
        .reset_index()
        .fillna(0)
    )

    for c in ["mean_fe_deviation", "std_fe_deviation"]:
        _register(c)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 7. Seasonal Features  (Family 3b — econometric)
# ═══════════════════════════════════════════════════════════════════════════

def build_seasonal_features(
    activity: pd.DataFrame,
    cutoff: pd.Timestamp = PREDICTION_CUTOFF,
    seasonal_volatility_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Seasonal deviation from personal baseline and timing of last flight.

    Features produced
    -----------------
    seasonal_volatility        : avg deviation from member's own monthly baseline
                                 (pass pre-computed df from nb03, or recompute here)
    last_flight_quarter        : calendar quarter of last flight (1–4)
    last_flight_month_of_year  : calendar month  of last flight (1–12)

    Parameters
    ----------
    activity               : leakage-safe monthly activity table
    cutoff                 : prediction boundary
    seasonal_volatility_df : optional pre-computed DataFrame with columns
                             [KEY, 'seasonal_volatility'] from notebook 03.
                             If None, volatility is recomputed here.

    Returns
    -------
    pd.DataFrame  (one row per KEY)
    """
    # ── Seasonal volatility ───────────────────────────────────────────────
    if seasonal_volatility_df is not None:
        sv = seasonal_volatility_df[[KEY, "seasonal_volatility"]].copy()
    else:
        act_12m = activity[
            activity["period"] >= cutoff - pd.DateOffset(months=11)
        ].copy()
        member_monthly = (
            act_12m.groupby([KEY, MONTH_COL])[FLIGHTS_COL].mean().reset_index()
        )
        member_avg = (
            act_12m.groupby(KEY)[FLIGHTS_COL].mean()
            .reset_index().rename(columns={FLIGHTS_COL: "member_avg"})
        )
        mm = member_monthly.merge(member_avg, on=KEY, how="left")
        mm["deviation"] = abs(mm[FLIGHTS_COL] - mm["member_avg"])
        sv = (
            mm.groupby(KEY)["deviation"].mean().reset_index()
            .rename(columns={"deviation": "seasonal_volatility"})
        )

    # ── Last flight timing ─────────────────────────────────────────────────
    last = (
        activity[activity[FLIGHTS_COL] > 0]
        .groupby(KEY)["period"].max().reset_index()
    )
    last["last_flight_quarter"]      = last["period"].dt.quarter
    last["last_flight_month_of_year"] = last["period"].dt.month

    out = sv.merge(
        last[[KEY, "last_flight_quarter", "last_flight_month_of_year"]],
        on=KEY, how="left",
    ).fillna(0)

    for c in ["seasonal_volatility", "last_flight_quarter", "last_flight_month_of_year"]:
        _register(c)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 8. Master Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_all_features(
    activity_raw: pd.DataFrame,
    members: pd.DataFrame,
    cohort: list,
    cutoff: pd.Timestamp = PREDICTION_CUTOFF,
    seasonal_volatility_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all feature families and return the final feature matrix.

    This is the single entry point for both the notebook pipeline and
    the Streamlit dashboard.  Calling this function guarantees that
    features are computed identically in both contexts.

    Parameters
    ----------
    activity_raw            : raw monthly activity table (NOT yet filtered)
    members                 : member profile / loyalty history table
    cohort                  : list of loyalty_number values to include
    cutoff                  : leakage boundary  (default = PREDICTION_CUTOFF)
    seasonal_volatility_df  : optional pre-computed seasonal volatility from nb03
    verbose                 : print progress messages

    Returns
    -------
    pd.DataFrame
        One row per member in cohort, columns = all engineered features
        (plus KEY column).  No nulls — all filled with 0 or median.
    """
    if verbose:
        print(f"build_all_features: {len(cohort):,} members, cutoff={cutoff.strftime('%Y-%m')}")

    # ── Step 0: enforce leakage boundary once ────────────────────────────
    activity = enforce_leakage_boundary(activity_raw, cutoff)
    activity = activity[activity[KEY].isin(cohort)].copy()

    # ── Step 1–7: build each feature family ──────────────────────────────
    if verbose: print("  [1/7] Raw activity features...")
    f1 = build_raw_activity_features(activity, cohort)

    if verbose: print("  [2/7] Hyperbolic discounting features...")
    f2 = build_hyperbolic_features(activity, cutoff)

    if verbose: print("  [3/7] Loss aversion features...")
    f3 = build_loss_aversion_features(activity, cutoff)

    if verbose: print("  [4/7] Status quo bias features...")
    f4 = build_status_quo_features(activity, members, cutoff)

    if verbose: print("  [5/7] Consistency features...")
    f5 = build_consistency_features(activity, cutoff, verbose=verbose)

    if verbose: print("  [6/7] Fixed-effects features...")
    f6 = build_fixed_effects_features(activity, members)

    if verbose: print("  [7/7] Seasonal features...")
    f7 = build_seasonal_features(activity, cutoff, seasonal_volatility_df)

    # ── Step 8: merge all families ────────────────────────────────────────
    base = pd.DataFrame({KEY: cohort})
    for fi in [f1, f2, f3, f4, f5, f6, f7]:
        if fi is not None and not fi.empty and KEY in fi.columns:
            base = base.merge(fi, on=KEY, how="left")

    # ── Step 9: demographic columns from members table ────────────────────
    demo_cols = [c for c in [
        CLV_COL, CARD_COL, "salary", "province",
        "marital_status", "education", "gender", "enrollment_year",
    ] if c in members.columns]

    if demo_cols:
        from sklearn.preprocessing import LabelEncoder
        demo_df = members[[KEY] + demo_cols].drop_duplicates(KEY).copy()

        # Label-encode categoricals
        for col in demo_df.select_dtypes(include="object").columns:
            if col == KEY:
                continue
            le = LabelEncoder()
            demo_df[col] = le.fit_transform(demo_df[col].astype(str).fillna("unknown"))
            _register(col)

        if CLV_COL in demo_df.columns:
            _register(CLV_COL)

        base = base.merge(demo_df, on=KEY, how="left")

    # CLV percentile rank
    if CLV_COL in base.columns:
        base["clv_percentile"] = base[CLV_COL].rank(pct=True)
        _register("clv_percentile")

    # ── Step 10: final null fill ──────────────────────────────────────────
    for col in base.columns:
        if col == KEY:
            continue
        if base[col].dtype == object:
            base[col] = base[col].fillna("unknown")
        else:
            base[col] = base[col].replace([np.inf, -np.inf], np.nan)
            base[col] = base[col].fillna(base[col].median())

    if verbose:
        print(f"  Done. Feature matrix: {base.shape[0]:,} rows × {base.shape[1]} columns")

    return base
