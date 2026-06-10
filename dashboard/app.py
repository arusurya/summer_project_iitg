import sys
from pathlib import Path
# ── Make src/ importable ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.model import (
    load_model,
    load_explainer,
    load_meta,
    predict_churn_proba,
    predict_risk_bucket,
    lookup_member,
    get_model_summary,
    compute_behavioral_clv,
)
from src.segments import (
    build_segment_summary,
    build_priority_list,
    build_roi_table,
    SEGMENT_COLORS,
    RISK_COLORS,
    RISK_BUCKET_ORDER,
    NUDGE_DEFINITIONS,
)

# Paths 
PROC  = ROOT / "data" / "processed"
MODEL = ROOT / "models"

# Page config
st.set_page_config(
    page_title="Loyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
#plotly template
PLOTLY_TEMPLATE = "plotly_dark"

APP_COLORS = {
    "risk": "#E45756",
    "loyal": "#4CAF7D",
    "moderate": "#9AA4B2",
    "accent": "#4C9BE8",
}
PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

def style_plotly(fig, height=460):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E8EDF2", size=13),
        margin=dict(l=20, r=20, t=55, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(gridcolor="#2A2F3A", zerolinecolor="#3A4352")
    fig.update_yaxes(gridcolor="#2A2F3A", zerolinecolor="#3A4352")
    return fig

# Custom CSS

st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #0E1117;
    color: #E8EDF2;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #17172A;
    border-right: 1px solid #2A2F3A;
}

[data-testid="stSidebar"] * {
    color: #E8EDF2 !important;
}

/* Main text */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: #E8EDF2;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #161A23;
    border: 1px solid #2A2F3A;
    border-radius: 8px;
    padding: 14px 16px;
}

div[data-testid="metric-container"] label {
    color: #AAB4C0 !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F5F7FA !important;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background-color: #1B202A !important;
    color: #F5F7FA !important;
    border-color: #303846 !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background-color: #1B202A;
    color: #F5F7FA;
    border: 1px solid #3A4352;
    border-radius: 8px;
}

.stButton button:hover,
.stDownloadButton button:hover {
    border-color: #4C9BE8;
    color: #FFFFFF;
}

/* Risk badge colours */
.badge-critical {
    background: #E45756;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
}

.badge-high {
    background: #F29E4C;
    color: #111827;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
}

.badge-medium {
    background: #F2C94C;
    color: #111827;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
}

.badge-low {
    background: #4C9BE8;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Nudge card */
.nudge-card {
    background: #162235;
    border: 1px solid #28405F;
    border-left: 4px solid #4C9BE8;
    border-radius: 8px;
    padding: 16px 20px;
    margin-top: 12px;
}

.nudge-card h4 {
    margin: 0 0 8px 0;
    color: #DCEBFF;
}

.nudge-card p {
    margin: 4px 0;
    font-size: 0.9rem;
    color: #D3D9E3;
}

/* Section headers */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #DCEBFF;
    border-bottom: 2px solid #4C9BE8;
    padding-bottom: 5px;
    margin-bottom: 14px;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #161A23;
    border: 1px solid #2A2F3A;
    border-radius: 8px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #11151D;
    border: 1px solid #2A2F3A;
    border-radius: 8px;
}

/* Info boxes */
[data-testid="stAlert"] {
    background-color: #162235;
    color: #DCEBFF;
    border: 1px solid #28405F;
}

/* Code blocks */
code {
    background-color: #1B202A !important;
    color: #F5F7FA !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Data & model loading  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model artefacts…")
def _load_artefacts():
    model     = load_model(MODEL)
    meta      = load_meta(MODEL)
    try:
        explainer = load_explainer(MODEL)
    except FileNotFoundError:
        explainer = None
    return model, meta, explainer


@st.cache_data(show_spinner="Loading member data…")
def _load_data():
    features  = pd.read_csv(PROC / "04_features.csv")
    segments  = pd.read_csv(PROC / "06_segments.csv")
    playbook  = pd.read_csv(PROC / "07_retention_playbook.csv")
    activity  = pd.read_csv(PROC / "02_activity_clean.csv")
    activity["period"] = pd.to_datetime(
        activity["year"].astype(str) + "-" +
        activity["month"].astype(str).str.zfill(2)
    )
    return features, segments, playbook, activity


def _risk_badge(bucket: str) -> str:
    cls = {
        "Critical (>75%)" : "badge-critical",
        "High (50-75%)"   : "badge-high",
        "Medium (25-50%)" : "badge-medium",
        "Low (<25%)"      : "badge-low",
    }.get(bucket, "badge-low")
    return f'<span class="{cls}">{bucket}</span>'


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Loyalty Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Member Lookup", "Segment Map",
         "Cohort Trends", "Nudge Engine"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Model summary in sidebar
    try:
        summary = get_model_summary(MODEL)
        st.markdown("**Model performance**")
        st.markdown(f"XGBoost CV AUC &nbsp; `{summary['xgb_auc_pct']}`")
        st.markdown(f"Decision threshold &nbsp; `{summary['threshold_pct']}`")
        st.markdown(f"Overall churn rate &nbsp; `{summary['churn_rate_pct']}`")
        st.markdown(f"Members analysed &nbsp; `{summary['n_members']:,}`")
    except Exception:
        st.info("Run notebooks 01–07 to populate model stats.")

    st.markdown("---")
    st.caption("Behavioural Airline Loyalty Intelligence")


# ══════════════════════════════════════════════════════════════════════════
# Load artefacts
# ══════════════════════════════════════════════════════════════════════════

try:
    model, meta, explainer = _load_artefacts()
    features, segments, playbook, activity = _load_data()
    FEATURE_COLS = meta["feature_cols"]
    DATA_LOADED  = True
except Exception as e:
    st.error(f"**Data not ready.** Run all notebooks first, then reload this page.\n\n`{e}`")
    st.stop()
    DATA_LOADED = False


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Member Lookup
# ══════════════════════════════════════════════════════════════════════════

if page == "Member Lookup":

    st.title("Member Lookup")
    st.caption(
        "Enter a loyalty number to see churn risk, behavioural signals, "
        "SHAP explanation, and the recommended retention action."
    )

    col_input, col_hint = st.columns([2, 3])
    with col_input:
        member_id_input = st.text_input(
            "Loyalty number", placeholder="e.g. 123456",
            label_visibility="collapsed"
        )
    with col_hint:
        if st.button("Show a random high-risk member"):
            hi_risk = playbook[
                playbook["risk_bucket"].isin(["Critical (>75%)", "High (50-75%)"])
            ]
            if not hi_risk.empty:
                member_id_input = str(hi_risk.sample(1)["loyalty_number"].values[0])
                st.session_state["lookup_id"] = member_id_input

    # Use session state so random button works
    if "lookup_id" in st.session_state and not member_id_input:
        member_id_input = st.session_state["lookup_id"]

    if not member_id_input:
        st.info("Enter a loyalty number above or click **Show a random high-risk member**.")
        st.stop()

    # Try numeric conversion
    try:
        member_id = int(member_id_input)
    except ValueError:
        member_id = member_id_input

    with st.spinner("Looking up member…"):
        result = lookup_member(
            member_id=member_id,
            features_df=features,
            segments_df=segments,
            playbook_df=playbook,
            model=model,
            explainer=explainer,
            feature_cols=FEATURE_COLS,
            model_dir=MODEL,
        )

    if not result["found"]:
        st.error(f"Member `{member_id}` not found. Check the loyalty number and try again.")
        st.stop()

    # ── Top KPI row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Churn probability",  f"{result['churn_pct']}%")
    c2.metric("CLV",                f"${result['clv']:,.0f}")
    c3.metric("Behavioral CLV",     f"${result['behavioral_clv']:,.0f}")
    c4.metric("Segment",            result["segment"])
    c5.metric("RFM segment",        result["rfm_segment"])

    st.markdown(
        f"**Risk level:** {_risk_badge(result['risk_bucket'])}",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    left, right = st.columns([1, 1])

    # ── Left: SHAP waterfall ──────────────────────────────────────────────
    with left:
        st.markdown('<p class="section-title">Why is this member at risk?</p>',
                    unsafe_allow_html=True)

        if not result["shap_waterfall"].empty:
            wf = result["shap_waterfall"].copy()
            wf_plot = wf.copy()
            wf_plot["direction"] = np.where(
                wf_plot["shap_value"] >= 0,
                "Increases churn risk",
                "Reduces churn risk"
            )
            wf_plot = wf_plot.sort_values("shap_value", ascending=True)

            fig = px.bar(
                wf_plot,
                x="shap_value",
                y="label",
                orientation="h",
                color="direction",
                color_discrete_map={
                    "Increases churn risk": "#E45756",
                    "Reduces churn risk": "#4C9BE8",
                },
                labels={
                    "shap_value": "Impact on churn risk",
                    "label": "",
                    "direction": "",
                },
                title=f"Member {member_id} churn drivers",
            )

            fig.add_vline(x=0, line_dash="dash", line_color="#9AA4B2")
            fig = style_plotly(fig, height=500)

            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            st.caption(
                f"Base rate: {result['explanation']['base_value']:.1%}  →  "
                f"This member: **{result['churn_pct']}%**"
            )
        else:
            st.info("SHAP explainer not available. Run notebook 05 to generate it.")

            # Fallback: show top features from predictions table
            if not result["shap_drivers"].empty:
                st.dataframe(result["shap_drivers"][
                    ["feature", "shap_value", "feature_value"]
                ].round(4), use_container_width=True)

    # ── Right: Nudge recommendation ───────────────────────────────────────
    with right:
        st.markdown('<p class="section-title">Recommended action</p>',
                    unsafe_allow_html=True)

        nudge = result["nudge"]
        if nudge:
            nudge_def = NUDGE_DEFINITIONS.get(nudge.get("nudge_type", ""), {})

            st.markdown(f"""
<div class="nudge-card">
  <h4> {nudge.get("nudge_type", "—")}</h4>
  <p><strong>Priority:</strong> {nudge.get("priority_tier", "—")}</p>
  <p><strong>Channel:</strong> {nudge.get("channel", "—")}</p>
  <p><strong>Timing:</strong> {nudge.get("timing", "—")}</p>
</div>
""", unsafe_allow_html=True)

            st.markdown("**Subject line**")
            st.code(nudge.get("subject_line", ""), language=None)

            st.markdown("**Message**")
            st.info(nudge.get("message_frame", ""))

            with st.expander("Why this intervention works (behavioural economics)"):
                st.markdown(f"**Mechanism:** {nudge.get('mechanism', '—')}")
                st.markdown(f"**Principle:** {nudge.get('be_principle', nudge_def.get('be_principle', '—'))}")
                st.markdown(f"**Expected lift:** {nudge.get('expected_lift', '—')}")
                st.markdown(f"**Success KPI:** {nudge.get('success_kpi', '—')}")
        else:
            st.info("No nudge record found for this member.")

    # ── Behavioural signal summary ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Behavioural signals</p>',
                unsafe_allow_html=True)

    sig_cols = [
        ("months_since_last_flight", "Months since last flight", "{:.0f}"),
        ("recency_ratio",            "Recency ratio (3m/12m)",   "{:.3f}"),
        ("loss_aversion_score",      "Loss aversion score",      "{:.3f}"),
        ("redemption_ratio",         "Redemption ratio",         "{:.3f}"),
        ("flight_trajectory",        "Flight trajectory",        "{:.2f}"),
        ("hyperbolic_flight_score",  "Hyperbolic flight score",  "{:.1f}"),
        ("seasonal_volatility",      "Seasonal volatility",      "{:.3f}"),
        ("tenure_months",            "Tenure (months)",          "{:.0f}"),
    ]

    member_row = features[features["loyalty_number"] == member_id]
    if not member_row.empty:
        sig_values = {}
        for col, label, fmt in sig_cols:
            if col in member_row.columns:
                val = member_row[col].values[0]
                sig_values[label] = fmt.format(val)

        if sig_values:
            sig_df = pd.DataFrame(
                list(sig_values.items()), columns=["Signal", "Value"]
            )
            n_per_row = 4
            rows = [sig_df.iloc[i:i+n_per_row] for i in range(0, len(sig_df), n_per_row)]
            for row_df in rows:
                row_cols = st.columns(len(row_df))
                for col_obj, (_, sig_row) in zip(row_cols, row_df.iterrows()):
                    col_obj.metric(sig_row["Signal"], sig_row["Value"])


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Segment Map
# ══════════════════════════════════════════════════════════════════════════

elif page == "Segment Map":

    st.title("Segment Map")
    st.caption("Churn risk vs Behavioral CLV — each dot is a member, coloured by segment.")

    seg_summary = build_segment_summary(segments)

    # ── Top metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total members",    f"{len(segments):,}")
    c2.metric("Segments",         f"{segments['cluster_label'].nunique()}")
    critical_n = (segments["risk_bucket"] == "Critical (>75%)").sum()
    c3.metric("Critical risk",    f"{critical_n:,}")
    total_bclv = segments["behavioral_clv"].sum() if "behavioral_clv" in segments.columns else 0
    c4.metric("Total behavioral CLV", f"${total_bclv:,.0f}")

    st.markdown("---")

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.markdown('<p class="section-title">Churn risk vs Behavioral CLV</p>',
                    unsafe_allow_html=True)

        plot_df = segments.copy()
        plot_df["churn_pct"] = plot_df["churn_prob"] * 100
        fig = px.scatter(
            plot_df,
            x="churn_pct",
            y="behavioral_clv",
            color="cluster_label",
            hover_data=["loyalty_number", "risk_bucket", "clv", "behavioral_clv"],
            labels={
                "churn_pct": "Churn probability (%)",
                "behavioral_clv": "Behavioral CLV ($)",
                "cluster_label": "Segment",
            },
            title="Churn risk vs Behavioral CLV",
        ) 

        fig.add_vline(x=50, line_dash="dash", line_color="#9AA4B2")

        fig.update_layout(
           template=PLOTLY_TEMPLATE,
           height=520,
           margin=dict(l=10, r=10, t=60, b=10),
       )

        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    with col_table:
        st.markdown('<p class="section-title">Segment summary</p>',
                    unsafe_allow_html=True)

        display_cols = [
            c for c in ["cluster_label", "n_members", "churn_rate",
                         "avg_churn_prob", "avg_clv", "avg_behavioral_clv",
                         "n_critical"]
            if c in seg_summary.columns
        ]
        fmt: dict = {}
        for c in display_cols:
            if "rate" in c or "prob" in c: fmt[c] = "{:.1%}"
            elif "clv" in c.lower():       fmt[c] = "${:,.0f}"
            elif c == "n_members":         fmt[c] = "{:,}"
        st.dataframe(
            seg_summary[display_cols].style.format(fmt),
            use_container_width=True, height=300,
        )

        # CLV gap chart
        if "avg_clv" in seg_summary.columns and "avg_behavioral_clv" in seg_summary.columns:
            st.markdown(
                '<p class="section-title" style="margin-top:16px">CLV gap by segment</p>',
                unsafe_allow_html=True
            )

            clv_gap = seg_summary[["cluster_label", "avg_clv", "avg_behavioral_clv"]].copy()
            clv_gap = clv_gap.melt(
                id_vars="cluster_label",
                value_vars=["avg_clv", "avg_behavioral_clv"],
                var_name="metric",
                value_name="value",
            )

            clv_gap["metric"] = clv_gap["metric"].map({
                "avg_clv": "Avg CLV",
                "avg_behavioral_clv": "Avg Behavioral CLV",
            })

            fig2 = px.bar(
                clv_gap,
                x="cluster_label",
                y="value",
                color="metric",
                barmode="group",
                labels={"cluster_label": "Segment", "value": "CLV ($)", "metric": ""},
                title="CLV gap by segment",
            )

            fig2.update_layout(
                template=PLOTLY_TEMPLATE,
                height=360,
                margin=dict(l=10, r=10, t=60, b=10),
            )

            st.plotly_chart(fig2, use_container_width=True)

            


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Cohort Trends
# ══════════════════════════════════════════════════════════════════════════

elif page == "Cohort Trends":

    st.title("Cohort Trends")
    st.caption(
        "Monthly flight activity over time, broken down by segment. "
        "Declining curves identify segments to prioritise."
    )

    # ── Filters ───────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        available_segs = sorted(segments["cluster_label"].dropna().unique())
        selected_segs  = st.multiselect(
            "Segments to show", available_segs,
            default=available_segs[:min(4, len(available_segs))],
        )
    with col_f2:
        metric_choice = st.selectbox(
            "Metric", ["total_flights", "points_accumulated", "distance"],
            format_func=lambda x: {
                "total_flights"     : "Flights per member",
                "points_accumulated": "Points accumulated per member",
                "distance"          : "Distance per member",
            }.get(x, x),
        )

    if not selected_segs:
        st.warning("Select at least one segment.")
        st.stop()

    # ── Merge activity with segment labels ────────────────────────────────
    seg_map    = segments[["loyalty_number", "cluster_label"]].drop_duplicates()
    act_segged = activity.merge(seg_map, on="loyalty_number", how="inner")
    act_segged = act_segged[act_segged["cluster_label"].isin(selected_segs)]

    metric_col = next(
        (c for c in [metric_choice, metric_choice + "_sum",
                      "total_flights"] if c in act_segged.columns),
        None,
    )

    if metric_col is None:
        st.warning(f"Column `{metric_choice}` not found in activity data.")
        st.stop()

    trend = (
        act_segged.groupby(["period", "cluster_label"])[metric_col]
        .mean().reset_index()
        .sort_values("period")
    )

    # ── Plot ──────────────────────────────────────────────────────────────
    trend_plot = trend.copy()

    fig = px.line(
       trend_plot,
       x="period",
       y=metric_col,
       color="cluster_label",
       markers=True,
       labels={
           "period": "Month",
            metric_col: metric_col.replace("_", " ").title() + " per member",
           "cluster_label": "Segment",
        },
        title="Monthly engagement trend by segment",
    ) 

    fig.add_vline(
        x=pd.Timestamp("2018-06-01"),
        line_dash="dash",
        line_color="#9AA4B2",
    ) 

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # ── Year-over-year comparison ─────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Year-on-year change by segment</p>',
                unsafe_allow_html=True)

    act_segged["year"] = act_segged["period"].dt.year
    yoy = (
        act_segged.groupby(["year", "cluster_label"])[metric_col]
        .mean().reset_index()
    )
    yoy_pivot = yoy.pivot(index="cluster_label", columns="year", values=metric_col)

    years = sorted(yoy_pivot.columns)
    if len(years) >= 2:
        yoy_pivot["YoY change (last 2 yrs)"] = (
            (yoy_pivot[years[-1]] - yoy_pivot[years[-2]])
            / yoy_pivot[years[-2]].replace(0, np.nan)
        )
        yoy_display = yoy_pivot.loc[
            yoy_pivot.index.isin(selected_segs)
        ].round(2)

        fmt_yoy = {c: "{:.2f}" for c in yoy_display.columns}
        fmt_yoy["YoY change (last 2 yrs)"] = "{:.1%}"
        st.dataframe(
            yoy_display.style.format(fmt_yoy)
            .background_gradient(subset=["YoY change (last 2 yrs)"],
                                  cmap="RdYlGn"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Nudge Engine
# ══════════════════════════════════════════════════════════════════════════

elif page == "Nudge Engine":

    st.title("Nudge Engine")
    st.caption(
        "Filter members by risk and segment, then export a prioritised "
        "outreach list with personalised message copy ready to send."
    )

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander(" Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            min_prob = st.slider(
                "Min churn probability", 0.0, 1.0, 0.5, 0.05,
                format="%.0f%%",
            )
        with fc2:
            available_segs = sorted(playbook["cluster_label"].dropna().unique())
            sel_segs = st.multiselect(
                "Segments", available_segs, default=available_segs,
            )
        with fc3:
            sel_buckets = st.multiselect(
                "Risk buckets", RISK_BUCKET_ORDER, default=RISK_BUCKET_ORDER[:2],
            )
        with fc4:
            top_n = st.number_input(
                "Max members to show", min_value=10,
                max_value=len(playbook), value=100, step=10,
            )

    # ── Build filtered list ───────────────────────────────────────────────
    plist = build_priority_list(
        playbook,
        n=int(top_n),
        min_churn_prob=float(min_prob),
        segments=sel_segs if sel_segs else None,
        risk_buckets=sel_buckets if sel_buckets else None,
    )

    # ── Summary metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Members in list", f"{len(plist):,}")

    if "behavioral_clv" in plist.columns:
        m2.metric("Behavioral CLV at risk",
                  f"${plist['behavioral_clv'].sum():,.0f}")
    elif "clv" in plist.columns:
        m2.metric("CLV at risk", f"${plist['clv'].sum():,.0f}")

    if "churn_prob" in plist.columns:
        m3.metric("Avg churn probability",
                  f"{plist['churn_prob'].mean():.1%}")

    if "nudge_type" in plist.columns:
        if "nudge_type" in plist.columns and not plist.empty:
            top_nudge = plist["nudge_type"].value_counts().index[0]
            m4.metric("Top nudge type", top_nudge[:22] + ("..." if len(top_nudge) > 22 else ""))
        else:
            m4.metric("Top nudge type", "—")

    st.markdown("---")

    # ── Priority list table ───────────────────────────────────────────────
    st.markdown('<p class="section-title">Priority outreach list</p>',
                unsafe_allow_html=True)

    if plist.empty:
        st.warning("No members match the current filters.")
    else:
        display_table_cols = [
            c for c in [
                "loyalty_number", "cluster_label", "risk_bucket",
                "churn_prob", "clv", "behavioral_clv",
                "priority_score", "nudge_type", "subject_line",
                "channel", "timing",
            ] if c in plist.columns
        ]
        fmt_table: dict = {}
        for c in display_table_cols:
            if c in ("churn_prob",):     fmt_table[c] = "{:.1%}"
            elif "clv" in c.lower():     fmt_table[c] = "${:,.0f}"
            elif c == "priority_score":  fmt_table[c] = "{:.1f}"

        st.dataframe(
            plist[display_table_cols].style.format(fmt_table),
            use_container_width=True,
            height=380,
        )

        # ── Export button ─────────────────────────────────────────────────
        csv_bytes = plist[display_table_cols].to_csv(index=False).encode()
        st.download_button(
            label="⬇Download outreach list as CSV",
            data=csv_bytes,
            file_name="priority_outreach_list.csv",
            mime="text/csv",
        )

    # ── Nudge breakdown chart ─────────────────────────────────────────────
    if not plist.empty and "nudge_type" in plist.columns:
        st.markdown("---")
        col_pie, col_roi = st.columns(2)

        with col_pie:
            st.markdown('<p class="section-title">Members per nudge type</p>',
                        unsafe_allow_html=True)
            nudge_counts = (
                plist["nudge_type"]
                .value_counts()
                .reset_index()
            )
            nudge_counts.columns = ["nudge_type", "members"]
            fig = px.bar(
                nudge_counts,
                x="members",
                y="nudge_type",
                orientation="h",
                labels={"members": "Members", "nudge_type": ""},
                title="Nudge type distribution",
            )
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                showlegend=False,
                height=360,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            
            fig.update_traces(marker_color="#4C9BE8")
            fig = style_plotly(fig, height=330)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            
        with col_roi:
            st.markdown('<p class="section-title">Estimated ROI by nudge</p>',
                        unsafe_allow_html=True)
            roi_df = build_roi_table(plist)
            roi_display = roi_df[
                ["nudge_type", "n_members", "clv_saved_est",
                 "campaign_cost_est", "roi_estimate"]
            ]
            st.dataframe(
                roi_display.style.format({
                    "clv_saved_est"    : "${:,.0f}",
                    "campaign_cost_est": "${:,.0f}",
                    "roi_estimate"     : "{:.0f}x",
                    "n_members"        : "{:,}",
                }),
                use_container_width=True,
                height=260,
            )

        # ── Per-nudge message preview ─────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">Message previews by nudge type</p>',
                    unsafe_allow_html=True)

        for nudge_key, nudge_def in NUDGE_DEFINITIONS.items():
            subset = plist[plist["nudge_type"] == nudge_key]
            if subset.empty:
                continue

            with st.expander(
                f"**{nudge_key}** — {len(subset):,} members  ·  "
                f"{nudge_def.get('priority_tier', '—')}",
                expanded=False,
            ):
                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown(f"**Channel:** {nudge_def.get('channel', '—')}")
                    st.markdown(f"**Timing:** {nudge_def.get('timing', '—')}")
                    st.markdown(f"**Expected lift:** {nudge_def.get('expected_lift', '—')}")
                    st.markdown(f"**KPI:** {nudge_def.get('success_kpi', '—')}")
                with c_right:
                    st.markdown(f"**Behavioural mechanism:**")
                    st.info(nudge_def.get("be_principle", "—"))

                # Show a sample subject line from the first matching member
                sample = subset.iloc[0]
                if "subject_line" in sample:
                    st.markdown("**Sample subject line:**")
                    st.code(sample["subject_line"], language=None)
                if "message_frame" in sample:
                    st.markdown("**Sample message:**")
                    st.markdown(f"> {sample['message_frame']}")
