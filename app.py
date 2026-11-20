"""AI Agent for Detecting Hidden Patient Deterioration Using ICU Vital Signs.

Streamlit web application entry point.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import streamlit as st

from agent import (
    VITALS,
    SCORES,
    SCORED_VITALS,
    REFERENCE_BANDS,
    news_scores,
    risk_category,
    get_risk_color,
    alert_at_hour,
    assess_patient_risk,
    compute_hourly_transitions,
    prepare_history,
)
from data_utils import (
    load_dataset,
    get_dataset_overview_stats,
    get_patient_records,
    get_available_patient_hours,
    get_reading_at_hour,
)
from visualizations import (
    GRAPH_TITLES,
    render_patient_graph,
    plot_dataset_vital_comparison,
    plot_dataset_correlation_heatmap,
    plot_dataset_news_vs_output,
)

# Page configuration
st.set_page_config(
    page_title="ICU Patient Deterioration Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern hospital clinical dashboard
CUSTOM_CSS = """
<style>
/* Modern Medical Typography & Colors */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Header Banner */
.main-header-banner {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #0f766e 100%);
    color: white;
    padding: 24px 30px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.15);
}
.main-header-banner h1 {
    color: white !important;
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header-banner p {
    color: #e0f2fe;
    font-size: 15px;
    margin: 0;
    font-weight: 500;
}

/* Clinical Metric Card */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.2s ease-in-out;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.08);
    border-color: #cbd5e1;
}
.metric-label {
    font-size: 12px;
    text-transform: uppercase;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.metric-val {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
}
.metric-unit {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 500;
    margin-left: 3px;
}
.metric-sub {
    font-size: 11px;
    margin-top: 5px;
    color: #475569;
}

/* Clinical Info Section Card */
.info-section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

/* Risk Badges */
.risk-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.risk-lower {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
}
.risk-mild {
    background-color: #fef9c3;
    color: #854d0e;
    border: 1px solid #fde047;
}
.risk-high {
    background-color: #ffedd5;
    color: #9a3412;
    border: 1px solid #fdba74;
}
.risk-very-high {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
}

/* Condition-Specific Blinking Indicators (Only ONE distinct color blinks per condition) */
@keyframes blink-beacon-green {
    0%, 100% {
        background-color: #22c55e;
        box-shadow: 0 0 5px #22c55e, 0 0 12px rgba(34, 197, 94, 0.8);
        opacity: 1;
        transform: scale(1.1);
    }
    50% {
        background-color: #bbf7d0;
        box-shadow: none;
        opacity: 0.35;
        transform: scale(0.88);
    }
}

@keyframes blink-beacon-yellow {
    0%, 100% {
        background-color: #eab308;
        box-shadow: 0 0 6px #eab308, 0 0 14px rgba(234, 179, 8, 0.85);
        opacity: 1;
        transform: scale(1.1);
    }
    50% {
        background-color: #fef08a;
        box-shadow: none;
        opacity: 0.35;
        transform: scale(0.88);
    }
}

@keyframes blink-beacon-orange {
    0%, 100% {
        background-color: #f97316;
        box-shadow: 0 0 7px #f97316, 0 0 16px rgba(249, 115, 22, 0.9);
        opacity: 1;
        transform: scale(1.12);
    }
    50% {
        background-color: #fed7aa;
        box-shadow: none;
        opacity: 0.35;
        transform: scale(0.88);
    }
}

@keyframes blink-beacon-red {
    0%, 100% {
        background-color: #ef4444;
        box-shadow: 0 0 8px #ef4444, 0 0 20px rgba(239, 68, 68, 0.95);
        opacity: 1;
        transform: scale(1.15);
    }
    50% {
        background-color: #fca5a5;
        box-shadow: none;
        opacity: 0.3;
        transform: scale(0.85);
    }
}

/* Pulsing Card Borders for Each Condition */
@keyframes pulse-card-green {
    0%, 100% { border-color: #22c55e; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.35); }
    50% { border-color: #86efac; box-shadow: 0 0 14px 4px rgba(34, 197, 94, 0.25); }
}

@keyframes pulse-card-yellow {
    0%, 100% { border-color: #eab308; box-shadow: 0 0 0 0 rgba(234, 179, 8, 0.4); }
    50% { border-color: #fde047; box-shadow: 0 0 16px 5px rgba(234, 179, 8, 0.3); }
}

@keyframes pulse-card-orange {
    0%, 100% { border-color: #f97316; box-shadow: 0 0 0 0 rgba(249, 115, 22, 0.45); }
    50% { border-color: #fb923c; box-shadow: 0 0 18px 6px rgba(249, 115, 22, 0.35); }
}

@keyframes pulse-card-red {
    0%, 100% { border-color: #ef4444; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.55); }
    50% { border-color: #f87171; box-shadow: 0 0 22px 8px rgba(239, 68, 68, 0.45); }
}

.beacon-dot {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}
.beacon-green { animation: blink-beacon-green 1.25s infinite ease-in-out; }
.beacon-yellow { animation: blink-beacon-yellow 1.1s infinite ease-in-out; }
.beacon-orange { animation: blink-beacon-orange 0.95s infinite ease-in-out; }
.beacon-red { animation: blink-beacon-red 0.8s infinite ease-in-out; }

.condition-alert-box {
    border-radius: 12px;
    padding: 18px 24px;
    margin: 20px 0;
    border: 2px solid;
}
.box-green {
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    border-color: #22c55e;
    animation: pulse-card-green 2.5s infinite ease-in-out;
    color: #14532d;
}
.box-yellow {
    background: linear-gradient(135deg, #fefce8 0%, #ffffff 100%);
    border-color: #eab308;
    animation: pulse-card-yellow 2.0s infinite ease-in-out;
    color: #713f12;
}
.box-orange {
    background: linear-gradient(135deg, #fff7ed 0%, #ffffff 100%);
    border-color: #f97316;
    animation: pulse-card-orange 1.6s infinite ease-in-out;
    color: #7c2d12;
}
.box-red {
    background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
    border-color: #ef4444;
    animation: pulse-card-red 1.2s infinite ease-in-out;
    color: #7f1d1d;
}

/* Graph Stepper Toolbar */
.graph-toolbar {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================
def init_session_state():
    """Ensure persistent session state variables exist."""
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Homepage"
    if "sidebar_nav_radio" not in st.session_state:
        st.session_state["sidebar_nav_radio"] = "Homepage"
    if "existing_graph_idx" not in st.session_state:
        st.session_state["existing_graph_idx"] = 1
    if "new_graph_idx" not in st.session_state:
        st.session_state["new_graph_idx"] = 1
    if "dataset_graph_idx" not in st.session_state:
        st.session_state["dataset_graph_idx"] = 1
    if "new_patients" not in st.session_state:
        st.session_state["new_patients"] = {}  # patient_id -> list of reading dicts


init_session_state()


def set_nav_page(page_name: str):
    """Callback triggered before widgets instantiate to set navigation state safely."""
    st.session_state["sidebar_nav_radio"] = page_name
    st.session_state["nav_page"] = page_name


# Load dataset once with caching
@st.cache_data
def get_data():
    return load_dataset()


try:
    dataset_df, dataset_std = get_data()
    overview_stats = get_dataset_overview_stats(dataset_df)
except Exception as e:
    st.error(f"Error loading ICU dataset: {e}")
    st.stop()


# ==============================================================================
# Sidebar Navigation
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏥 ICU Monitoring System")
    st.markdown(
        "<div style='font-size:12px;color:#64748b;margin-bottom:15px;'>"
        "Rule-Based Intelligent Agent for Detecting Hidden Deterioration</div>",
        unsafe_allow_html=True,
    )

    nav_options = [
        "Homepage",
        "Existing Patient Monitoring",
        "New Patient Monitoring",
        "Dataset Analysis",
        "About the Project",
    ]

    selected_page = st.radio(
        "Navigation",
        nav_options,
        key="sidebar_nav_radio",
    )
    if selected_page != st.session_state.get("nav_page"):
        st.session_state["nav_page"] = selected_page
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📈 Dataset Quick Facts")
    st.markdown(f"• **Records:** `{overview_stats['total_records']:,}`")
    st.markdown(f"• **Unique Stays:** `{overview_stats['unique_stays']:,}`")
    st.markdown(f"• **Output 0:** `{overview_stats['output_0']:,}` ({overview_stats['output_0_pct']}%)")
    st.markdown(f"• **Output 1:** `{overview_stats['output_1']:,}` ({overview_stats['output_1_pct']}%)")
    st.markdown(f"• **Columns:** `{overview_stats['num_columns']}`")

    st.markdown("---")
    st.caption("Prototypes for educational & academic research. Not a certified clinical medical device.")


# ==============================================================================
# PAGE 1: HOMEPAGE
# ==============================================================================
def render_homepage():
    st.markdown(
        """
        <div class="main-header-banner">
            <h1>AI Agent for Detecting Hidden Patient Deterioration</h1>
            <p>Rule-Based ICU Patient Monitoring and Early Warning System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Welcome to the clinical decision-support portal for ICU vital-sign monitoring. 
        This rule-based intelligent agent calculates continuous **Modified NEWS (National Early Warning Score)**,
        analyzes 3-hour deterioration trajectories, alerts on rapid vital-sign volatility, and provides comprehensive
        individualized visual trend diagnostics.
        """
    )

    st.markdown("### 📊 Dynamic Dataset Overview")
    st.caption("Live metrics read directly from the original 104,609 patient monitoring records.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-val">{overview_stats['total_records']:,}</div>
                <div class="metric-sub">Hourly Observations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ICU Stays</div>
                <div class="metric-val">{overview_stats['unique_stays']:,}</div>
                <div class="metric-sub">Unique Patients</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Columns</div>
                <div class="metric-val">{overview_stats['num_columns']}</div>
                <div class="metric-sub">Vitals & Scores</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Output = 0</div>
                <div class="metric-val" style="color:#0284c7;">{overview_stats['output_0']:,}</div>
                <div class="metric-sub">Stable ({overview_stats['output_0_pct']}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Output = 1</div>
                <div class="metric-val" style="color:#ea580c;">{overview_stats['output_1']:,}</div>
                <div class="metric-sub">Deteriorated ({overview_stats['output_1_pct']}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Portal Modules")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="info-section-card">
                <h3 style="color:#0284c7;margin-top:0;">🩺 Existing Patient Monitoring</h3>
                <p style="color:#475569;font-size:14px;">
                    Search and inspect any of the 2,153 patient stays in the dataset. Examine hourly vital signs,
                    step through 11 full-screen interactive graphs, assess 3-hour trend volatility, and detect early deterioration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Existing Patient Monitoring →",
            key="btn_home_existing",
            on_click=set_nav_page,
            args=("Existing Patient Monitoring",),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="info-section-card">
                <h3 style="color:#0f766e;margin-top:0;">📊 Dataset Analysis Dashboard</h3>
                <p style="color:#475569;font-size:14px;">
                    Explore population-level distributions, examine vital sign differences between Output 0 and Output 1,
                    inspect correlation heatmaps, and evaluate the relationship between NEWS and deterioration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Dataset Analysis →",
            key="btn_home_dataset",
            on_click=set_nav_page,
            args=("Dataset Analysis",),
            use_container_width=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="info-section-card">
                <h3 style="color:#0284c7;margin-top:0;">➕ New Patient Monitoring</h3>
                <p style="color:#475569;font-size:14px;">
                    Enter clinical vital signs for a newly admitted patient. Record multiple sequential hours in session
                    memory, observe automatic NEWS calculations, and run the complete 11-graph diagnostic sequence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open New Patient Monitoring →",
            key="btn_home_new",
            on_click=set_nav_page,
            args=("New Patient Monitoring",),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="info-section-card">
                <h3 style="color:#475569;margin-top:0;">ℹ️ About the Project</h3>
                <p style="color:#475569;font-size:14px;">
                    Learn about the rule-based clinical scoring rules, the 4 NEWS components, reference bands,
                    immediate warning criteria, and ethical considerations in healthcare decision-support agents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Learn About the Project →",
            key="btn_home_about",
            on_click=set_nav_page,
            args=("About the Project",),
            use_container_width=True,
        )


# ==============================================================================
# COMPONENT: Patient Details View
# ==============================================================================
def render_patient_record_cards(row: pd.Series, existing: bool = True):
    """Render medical cards for vital signs and calculated vs recorded NEWS scores."""
    calc_scores = news_scores(row)
    curr_score = calc_scores["modified_news2"]
    category = risk_category(curr_score)

    badge_class = {
        "LOWER RISK": "risk-lower",
        "MILD ABNORMALITY / INCREASED RISK": "risk-mild",
        "HIGH RISK": "risk-high",
        "VERY HIGH RISK": "risk-very-high",
    }.get(category, "risk-lower")

    badge_beacon = {
        "LOWER RISK": "beacon-green",
        "MILD ABNORMALITY / INCREASED RISK": "beacon-yellow",
        "HIGH RISK": "beacon-orange",
        "VERY HIGH RISK": "beacon-red",
    }.get(category, "beacon-green")

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
            <div>
                <span style="font-size:18px; font-weight:800; color:#0f172a;">Patient Stay ID: {int(row['patientunitstayid'])}</span>
                <span style="color:#64748b; font-size:14px; margin-left:12px;">Age: <b>{row['age']}</b> | Gender: <b>{row['gender']}</b> | Selected Hour: <b>Hour {int(row['hour'])}</b></span>
            </div>
            <div>
                <span class="risk-badge {badge_class}">
                    <span class="beacon-dot {badge_beacon}" style="width:10px;height:10px;margin-right:6px;"></span>
                    {category} (NEWS: {curr_score}/12)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Vital Signs Row
    v_cols = st.columns(7)
    for idx, (k, (lbl, unit)) in enumerate(VITALS.items()):
        val = float(row[k])
        is_news = k in SCORED_VITALS
        news_comp_score = calc_scores[SCORED_VITALS[k]] if is_news else None
        comp_str = f"NEWS: +{news_comp_score}" if is_news else "Ref Vital"

        with v_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label" title="{lbl}">{lbl[:14]}</div>
                    <div class="metric-val">{val:.1f}<span class="metric-unit">{unit}</span></div>
                    <div class="metric-sub" style="color:{'#dc2626' if news_comp_score and news_comp_score >= 2 else '#64748b'};">
                        {comp_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Scoring Comparison Section
    s_col1, s_col2 = st.columns([1, 1])

    with s_col1:
        st.markdown("##### 🧮 NEWS Component Breakdown (Agreed Rules)")
        comp_data = {
            "NEWS Component": [
                "Heart Rate (news_hr)",
                "Oxygen Saturation (news_spo2)",
                "Respiratory Rate (news_rr)",
                "Systolic BP (news_sbp)",
                "Total Recalculated Modified NEWS",
            ],
            "Observed Value": [
                f"{float(row['heart_rate']):.1f} bpm",
                f"{float(row['spo2']):.1f} %",
                f"{float(row['respiratory_rate']):.1f} br/min",
                f"{float(row['systolic_bp']):.1f} mmHg",
                "—",
            ],
            "Calculated Score": [
                calc_scores["news_hr"],
                calc_scores["news_spo2"],
                calc_scores["news_rr"],
                calc_scores["news_sbp"],
                f"{calc_scores['modified_news2']} / 12 theoretical",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

    with s_col2:
        if existing:
            st.markdown("##### 📋 Recorded vs Recalculated Values")
            rec_news = int(row["modified_news2"])
            rec_target = int(row["deterioration_target"])
            rec_curr = int(row.get("currently_deteriorated", 0))

            diff_detected = any(int(row[c]) != calc_scores[c] for c in SCORES) or rec_news != curr_score

            rec_data = {
                "Metric": [
                    "Recorded Modified NEWS (Dataset)",
                    "Recalculated Modified NEWS (Agreed Rules)",
                    "Recorded Deterioration Target (0/1)",
                    "Recorded Currently Deteriorated (0/1)",
                ],
                "Recorded Value": [
                    f"{rec_news} (Dataset range 0-4)",
                    f"{curr_score} / 12 (Theoretical range 0-12)",
                    f"{rec_target} ({'Deteriorated' if rec_target == 1 else 'Stable'})",
                    f"{rec_curr}",
                ],
            }
            st.dataframe(pd.DataFrame(rec_data), hide_index=True, use_container_width=True)
            if diff_detected:
                st.info(
                    "ℹ️ **Note on Dataset Differences:** Stored dataset scores max out at 4 due to the original pre-processing binning, "
                    "whereas the recalculated engine evaluates the complete theoretical clinical range (0 to 12)."
                )
        else:
            st.markdown("##### 📝 New Patient Metadata")
            meta_data = {
                "Parameter": ["Patient ID", "Age", "Gender", "Reading Hour", "Calculated Mean BP"],
                "Value": [
                    str(row["patientunitstayid"]),
                    f"{row['age']} yrs",
                    str(row["gender"]),
                    f"Hour {int(row['hour'])}",
                    f"{float(row['mean_bp']):.2f} mmHg",
                ],
            }
            st.dataframe(pd.DataFrame(meta_data), hide_index=True, use_container_width=True)


# ==============================================================================
# COMPONENT: Sequential Graph Viewer (One Large Graph at a Time)
# ==============================================================================
def render_sequential_graph_viewer(
    patient_df: pd.DataFrame,
    selected_hour: int,
    session_key: str,
    dataset_std: Dict[str, float],
    existing: bool = True,
):
    """Display ONE large complete interactive graph at a time with Previous/Next controls."""
    st.markdown("---")
    st.markdown("### 📈 Interactive Patient Diagnostic Graphs")
    st.caption("Inspect individual clinical curves one by one. Use the toolbar below to navigate.")

    curr_idx = st.session_state[session_key]

    # Graph Stepper Toolbar
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2.5, 1])

    with col_nav1:
        if st.button("◀ Previous Graph", key=f"{session_key}_prev", use_container_width=True):
            if curr_idx > 1:
                st.session_state[session_key] = curr_idx - 1
                st.rerun()
            else:
                st.session_state[session_key] = 11
                st.rerun()

    with col_nav2:
        # Quick jump dropdown
        selected_title = st.selectbox(
            "Jump to Graph",
            options=GRAPH_TITLES,
            index=curr_idx - 1,
            key=f"{session_key}_jump_select",
            label_visibility="collapsed",
        )
        new_idx = GRAPH_TITLES.index(selected_title) + 1
        if new_idx != curr_idx:
            st.session_state[session_key] = new_idx
            st.rerun()

    with col_nav3:
        if st.button("Next Graph ▶", key=f"{session_key}_next", use_container_width=True):
            if curr_idx < 11:
                st.session_state[session_key] = curr_idx + 1
                st.rerun()
            else:
                st.session_state[session_key] = 1
                st.rerun()

    # Progress Indicator
    st.markdown(
        f"<div style='text-align:center; font-weight:700; color:#0284c7; margin: 6px 0 16px 0;'>"
        f"Graph {curr_idx} of 11: {GRAPH_TITLES[curr_idx - 1]}</div>",
        unsafe_allow_html=True,
    )

    # Render Current Large Graph
    fig = render_patient_graph(
        graph_idx=curr_idx,
        patient_df=patient_df,
        selected_hour=selected_hour,
        dataset_std=dataset_std,
        existing=existing,
    )
    st.plotly_chart(fig, use_container_width=True)

    # If Graph 8 (Modified NEWS Score vs Hour) is displayed, show consecutive hourly changes callout
    if curr_idx == 8:
        history = prepare_history(patient_df, selected_hour)
        transitions = compute_hourly_transitions(history, selected_hour)
        if transitions:
            st.markdown("##### ⏱️ Consecutive Hourly Score Changes (Present Hour to Next Hour)")
            t_cols = st.columns(min(len(transitions), 4))
            for i, t in enumerate(transitions):
                col_i = t_cols[i % len(t_cols)]
                with col_i:
                    st.markdown(
                        f"""
                        <div style="background:#ffffff; border:1px solid #e2e8f0; border-top:3px solid {t['color']}; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                            <div style="font-size:12px; color:#64748b; font-weight:700;">Hour {t['from_hour']} ➔ Hour {t['to_hour']}</div>
                            <div style="font-size:16px; font-weight:800; color:{t['color']}; margin:3px 0;">{t['trend_badge']}</div>
                            <div style="font-size:12px; color:#334155;">Score: {t['from_score']} ➔ {t['to_score']}/12</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ==============================================================================
# COMPONENT: Historical Trend & Risk Assessment Report
# ==============================================================================
def render_risk_assessment_report(patient_df: pd.DataFrame, selected_hour: int):
    """Display comprehensive rule-based assessment, 3-hour trend changes, and condition-specific blinking alert."""
    st.markdown("---")
    st.markdown("### 🚨 Rule-Based Condition Assessment & Historical Trend Analysis")

    assessment = assess_patient_risk(patient_df, selected_hour)
    if not assessment:
        st.warning("Insufficient record history to perform risk assessment.")
        return

    curr_score = assessment["current_score"]
    category = assessment["risk_category"]
    trend = assessment["trend"]
    is_immediate = assessment["is_immediate"]
    previous_readings = assessment["previous_readings"]
    score_change = assessment["score_change"]
    baseline_hour = assessment["baseline_hour"]

    # DETERMINE THE EXACT SINGLE COLOR AND BANNER TO BLINK:
    # 1. Very High (score >= 7) OR Immediate Warning: RED
    # 2. High (score 5-6): ORANGE
    # 3. Mild Abnormality (score 3-4): YELLOW/AMBER
    # 4. Normal / Lower Risk (score 0-2): GREEN
    if is_immediate or curr_score >= 7:
        box_class = "box-red"
        beacon_class = "beacon-red"
        icon_symbol = "🚨"
        if is_immediate:
            alert_title = "IMMEDIATE WARNING: NEWS Score Is Increasing"
            alert_desc = (
                f"<b>Critical Deterioration Alert:</b> The patient's Modified NEWS score is <b>{curr_score}/12</b> with rapid escalation "
                f"(Score change: <b>{score_change:+d}</b>). Immediate senior medical review and vital-sign stabilization required."
            )
        else:
            alert_title = "CRITICAL ALERT: Very High Risk Deterioration"
            alert_desc = f"<b>Severe Physiological Instability:</b> Modified NEWS score is <b>{curr_score}/12</b>. Immediate senior clinical emergency response required."
    elif curr_score >= 5:
        box_class = "box-orange"
        beacon_class = "beacon-orange"
        icon_symbol = "🟠"
        alert_title = "HIGH RISK CONDITION: Urgent Clinical Attention Required"
        alert_desc = f"<b>High Risk Alert:</b> The patient's Modified NEWS score has reached <b>{curr_score}/12</b>. Urgent medical assessment and continuous monitoring are indicated."
    elif curr_score >= 3:
        box_class = "box-yellow"
        beacon_class = "beacon-yellow"
        icon_symbol = "🟡"
        alert_title = "MILD ABNORMALITY / INCREASED RISK: Close Monitoring Warranted"
        alert_desc = f"<b>Abnormality Notice:</b> Modified NEWS score is <b>{curr_score}/12</b>. Physiological parameters deviate from normal. Increase observation frequency."
    else:
        box_class = "box-green"
        beacon_class = "beacon-green"
        icon_symbol = "🟢"
        alert_title = "NORMAL CONDITION: Lower Physiological Risk"
        alert_desc = f"<b>Stable Baseline:</b> Modified NEWS score is <b>{curr_score}/12</b>. All vital signs are within normal ranges. Continue routine ICU monitoring."

    st.markdown(
        f"""
        <div class="condition-alert-box {box_class}">
            <div style="font-size:18px; font-weight:800; margin-bottom:8px; display:flex; align-items:center;">
                <span class="beacon-dot {beacon_class}"></span>
                <span>{icon_symbol} {alert_title}</span>
            </div>
            <div style="font-size:14px; line-height:1.5;">
                {alert_desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary Cards
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Current Modified NEWS</div>
                <div class="metric-val" style="color:{get_risk_color(curr_score)};">{curr_score} <span class="metric-unit">/ 12</span></div>
                <div class="metric-sub">{category}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r2:
        trend_label = {
            "immediate": "Rapid Escalation",
            "increasing": "Increasing",
            "decreasing": "Improving (Decreasing)",
            "stable": "Stable / Unchanged",
            "no_history": "Insufficient Prior Data",
        }.get(trend, trend)
        trend_color = "#dc2626" if trend in ["immediate", "increasing"] else ("#16a34a" if trend == "decreasing" else "#0284c7")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">3-Hour Trajectory</div>
                <div class="metric-val" style="color:{trend_color}; font-size:20px;">{trend_label}</div>
                <div class="metric-sub">Trajectory Status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r3:
        delta_str = f"{score_change:+d}" if baseline_hour is not None else "N/A"
        delta_sub = f"From Hour {baseline_hour}" if baseline_hour is not None else "No prior 3h reading"
        delta_color = "#dc2626" if score_change > 0 else ("#16a34a" if score_change < 0 else "#0f172a")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Score Change (Δ)</div>
                <div class="metric-val" style="color:{delta_color};">{delta_str}</div>
                <div class="metric-sub">{delta_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r4:
        prev_str = ", ".join(f"h{h}={s}" for h, s in previous_readings) if previous_readings else "None"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">3-Hour History</div>
                <div class="metric-val" style="font-size:16px; font-weight:700; color:#334155;">{prev_str}</div>
                <div class="metric-sub">Observed Previous Scores</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Consecutive Hourly Transitions Breakdown (Hour by Hour Analysis)
    transitions = assessment.get("hourly_transitions", [])
    if transitions:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### ⏱️ Step-by-Step Hourly Score Progression (Present Hour to Next Hour)")
        st.caption(
            "Tracking consecutive hourly intervals: whether the Modified NEWS score is **increasing**, **decreasing**, or **not increased (stable)**, by how many points, and the contributing vital sign changes:"
        )

        for t in transitions:
            t_icon = "📈" if t["delta"] > 0 else ("📉" if t["delta"] < 0 else "⏸️")
            border_color = t["color"]
            bg_color = "#fef2f2" if t["delta"] > 0 else ("#f0fdf4" if t["delta"] < 0 else "#f8fafc")
            delta_str = f"+{t['delta']} pts (INCREASED)" if t['delta'] > 0 else (f"{t['delta']} pts (DECREASED)" if t['delta'] < 0 else "0 pts (NOT INCREASED / STABLE)")

            st.markdown(
                f"""
                <div style="background:{bg_color}; border-left: 4px solid {border_color}; border-radius: 8px; padding: 12px 18px; margin-bottom: 10px; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                        <span style="font-weight:700; font-size:15px; color:#0f172a;">
                            {t_icon} Hour {t['from_hour']} ➔ Hour {t['to_hour']} Transition
                        </span>
                        <span style="font-weight:700; font-size:13px; color:{t['color']};">
                            {delta_str}
                        </span>
                    </div>
                    <div style="font-size:13.5px; color:#334155; margin-top:5px;">
                        <b>Modified NEWS:</b> {t['from_score']} ➔ {t['to_score']} / 12 &nbsp;|&nbsp; {t['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Detailed Tabular Breakdown
        t_records = []
        for t in transitions:
            status_text = "Increasing" if t["delta"] > 0 else ("Decreasing" if t["delta"] < 0 else "Not Increased (Stable)")
            t_records.append({
                "Transition Interval": f"Hour {t['from_hour']} ➔ Hour {t['to_hour']}",
                "Start Score": f"{t['from_score']}/12",
                "End Score": f"{t['to_score']}/12",
                "Score Delta (Δ)": f"{t['delta']:+d} pts" if t["delta"] != 0 else "0 pts",
                "Trajectory Status": status_text,
                "Detailed Physiological Explanation": t["description"],
            })
        st.dataframe(pd.DataFrame(t_records), hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Rule-Based Messages
    st.markdown("##### 📋 Agent Decision Logs & Protocol Actions")
    for msg in assessment["warning_messages"]:
        if "IMMEDIATE WARNING" in msg:
            st.error(f"🚨 {msg}")
        elif "VERY HIGH" in msg or "HIGH" in msg:
            st.warning(f"⚠️ {msg}")
        elif "MILD" in msg:
            st.info(f"ℹ️ {msg}")
        else:
            st.success(f"✅ {msg}")

    st.caption("ℹ️ Rule-based monitoring prototype based on NHS NEWS2 protocols. Does not replace professional clinical assessment.")


# ==============================================================================
# PAGE 2: EXISTING PATIENT MONITORING
# ==============================================================================
def render_existing_patient_page():
    st.markdown("## 🩺 Existing Patient Monitoring")
    st.markdown(
        "Search for an ICU patient stay, select an observation hour, and analyze patient deterioration."
    )

    # Patient ID selection
    col_input1, col_input2 = st.columns([2, 1])

    with col_input1:
        patient_id_input = st.number_input(
            "Enter Patient Stay ID (patientunitstayid):",
            min_value=1,
            value=141765,  # Benchmark patient ID requested by user
            step=1,
            key="existing_patient_id_input_widget",
            help="Example IDs with multiple hours: 141765, 141168, 141233",
        )

    patient_id_val = int(patient_id_input)

    # Automatically reset graph index if patient ID changes
    if st.session_state.get("active_existing_id") != patient_id_val:
        st.session_state["active_existing_id"] = patient_id_val
        st.session_state["existing_graph_idx"] = 1

    with col_input2:
        st.write("")
        st.write("")
        st.markdown(
            "<span style='font-size:13px;color:#64748b;'>Suggested Benchmark ID: <b>141765</b> (Hour 4)</span>",
            unsafe_allow_html=True,
        )

    # Strictly retrieve ONLY this patient's records
    patient_df = get_patient_records(dataset_df, patient_id_val)

    if patient_df.empty:
        st.error(f"❌ Patient ID `{patient_id_val}` was not found in the dataset. Please enter a valid patient stay ID.")
        return

    available_hours = get_available_patient_hours(patient_df)

    # Hour selection with dynamic key tied to patient ID
    st.markdown("---")
    col_h1, col_h2 = st.columns([2, 3])
    with col_h1:
        default_idx = available_hours.index(4) if (patient_id_val == 141765 and 4 in available_hours) else 0
        selected_hour = st.selectbox(
            "Select Observation Hour to Inspect:",
            options=available_hours,
            index=default_idx,
            format_func=lambda h: f"Hour {h}",
            key=f"existing_hour_select_widget_{patient_id_val}",
        )

    with col_h2:
        st.write("")
        st.write("")
        st.info(f"Patient has **{len(available_hours)}** total recorded hours (from Hour {available_hours[0]} to Hour {available_hours[-1]}).")

    selected_record = get_reading_at_hour(patient_df, selected_hour)
    if selected_record is None:
        st.error("Could not retrieve patient data for selected hour.")
        return

    # 1. Show Patient Details & NEWS Breakdown
    render_patient_record_cards(selected_record, existing=True)

    # 2. Sequential Graph Viewer (1 to 11)
    render_sequential_graph_viewer(
        patient_df=patient_df,
        selected_hour=selected_hour,
        session_key="existing_graph_idx",
        dataset_std=dataset_std,
        existing=True,
    )

    # 3. Historical Trend Assessment & Single-Color Blinking Alert
    render_risk_assessment_report(patient_df, selected_hour)


# ==============================================================================
# PAGE 3: NEW PATIENT MONITORING
# ==============================================================================
def render_new_patient_page():
    st.markdown("## ➕ New Patient Monitoring (Manual Intake)")
    st.markdown(
        "Enter vital signs for a newly admitted patient. Enter multiple hours to evaluate 3-hour trends and trigger early warning alerts."
    )

    known_ids = set(dataset_df["patientunitstayid"].unique())
    existing_new_ids = sorted(list(st.session_state["new_patients"].keys()))

    # Clean Mode Selection to prevent ID collisions
    if existing_new_ids:
        mode = st.radio(
            "Select New Patient Workflow:",
            ["📂 Select Existing Registered New Patient", "➕ Register Another New Patient"],
            horizontal=True,
            key="new_patient_workflow_mode",
        )
    else:
        mode = "➕ Register Another New Patient"

    if mode == "📂 Select Existing Registered New Patient" and existing_new_ids:
        active_patient_id = st.selectbox(
            "Choose Registered Patient ID:",
            options=existing_new_ids,
            format_func=lambda pid: f"Patient ID {pid} ({len(st.session_state['new_patients'][pid])} hourly readings)",
            key="select_existing_new_patient_box",
        )
    else:
        # Generate default unused ID
        def_id = 990001
        while def_id in known_ids or def_id in existing_new_ids:
            def_id += 1

        active_patient_id = st.number_input(
            "Enter New Patient ID:",
            min_value=1,
            value=def_id,
            step=1,
            key="register_new_patient_id_entry",
            help="Choose an ID distinct from existing dataset stays.",
        )
        if int(active_patient_id) in known_ids:
            st.warning(f"⚠️ Patient ID `{active_patient_id}` already exists in the original dataset. Please choose a unique new ID.")

    active_patient_id = int(active_patient_id)

    # Reset graph index if active patient changes
    if st.session_state.get("active_new_id") != active_patient_id:
        st.session_state["active_new_id"] = active_patient_id
        st.session_state["new_graph_idx"] = 1

    current_patient_readings = st.session_state["new_patients"].get(active_patient_id, [])

    # Intake Form for active_patient_id
    with st.expander(f"📝 Record Hourly Reading for Patient {active_patient_id}", expanded=(len(current_patient_readings) == 0)):
        with st.form(f"new_reading_form_{active_patient_id}"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                def_age = current_patient_readings[0]["age"] if current_patient_readings else 65
                def_gender = current_patient_readings[0]["gender"] if current_patient_readings else "male"
                gender_opts = ["male", "female", "other"]
                gender_idx = gender_opts.index(def_gender) if def_gender in gender_opts else 0

                age = st.number_input("Age (years):", min_value=1, max_value=120, value=int(def_age), step=1)
                gender = st.selectbox("Gender:", gender_opts, index=gender_idx)
                
                next_hour = max([r["hour"] for r in current_patient_readings]) + 1 if current_patient_readings else 0
                hour = st.number_input(
                    "Observation Hour:",
                    min_value=0,
                    max_value=500,
                    value=int(next_hour),
                    step=1,
                )

            with f_col2:
                hr = st.number_input("Heart Rate (bpm):", min_value=1.0, max_value=300.0, value=82.0, step=1.0)
                spo2 = st.number_input("Oxygen Saturation - SpO2 (%):", min_value=0.0, max_value=100.0, value=96.0, step=1.0)
                temp = st.number_input("Temperature (°C):", min_value=25.0, max_value=45.0, value=37.0, step=0.1)

            with f_col3:
                rr = st.number_input("Respiratory Rate (breaths/min):", min_value=1.0, max_value=100.0, value=16.0, step=1.0)
                sbp = st.number_input("Systolic BP (mmHg):", min_value=20.0, max_value=300.0, value=120.0, step=1.0)
                dbp = st.number_input("Diastolic BP (mmHg):", min_value=10.0, max_value=250.0, value=75.0, step=1.0)

            submitted = st.form_submit_button("Record Reading", use_container_width=True)
            if submitted:
                existing_hours = [r["hour"] for r in current_patient_readings]
                if hour in existing_hours:
                    st.error(f"Hour {hour} already recorded for Patient {active_patient_id}! Please choose a different hour.")
                else:
                    mean_bp = round((sbp + 2.0 * dbp) / 3.0, 2)
                    new_row = {
                        "patientunitstayid": active_patient_id,
                        "hour": int(hour),
                        "age": int(age),
                        "gender": gender,
                        "heart_rate": float(hr),
                        "spo2": float(spo2),
                        "temperature": float(temp),
                        "respiratory_rate": float(rr),
                        "systolic_bp": float(sbp),
                        "diastolic_bp": float(dbp),
                        "mean_bp": mean_bp,
                    }
                    new_row.update(news_scores(new_row))

                    if active_patient_id not in st.session_state["new_patients"]:
                        st.session_state["new_patients"][active_patient_id] = []
                    st.session_state["new_patients"][active_patient_id].append(new_row)
                    st.session_state["new_patients"][active_patient_id].sort(key=lambda x: x["hour"])
                    st.success(f"✅ Hourly reading for Hour {hour} added for Patient {active_patient_id}!")
                    st.rerun()

    # Retrieve strictly THIS active patient's records
    patient_records_list = st.session_state["new_patients"].get(active_patient_id, [])

    if not patient_records_list:
        st.info(f"No readings recorded yet for Patient {active_patient_id}. Add your first reading using the form above.")
        return

    # Strictly isolated DataFrame containing ONLY active_patient_id
    new_patient_df = pd.DataFrame(patient_records_list).sort_values("hour")

    # Hour selection
    st.markdown("---")
    st.markdown(f"#### 🕒 Select Reading Hour to Inspect (Patient {active_patient_id})")
    hours_available = sorted(new_patient_df["hour"].unique().tolist())

    col_nh1, col_nh2 = st.columns([2, 3])
    with col_nh1:
        selected_new_hour = st.selectbox(
            "Patient Observation Hour:",
            options=hours_available,
            index=len(hours_available) - 1,  # Default to latest hour
            format_func=lambda h: f"Hour {h}",
            key=f"new_hour_select_widget_{active_patient_id}",
        )

    with col_nh2:
        st.write("")
        st.write("")
        if st.button(f"🗑️ Clear Readings for Patient {active_patient_id}", key=f"clear_patient_button_{active_patient_id}"):
            del st.session_state["new_patients"][active_patient_id]
            st.rerun()

    selected_new_record = new_patient_df.loc[new_patient_df["hour"] == selected_new_hour].iloc[0]

    # 1. Show Patient Details & NEWS Breakdown
    render_patient_record_cards(selected_new_record, existing=False)

    # 2. Sequential Graph Viewer (1 to 11) strictly using new_patient_df
    render_sequential_graph_viewer(
        patient_df=new_patient_df,
        selected_hour=selected_new_hour,
        session_key="new_graph_idx",
        dataset_std=dataset_std,
        existing=False,
    )

    # 3. Historical Trend Assessment & Single-Color Blinking Alerts
    render_risk_assessment_report(new_patient_df, selected_new_hour)


# ==============================================================================
# PAGE 4: DATASET ANALYSIS DASHBOARD
# ==============================================================================
def render_dataset_analysis_page():
    st.markdown("## 📊 ICU Dataset Population Analysis")
    st.markdown(
        "Comprehensive exploratory analysis of the complete original dataset (104,609 observations across 2,153 ICU stays)."
    )

    # Overview Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Hourly Records</div>
                <div class="metric-val">{overview_stats['total_records']:,}</div>
                <div class="metric-sub">Complete un-balanced set</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">ICU Patient Stays</div>
                <div class="metric-val">{overview_stats['unique_stays']:,}</div>
                <div class="metric-sub">Unique admissions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Output 0 (Non-Deteriorated)</div>
                <div class="metric-val" style="color:#0284c7;">{overview_stats['output_0']:,}</div>
                <div class="metric-sub">{overview_stats['output_0_pct']}% of dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Output 1 (Deteriorated)</div>
                <div class="metric-val" style="color:#ea580c;">{overview_stats['output_1']:,}</div>
                <div class="metric-sub">{overview_stats['output_1_pct']}% of dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped Means Comparison Table
    st.markdown("### 📋 Mean Vital Signs Grouped by Output Class")
    means_df = overview_stats["class_means"].copy()
    means_df.index = [f"Output {idx} ({'Deteriorated' if idx == 1 else 'Stable'})" for idx in means_df.index]
    means_df.columns = [f"{VITALS[c][0]} ({VITALS[c][1]})" for c in means_df.columns]
    st.dataframe(means_df.style.format("{:.2f}"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Dataset Graphs Navigation
    st.markdown("### 📈 Interactive Dataset Visualizations")
    st.caption("Inspect population-level distributions individually. Choose a visualization below.")

    dataset_graph_options = [
        "1. Heart Rate: Output 0 vs Output 1 Boxplot",
        "2. SpO₂: Output 0 vs Output 1 Boxplot",
        "3. Temperature: Output 0 vs Output 1 Boxplot",
        "4. Respiratory Rate: Output 0 vs Output 1 Boxplot",
        "5. Systolic BP: Output 0 vs Output 1 Boxplot",
        "6. Diastolic BP: Output 0 vs Output 1 Boxplot",
        "7. Mean BP: Output 0 vs Output 1 Boxplot",
        "8. Vital-Sign Pearson Correlation Matrix Heatmap",
        "9. Modified NEWS Score vs Deterioration Target Bar Chart",
    ]

    curr_ds_idx = st.session_state["dataset_graph_idx"]

    # Navigation toolbar for dataset graphs
    d_col1, d_col2, d_col3 = st.columns([1, 2.5, 1])
    with d_col1:
        if st.button("◀ Previous Graph", key="ds_prev", use_container_width=True):
            if curr_ds_idx > 1:
                st.session_state["dataset_graph_idx"] = curr_ds_idx - 1
                st.rerun()
            else:
                st.session_state["dataset_graph_idx"] = 9
                st.rerun()

    with d_col2:
        selected_ds_title = st.selectbox(
            "Jump to Dataset Graph",
            options=dataset_graph_options,
            index=curr_ds_idx - 1,
            key="ds_jump_select",
            label_visibility="collapsed",
        )
        new_ds_idx = dataset_graph_options.index(selected_ds_title) + 1
        if new_ds_idx != curr_ds_idx:
            st.session_state["dataset_graph_idx"] = new_ds_idx
            st.rerun()

    with d_col3:
        if st.button("Next Graph ▶", key="ds_next", use_container_width=True):
            if curr_ds_idx < 9:
                st.session_state["dataset_graph_idx"] = curr_ds_idx + 1
                st.rerun()
            else:
                st.session_state["dataset_graph_idx"] = 1
                st.rerun()

    st.markdown(
        f"<div style='text-align:center; font-weight:700; color:#0284c7; margin: 6px 0 16px 0;'>"
        f"Visualization {curr_ds_idx} of 9: {dataset_graph_options[curr_ds_idx - 1]}</div>",
        unsafe_allow_html=True,
    )

    # Render selected dataset graph
    if curr_ds_idx == 1:
        st.plotly_chart(plot_dataset_vital_comparison(dataset_df, "heart_rate"), use_container_width=True)
    elif curr_ds_idx == 2:
        st.plotly_chart(plot_dataset_vital_comparison(dataset_df, "spo2"), use_container_width=True)
    elif curr_ds_idx == 3:
        st.plotly_chart(plot_dataset_vital_comparison(dataset_df, "temperature"), use_container_width=True)
    elif curr_ds_idx == 4:
        st.plotly_chart(plot_dataset_vital_comparison(dataset_df, "respiratory_rate"), use_container_width=True)
    elif curr_ds_idx == 5:
        st.plotly_chart(plot_dataset_vital_comparison(dataset_df, "systolic_bp"), use_container_width=True)
    elif curr_ds_idx == 6:
        st.plotly_chart(plot_dataset_vital_comparison(dataset_df, "diastolic_bp"), use_container_width=True)
    elif curr_ds_idx == 7:
        st.plotly_chart(plot_dataset_vital_comparison(dataset_df, "mean_bp"), use_container_width=True)
    elif curr_ds_idx == 8:
        st.plotly_chart(plot_dataset_correlation_heatmap(overview_stats["corr_matrix"]), use_container_width=True)
    elif curr_ds_idx == 9:
        st.plotly_chart(plot_dataset_news_vs_output(overview_stats["grouped_news"]), use_container_width=True)


# ==============================================================================
# PAGE 5: ABOUT THE PROJECT
# ==============================================================================
def render_about_page():
    st.markdown("## ℹ️ About the Project")
    st.markdown(
        """
        ### AI Agent for Detecting Hidden Patient Deterioration Using ICU Vital Signs
        **Rule-Based Clinical Intelligence & Early Warning Prototype**
        """
    )

    st.markdown(
        """
        In Intensive Care Units (ICU), delayed recognition of subtle patient deterioration is a leading cause 
        of preventable adverse events, cardiac arrests, and unplanned transfers. Machine learning models often suffer
        from 'black-box' opacity and domain shifts in clinical environments.

        This system leverages a **transparent, rule-based clinical intelligent agent** built on 
        the internationally validated **National Early Warning Score (NEWS2)** framework. 
        It couples deterministic scoring with **dynamic 3-hour trend monitoring** to detect rapid score escalation.
        """
    )

    st.markdown("---")
    st.markdown("### 🧮 NEWS Component Scoring Matrix")

    news_matrix = {
        "Physiological Parameter": [
            "Respiration Rate (breaths/min)",
            "SpO2 Oxygen Saturation (%)",
            "Systolic Blood Pressure (mmHg)",
            "Heart Rate (beats/min)",
        ],
        "Score 3": ["≤ 8 or ≥ 25", "≤ 91", "≤ 90 or ≥ 220", "≤ 40 or ≥ 131"],
        "Score 2": ["21 – 24", "92 – 93", "91 – 100", "111 – 130"],
        "Score 1": ["9 – 11", "94 – 95", "101 – 110", "41 – 50 or 91 – 110"],
        "Score 0 (Normal)": ["12 – 20", "≥ 96", "111 – 219", "51 – 90"],
    }
    st.dataframe(pd.DataFrame(news_matrix), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🚦 Risk Stratification & Clinical Response")

    risk_tiers = {
        "Modified NEWS Score": ["0 – 2", "3 – 4", "5 – 6", "7 – 12"],
        "Clinical Risk Category": [
            "LOWER RISK",
            "MILD ABNORMALITY / INCREASED RISK",
            "HIGH RISK",
            "VERY HIGH RISK",
        ],
        "Recommended Monitoring & Response": [
            "Continue routine ICU/ward vital-sign monitoring as prescribed.",
            "Inform registered nurse; increase monitoring frequency; evaluate clinical trend.",
            "Urgent medical review by ICU registrar or rapid response team; continuous monitoring.",
            "Immediate emergency senior clinical assessment; prepare for invasive stabilization or resuscitation.",
        ],
    }
    st.dataframe(pd.DataFrame(risk_tiers), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚡ Trend Volatility Rules (Immediate Early Warning)")
    st.markdown(
        """
        A static score alone can miss dynamic decompensation. The agent computes:
        1. **Rapid Delta Condition:** A score escalation of **≥ 2 points** compared to the earliest reading within the past 3 hours.
        2. **Monotonic Rising Condition:** A continuous non-decreasing trajectory over the 3-hour window with total change **≥ 1 point**.

        When triggered, the portal issues an **IMMEDIATE WARNING** with animated visual beacon alerts.
        """
    )

    st.markdown("---")
    st.markdown("### ⚠️ Academic & Clinical Disclaimer")
    st.info(
        "**Disclaimer:** This software is an educational prototype developed for research and demonstration purposes. "
        "It is not a certified medical device, does not provide autonomous clinical diagnoses, and should never supersede "
        "the judgment, direct evaluation, and clinical decisions of licensed healthcare professionals."
    )


# ==============================================================================
# Page Router
# ==============================================================================
page = st.session_state["nav_page"]

if page == "Homepage":
    render_homepage()
elif page == "Existing Patient Monitoring":
    render_existing_patient_page()
elif page == "New Patient Monitoring":
    render_new_patient_page()
elif page == "Dataset Analysis":
    render_dataset_analysis_page()
elif page == "About the Project":
    render_about_page()
