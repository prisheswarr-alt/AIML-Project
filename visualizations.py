"""Interactive Plotly visualizations for the ICU Patient Deterioration Monitoring website.

Features 11 specialized patient graphs and complete dataset overview visualizations.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent import (
    VITALS,
    SCORES,
    SCORED_VITALS,
    REFERENCE_BANDS,
    news_scores,
    risk_category,
    get_risk_color,
    alert_at_hour,
    prepare_history,
    visible_history,
    add_calculated_scores,
)

# Shared layout styling for modern clinical theme
THEME_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=13),
    margin=dict(l=60, r=40, t=60, b=50),
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#e2e8f0",
        borderwidth=1,
    ),
)


# ==============================================================================
# 1-7: Single Vital Sign vs Hour
# ==============================================================================
def plot_vital_sign_chart(
    visible_df: pd.DataFrame,
    selected_hour: int,
    vital_key: str,
    existing: bool = False,
) -> go.Figure:
    """Generate interactive chart for an individual vital sign over time."""
    label, unit = VITALS[vital_key]
    lower, upper = REFERENCE_BANDS[vital_key]
    is_news = vital_key in SCORED_VITALS

    fig = go.Figure()

    # Predefined reference/normal band
    fig.add_hrect(
        y0=lower,
        y1=upper,
        fillcolor="rgba(40, 167, 69, 0.12)",
        line_width=1,
        line_color="rgba(40, 167, 69, 0.35)",
        line_dash="dot",
        annotation_text="NEWS Score-0 Band" if is_news else "Normal Reference Band",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#2e7d32",
    )

    # Historical values line
    fig.add_trace(
        go.Scatter(
            x=visible_df["hour"],
            y=visible_df[vital_key],
            mode="lines+markers",
            name="Recorded Vital Sign",
            line=dict(color="#0284c7", width=3),
            marker=dict(size=8, color="#0284c7", symbol="circle"),
            hovertemplate=f"<b>Hour %{{x}}</b><br>{label}: %{{y:.2f}} {unit}<extra></extra>",
        )
    )

    # Highlight selected hour
    selected_record = visible_df.loc[visible_df["hour"] == selected_hour]
    if not selected_record.empty:
        val = float(selected_record.iloc[0][vital_key])
        fig.add_trace(
            go.Scatter(
                x=[selected_hour],
                y=[val],
                mode="markers",
                name=f"Selected Hour ({selected_hour})",
                marker=dict(
                    color="#dc2626",
                    size=14,
                    symbol="circle",
                    line=dict(color="#ffffff", width=2.5),
                ),
                hovertemplate=f"<b>Selected Hour: {selected_hour}</b><br>{label}: {val:.2f} {unit}<extra></extra>",
            )
        )

    # Determine y-axis range with margin
    y_min = min(float(visible_df[vital_key].min()), lower)
    y_max = max(float(visible_df[vital_key].max()), upper)
    padding = (y_max - y_min) * 0.15 if y_max > y_min else 5.0

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"<b>{label} vs Hour</b> (Current Hour: {selected_hour})",
            font=dict(size=17, color="#0f172a"),
        ),
        xaxis=dict(
            title="Monitoring Hour",
            tickmode="linear",
            dtick=1 if len(visible_df) <= 15 else 2,
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        yaxis=dict(
            title=f"{label} ({unit})",
            range=[y_min - padding, y_max + padding],
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        height=520,
    )

    return fig


# ==============================================================================
# 8: Modified NEWS Score vs Hour
# ==============================================================================
def plot_news_score_chart(
    visible_df: pd.DataFrame,
    selected_hour: int,
    existing: bool = False,
) -> go.Figure:
    """Generate Modified NEWS Score trend vs Hour (0-12 scale) with optional recorded target."""
    scores = visible_df["calculated_news"]
    max_score = max(12, int(scores.max()))
    if existing and "modified_news2" in visible_df.columns:
        max_score = max(max_score, int(visible_df["modified_news2"].max()))

    if existing and "deterioration_target" in visible_df.columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Add risk zones shading
    fig.add_hrect(y0=0, y1=2, fillcolor="rgba(40, 167, 69, 0.08)", line_width=0, layer="below")
    fig.add_hrect(y0=2, y1=4, fillcolor="rgba(245, 158, 11, 0.08)", line_width=0, layer="below")
    fig.add_hrect(y0=4, y1=6, fillcolor="rgba(234, 88, 12, 0.08)", line_width=0, layer="below")
    fig.add_hrect(y0=6, y1=max_score + 1, fillcolor="rgba(220, 38, 38, 0.08)", line_width=0, layer="below")

    # Calculate consecutive hourly progression for each point
    scores_list = visible_df["calculated_news"].tolist()
    hours_list = visible_df["hour"].tolist()
    hover_notes = []
    for i in range(len(scores_list)):
        if i == 0:
            hover_notes.append("Initial baseline reading")
        else:
            diff = scores_list[i] - scores_list[i - 1]
            h_prev = hours_list[i - 1]
            h_curr = hours_list[i]
            if diff > 0:
                hover_notes.append(f"Hr {h_prev} ➔ {h_curr}: <b>Increased by +{diff}</b>")
            elif diff < 0:
                hover_notes.append(f"Hr {h_prev} ➔ {h_curr}: <b>Decreased by {abs(diff)}</b>")
            else:
                hover_notes.append(f"Hr {h_prev} ➔ {h_curr}: <b>Not Increased (Stable, 0 pts)</b>")

    # Calculated NEWS trace
    trace_calc = go.Scatter(
        x=visible_df["hour"],
        y=scores,
        customdata=hover_notes,
        mode="lines+markers",
        name="Calculated Modified NEWS (0-12)",
        line=dict(color="#7c3aed", width=3),
        marker=dict(size=8, color="#7c3aed"),
        hovertemplate="<b>Hour %{x}</b><br>Calculated NEWS: <b>%{y}/12</b><br>Trajectory: %{customdata}<extra></extra>",
    )

    if existing and "deterioration_target" in visible_df.columns:
        fig.add_trace(trace_calc, secondary_y=False)
    else:
        fig.add_trace(trace_calc)

    # If existing patient, add recorded modified NEWS and deterioration target
    if existing and "modified_news2" in visible_df.columns:
        trace_rec = go.Scatter(
            x=visible_df["hour"],
            y=visible_df["modified_news2"],
            mode="lines+markers",
            name="Recorded Modified NEWS (Dataset)",
            line=dict(color="#0d9488", width=2.5, dash="dash"),
            marker=dict(size=7, color="#0d9488", symbol="x"),
            hovertemplate="<b>Hour %{x}</b><br>Recorded NEWS: %{y}<extra></extra>",
        )
        if "deterioration_target" in visible_df.columns:
            fig.add_trace(trace_rec, secondary_y=False)
        else:
            fig.add_trace(trace_rec)

    if existing and "deterioration_target" in visible_df.columns:
        trace_target = go.Scatter(
            x=visible_df["hour"],
            y=visible_df["deterioration_target"],
            mode="lines+markers",
            name="Deterioration Target (0/1)",
            line=dict(color="#0f172a", width=2, dash="dot", shape="hv"),
            marker=dict(size=6, color="#0f172a"),
            hovertemplate="<b>Hour %{x}</b><br>Recorded Target: %{y}<extra></extra>",
        )
        fig.add_trace(trace_target, secondary_y=True)

    # Highlight selected hour
    selected_record = visible_df.loc[visible_df["hour"] == selected_hour]
    if not selected_record.empty:
        curr_score = int(selected_record.iloc[0]["calculated_news"])
        trace_sel = go.Scatter(
            x=[selected_hour],
            y=[curr_score],
            mode="markers",
            name=f"Current Score ({curr_score}/12)",
            marker=dict(
                color="#dc2626",
                size=14,
                symbol="circle",
                line=dict(color="#ffffff", width=2.5),
            ),
            hovertemplate=f"<b>Selected Hour: {selected_hour}</b><br>Score: {curr_score}/12<extra></extra>",
        )
        if existing and "deterioration_target" in visible_df.columns:
            fig.add_trace(trace_sel, secondary_y=False)
        else:
            fig.add_trace(trace_sel)

    subtitle = "Observed dataset values vs Recalculated 0-12 score" if existing else "Manual entries; full 0-12 theoretical range"
    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"<b>Modified NEWS Score Trend</b> | Hour {selected_hour}<br><span style='font-size:12px;color:#64748b;'>{subtitle}</span>",
            font=dict(size=17, color="#0f172a"),
        ),
        xaxis=dict(
            title="Monitoring Hour",
            tickmode="linear",
            dtick=1 if len(visible_df) <= 15 else 2,
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        yaxis=dict(
            title="Modified NEWS Score",
            range=[-0.3, max_score + 0.8],
            dtick=1,
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        height=520,
    )

    if existing and "deterioration_target" in visible_df.columns:
        fig.update_yaxes(
            title_text="Recorded Deterioration Target (0 / 1)",
            range=[-0.1, 1.2],
            dtick=1,
            secondary_y=True,
            showgrid=False,
        )

    return fig


# ==============================================================================
# 9: Previous Three-Hour Vital-Sign Comparison
# ==============================================================================
def plot_previous_3_hours_chart(
    history_df: pd.DataFrame,
    selected_hour: int,
    dataset_std: Dict[str, float],
) -> go.Figure:
    """Standardized bar chart comparing vital sign changes across the previous 3 hours."""
    window = history_df.loc[
        (history_df["hour"] >= selected_hour - 3) & (history_df["hour"] <= selected_hour)
    ].sort_values("hour")

    fig = go.Figure()

    if len(window) < 2:
        fig.add_annotation(
            text="<b>Insufficient 3-Hour History</b><br>Only one reading available in the [Hour - 3, Hour] window.<br>Cannot calculate standard-deviation vital deltas.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#64748b"),
            align="center",
        )
        fig.update_layout(
            **THEME_LAYOUT,
            title=dict(
                text=f"<b>Previous Three-Hour Vital-Sign Changes</b> | Hour {selected_hour}",
                font=dict(size=17, color="#0f172a"),
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500,
        )
        return fig

    baseline = window.iloc[0]
    current = window.iloc[-1]
    baseline_hour = int(baseline["hour"])

    columns = list(VITALS.keys())
    labels = [VITALS[c][0] for c in columns]
    units = [VITALS[c][1] for c in columns]

    raw_deltas = [float(current[c]) - float(baseline[c]) for c in columns]
    scaled_deltas = []
    for c, delta in zip(columns, raw_deltas):
        std_val = float(dataset_std.get(c, 1.0))
        if not np.isfinite(std_val) or std_val < 1e-9:
            std_val = 1.0
        scaled_deltas.append(delta / std_val)

    # Bar colors: amber/orange for increase, teal/blue for decrease
    colors = ["#ea580c" if val >= 0 else "#0284c7" for val in scaled_deltas]

    text_labels = [
        f"{delta:+.2f} {unit} ({scaled:+.2f} SD)"
        for delta, unit, scaled in zip(raw_deltas, units, scaled_deltas)
    ]

    fig.add_trace(
        go.Bar(
            y=labels,
            x=scaled_deltas,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#ffffff", width=1.5)),
            text=text_labels,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Standardized Change: %{x:.2f} SD<br>Raw Delta: %{text}<extra></extra>",
            cliponaxis=False,
        )
    )

    # Vertical reference line at 0
    fig.add_vline(x=0, line_width=1.5, line_color="#334155")

    max_abs = max(1.0, max(abs(s) for s in scaled_deltas))
    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"<b>Vital-Sign Changes: Hour {baseline_hour} to Hour {selected_hour}</b><br><span style='font-size:12px;color:#64748b;'>Normalized by Dataset Standard Deviation (z-score change)</span>",
            font=dict(size=17, color="#0f172a"),
        ),
        xaxis=dict(
            title="Change Divided by Dataset Standard Deviation (Standardized across units)",
            range=[-max_abs * 1.45, max_abs * 1.45],
            gridcolor="#f1f5f9",
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="#f1f5f9",
        ),
        height=520,
    )

    return fig


# ==============================================================================
# 10: Patient Abnormality Heatmap
# ==============================================================================
def plot_abnormality_heatmap_chart(
    visible_df: pd.DataFrame,
    selected_hour: int,
) -> go.Figure:
    """Heatmap showing vital signs scores (0-3 for NEWS vitals, 0/1 for reference bands) over time."""
    columns = list(VITALS.keys())
    row_labels = [VITALS[c][0] for c in columns]
    hours = [int(h) for h in visible_df["hour"]]

    values = np.zeros((len(columns), len(visible_df)), dtype=int)
    annotations_text = []

    for i, col in enumerate(columns):
        row_text = []
        for j, (_, row) in enumerate(visible_df.iterrows()):
            raw = float(row[col])
            scores = news_scores(row)
            if col in SCORED_VITALS:
                score = scores[SCORED_VITALS[col]]
                values[i, j] = score
                row_text.append(f"{raw:.1f}<br><span style='font-size:9px;'>Score: {score}</span>")
            else:
                low, high = REFERENCE_BANDS[col]
                dev = 1 if raw < low or raw > high else 0
                values[i, j] = dev
                status = "Abnormal" if dev else "Normal"
                row_text.append(f"{raw:.1f}<br><span style='font-size:9px;'>{status}</span>")
        annotations_text.append(row_text)

    # Discrete color palette matching clinical alert thresholds
    # 0 -> Greenish-White, 1 -> Yellow/Cream, 2 -> Orange, 3 -> Coral Red
    colorscale = [
        [0.0, "#dcfce7"],   # 0
        [0.33, "#fef08a"],  # 1
        [0.66, "#fed7aa"],  # 2
        [1.0, "#fca5a5"],   # 3
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=[f"Hr {h}" for h in hours],
            y=row_labels,
            text=annotations_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            colorbar=dict(
                title=dict(text="Severity", side="top"),
                tickvals=[0, 1, 2, 3],
                ticktext=["0 (Normal)", "1 (Mild)", "2 (Moderate)", "3 (Severe)"],
                len=0.75,
            ),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b> at %{x}<br>Cell Value: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"<b>Vital-Sign Abnormality Heatmap</b> | Up to Hour {selected_hour}<br><span style='font-size:12px;color:#64748b;'>Rows 1-4: NEWS components (0-3) | Rows 5-7: 0 Normal / 1 Outside Clinical Reference Band</span>",
            font=dict(size=17, color="#0f172a"),
        ),
        xaxis=dict(title="Monitoring Hour", tickangle=0),
        yaxis=dict(autorange="reversed"),
        height=540,
    )

    return fig


# ==============================================================================
# 11: Patient Warning Timeline
# ==============================================================================
def plot_warning_timeline_chart(
    visible_df: pd.DataFrame,
    full_history: pd.DataFrame,
    selected_hour: int,
) -> go.Figure:
    """Timeline chart mapping hourly risk levels, score progression, and immediate warning triggers."""
    scores = visible_df["calculated_news"].to_numpy(dtype=int)
    hours = visible_df["hour"].to_numpy(dtype=int)
    max_score = max(12, int(scores.max()))

    fig = go.Figure()

    # Base progression line
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=scores,
            mode="lines",
            name="Score Trajectory",
            line=dict(color="#94a3b8", width=2.5),
            hoverinfo="skip",
        )
    )

    # Color scatter markers by category
    categories = [risk_category(s) for s in scores]
    marker_colors = [get_risk_color(s) for s in scores]

    fig.add_trace(
        go.Scatter(
            x=hours,
            y=scores,
            mode="markers",
            name="Hourly Risk Assessment",
            marker=dict(size=12, color=marker_colors, line=dict(color="#ffffff", width=2)),
            text=categories,
            hovertemplate="<b>Hour %{x}</b><br>Score: %{y}/12<br>Category: %{text}<extra></extra>",
        )
    )

    # Find immediate warning hours
    warning_hours = []
    warning_scores = []
    for h, s in zip(hours, scores):
        status = alert_at_hour(full_history, int(h))
        if status == "immediate":
            warning_hours.append(h)
            warning_scores.append(s)

    if warning_hours:
        fig.add_trace(
            go.Scatter(
                x=warning_hours,
                y=warning_scores,
                mode="markers",
                name="Immediate Warning Triggered",
                marker=dict(
                    symbol="triangle-up",
                    size=20,
                    color="rgba(220, 38, 38, 0.2)",
                    line=dict(color="#dc2626", width=2.5),
                ),
                hovertemplate="<b>⚠️ IMMEDIATE WARNING TRIGGERED</b><br>Hour: %{x}<br>Score: %{y}/12<br>Rapid score increase detected<extra></extra>",
            )
        )

    # Highlight current selected hour
    curr_rec = visible_df.loc[visible_df["hour"] == selected_hour]
    if not curr_rec.empty:
        curr_score = int(curr_rec.iloc[0]["calculated_news"])
        fig.add_trace(
            go.Scatter(
                x=[selected_hour],
                y=[curr_score],
                mode="markers",
                name=f"Current Hour ({selected_hour})",
                marker=dict(
                    size=18,
                    symbol="circle-open",
                    line=dict(color="#0f172a", width=3),
                ),
                hovertemplate=f"<b>Current Inspection Point</b><br>Hour: {selected_hour}<br>Score: {curr_score}/12<extra></extra>",
            )
        )

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"<b>Patient Warning Timeline</b> | Up to Hour {selected_hour}<br><span style='font-size:12px;color:#64748b;'>Hourly risk categories with immediate rapid-deterioration triggers (▲)</span>",
            font=dict(size=17, color="#0f172a"),
        ),
        xaxis=dict(
            title="Monitoring Hour",
            tickmode="linear",
            dtick=1 if len(visible_df) <= 15 else 2,
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="Modified NEWS Score (0-12)",
            range=[-0.4, max_score + 1.2],
            dtick=1,
            gridcolor="#f1f5f9",
        ),
        height=520,
    )

    return fig


# ==============================================================================
# Helper to render the appropriate patient graph by index (1 to 11)
# ==============================================================================
GRAPH_TITLES = [
    "1. Heart Rate vs Hour",
    "2. SpO₂ vs Hour",
    "3. Temperature vs Hour",
    "4. Respiratory Rate vs Hour",
    "5. Systolic BP vs Hour",
    "6. Diastolic BP vs Hour",
    "7. Mean BP vs Hour",
    "8. Modified NEWS Score vs Hour",
    "9. Previous Three-Hour Vital-Sign Comparison",
    "10. Patient Abnormality Heatmap",
    "11. Patient Warning Timeline",
]


def render_patient_graph(
    graph_idx: int,
    patient_df: pd.DataFrame,
    selected_hour: int,
    dataset_std: Dict[str, float],
    existing: bool = False,
) -> go.Figure:
    """Route to the specific graph function based on index (1-11)."""
    history = prepare_history(patient_df, selected_hour)
    visible = add_calculated_scores(visible_history(patient_df, selected_hour))

    if graph_idx == 1:
        return plot_vital_sign_chart(visible, selected_hour, "heart_rate", existing)
    elif graph_idx == 2:
        return plot_vital_sign_chart(visible, selected_hour, "spo2", existing)
    elif graph_idx == 3:
        return plot_vital_sign_chart(visible, selected_hour, "temperature", existing)
    elif graph_idx == 4:
        return plot_vital_sign_chart(visible, selected_hour, "respiratory_rate", existing)
    elif graph_idx == 5:
        return plot_vital_sign_chart(visible, selected_hour, "systolic_bp", existing)
    elif graph_idx == 6:
        return plot_vital_sign_chart(visible, selected_hour, "diastolic_bp", existing)
    elif graph_idx == 7:
        return plot_vital_sign_chart(visible, selected_hour, "mean_bp", existing)
    elif graph_idx == 8:
        return plot_news_score_chart(visible, selected_hour, existing)
    elif graph_idx == 9:
        return plot_previous_3_hours_chart(history, selected_hour, dataset_std)
    elif graph_idx == 10:
        return plot_abnormality_heatmap_chart(visible, selected_hour)
    elif graph_idx == 11:
        return plot_warning_timeline_chart(visible, history, selected_hour)
    else:
        raise ValueError(f"Invalid graph index: {graph_idx}. Must be between 1 and 11.")


# ==============================================================================
# DATASET ANALYSIS VISUALIZATIONS
# ==============================================================================
def plot_dataset_vital_comparison(df: pd.DataFrame, vital_key: str) -> go.Figure:
    """Generate box plot for vital sign by deterioration target (0 vs 1)."""
    label, unit = VITALS[vital_key]

    fig = go.Figure()

    group_0 = df.loc[df["deterioration_target"] == 0, vital_key].dropna()
    group_1 = df.loc[df["deterioration_target"] == 1, vital_key].dropna()

    fig.add_trace(
        go.Box(
            y=group_0,
            name="Output 0 (Stable / Non-Deteriorated)",
            marker_color="#0284c7",
            boxmean=True,
            hovertemplate=f"Output 0<br>{label}: %{{y:.2f}} {unit}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Box(
            y=group_1,
            name="Output 1 (Deterioration Detected)",
            marker_color="#ea580c",
            boxmean=True,
            hovertemplate=f"Output 1<br>{label}: %{{y:.2f}} {unit}<extra></extra>",
        )
    )

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text=f"<b>Dataset Distribution: {label} vs Recorded Output</b>",
            font=dict(size=17, color="#0f172a"),
        ),
        yaxis=dict(title=f"{label} ({unit})", gridcolor="#f1f5f9"),
        height=500,
    )

    return fig


def plot_dataset_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Generate correlation heatmap across the 7 ICU vital signs."""
    names = [VITALS[col][0] for col in corr_df.columns]

    z_vals = corr_df.values
    text_vals = [[f"{val:.2f}" for val in row] for row in z_vals]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=names,
            y=names,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Pearson Correlation", len=0.8),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text="<b>Pearson Correlation Matrix: Seven Primary ICU Vital Signs</b>",
            font=dict(size=17, color="#0f172a"),
        ),
        xaxis=dict(tickangle=30),
        yaxis=dict(autorange="reversed"),
        height=540,
    )

    return fig


def plot_dataset_news_vs_output(grouped_news_df: pd.DataFrame) -> go.Figure:
    """Generate bar chart showing the percentage of output 1 records at each NEWS score."""
    scores = grouped_news_df["modified_news2"]
    rates = grouped_news_df["percent_deteriorated"]
    counts = grouped_news_df["count"]

    fig = go.Figure()

    text_annotations = [f"{rate:.1f}%<br>(n={cnt:,})" for rate, cnt in zip(rates, counts)]

    fig.add_trace(
        go.Bar(
            x=scores,
            y=rates,
            marker_color="#0284c7",
            text=text_annotations,
            textposition="outside",
            hovertemplate="<b>NEWS Score: %{x}</b><br>Deterioration Rate: %{y:.1f}%<br>Sample Count: %{text}<extra></extra>",
            cliponaxis=False,
        )
    )

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(
            text="<b>Percentage of Deterioration (Output 1) by Recorded Modified NEWS Score</b><br><span style='font-size:12px;color:#64748b;'>Descriptive association across 104,609 records</span>",
            font=dict(size=17, color="#0f172a"),
        ),
        xaxis=dict(title="Recorded Modified NEWS Score", tickmode="linear", dtick=1, gridcolor="#f1f5f9"),
        yaxis=dict(
            title="Percentage with Output = 1 (%)",
            range=[0, max(rates) * 1.35 if len(rates) else 100],
            gridcolor="#f1f5f9",
        ),
        height=500,
    )

    return fig
