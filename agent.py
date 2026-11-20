"""Rule-Based Intelligent Agent for ICU Patient Deterioration Monitoring.

Preserves exact scoring algorithms, thresholds, risk categories, and trend
detection rules from the original implementation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Core Vital Signs Dictionary: key -> (Display Label, Unit)
VITALS: Dict[str, Tuple[str, str]] = {
    "heart_rate": ("Heart rate", "bpm"),
    "spo2": ("Oxygen saturation (SpO2)", "%"),
    "temperature": ("Temperature", "°C"),
    "respiratory_rate": ("Respiratory rate", "breaths/min"),
    "systolic_bp": ("Systolic BP", "mmHg"),
    "diastolic_bp": ("Diastolic BP", "mmHg"),
    "mean_bp": ("Mean BP", "mmHg"),
}

# The 4 NEWS components
SCORES: Tuple[str, ...] = ("news_hr", "news_spo2", "news_rr", "news_sbp")

# Mapping from vital column to corresponding NEWS score column
SCORED_VITALS: Dict[str, str] = {
    "heart_rate": "news_hr",
    "spo2": "news_spo2",
    "respiratory_rate": "news_rr",
    "systolic_bp": "news_sbp",
}

# Predefined reference bands:
# For scored vitals, these correspond to the NEWS 0 score band.
# For others, these represent standard clinical reference bands.
REFERENCE_BANDS: Dict[str, Tuple[float, float]] = {
    "heart_rate": (51.0, 90.0),
    "spo2": (96.0, 100.0),
    "temperature": (36.0, 38.0),
    "respiratory_rate": (12.0, 20.0),
    "systolic_bp": (111.0, 219.0),
    "diastolic_bp": (60.0, 80.0),
    "mean_bp": (70.0, 105.0),
}


def news_scores(row: Any) -> Dict[str, int]:
    """Calculate the 4 NEWS components and total modified NEWS score.

    Thresholds:
      - Heart rate: <= 40 or >= 131 -> 3; <= 50 -> 1; <= 90 -> 0; <= 110 -> 1; else -> 2
      - SpO2: <= 91 -> 3; <= 93 -> 2; <= 95 -> 1; else -> 0
      - Respiratory rate: <= 8 or >= 25 -> 3; <= 11 -> 1; <= 20 -> 0; else -> 2
      - Systolic BP: <= 90 or >= 220 -> 3; <= 100 -> 2; <= 110 -> 1; else -> 0

    Total Modified NEWS = news_hr + news_spo2 + news_rr + news_sbp (range 0 to 12).
    """
    hr = float(row["heart_rate"])
    spo2 = float(row["spo2"])
    rr = float(row["respiratory_rate"])
    sbp = float(row["systolic_bp"])

    hr_score = 3 if hr <= 40 or hr >= 131 else (1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2)
    spo2_score = 3 if spo2 <= 91 else (2 if spo2 <= 93 else 1 if spo2 <= 95 else 0)
    rr_score = 3 if rr <= 8 or rr >= 25 else (1 if rr <= 11 else 0 if rr <= 20 else 2)
    sbp_score = 3 if sbp <= 90 or sbp >= 220 else (2 if sbp <= 100 else 1 if sbp <= 110 else 0)

    return {
        "news_hr": hr_score,
        "news_spo2": spo2_score,
        "news_rr": rr_score,
        "news_sbp": sbp_score,
        "modified_news2": hr_score + spo2_score + rr_score + sbp_score,
    }


def risk_category(score: int) -> str:
    """Classify the modified NEWS score into project-defined risk categories:

    0 - 2: LOWER RISK
    3 - 4: MILD ABNORMALITY / INCREASED RISK
    5 - 6: HIGH RISK
    7 - 12: VERY HIGH RISK
    """
    if score <= 2:
        return "LOWER RISK"
    if score <= 4:
        return "MILD ABNORMALITY / INCREASED RISK"
    if score <= 6:
        return "HIGH RISK"
    return "VERY HIGH RISK"


def get_risk_color(score: int) -> str:
    """Return hex color corresponding to risk level."""
    if score <= 2:
        return "#28a745"  # Green
    if score <= 4:
        return "#f0ad4e"  # Amber / Yellow
    if score <= 6:
        return "#e67e22"  # Orange
    return "#dc3545"      # Crimson / Red


def prepare_history(patient: pd.DataFrame, selected_hour: int) -> pd.DataFrame:
    """Filter records up to the selected hour sorted by hour (no future data leakage)."""
    return patient.loc[patient["hour"] <= selected_hour].sort_values("hour").copy()


def visible_history(patient: pd.DataFrame, selected_hour: int) -> pd.DataFrame:
    """Return visible historical window (selected hour and up to 12 preceding hours)."""
    history = prepare_history(patient, selected_hour)
    return history.loc[history["hour"] >= selected_hour - 12].copy()


def add_calculated_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Append recalculated modified NEWS score to each row in the DataFrame."""
    data = frame.copy()
    data["calculated_news"] = [news_scores(r)["modified_news2"] for _, r in data.iterrows()]
    return data


def alert_at_hour(history: pd.DataFrame, hour: int) -> str:
    """Determine the trend and alert state at a given hour using up to 3 preceding hours.

    Returns:
      - 'unavailable': no history at or before hour
      - 'no_history': no readings in the previous 3 hours
      - 'immediate': score change >= 2 or strictly non-decreasing with total change >= 1
      - 'increasing': score increased (> 0)
      - 'decreasing': score decreased (< 0)
      - 'stable': score unchanged
    """
    period = prepare_history(history, hour)
    if period.empty:
        return "unavailable"
    current_score = news_scores(period.iloc[-1])["modified_news2"]
    previous = period.loc[(period["hour"] >= hour - 3) & (period["hour"] < hour)]
    if previous.empty:
        return "no_history"
    previous_scores = [news_scores(r)["modified_news2"] for _, r in previous.iterrows()]
    change = current_score - previous_scores[0]
    values = previous_scores + [current_score]
    increasing = all(b >= a for a, b in zip(values, values[1:])) and any(
        b > a for a, b in zip(values, values[1:])
    )
    if change >= 2 or (increasing and change >= 1):
        return "immediate"
    if change > 0:
        return "increasing"
    if change < 0:
        return "decreasing"
    return "stable"


def compute_hourly_transitions(history: pd.DataFrame, selected_hour: int) -> List[Dict[str, Any]]:
    """Compute step-by-step consecutive hourly transitions up to selected_hour.

    Evaluates whether the modified score increased, decreased, or did not increase (stable)
    between hour t and hour t+1, along with the score delta and contributing vital changes.
    """
    period = prepare_history(history, selected_hour)
    if len(period) < 1:
        return []

    rows = [r for _, r in period.iterrows()]
    transitions: List[Dict[str, Any]] = []

    for i in range(len(rows) - 1):
        prev_r = rows[i]
        curr_r = rows[i + 1]

        h_prev = int(prev_r["hour"])
        h_curr = int(curr_r["hour"])

        s_prev = news_scores(prev_r)
        s_curr = news_scores(curr_r)

        score_prev = s_prev["modified_news2"]
        score_curr = s_curr["modified_news2"]
        delta = score_curr - score_prev

        # Culprit vitals that changed
        reasons: List[str] = []
        for v_col, s_col in SCORED_VITALS.items():
            comp_diff = s_curr[s_col] - s_prev[s_col]
            v_label, v_unit = VITALS[v_col]
            val_prev = float(prev_r[v_col])
            val_curr = float(curr_r[v_col])
            if comp_diff > 0:
                reasons.append(f"{v_label}: {val_prev:.1f} -> {val_curr:.1f} {v_unit} (+{comp_diff} pts)")
            elif comp_diff < 0:
                reasons.append(f"{v_label}: {val_prev:.1f} -> {val_curr:.1f} {v_unit} ({comp_diff} pts)")

        if delta > 0:
            trend_label = "INCREASING"
            trend_badge = f"📈 Increased (+{delta})"
            color = "#dc2626"
            reason_text = " (" + ", ".join(reasons) + ")" if reasons else ""
            desc = f"From Hour {h_prev} to Hour {h_curr}: Score increased by +{delta} (from {score_prev} to {score_curr}){reason_text}."
        elif delta < 0:
            trend_label = "DECREASING"
            trend_badge = f"📉 Decreased ({delta})"
            color = "#16a34a"
            reason_text = " (" + ", ".join(reasons) + ")" if reasons else ""
            desc = f"From Hour {h_prev} to Hour {h_curr}: Score decreased by {abs(delta)} (from {score_prev} to {score_curr}){reason_text}."
        else:
            trend_label = "STABLE / NOT INCREASED"
            trend_badge = "⏸️ Not Increased (Stable)"
            color = "#0284c7"
            desc = f"From Hour {h_prev} to Hour {h_curr}: Score did not increase (remained stable at {score_curr})."

        transitions.append({
            "from_hour": h_prev,
            "to_hour": h_curr,
            "from_score": score_prev,
            "to_score": score_curr,
            "delta": delta,
            "trend_label": trend_label,
            "trend_badge": trend_badge,
            "color": color,
            "reasons": reasons,
            "description": desc,
        })

    return transitions


def assess_patient_risk(patient: pd.DataFrame, selected_hour: int) -> Dict[str, Any]:
    """Perform comprehensive condition assessment for the patient at the selected hour."""
    history = prepare_history(patient, selected_hour)
    if history.empty:
        return {}
    current = history.iloc[-1]
    calculated = news_scores(current)
    current_score = calculated["modified_news2"]
    category = risk_category(current_score)
    trend = alert_at_hour(history, selected_hour)

    previous = history.loc[(history["hour"] >= selected_hour - 3) & (history["hour"] < selected_hour)]
    previous_readings: List[Tuple[int, int]] = []
    change = 0
    baseline_hour = None
    if not previous.empty:
        previous_scores = [
            (int(r["hour"]), news_scores(r)["modified_news2"])
            for _, r in previous.iterrows()
        ]
        previous_readings = previous_scores
        baseline_hour = previous_scores[0][0]
        change = current_score - previous_scores[0][1]

    # Generate rule-based warning message
    warning_messages: List[str] = []
    if trend == "immediate":
        warning_messages.append("IMMEDIATE WARNING: NEWS score is increasing rapidly. Prompt clinical review is strongly advised.")
    elif trend == "increasing":
        warning_messages.append("WARNING: NEWS score has increased; patient reassessment is recommended.")
    elif trend == "decreasing":
        warning_messages.append("TREND: NEWS score has decreased in the 3-hour comparison window.")
    elif trend == "stable":
        warning_messages.append("TREND: NEWS score is unchanged relative to the earliest reading in the 3-hour window.")
    else:
        warning_messages.append("TREND: Insufficient historical readings in the previous 3 hours to establish a rate of change.")

    if current_score >= 7:
        warning_messages.append("VERY HIGH SCORE (>=7): Seek immediate senior clinical assessment and emergency intervention.")
    elif current_score >= 5:
        warning_messages.append("HIGH SCORE (5-6): Urgent clinical assessment and increased monitoring frequency required.")
    elif current_score >= 3:
        warning_messages.append("MILD ABNORMALITY (3-4): Abnormal findings warrant clinical review and repeat monitoring.")
    else:
        warning_messages.append("LOWER RISK (0-2): Continue routine ward/ICU monitoring as clinically indicated.")

    return {
        "patient_id": current["patientunitstayid"],
        "selected_hour": int(selected_hour),
        "current_score": current_score,
        "risk_category": category,
        "trend": trend,
        "is_immediate": (trend == "immediate"),
        "previous_readings": previous_readings,
        "score_change": change,
        "baseline_hour": baseline_hour,
        "warning_messages": warning_messages,
        "calculated_components": calculated,
        "current_row": current,
        "hourly_transitions": compute_hourly_transitions(patient, selected_hour),
    }
