"""Dataset loading, validation, caching, and retrieval utilities for ICU Patient Deterioration Monitoring.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st

from agent import VITALS, SCORES

# Potential file names / locations for the dataset
DATASET_CANDIDATES = [
    Path("rule_based_agent_dataset (1).csv"),
    Path("rule_based_agent_dataset (1)(2).csv"),
    Path("rule_based_agent_dataset.csv"),
    Path(r"c:\Users\P.Risheswar Reddy\Desktop\AMRITA\SEM-3\AIML\Project\rule_based_agent_dataset (1).csv"),
    Path(r"C:\Users\P.Risheswar Reddy\Downloads\rule_based_patient_monitor_complete\rule_based_agent_dataset (1)(2).csv"),
]


def resolve_dataset_path() -> Optional[Path]:
    """Find the valid dataset path among potential candidate locations."""
    # First check relative to current working directory
    for candidate in DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    # Check in same directory as this file
    this_dir = Path(__file__).resolve().parent
    for name in ["rule_based_agent_dataset (1).csv", "rule_based_agent_dataset (1)(2).csv", "rule_based_agent_dataset.csv"]:
        p = this_dir / name
        if p.is_file():
            return p.resolve()
    return None


@st.cache_data(show_spinner=True)
def load_dataset(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load, validate, and cache the ICU patient monitoring dataset.

    Returns the loaded DataFrame and the standard deviation of each vital sign.
    """
    if csv_path:
        target_path = Path(csv_path)
    else:
        target_path = resolve_dataset_path()

    if target_path is None or not target_path.is_file():
        raise FileNotFoundError(
            "Dataset CSV could not be found. Please ensure 'rule_based_agent_dataset (1).csv' "
            "is located in the project directory."
        )

    df = pd.read_csv(target_path)

    # Ensure required columns exist
    required_cols = {
        "patientunitstayid",
        "hour",
        "age",
        "gender",
        "deterioration_target",
        "currently_deteriorated",
        "modified_news2",
        *SCORES,
        *VITALS,
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Clean / cast types
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["deterioration_target"] = df["deterioration_target"].astype(int)
    df["modified_news2"] = df["modified_news2"].astype(int)

    # Precalculate vital sign standard deviations
    vital_cols = list(VITALS.keys())
    std_series = df[vital_cols].std().to_dict()

    return df, std_series


@st.cache_data
def get_dataset_overview_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute dataset summary metrics dynamically from the raw DataFrame."""
    total_records = len(df)
    unique_stays = int(df["patientunitstayid"].nunique())
    num_columns = len(df.columns)

    target_counts = df["deterioration_target"].value_counts().to_dict()
    output_0 = int(target_counts.get(0, 0))
    output_1 = int(target_counts.get(1, 0))

    vital_cols = list(VITALS.keys())
    vital_summary = df[vital_cols].describe().round(2).to_dict()

    # Means by deterioration target
    class_means = df.groupby("deterioration_target")[vital_cols].mean().round(2)

    # Pearson correlation matrix
    corr_matrix = df[vital_cols].corr().round(3)

    # Target rate by modified NEWS
    grouped_news = (
        df.groupby("modified_news2")["deterioration_target"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped_news["percent_deteriorated"] = (grouped_news["mean"] * 100).round(1)

    return {
        "total_records": total_records,
        "unique_stays": unique_stays,
        "num_columns": num_columns,
        "output_0": output_0,
        "output_1": output_1,
        "output_0_pct": round((output_0 / total_records) * 100, 1),
        "output_1_pct": round((output_1 / total_records) * 100, 1),
        "vital_summary": vital_summary,
        "class_means": class_means,
        "corr_matrix": corr_matrix,
        "grouped_news": grouped_news,
    }


def get_patient_records(df: pd.DataFrame, patient_id: int) -> pd.DataFrame:
    """Retrieve all records for a given patient ID sorted chronologically by hour."""
    patient = df.loc[df["patientunitstayid"] == patient_id].sort_values("hour").copy()
    return patient


def get_available_patient_hours(patient_df: pd.DataFrame) -> List[int]:
    """Retrieve sorted list of available hourly reading timestamps for the patient."""
    if patient_df.empty:
        return []
    return sorted(patient_df["hour"].unique().tolist())


def get_reading_at_hour(patient_df: pd.DataFrame, hour: int) -> Optional[pd.Series]:
    """Get the specific record for a patient at the given hour."""
    subset = patient_df.loc[patient_df["hour"] == hour]
    if subset.empty:
        return None
    return subset.iloc[0]
