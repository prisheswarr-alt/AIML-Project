"""Automated unit and integration test suite for the ICU Patient Deterioration Monitoring app.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project directory is in path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

from agent import (
    VITALS,
    SCORES,
    news_scores,
    risk_category,
    alert_at_hour,
    assess_patient_risk,
)
from data_utils import (
    load_dataset,
    get_dataset_overview_stats,
    get_patient_records,
    get_available_patient_hours,
    get_reading_at_hour,
)
from visualizations import (
    render_patient_graph,
    plot_dataset_vital_comparison,
    plot_dataset_correlation_heatmap,
    plot_dataset_news_vs_output,
)


def run_tests():
    print("==================================================")
    print("STARTING ICU PATIENT MONITOR VERIFICATION TESTS")
    print("==================================================")

    # 1. Dataset loading test
    print("\n[1] Testing dataset loading and dynamic statistics...")
    csv_file = project_dir / "rule_based_agent_dataset (1).csv"
    assert csv_file.is_file(), f"Dataset file not found at {csv_file}"

    df, std_dict = load_dataset(str(csv_file))
    print(f"  Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
    assert len(df) == 104609, f"Expected 104609 rows, got {len(df)}"
    assert df["patientunitstayid"].nunique() == 2153, f"Expected 2153 unique stays, got {df['patientunitstayid'].nunique()}"

    stats = get_dataset_overview_stats(df)
    assert stats["total_records"] == 104609
    assert stats["unique_stays"] == 2153
    assert stats["output_0"] == 82889, f"Expected 82889 output 0, got {stats['output_0']}"
    assert stats["output_1"] == 21720, f"Expected 21720 output 1, got {stats['output_1']}"
    print(f"  Dynamic stats confirmed: Output 0 = {stats['output_0']:,}, Output 1 = {stats['output_1']:,}")

    # 2. Existing patient lookup test: Patient 141765 at Hour 4
    print("\n[2] Testing existing patient 141765 at Hour 4...")
    patient = get_patient_records(df, 141765)
    assert not patient.empty, "Patient 141765 not found!"
    available_hours = get_available_patient_hours(patient)
    print(f"  Patient 141765 has {len(available_hours)} available hours (first 10: {available_hours[:10]})")
    assert 4 in available_hours, "Hour 4 not found in available hours for patient 141765!"

    row_h4 = get_reading_at_hour(patient, 4)
    assert row_h4 is not None, "Failed to retrieve row at Hour 4"
    scores_h4 = news_scores(row_h4)
    print(f"  Hour 4 Vitals: HR={row_h4['heart_rate']}, SpO2={row_h4['spo2']}, RR={row_h4['respiratory_rate']}, SBP={row_h4['systolic_bp']}")
    print(f"  Recalculated NEWS components: {scores_h4}")
    assert "modified_news2" in scores_h4

    assessment_h4 = assess_patient_risk(patient, 4)
    print(f"  Risk Category: {assessment_h4['risk_category']}")
    print(f"  Score Trend: {assessment_h4['trend']}")
    print(f"  Score Change: {assessment_h4['score_change']}")
    print(f"  Warning Messages: {assessment_h4['warning_messages']}")

    # 3. Test all 11 patient graphs generation for Patient 141765 at Hour 4
    print("\n[3] Testing generation of all 11 sequential patient graphs...")
    for idx in range(1, 12):
        fig = render_patient_graph(
            graph_idx=idx,
            patient_df=patient,
            selected_hour=4,
            dataset_std=std_dict,
            existing=True,
        )
        assert fig is not None, f"Graph {idx} failed to generate"
        print(f"  [OK] Graph {idx}/11 generated successfully: {fig.layout.title.text[:40]}...")

    # 4. Test New Patient Intake & Multi-Hour Trend
    print("\n[4] Testing new patient multi-hour intake and alerts...")
    # Simulate a new patient whose vitals deteriorate over 4 hours
    new_records = [
        # Hour 0: normal
        {"patientunitstayid": 999901, "hour": 0, "age": 60, "gender": "male",
         "heart_rate": 75.0, "spo2": 98.0, "temperature": 37.0, "respiratory_rate": 16.0,
         "systolic_bp": 120.0, "diastolic_bp": 80.0, "mean_bp": 93.33},
        # Hour 1: mild increase
        {"patientunitstayid": 999901, "hour": 1, "age": 60, "gender": "male",
         "heart_rate": 95.0, "spo2": 95.0, "temperature": 37.5, "respiratory_rate": 20.0,
         "systolic_bp": 115.0, "diastolic_bp": 75.0, "mean_bp": 88.33},
        # Hour 2: worsening
        {"patientunitstayid": 999901, "hour": 2, "age": 60, "gender": "male",
         "heart_rate": 112.0, "spo2": 93.0, "temperature": 38.2, "respiratory_rate": 24.0,
         "systolic_bp": 105.0, "diastolic_bp": 70.0, "mean_bp": 81.67},
        # Hour 3: critical deterioration
        {"patientunitstayid": 999901, "hour": 3, "age": 60, "gender": "male",
         "heart_rate": 135.0, "spo2": 89.0, "temperature": 38.9, "respiratory_rate": 28.0,
         "systolic_bp": 88.0, "diastolic_bp": 55.0, "mean_bp": 66.0},
    ]
    for r in new_records:
        r.update(news_scores(r))

    new_df = pd.DataFrame(new_records)
    new_assessment = assess_patient_risk(new_df, 3)
    print(f"  New patient Hour 3 Modified NEWS: {new_assessment['current_score']} / 12")
    print(f"  New patient Trend: {new_assessment['trend']}")
    print(f"  New patient Immediate Warning: {new_assessment['is_immediate']}")
    assert new_assessment["is_immediate"] is True, "Expected immediate warning for rapid deterioration!"
    print("  [OK] Immediate warning correctly triggered for deteriorating new patient!")

    # Test all 11 graphs on new patient
    for idx in range(1, 12):
        fig = render_patient_graph(
            graph_idx=idx,
            patient_df=new_df,
            selected_hour=3,
            dataset_std=std_dict,
            existing=False,
        )
        assert fig is not None

    # 5. Test Dataset Analysis Visualizations
    print("\n[5] Testing dataset analysis charts...")
    for v_key in VITALS.keys():
        f_box = plot_dataset_vital_comparison(df, v_key)
        assert f_box is not None
    print("  [OK] All 7 vital-sign box plots generated successfully")

    f_corr = plot_dataset_correlation_heatmap(stats["corr_matrix"])
    assert f_corr is not None
    print("  [OK] Correlation heatmap generated successfully")

    f_news_bar = plot_dataset_news_vs_output(stats["grouped_news"])
    assert f_news_bar is not None
    print("  [OK] NEWS vs Deterioration Target bar chart generated successfully")

    print("\n==================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
