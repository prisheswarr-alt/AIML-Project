# AI Agent for Detecting Hidden Patient Deterioration Using ICU Vital Signs

A modern, professional, interactive clinical dashboard and rule-based intelligent agent for early detection of patient deterioration in Intensive Care Units (ICU).

---

## 🌟 Overview

In the ICU, subtle physiological decompensations often precede catastrophic events such as cardiac arrest, septic shock, or acute respiratory failure. This project converts an established Python rule-based monitoring agent into a full-stack, responsive healthcare dashboard powered by **Streamlit**, **Plotly**, **Pandas**, and **NumPy**.

### Key System Highlights
- **Exact Clinical Logic Preservation:** Implements the NHS NEWS2 (National Early Warning Score) component scoring without machine-learning opacity.
- **Dynamic 3-Hour Trend Volatility Engine:** Monitors rolling multi-hour trajectories and detects rapid deterioration (score changes ≥ 2 or monotonic increases ≥ 1).
- **Flashing Visual Beacon Warnings:** CSS animated pulsing and color-blinking warnings to highlight critical patient deterioration.
- **11 Full-Scale Sequential Diagnostic Visualizations:** Step-by-step interactive navigation through individual vital curves, normalized delta comparisons, abnormality heatmaps, and timeline trackers.
- **Dual Monitoring Modes:** Full support for existing patient records (104,609 observations) and interactive intake of new admissions.
- **Dataset Population Analytics:** In-depth distributions, boxplots across outcome classes, and correlation matrices computed dynamically from real data.

---

## 🏗️ Project Architecture

```
icu_patient_monitor/
├── app.py                      # Main Streamlit web application & UI routing
├── agent.py                    # Rule-based NEWS scoring engine & trend algorithms
├── data_utils.py               # Dataset loading, caching, and patient record retrieval
├── visualizations.py           # 11 interactive Plotly patient graphs & dataset charts
├── requirements.txt            # Python dependencies
├── rule_based_agent_dataset (1).csv  # Real ICU vital-signs dataset
├── .streamlit/
│   └── config.toml             # Streamlit visual theme configuration
└── README.md                   # Documentation and usage guide
```

---

## ⚙️ Prerequisites & Setup (Windows)

### 1. Requirements
- Python 3.10+ (tested on Python 3.13)
- PowerShell or Command Prompt

### 2. Navigate to Project Directory
```powershell
cd "C:\Users\P.Risheswar Reddy\.gemini\antigravity\scratch\icu_patient_monitor"
```

### 3. Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit web dashboard with:

```powershell
.\.venv\Scripts\streamlit run app.py
```

Or if your virtual environment is already activated:
```powershell
streamlit run app.py
```

The web application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🩺 System Capabilities & Verification Guide

### 1. Homepage & Live Statistics
- Dynamic reading of the 104,609 records from the original CSV.
- Overview of Output 0 (82,889 stable stays) vs Output 1 (21,720 deterioration cases).
- Quick navigation tiles to all submodules.

### 2. Existing Patient Monitoring
- **Benchmark Patient:** Enter Patient Stay ID `141765` and select observation `Hour 4`.
- View vital signs with unit tags, score breakdowns, and comparison against recorded dataset scores.
- **11 Interactive Sequential Graphs:**
  1. Heart Rate vs Hour (with NEWS-0 band)
  2. SpO₂ vs Hour (with hypoxemia reference band)
  3. Temperature vs Hour
  4. Respiratory Rate vs Hour
  5. Systolic BP vs Hour
  6. Diastolic BP vs Hour
  7. Mean BP vs Hour
  8. Modified NEWS Score vs Hour (0-12 scale with dual-axis recorded outcome)
  9. Previous 3-Hour Standardized Vital-Sign Delta Bar Chart
  10. Abnormality Score Heatmap
  11. Patient Warning Timeline
- **Sequential Stepper:** Navigate with **◀ Previous Graph** and **Next Graph ▶** or use the dropdown jump selector.
- **Risk Assessment:** Immediate flashing warning if score delta ≥ 2 or rising monotonically.

### 3. New Patient Monitoring
- Enter custom patient admissions with Age, Gender, and initial vital signs.
- Enter multiple hours sequentially to observe trajectory formation.
- Automatically calculates Mean Arterial Pressure ($MAP = \frac{SBP + 2 \times DBP}{3}$) and NEWS components.
- Stored safely in session state without modifying the original dataset file.

### 4. Dataset Analysis Dashboard
- Side-by-side distribution analysis of Output 0 vs Output 1 for each of the 7 vital signs.
- Complete 7x7 Pearson correlation heatmap.
- Percentage deterioration rate per NEWS score level.

---

## ⚠️ Clinical Disclaimer
This system is an **academic research prototype and educational demonstration tool** based on standard NHS NEWS2 guidelines. It is not an FDA/CE-certified diagnostic medical device and must never substitute for licensed medical judgment or professional clinical assessment.
