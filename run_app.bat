@echo off
title ICU Patient Deterioration Monitoring Portal
cd /d "%~dp0"

echo ===================================================================
echo Starting ICU Patient Deterioration Monitoring Web Dashboard...
echo ===================================================================

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)

pause
