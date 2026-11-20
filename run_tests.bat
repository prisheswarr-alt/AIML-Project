@echo off
title ICU Patient Deterioration Verification Tests
cd /d "%~dp0"

echo Running comprehensive automated verification test suite...
".venv\Scripts\python.exe" test_app.py

pause
