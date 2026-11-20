@echo off
title MEMO-EXI Streamlit Dashboard
echo ========================================================
echo Starting MEMO-EXI Adaptive Temporal Memory Dashboard...
echo ========================================================

cd /d "%~dp0"

IF EXIST "C:\Users\prana\miniconda3\python.exe" (
    "C:\Users\prana\miniconda3\python.exe" -m streamlit run app.py
) ELSE (
    python -m streamlit run app.py
)

pause
