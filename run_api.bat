@echo off
title MEMO-EXI FastAPI Backend
echo ========================================================
echo Starting MEMO-EXI FastAPI REST API Server...
echo ========================================================

cd /d "%~dp0"

IF EXIST "C:\Users\prana\miniconda3\python.exe" (
    "C:\Users\prana\miniconda3\python.exe" -m uvicorn memo_exi.api.main:app --host 0.0.0.0 --port 8000 --reload
) ELSE (
    python -m uvicorn memo_exi.api.main:app --host 0.0.0.0 --port 8000 --reload
)

pause
