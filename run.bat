@echo off
cd /d "%~dp0"
echo Starting CAO 2026 Point Predictor...

REM Build DB if it doesn't exist
if not exist predictor.db (
    echo Database not found - running ingestion...
    python ingest.py
    if errorlevel 1 (
        echo Ingestion failed. Check that Data\ folder contains the source files.
        pause
        exit /b 1
    )
)

REM Kill anything already on port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501 "') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM Launch Streamlit
python -m streamlit run app.py --server.port 8501
pause
