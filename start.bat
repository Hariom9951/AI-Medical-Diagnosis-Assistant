@echo off
REM =============================================================
REM start.bat — AI Medical Diagnosis Assistant (Windows launcher)
REM =============================================================
REM Windows-compatible script to run both services locally
REM (outside Docker, for dev testing on Windows).
REM
REM To run the Docker image instead, use:
REM   docker-compose up --build
REM =============================================================

echo =========================================================
echo   AI Medical Diagnosis Assistant - Starting Services
echo =========================================================
echo.

REM ─── Activate venv if it exists ─────────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment ...
    call venv\Scripts\activate.bat
) else (
    echo [WARN] No venv found — using system Python.
)

echo.
echo [START] Launching FastAPI on port 8000 (background) ...
start "FastAPI-Server" /B uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

REM Wait 5 seconds for FastAPI to initialise
timeout /t 5 /nobreak > nul

echo [START] Launching Streamlit on port 8501 ...
streamlit run src/frontend/app.py ^
    --server.port 8501 ^
    --server.address 0.0.0.0 ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --server.fileWatcherType none

REM If Streamlit exits, also stop FastAPI
echo [INFO] Streamlit exited. Stopping all services ...
taskkill /F /FI "WindowTitle eq FastAPI-Server*" 2>nul
exit /b 0
