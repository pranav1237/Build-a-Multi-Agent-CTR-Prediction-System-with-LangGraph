@echo off
echo ===================================================
echo   Multi-Agent CTR Prediction System with LangGraph
echo ===================================================
echo.
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Installing dependencies in venv...
call venv\Scripts\pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Dependency installation failed. Trying to start anyway...
)

echo.
echo Starting FastAPI Web Server from venv at http://localhost:8000 ...
call venv\Scripts\python -m uvicorn backend.main:server --host 127.0.0.1 --port 8000 --reload
pause
