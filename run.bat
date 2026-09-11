@echo off
REM =========================================================================
REM Crypto & P2P Forensic Intelligence Dashboard - One-Click Launcher (Windows)
REM =========================================================================

echo [INFO] Initializing Crypto Forensic Intelligence Environment...

REM Check Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    pause
    exit /b 1
)

REM Verify and generate synthetic data if missing
if not exist "data\transactions.csv" (
    echo [INFO] Generating realistic synthetic transaction dataset...
    python generate_synthetic_data.py
)

REM Verify and train model if missing
if not exist "models\isolation_forest.joblib" (
    echo [INFO] Training Isolation Forest and calibrating risk scores...
    python train_model.py
)

echo [INFO] Launching Streamlit Forensic Dashboard on http://localhost:8501 ...
streamlit run app.py --server.port 8501 --server.headless false

pause
