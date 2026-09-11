#!/usr/bin/env bash
# =========================================================================
# Crypto & P2P Forensic Intelligence Dashboard - One-Click Launcher (POSIX)
# =========================================================================

set -e

echo "[INFO] Initializing Crypto Forensic Intelligence Environment..."

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Ensure data directory exists
mkdir -p data models

# Verify and generate synthetic data if missing
if [ ! -f "data/transactions.csv" ]; then
    echo "[INFO] Generating realistic synthetic transaction dataset..."
    python3 generate_synthetic_data.py
fi

# Verify and train model if missing
if [ ! -f "models/isolation_forest.joblib" ]; then
    echo "[INFO] Training Isolation Forest and calibrating risk scores..."
    python3 train_model.py
fi

echo "[INFO] Launching Streamlit Forensic Dashboard on http://localhost:8501 ..."
streamlit run app.py --server.port 8501
