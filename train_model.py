"""
train_model.py
--------------
Unsupervised AI/ML Anomaly Detection & Calibrated Risk Scoring Pipeline.
Features:
- Robust feature scaling and preprocessing pipeline
- Isolation Forest ensemble (contamination=0.10, n_estimators=150)
- Inverts decision function and maps to a calibrated 0-100 integer Risk Score
- Categorizes transactions into Risk Tiers: [CRITICAL, HIGH, MEDIUM, LOW]
- Exports trained models, scalers, and scored dataset to models/ directory.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from ingestion_pipeline import IngestionPipeline
from graph_engine import HeterogeneousGraphEngine
from feature_engineering import FeatureEngineeringMatrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForensicModelTrainer")

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ForensicRiskModel:
    """
    Production-grade Unsupervised Isolation Forest model with inverted decision-function
    scoring, calibrated 0-100 risk normalization, and risk tier assignments.
    """

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.risk_normalizer = MinMaxScaler(feature_range=(0, 100))
        self.model = IsolationForest(
            n_estimators=150,
            max_samples=0.85,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_names: list = []
        self.is_trained: bool = False

    @staticmethod
    def get_risk_tier(score: int) -> str:
        """Categorize 0-100 integer risk score into standard compliance tiers."""
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 35:
            return "MEDIUM"
        return "LOW"

    def fit_and_score(self, X: pd.DataFrame, df_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Trains Isolation Forest, inverts decision function, scales to 0-100, and enriches dataframe.
        """
        logger.info(f"Training Isolation Forest anomaly model on {len(X)} samples with {X.shape[1]} features...")
        self.feature_names = list(X.columns)

        # Scale continuous features
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.model.fit(X_scaled)
        self.is_trained = True

        # Invert decision function: lower decision_function = more anomalous -> higher raw anomaly score
        raw_decision = self.model.decision_function(X_scaled)
        raw_anomaly_scores = -raw_decision  # Higher = more anomalous

        # Calibrate raw anomaly scores to 0-100 Risk Score range
        risk_scores_float = self.risk_normalizer.fit_transform(raw_anomaly_scores.reshape(-1, 1)).flatten()
        risk_scores = np.clip(np.round(risk_scores_float), 0, 100).astype(int)

        # Predict binary anomaly labels (-1 = anomaly, 1 = normal)
        anomaly_preds = self.model.predict(X_scaled)
        is_anomaly_flag = np.where(anomaly_preds == -1, 1, 0)

        # Enrich metadata DataFrame
        scored_df = df_metadata.copy()
        scored_df["raw_anomaly_score"] = raw_anomaly_scores
        scored_df["risk_score"] = risk_scores
        scored_df["is_anomaly"] = is_anomaly_flag
        scored_df["risk_tier"] = [self.get_risk_tier(s) for s in risk_scores]

        # Log Distribution Summary
        tier_counts = scored_df["risk_tier"].value_counts().to_dict()
        logger.info(f"Model Training Complete! Risk Tier Distribution: {tier_counts}")
        return scored_df, X_scaled

    def save(self, filepath_prefix: Path = MODELS_DIR):
        """Save model artifacts to disk."""
        logger.info(f"Saving model artifacts to {filepath_prefix}...")
        joblib.dump(self.model, filepath_prefix / "isolation_forest.joblib")
        joblib.dump(self.scaler, filepath_prefix / "scaler.joblib")
        joblib.dump(self.risk_normalizer, filepath_prefix / "risk_normalizer.joblib")

        meta = {
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_features": len(self.feature_names),
            "risk_tiers": {
                "CRITICAL": ">= 80",
                "HIGH": "60 - 79",
                "MEDIUM": "35 - 59",
                "LOW": "< 35"
            }
        }
        with open(filepath_prefix / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("[+] All model artifacts successfully persisted.")

    def load(self, filepath_prefix: Path = MODELS_DIR):
        """Load trained model artifacts from disk."""
        self.model = joblib.load(filepath_prefix / "isolation_forest.joblib")
        self.scaler = joblib.load(filepath_prefix / "scaler.joblib")
        self.risk_normalizer = joblib.load(filepath_prefix / "risk_normalizer.joblib")
        with open(filepath_prefix / "feature_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.feature_names = meta["feature_names"]
        self.is_trained = True
        logger.info("[+] Loaded model artifacts from disk.")


def run_training_pipeline() -> Tuple[pd.DataFrame, ForensicRiskModel, pd.DataFrame]:
    """Execute end-to-end training pipeline and return scored dataset."""
    pipeline = IngestionPipeline()
    csv_path = DATA_DIR / "transactions.csv"
    if not csv_path.exists():
        from generate_synthetic_data import export_synthetic_data
        export_synthetic_data()

    txs = pipeline.ingest_file(csv_path)

    graph_eng = HeterogeneousGraphEngine()
    fe_matrix = FeatureEngineeringMatrix(graph_eng)
    X, full_df = fe_matrix.extract_features(txs, fit_embedder=True)

    risk_model = ForensicRiskModel(contamination=0.10, random_state=42)
    scored_df, X_scaled = risk_model.fit_and_score(X, full_df)
    risk_model.save(MODELS_DIR)

    # Save scored dataset for fast app loading
    scored_csv = DATA_DIR / "scored_transactions.csv"
    scored_df.to_csv(scored_csv, index=False)
    logger.info(f"[+] Saved scored transactions dataset to {scored_csv}")

    return scored_df, risk_model, X


if __name__ == "__main__":
    scored_data, model, X_mat = run_training_pipeline()
    print("\n=== Forensic Risk Score Profile ===")
    print(scored_data["risk_tier"].value_counts())
    print("\nTop 5 High-Risk Transactions:")
    print(scored_data[["tx_id", "risk_score", "risk_tier", "pattern_label", "src_org"]].sort_values(by="risk_score", ascending=False).head())
