"""
Smart Traffic Volume Prediction - Core Model Monitoring Module
CRED Data Science Internship Portfolio Component

Implements Data Drift (PSI), Prediction Drift, Model Performance Tracking,
SQLite Prediction Logging, Model Alerting, and Synthetic Drift Testing.
"""

import json
import os
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_psi(expected, actual, num_bins=10, eps=1e-4):
    """
    Calculate Population Stability Index (PSI) between reference (expected)
    and current (actual) arrays.
    
    PSI Rule of Thumb:
      - PSI < 0.10: No significant distribution change (Stable)
      - 0.10 <= PSI < 0.25: Moderate distribution change (Slight Shift)
      - PSI >= 0.25: Significant distribution change (Action Needed)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin breakpoints using quantiles of expected data
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)  # Handle duplicate quantile edges

    if len(bins) <= 1:
        bins = np.array([expected.min() - 1, expected.max() + 1])

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Handle zero counts cleanly with epsilon
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def calculate_wasserstein(expected, actual):
    """Calculate Wasserstein Distance (Earth Mover's Distance)."""
    return float(stats.wasserstein_distance(expected, actual))


class FeatureDriftMonitor:
    """Monitors feature distribution drift against baseline reference dataset."""

    def __init__(self, reference_df, required_features=None):
        if reference_df is None or len(reference_df) == 0:
            raise ValueError("Reference dataset cannot be empty or None.")
        
        self.reference_df = reference_df
        self.required_features = required_features or ['hour', 'day', 'month', 'weekday', 'is_rush']
        
        # Verify reference dataset contains required features
        missing_ref = [f for f in self.required_features if f not in self.reference_df.columns]
        if missing_ref:
            raise ValueError(f"Reference dataset missing required features: {missing_ref}")

    def analyze_drift(self, current_df):
        """
        Calculates feature distribution drift metrics (PSI, Wasserstein Distance)
        for all required features in current_df.
        """
        if current_df is None or len(current_df) == 0:
            raise ValueError("Current dataset for monitoring cannot be empty or None.")
        
        missing_curr = [f for f in self.required_features if f not in current_df.columns]
        if missing_curr:
            raise ValueError(f"Current monitoring dataset missing required features: {missing_curr}")

        results = []
        for feature in self.required_features:
            ref_feat = self.reference_df[feature].dropna()
            curr_feat = current_df[feature].dropna()

            # Check invalid range
            if feature == 'hour' and ((curr_feat < 0) | (curr_feat > 23)).any():
                raise ValueError("Invalid feature value: 'hour' must be between 0 and 23.")
            if feature == 'month' and ((curr_feat < 1) | (curr_feat > 12)).any():
                raise ValueError("Invalid feature value: 'month' must be between 1 and 12.")

            psi_score = calculate_psi(ref_feat, curr_feat)
            wasserstein_score = calculate_wasserstein(ref_feat, curr_feat)

            if psi_score < 0.10:
                status = "No Drift"
                threshold = "PSI < 0.10"
                drift_detected = False
            elif psi_score < 0.25:
                status = "Moderate Shift"
                threshold = "0.10 <= PSI < 0.25"
                drift_detected = True
            else:
                status = "Significant Drift"
                threshold = "PSI >= 0.25"
                drift_detected = True

            results.append({
                "Feature": feature,
                "PSI Metric": round(psi_score, 4),
                "Wasserstein Distance": round(wasserstein_score, 4),
                "Threshold": threshold,
                "Drift Status": status,
                "Drift Detected": drift_detected
            })

        return pd.DataFrame(results)


class PredictionDriftMonitor:
    """Monitors output prediction distribution drift."""

    def analyze_drift(self, ref_predictions, current_predictions):
        ref_preds = np.asarray(ref_predictions, dtype=float)
        curr_preds = np.asarray(current_predictions, dtype=float)

        if len(ref_preds) == 0 or len(curr_preds) == 0:
            raise ValueError("Prediction arrays cannot be empty.")

        psi = calculate_psi(ref_preds, curr_preds)
        w_dist = calculate_wasserstein(ref_preds, curr_preds)
        drift_detected = (psi >= 0.10)

        stats_summary = {
            "Reference Mean": round(float(np.mean(ref_preds)), 2),
            "Current Mean": round(float(np.mean(curr_preds)), 2),
            "Reference Median": round(float(np.median(ref_preds)), 2),
            "Current Median": round(float(np.median(curr_preds)), 2),
            "Reference Std": round(float(np.std(ref_preds)), 2),
            "Current Std": round(float(np.std(curr_preds)), 2),
            "Reference Q1 (25%)": round(float(np.percentile(ref_preds, 25)), 2),
            "Current Q1 (25%)": round(float(np.percentile(curr_preds, 25)), 2),
            "Reference Q3 (75%)": round(float(np.percentile(ref_preds, 75)), 2),
            "Current Q3 (75%)": round(float(np.percentile(curr_preds, 75)), 2),
            "Prediction PSI": round(psi, 4),
            "Wasserstein Distance": round(w_dist, 4),
            "Prediction Drift Status": "Drift Detected" if drift_detected else "Stable",
            "Drift Detected": drift_detected
        }

        return stats_summary


class PerformanceMonitor:
    """Evaluates predictive accuracy when ground-truth actuals are available."""

    def evaluate_performance(self, actuals, predictions):
        if actuals is None or predictions is None:
            raise ValueError("Actuals and predictions cannot be None.")
        
        y_true = np.asarray(actuals, dtype=float)
        y_pred = np.asarray(predictions, dtype=float)

        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Actuals and predictions cannot be empty.")
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of actuals and predictions must match.")

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            "Evaluated Samples": len(y_true),
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 2),
            "RMSE": round(float(rmse), 4),
            "R2 Score": round(float(r2), 6)
        }


class ModelAlertManager:
    """Manages model status alerts based on monitored performance thresholds."""

    def __init__(self, rmse_threshold=550.0):
        self.rmse_threshold = rmse_threshold

    def check_alert(self, rmse):
        if rmse > self.rmse_threshold:
            return {
                "Alert Triggered": True,
                "Status": "Performance Degradation Detected",
                "Message": f"Monitored RMSE ({rmse:.2f}) exceeds performance threshold ({self.rmse_threshold:.2f})."
            }
        else:
            return {
                "Alert Triggered": False,
                "Status": "Performance Within Threshold",
                "Message": f"Monitored RMSE ({rmse:.2f}) is within acceptable threshold ({self.rmse_threshold:.2f})."
            }


class PredictionLogger:
    """Logs inference requests into SQLite users.db database."""

    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            hour INTEGER,
            day INTEGER,
            month INTEGER,
            weekday INTEGER,
            is_rush INTEGER,
            predicted_traffic_volume REAL,
            actual_traffic_volume REAL,
            model_version TEXT
        )
        """)
        conn.commit()
        conn.close()

    def log_predictions(self, df_input, predictions, actuals=None, model_version="v1.0.0"):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        preds = np.asarray(predictions, dtype=float)

        for i in range(len(df_input)):
            row = df_input.iloc[i]
            p_val = float(preds[i])
            a_val = float(actuals.iloc[i]) if (actuals is not None and i < len(actuals)) else None

            c.execute("""
            INSERT INTO prediction_logs 
            (timestamp, hour, day, month, weekday, is_rush, predicted_traffic_volume, actual_traffic_volume, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_str,
                int(row['hour']),
                int(row['day']),
                int(row['month']),
                int(row['weekday']),
                int(row['is_rush']),
                p_val,
                a_val,
                model_version
            ))

        conn.commit()
        conn.close()

    def get_logged_predictions(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        df_logs = pd.read_sql_query(f"SELECT * FROM prediction_logs ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        return df_logs


class SyntheticDriftGenerator:
    """Generates controlled synthetic feature drift for software testing and demonstration."""

    @staticmethod
    def generate_drifted_sample(df, target_hour=8, force_rush=1):
        """
        Creates a controlled distribution shift by setting hours to peak rush hour (e.g. 8 AM)
        and forcing is_rush indicator to 1.
        """
        drifted = df.copy()
        if 'hour' in drifted.columns:
            drifted['hour'] = target_hour
        if 'is_rush' in drifted.columns:
            drifted['is_rush'] = force_rush
        return drifted



def generate_monitoring_report(reference_df, current_df, model, rmse_threshold=550.0, dataset_label="Historical Simulation"):
    """Generates a complete structured monitoring report."""

    feature_monitor = FeatureDriftMonitor(reference_df)
    feature_drift_df = feature_monitor.analyze_drift(current_df)

    ref_preds = model.predict(reference_df[['hour', 'day', 'month', 'weekday', 'is_rush']])
    curr_preds = model.predict(current_df[['hour', 'day', 'month', 'weekday', 'is_rush']])

    pred_monitor = PredictionDriftMonitor()
    pred_drift_summary = pred_monitor.analyze_drift(ref_preds, curr_preds)

    perf_summary = None
    alert_summary = None

    if 'traffic_volume' in current_df.columns:
        perf_monitor = PerformanceMonitor()
        perf_summary = perf_monitor.evaluate_performance(current_df['traffic_volume'], curr_preds)

        alert_manager = ModelAlertManager(rmse_threshold=rmse_threshold)
        alert_summary = alert_manager.check_alert(perf_summary['RMSE'])

    # Load metadata if available
    meta_path = "monitoring/model_metadata.json"
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_label": dataset_label,
        "monitored_samples": len(current_df),
        "feature_drift": feature_drift_df.to_dict(orient="records"),
        "prediction_drift": pred_drift_summary,
        "performance_metrics": perf_summary,
        "alert": alert_summary,
        "model_metadata": metadata
    }

    return report
