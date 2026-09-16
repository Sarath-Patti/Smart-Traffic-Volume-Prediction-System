"""
Automated Test Suite for Traffic Volume Prediction Model Monitoring System
Covers normal inputs, missing features, invalid values, synthetic drift,
prediction drift, performance alerts, and empty datasets.
"""

import unittest
import pandas as pd
import numpy as np
import os
import joblib

from monitoring.monitor import (
    calculate_psi,
    calculate_wasserstein,
    FeatureDriftMonitor,
    PredictionDriftMonitor,
    PerformanceMonitor,
    ModelAlertManager,
    PredictionLogger,
    SyntheticDriftGenerator,
    generate_monitoring_report
)


class TestModelMonitoringSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up reference data and model for testing."""
        ref_path = "monitoring/reference_data.csv"
        if os.path.exists(ref_path):
            cls.reference_df = pd.read_csv(ref_path)
        else:
            # Fallback synthetic reference dataset if file doesn't exist
            np.random.seed(42)
            cls.reference_df = pd.DataFrame({
                'hour': np.random.randint(0, 24, 1000),
                'day': np.random.randint(1, 31, 1000),
                'month': np.random.randint(1, 13, 1000),
                'weekday': np.random.randint(0, 7, 1000),
                'is_rush': np.random.choice([0, 1], 1000),
                'traffic_volume': np.random.randint(500, 5000, 1000)
            })

        cls.normal_df = cls.reference_df.sample(n=500, random_state=42).copy()
        
        # Load trained Random Forest model if available
        model_path = "saved_models/Random_Forest.pkl"
        if os.path.exists(model_path):
            cls.model = joblib.load(model_path)
        else:
            cls.model = None

    def test_1_normal_monitoring_input(self):
        """Test monitoring execution with normal valid inputs."""
        monitor = FeatureDriftMonitor(self.reference_df)
        drift_df = monitor.analyze_drift(self.normal_df)

        self.assertIsInstance(drift_df, pd.DataFrame)
        self.assertEqual(len(drift_df), 5)
        self.assertIn("PSI Metric", drift_df.columns)
        self.assertIn("Drift Status", drift_df.columns)
        
        # Normal sample should have low PSI (No Drift or minor shift)
        for _, row in drift_df.iterrows():
            self.assertGreaterEqual(row["PSI Metric"], 0.0)

    def test_2_missing_required_feature(self):
        """Test that missing required feature raises descriptive ValueError."""
        incomplete_df = self.normal_df.drop(columns=['is_rush'])
        monitor = FeatureDriftMonitor(self.reference_df)

        with self.assertRaises(ValueError) as ctx:
            monitor.analyze_drift(incomplete_df)
        self.assertIn("missing required features", str(ctx.exception))

    def test_3_invalid_feature_values(self):
        """Test that out-of-bound feature values raise descriptive ValueError."""
        invalid_df = self.normal_df.copy()
        invalid_df.iloc[0, invalid_df.columns.get_loc('hour')] = 99  # Invalid hour
        
        monitor = FeatureDriftMonitor(self.reference_df)
        with self.assertRaises(ValueError) as ctx:
            monitor.analyze_drift(invalid_df)
        self.assertIn("Invalid feature value", str(ctx.exception))

    def test_4_synthetic_feature_drift(self):
        """Test that synthetic feature shift is correctly detected as drift."""
        drifted_df = SyntheticDriftGenerator.generate_drifted_sample(
            self.normal_df, target_hour=8, force_rush=1
        )
        
        monitor = FeatureDriftMonitor(self.reference_df)
        drift_df = monitor.analyze_drift(drifted_df)

        hour_row = drift_df[drift_df['Feature'] == 'hour'].iloc[0]
        self.assertTrue(hour_row['Drift Detected'])
        self.assertGreaterEqual(hour_row['PSI Metric'], 0.10)

    def test_5_prediction_drift(self):
        """Test prediction drift calculation between reference and current batches."""
        ref_preds = np.random.normal(3000, 1000, 1000)
        curr_preds_shifted = np.random.normal(5000, 1000, 1000)

        pred_monitor = PredictionDriftMonitor()
        summary = pred_monitor.analyze_drift(ref_preds, curr_preds_shifted)

        self.assertTrue(summary["Drift Detected"])
        self.assertGreaterEqual(summary["Prediction PSI"], 0.10)
        self.assertIn("Reference Mean", summary)
        self.assertIn("Current Mean", summary)

    def test_6_performance_degradation_alert(self):
        """Test performance monitoring and degradation alert triggering."""
        perf_monitor = PerformanceMonitor()
        
        actuals = np.array([3000, 4000, 5000, 2000, 1000])
        bad_preds = np.array([1000, 1000, 1000, 5000, 5000])  # High errors

        metrics = perf_monitor.evaluate_performance(actuals, bad_preds)
        self.assertGreater(metrics["RMSE"], 0)

        alert_manager = ModelAlertManager(rmse_threshold=550.0)
        alert = alert_manager.check_alert(metrics["RMSE"])

        self.assertTrue(alert["Alert Triggered"])
        self.assertEqual(alert["Status"], "Performance Degradation Detected")

    def test_7_empty_monitoring_dataset(self):
        """Test that empty monitoring DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            FeatureDriftMonitor(self.reference_df).analyze_drift(empty_df)


if __name__ == "__main__":
    unittest.main()
