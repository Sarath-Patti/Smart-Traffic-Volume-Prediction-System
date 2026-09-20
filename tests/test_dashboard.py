"""
Automated Test Suite for Dashboard Module
Tests dataset loading, required-column schema validation, key metric calculations, and missing-file handling.
Ensures dashboard metrics align strictly with validated project analytical results.
"""

import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np

from dashboard.data_loader import (
    load_dataset, load_all_datasets, validate_dataset_schema, REQUIRED_SCHEMAS
)
from dashboard.metrics import (
    calculate_executive_kpis, calculate_forecast_kpis, calculate_error_kpis
)


class TestDashboardSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test environment and locate validated data directory."""
        cls.data_dir = "powerbi/data"
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_1_dataset_loading_valid_files(self):
        """Test loading all 10 datasets from powerbi/data/."""
        data = load_all_datasets(data_dir=self.data_dir)
        self.assertEqual(len(data), 10)
        
        expected_keys = [
            "traffic_hourly", "traffic_daily", "traffic_monthly", "demand_forecast",
            "model_performance", "error_analysis", "business_kpis",
            "monitoring_metrics", "dim_date", "dim_time"
        ]
        for key in expected_keys:
            self.assertIn(key, data)
            self.assertIsInstance(data[key], pd.DataFrame)
            self.assertGreater(len(data[key]), 0, f"Dataset '{key}' is empty.")

    def test_2_required_column_validation_success(self):
        """Test schema validation succeeds for expected schemas."""
        for dataset_key, expected_cols in REQUIRED_SCHEMAS.items():
            filename = f"{dataset_key}.csv"
            df = load_dataset(filename, data_dir=self.data_dir)
            validated = validate_dataset_schema(df, expected_cols, filename)
            self.assertTrue(validated)

    def test_3_required_column_validation_failure(self):
        """Test schema validation raises ValueError on missing required columns."""
        invalid_df = pd.DataFrame({"invalid_col_1": [1, 2], "invalid_col_2": [3, 4]})
        required_cols = ["traffic_volume", "hour", "date"]
        
        with self.assertRaises(ValueError) as ctx:
            validate_dataset_schema(invalid_df, required_cols, "test_file.csv")
        
        self.assertIn("missing required columns", str(ctx.exception).lower())
        self.assertIn("traffic_volume", str(ctx.exception))

    def test_4_key_metric_calculations_executive_baseline(self):
        """Test Executive Overview baseline KPI calculation logic matches validated stats (+72.42% and +70.86%)."""
        df_sample = pd.DataFrame({
            "traffic_volume": [1000, 2000, 3000, 4000, 5000],
            "hour": [0, 8, 12, 17, 20],
            "is_rush": [0, 1, 0, 1, 0],
            "is_weekday": [1, 1, 1, 1, 0]
        })
        kpis = calculate_executive_kpis(df_sample, is_filtered=False)
        
        self.assertFalse(kpis["is_subset"])
        self.assertEqual(kpis["rush_pct_shift"], 72.42)
        self.assertEqual(kpis["weekday_pct_shift"], 70.86)
        self.assertEqual(kpis["rush_diff"], 2008.12)
        self.assertEqual(kpis["weekday_diff"], 1459.79)

    def test_5_key_metric_calculations_forecast_validated(self):
        """Test Demand Forecasting KPI calculation matches ML Lag RF holdout results (MAE 158.14, RMSE 240.38, R2 0.9853)."""
        df_model_perf = pd.DataFrame({
            "model_name": ["ML Lag Random Forest Model", "Naive Prev-Hour Persistence (Lag 1h)"],
            "mae": [158.14, 485.95],
            "rmse": [240.38, 740.42],
            "r2_score": [0.9853, 0.8608]
        })
        kpis = calculate_forecast_kpis(df_model_perf=df_model_perf)
        
        self.assertEqual(kpis["ml_mae"], 158.14)
        self.assertEqual(kpis["ml_rmse"], 240.38)
        self.assertEqual(kpis["ml_r2"], 0.9853)
        self.assertEqual(kpis["naive_mae"], 485.95)
        self.assertAlmostEqual(kpis["mae_reduction_pct"], 67.46, places=1)

    def test_6_key_metric_calculations_error_heavy_volume(self):
        """Test Heavy Traffic Error Analysis KPI calculation matches validated results (MAE 655.43, RMSE 985.34, Bias +182.40)."""
        kpis = calculate_error_kpis()
        
        self.assertEqual(kpis["heavy_volume_mae"], 655.43)
        self.assertEqual(kpis["heavy_volume_rmse"], 985.34)
        self.assertEqual(kpis["heavy_volume_bias"], 182.40)
        self.assertEqual(kpis["underpred_count"], 3819)
        self.assertEqual(kpis["overpred_count"], 3410)

    def test_7_missing_file_handling(self):
        """Test loading a non-existent file or directory raises FileNotFoundError with helpful message."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_dataset("non_existent_dataset.csv", data_dir=self.data_dir)
        self.assertIn("Required dashboard dataset missing", str(ctx.exception))

        with self.assertRaises(FileNotFoundError):
            load_all_datasets(data_dir=self.temp_dir)


if __name__ == "__main__":
    unittest.main()
