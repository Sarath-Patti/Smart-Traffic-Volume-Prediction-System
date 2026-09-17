"""
Automated Test Suite for Data Science Analysis Extension
Covers SQL analytics, statistical analysis, lag/rolling feature generation,
leakage prevention, forecasting baselines, and error segmentation.
"""

import unittest
import os
import sqlite3
import pandas as pd
import numpy as np

from analysis.sql_analysis import SQLAnalyticsEngine
from analysis.statistical_analysis import (
    get_descriptive_stats,
    calculate_confidence_intervals,
    calculate_correlations_with_pvalues,
    test_weekday_vs_weekend,
    test_rush_vs_non_rush,
    test_weather_anova
)
from forecasting.forecasting import create_forecasting_features, evaluate_forecasting_models
from analysis.error_analysis import SegmentedErrorAnalyzer


class TestDataScienceAnalysisSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up dataset for testing."""
        cls.csv_path = "datafile.csv"
        df = pd.read_csv(cls.csv_path).drop_duplicates()
        df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
        df_sorted = df.sort_values(by='date_time').reset_index(drop=True)
        df_sorted['hour'] = df_sorted['date_time'].dt.hour
        df_sorted['day'] = df_sorted['date_time'].dt.day
        df_sorted['month'] = df_sorted['date_time'].dt.month
        df_sorted['weekday'] = df_sorted['date_time'].dt.weekday
        df_sorted['is_rush'] = df_sorted['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
        
        cls.df = df_sorted.head(1000).copy()

    def test_1_sql_analytics_execution(self):
        """Test SQL database setup and view queries."""
        engine = SQLAnalyticsEngine(db_path="traffic_analytics.db", csv_path=self.csv_path)
        daily_df = engine.get_query_result("v_daily_traffic_summary")
        buckets_df = engine.get_query_result("v_traffic_volume_buckets")

        self.assertIsInstance(daily_df, pd.DataFrame)
        self.assertGreater(len(daily_df), 0)
        self.assertIn("avg_traffic_volume", daily_df.columns)

        self.assertIsInstance(buckets_df, pd.DataFrame)
        self.assertEqual(len(buckets_df), 3)

    def test_2_statistical_analysis_functions(self):
        """Test descriptive stats, confidence intervals, correlation p-values, and hypothesis tests."""
        desc = get_descriptive_stats(self.df)
        self.assertIn("Mean", desc)
        self.assertIn("Median", desc)
        self.assertGreater(desc["Mean"], 0)

        ci = calculate_confidence_intervals(self.df)
        self.assertIn("Parametric t-CI", ci)
        self.assertLess(ci["Parametric t-CI"][0], ci["Sample Mean"])
        self.assertGreater(ci["Parametric t-CI"][1], ci["Sample Mean"])

        corr_df = calculate_correlations_with_pvalues(self.df)
        self.assertIn("Pearson r", corr_df.columns)
        self.assertIn("Pearson p-value", corr_df.columns)

        rush_res = test_rush_vs_non_rush(self.df)
        self.assertIn("Welch t-statistic", rush_res)
        self.assertIn("Cohen's d Effect Size", rush_res)

        weather_res = test_weather_anova(self.df)
        self.assertIn("ANOVA F-statistic", weather_res)

    def test_3_lag_feature_generation_no_leakage(self):
        """Test lag and rolling feature creation and verify shift(1) prevents lookahead leakage."""
        df_fc = create_forecasting_features(self.df)

        self.assertIn("lag_1h", df_fc.columns)
        self.assertIn("lag_24h", df_fc.columns)
        self.assertIn("rolling_mean_24h", df_fc.columns)

        # Verify no NaN values in clean forecasting dataset
        self.assertEqual(df_fc['rolling_mean_24h'].isnull().sum(), 0)

        # Leakage Verification:
        # Check that rolling_mean_24h at row 25 is mean of rows 1..24 (excluding row 25 itself!)
        raw_vals = df_fc['traffic_volume'].iloc[:24].values
        expected_rolling_mean = np.mean(raw_vals)
        actual_rolling_mean = df_fc['rolling_mean_24h'].iloc[24]

        self.assertAlmostEqual(actual_rolling_mean, expected_rolling_mean, places=2)

    def test_4_forecasting_evaluation(self):
        """Test naive baselines and ML forecasting Random Forest model evaluation."""
        res_df, df_fc, model_fc, test_tuple = evaluate_forecasting_models(self.csv_path)

        self.assertEqual(len(res_df), 3)
        self.assertIn("Naive Prev-Hour Baseline (Lag 1h)", res_df["Model Name"].values)
        self.assertIn("Random Forest Forecasting (Temporal + Lags)", res_df["Model Name"].values)
        
        rf_row = res_df[res_df["Model Type"] == "ML Forecasting Model"].iloc[0]
        self.assertGreater(rf_row["R2 Score"], 0.80)

    def test_5_error_segmentation(self):
        """Test operational error segmentation calculations."""
        analyzer = SegmentedErrorAnalyzer(model_path="saved_models/Random_Forest.pkl", csv_path=self.csv_path)
        bucket_err = analyzer.analyze_volume_buckets()
        hourly_err = analyzer.analyze_segment("hour", "Hour of Day")

        self.assertIsInstance(bucket_err, pd.DataFrame)
        self.assertIn("Mean Bias Error", bucket_err.columns)
        self.assertIn("Bias Direction", bucket_err.columns)

        self.assertIsInstance(hourly_err, pd.DataFrame)
        self.assertEqual(len(hourly_err), 24)

    def test_6_empty_or_invalid_inputs(self):
        """Test error handling for invalid datasets."""
        empty_df = pd.DataFrame()
        with self.assertRaises(Exception):
            get_descriptive_stats(empty_df)


if __name__ == "__main__":
    unittest.main()
