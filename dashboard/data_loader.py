"""
Traffic Intelligence Dashboard - Data Loader Module
Loads, caches, and validates Power BI data layer CSV datasets from powerbi/data/.
"""

import os
import pandas as pd

# Required column schemas for validation
REQUIRED_SCHEMAS = {
    "traffic_hourly": ["date_time", "hour", "weekday", "is_weekday", "is_rush", "traffic_volume", "volume_bucket"],
    "traffic_daily": ["date", "year", "month", "total_traffic_volume", "avg_traffic_volume"],
    "traffic_monthly": ["year", "month", "year_month", "avg_traffic_volume", "total_traffic_volume"],
    "demand_forecast": ["date_time", "hour", "actual_traffic_volume", "naive_lag1h_pred", "ml_rf_forecast_pred", "ml_rf_error", "ml_rf_abs_error"],
    "model_performance": ["evaluation_strategy", "model_name", "mae", "rmse", "r2_score"],
    "error_analysis": ["segment_type", "segment_value", "segment_label", "sample_count", "mae", "rmse", "mean_bias_error"],
    "business_kpis": ["kpi_category", "kpi_name", "kpi_value"],
    "monitoring_metrics": ["metric_type", "feature_name", "psi_score", "drift_status", "rmse_threshold", "monitored_rmse"],
    "dim_date": ["date", "year", "month", "day", "is_weekday"],
    "dim_time": ["hour", "time_label", "time_of_day_period", "is_rush_hour"]
}


def _raw_read_csv(filepath):
    """Raw pandas read_csv helper."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required dashboard dataset missing: '{filepath}'")
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset '{filepath}' is empty.")
    return df


try:
    import streamlit as st
    @st.cache_data(show_spinner=False)
    def _cached_read_csv(filepath):
        return _raw_read_csv(filepath)
except ImportError:
    _cached_read_csv = _raw_read_csv


def validate_dataset_schema(df, required_columns, dataset_name):
    """Validates that a DataFrame contains all required columns."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset '{dataset_name}' missing required columns: {missing}")
    return True


def load_dataset(dataset_name, data_dir="powerbi/data"):
    """
    Loads a dataset by name from data_dir.
    Applies schema validation and returns clean DataFrame.
    """
    filename = f"{dataset_name}.csv" if not dataset_name.endswith(".csv") else dataset_name
    key_name = dataset_name.replace(".csv", "")
    filepath = os.path.join(data_dir, filename)

    df = _cached_read_csv(filepath)

    if key_name in REQUIRED_SCHEMAS:
        validate_dataset_schema(df, REQUIRED_SCHEMAS[key_name], key_name)

    # Date parsing where applicable
    if "date_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date_time"]):
        df["date_time"] = pd.to_datetime(df["date_time"])
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    return df


def load_all_datasets(data_dir="powerbi/data"):
    """Loads all 10 dashboard datasets into a dictionary."""
    datasets = {}
    for key in REQUIRED_SCHEMAS.keys():
        datasets[key] = load_dataset(key, data_dir=data_dir)
    return datasets
