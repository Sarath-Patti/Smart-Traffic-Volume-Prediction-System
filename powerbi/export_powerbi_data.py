"""
Smart Traffic Volume Prediction - Power BI Data Export Pipeline
Generates clean, reproducible, production-grade CSV datasets for Power BI dashboard ingestion.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def generate_powerbi_datasets(csv_path="datafile.csv", model_path="saved_models/Random_Forest.pkl", output_dir="powerbi/data"):
    """Generates all Power BI ready CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Clean Raw Dataset
    df = pd.read_csv(csv_path).drop_duplicates()
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    df = df.sort_values(by='date_time').reset_index(drop=True)
    
    df['date'] = df['date_time'].dt.date
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour
    df['weekday'] = df['date_time'].dt.weekday
    df['is_weekday'] = df['weekday'].apply(lambda x: 1 if x < 5 else 0)
    df['is_rush'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
    
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    weekday_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
    
    df['month_name'] = df['month'].map(month_map)
    df['weekday_name'] = df['weekday'].map(weekday_map)
    
    def assign_volume_bucket(v):
        if v < 2000:
            return "Low (<2000)"
        elif v < 5000:
            return "Moderate (2000-4999)"
        else:
            return "Heavy (>=5000)"
            
    df['temp_celsius'] = (df['temp'] - 273.15).round(2)
    df['volume_bucket'] = df['traffic_volume'].apply(assign_volume_bucket)

    # -------------------------------------------------------------------------
    # DATASET 1: traffic_hourly.csv
    # -------------------------------------------------------------------------
    hourly_cols = [
        'date_time', 'date', 'year', 'month', 'month_name', 'day', 'hour',
        'weekday', 'weekday_name', 'is_weekday', 'is_rush', 'temp_celsius', 'temp',
        'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description',
        'traffic_volume', 'volume_bucket'
    ]
    traffic_hourly = df[hourly_cols].copy()
    traffic_hourly.to_csv(os.path.join(output_dir, "traffic_hourly.csv"), index=False)

    # -------------------------------------------------------------------------
    # DATASET 2: traffic_daily.csv
    # -------------------------------------------------------------------------
    traffic_daily = df.groupby('date').agg(
        year=('year', 'first'),
        month=('month', 'first'),
        month_name=('month_name', 'first'),
        day=('day', 'first'),
        weekday=('weekday', 'first'),
        weekday_name=('weekday_name', 'first'),
        is_weekday=('is_weekday', 'first'),
        total_traffic_volume=('traffic_volume', 'sum'),
        avg_traffic_volume=('traffic_volume', 'mean'),
        min_traffic_volume=('traffic_volume', 'min'),
        max_traffic_volume=('traffic_volume', 'max'),
        record_count=('traffic_volume', 'count')
    ).reset_index()
    traffic_daily['avg_traffic_volume'] = traffic_daily['avg_traffic_volume'].round(2)
    traffic_daily.to_csv(os.path.join(output_dir, "traffic_daily.csv"), index=False)

    # -------------------------------------------------------------------------
    # DATASET 3: traffic_monthly.csv
    # -------------------------------------------------------------------------
    traffic_monthly = df.groupby(['year', 'month']).agg(
        month_name=('month_name', 'first'),
        total_traffic_volume=('traffic_volume', 'sum'),
        avg_traffic_volume=('traffic_volume', 'mean'),
        peak_hourly_volume=('traffic_volume', 'max'),
        record_count=('traffic_volume', 'count')
    ).reset_index()
    traffic_monthly['year_month'] = traffic_monthly['year'].astype(str) + "-" + traffic_monthly['month'].astype(str).str.zfill(2)
    traffic_monthly['avg_traffic_volume'] = traffic_monthly['avg_traffic_volume'].round(2)
    traffic_monthly = traffic_monthly[['year', 'month', 'year_month', 'month_name', 'avg_traffic_volume', 'total_traffic_volume', 'peak_hourly_volume', 'record_count']]
    traffic_monthly.to_csv(os.path.join(output_dir, "traffic_monthly.csv"), index=False)

    # -------------------------------------------------------------------------
    # DATASET 4: demand_forecast.csv
    # -------------------------------------------------------------------------
    from forecasting.forecasting import create_forecasting_features, evaluate_forecasting_models
    res_df, df_fc, model_fc, (X_test, y_test, preds_ml) = evaluate_forecasting_models(csv_path)
    
    test_idx = X_test.index
    test_rows = df_fc.loc[test_idx].copy()
    
    test_rows['actual_traffic_volume'] = y_test.values
    test_rows['naive_lag1h_pred'] = test_rows['lag_1h'].values
    test_rows['naive_lag24h_pred'] = test_rows['lag_24h'].values
    test_rows['ml_rf_forecast_pred'] = preds_ml
    
    test_rows['naive_lag1h_error'] = test_rows['actual_traffic_volume'] - test_rows['naive_lag1h_pred']
    test_rows['naive_lag1h_abs_error'] = np.abs(test_rows['naive_lag1h_error'])
    
    test_rows['ml_rf_error'] = test_rows['actual_traffic_volume'] - test_rows['ml_rf_forecast_pred']
    test_rows['ml_rf_abs_error'] = np.abs(test_rows['ml_rf_error'])
    test_rows['ml_rf_squared_error'] = test_rows['ml_rf_error'] ** 2
    
    fc_cols = [
        'date_time', 'hour', 'weekday', 'is_rush', 'actual_traffic_volume',
        'naive_lag1h_pred', 'naive_lag24h_pred', 'ml_rf_forecast_pred',
        'naive_lag1h_error', 'naive_lag1h_abs_error',
        'ml_rf_error', 'ml_rf_abs_error', 'ml_rf_squared_error'
    ]
    demand_forecast = test_rows[fc_cols].copy()
    demand_forecast.to_csv(os.path.join(output_dir, "demand_forecast.csv"), index=False)

    # -------------------------------------------------------------------------
    # DATASET 5: model_performance.csv
    # -------------------------------------------------------------------------
    from analysis.error_analysis import SegmentedErrorAnalyzer
    analyzer = SegmentedErrorAnalyzer(model_path=model_path, csv_path=csv_path)
    eval_df = analyzer.test_df
    
    rf_holdout_mae = mean_absolute_error(eval_df['traffic_volume'], eval_df['predicted_volume'])
    rf_holdout_rmse = np.sqrt(mean_squared_error(eval_df['traffic_volume'], eval_df['predicted_volume']))
    rf_holdout_r2 = r2_score(eval_df['traffic_volume'], eval_df['predicted_volume'])

    ml_fc_mae = mean_absolute_error(demand_forecast['actual_traffic_volume'], demand_forecast['ml_rf_forecast_pred'])
    ml_fc_rmse = np.sqrt(mean_squared_error(demand_forecast['actual_traffic_volume'], demand_forecast['ml_rf_forecast_pred']))
    ml_fc_r2 = r2_score(demand_forecast['actual_traffic_volume'], demand_forecast['ml_rf_forecast_pred'])

    naive1_mae = mean_absolute_error(demand_forecast['actual_traffic_volume'], demand_forecast['naive_lag1h_pred'])
    naive1_rmse = np.sqrt(mean_squared_error(demand_forecast['actual_traffic_volume'], demand_forecast['naive_lag1h_pred']))
    naive1_r2 = r2_score(demand_forecast['actual_traffic_volume'], demand_forecast['naive_lag1h_pred'])

    perf_rows = [
        {
            "evaluation_strategy": "Chronological Holdout (5 Calendar Features)",
            "model_name": "Random Forest Regressor (Production Model)",
            "features_used": "hour, day, month, weekday, is_rush",
            "train_test_split": "85/15 Time-Sorted",
            "mae": round(rf_holdout_mae, 2),
            "rmse": round(rf_holdout_rmse, 2),
            "r2_score": round(rf_holdout_r2, 4)
        },
        {
            "evaluation_strategy": "Chronological Forecasting (Lags + Rolling)",
            "model_name": "ML Lag Random Forest Model",
            "features_used": "lags (1h, 24h, 168h), rolling (mean, std, ewma), calendar",
            "train_test_split": "85/15 Time-Sorted",
            "mae": round(ml_fc_mae, 2),
            "rmse": round(ml_fc_rmse, 2),
            "r2_score": round(ml_fc_r2, 4)
        },
        {
            "evaluation_strategy": "Chronological Forecasting (Naive Baseline)",
            "model_name": "Naive Prev-Hour Persistence (Lag 1h)",
            "features_used": "lag_1h",
            "train_test_split": "85/15 Time-Sorted",
            "mae": round(naive1_mae, 2),
            "rmse": round(naive1_rmse, 2),
            "r2_score": round(naive1_r2, 4)
        }
    ]
    model_performance = pd.DataFrame(perf_rows)
    model_performance.to_csv(os.path.join(output_dir, "model_performance.csv"), index=False)

    # -------------------------------------------------------------------------
    # DATASET 6: error_analysis.csv
    # -------------------------------------------------------------------------
    err_bucket = analyzer.analyze_volume_buckets()
    err_bucket.rename(columns={'Volume Bucket': 'Segment Value'}, inplace=True)
    err_bucket['Segment Type'] = 'Traffic Volume Bucket'

    err_hour = analyzer.analyze_segment("hour", "Hour of Day")
    err_rush = analyzer.analyze_segment("is_rush", "Rush Hour Status")
    err_weekday = analyzer.analyze_segment("weekday", "Day of Week")
    
    def standardize_err_df(df_in):
        df_out = df_in.rename(columns={
            'Segment Type': 'segment_type',
            'Segment Value': 'segment_value',
            'Sample Count (N)': 'sample_count',
            'MAE': 'mae',
            'RMSE': 'rmse',
            'Mean Bias Error': 'mean_bias_error',
            'Bias Direction': 'bias_direction'
        })
        df_out['segment_label'] = df_out['segment_value'].astype(str)
        return df_out[['segment_type', 'segment_value', 'segment_label', 'sample_count', 'mae', 'rmse', 'mean_bias_error', 'bias_direction']]

    err_combined = pd.concat([
        standardize_err_df(err_bucket),
        standardize_err_df(err_hour),
        standardize_err_df(err_rush),
        standardize_err_df(err_weekday)
    ], ignore_index=True)
    err_combined.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

    # -------------------------------------------------------------------------
    # DATASET 7: business_kpis.csv
    # -------------------------------------------------------------------------
    from analysis.business_analysis import BusinessDecisionAnalyzer
    biz_analyzer = BusinessDecisionAnalyzer(csv_path=csv_path, model_path=model_path)
    biz_res = biz_analyzer.generate_summary_report()
    
    kpi_file = "results/business/business_kpis.csv"
    if os.path.exists(kpi_file):
        df_kpi = pd.read_csv(kpi_file)
        df_kpi.rename(columns={
            "KPI Category": "kpi_category",
            "Metric": "kpi_name",
            "Value": "kpi_value"
        }, inplace=True)
        df_kpi.to_csv(os.path.join(output_dir, "business_kpis.csv"), index=False)

    # -------------------------------------------------------------------------
    # DATASET 8: monitoring_metrics.csv
    # -------------------------------------------------------------------------
    from monitoring.monitor import generate_monitoring_report
    ref_df = pd.read_csv("monitoring/reference_data.csv")
    split_idx = int(len(df) * 0.85)
    current_df = df.iloc[split_idx:].copy().sample(n=1000, random_state=42)
    model = joblib.load(model_path)
    
    report = generate_monitoring_report(ref_df, current_df, model, rmse_threshold=550.0, dataset_label="2018 Test Holdout")
    
    mon_rows = []
    for feat in report["feature_drift"]:
        fname = feat["Feature"]
        ref_mean = float(ref_df[fname].mean()) if fname in ref_df.columns else 0.0
        curr_mean = float(current_df[fname].mean()) if fname in current_df.columns else 0.0
        mon_rows.append({
            "metric_type": "Feature Drift (PSI)",
            "feature_name": fname,
            "reference_mean": round(ref_mean, 2),
            "current_mean": round(curr_mean, 2),
            "psi_score": feat["PSI Metric"],
            "drift_status": feat["Drift Status"],
            "rmse_threshold": 550.0,
            "monitored_rmse": report["performance_metrics"]["RMSE"] if report["performance_metrics"] else np.nan,
            "alert_triggered": report["alert"]["Alert Triggered"] if report["alert"] else False,
            "alert_message": report["alert"]["Message"] if report["alert"] else "Normal"
        })
        
    p_drift = report["prediction_drift"]
    mon_rows.append({
        "metric_type": "Prediction Drift (PSI)",
        "feature_name": "traffic_volume_prediction",
        "reference_mean": p_drift["Reference Mean"],
        "current_mean": p_drift["Current Mean"],
        "psi_score": p_drift["Prediction PSI"],
        "drift_status": p_drift["Prediction Drift Status"],
        "rmse_threshold": 550.0,
        "monitored_rmse": report["performance_metrics"]["RMSE"] if report["performance_metrics"] else np.nan,
        "alert_triggered": report["alert"]["Alert Triggered"] if report["alert"] else False,
        "alert_message": report["alert"]["Message"] if report["alert"] else "Normal"
    })
    
    monitoring_df = pd.DataFrame(mon_rows)
    monitoring_df.to_csv(os.path.join(output_dir, "monitoring_metrics.csv"), index=False)

    # -------------------------------------------------------------------------
    # DIMENSION TABLES: dim_date.csv & dim_time.csv
    # -------------------------------------------------------------------------
    dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    dim_date = pd.DataFrame({
        'date': dates.date,
        'year': dates.year,
        'quarter': dates.quarter,
        'month': dates.month,
        'month_name': dates.month_name(),
        'day': dates.day,
        'day_of_week': dates.weekday,
        'weekday_name': dates.day_name(),
        'is_weekday': [1 if d < 5 else 0 for d in dates.weekday],
        'week_of_year': dates.isocalendar().week
    })
    dim_date.to_csv(os.path.join(output_dir, "dim_date.csv"), index=False)

    hours = list(range(24))
    def get_time_period(h):
        if h in [7, 8, 9]:
            return "Morning Rush"
        elif h in [17, 18, 19]:
            return "Evening Rush"
        elif 10 <= h <= 16:
            return "Midday Off-Peak"
        elif 20 <= h <= 23:
            return "Late Evening"
        else:
            return "Overnight Low"

    dim_time = pd.DataFrame({
        'hour': hours,
        'time_label': [f"{h:02d}:00" for h in hours],
        'time_of_day_period': [get_time_period(h) for h in hours],
        'is_rush_hour': [1 if h in [7, 8, 9, 17, 18, 19] else 0 for h in hours]
    })
    dim_time.to_csv(os.path.join(output_dir, "dim_time.csv"), index=False)

    print(f"✅ All 10 Power BI datasets exported successfully to '{output_dir}/'.")


if __name__ == "__main__":
    generate_powerbi_datasets()
