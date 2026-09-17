"""
Smart Traffic Volume Prediction - Traffic Demand Forecasting Engine
Chronological lag and rolling feature engineering (shift(1) no leakage),
naive baselines, and ML Random Forest forecasting model evaluation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_forecasting_features(df):
    """
    Constructs chronological lag and rolling features.
    
    IMPORTANT LEAKAGE PREVENTION:
    All rolling features apply .shift(1) BEFORE window aggregation to ensure 
    that the target value at time t is never included in rolling statistics.
    """
    df_sorted = df.drop_duplicates().copy()
    df_sorted['date_time'] = pd.to_datetime(df_sorted['date_time'], dayfirst=True)
    df_sorted = df_sorted.sort_values(by='date_time').reset_index(drop=True)

    df_sorted['hour'] = df_sorted['date_time'].dt.hour
    df_sorted['day'] = df_sorted['date_time'].dt.day
    df_sorted['month'] = df_sorted['date_time'].dt.month
    df_sorted['weekday'] = df_sorted['date_time'].dt.weekday
    df_sorted['is_rush'] = df_sorted['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)

    # 1. Lag features
    df_sorted['lag_1h'] = df_sorted['traffic_volume'].shift(1)
    df_sorted['lag_24h'] = df_sorted['traffic_volume'].shift(24)

    # 2. Rolling features (shift by 1 to prevent lookahead leakage)
    df_sorted['rolling_mean_24h'] = df_sorted['traffic_volume'].shift(1).rolling(24).mean()
    df_sorted['rolling_std_24h'] = df_sorted['traffic_volume'].shift(1).rolling(24).std()
    df_sorted['ewma_24h'] = df_sorted['traffic_volume'].shift(1).ewm(span=24).mean()

    # Drop rows where lag features are NaN (first 24 rows)
    fc_cols = ['lag_1h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h', 'ewma_24h']
    df_fc = df_sorted.dropna(subset=fc_cols).reset_index(drop=True)
    
    return df_fc


def evaluate_forecasting_models(csv_path="datafile.csv"):
    """
    Evaluates forecasting performance using an out-of-time 85/15 chronological split.
    Compares Naive Baselines against ML Lag Forecasting Random Forest.
    """
    df = pd.read_csv(csv_path)
    df_fc = create_forecasting_features(df)

    split_idx = int(len(df_fc) * 0.85)

    features_fc = ['hour', 'day', 'month', 'weekday', 'is_rush', 'lag_1h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h', 'ewma_24h']
    X_fc = df_fc[features_fc]
    y_fc = df_fc['traffic_volume']

    X_train_fc, X_test_fc = X_fc.iloc[:split_idx], X_fc.iloc[split_idx:]
    y_train_fc, y_test_fc = y_fc.iloc[:split_idx], y_fc.iloc[split_idx:]

    results = []

    # 1. Naive Previous-Hour Baseline (lag_1h)
    pred_lag1 = X_test_fc['lag_1h']
    mae_lag1 = mean_absolute_error(y_test_fc, pred_lag1)
    mse_lag1 = mean_squared_error(y_test_fc, pred_lag1)
    rmse_lag1 = np.sqrt(mse_lag1)
    r2_lag1 = r2_score(y_test_fc, pred_lag1)
    results.append({
        "Model Type": "Baseline",
        "Model Name": "Naive Prev-Hour Baseline (Lag 1h)",
        "MAE": round(float(mae_lag1), 4),
        "MSE": round(float(mse_lag1), 2),
        "RMSE": round(float(rmse_lag1), 4),
        "R2 Score": round(float(r2_lag1), 6)
    })

    # 2. Naive Same-Hour-Previous-Day Baseline (lag_24h)
    pred_lag24 = X_test_fc['lag_24h']
    mae_lag24 = mean_absolute_error(y_test_fc, pred_lag24)
    mse_lag24 = mean_squared_error(y_test_fc, pred_lag24)
    rmse_lag24 = np.sqrt(mse_lag24)
    r2_lag24 = r2_score(y_test_fc, pred_lag24)
    results.append({
        "Model Type": "Baseline",
        "Model Name": "Naive Prev-Day Baseline (Lag 24h)",
        "MAE": round(float(mae_lag24), 4),
        "MSE": round(float(mse_lag24), 2),
        "RMSE": round(float(rmse_lag24), 4),
        "R2 Score": round(float(r2_lag24), 6)
    })

    # 3. ML Forecasting Model (Random Forest with Lag/Rolling Features)
    rf_fc = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_fc.fit(X_train_fc, y_train_fc)
    pred_rf_fc = rf_fc.predict(X_test_fc)

    mae_rf = mean_absolute_error(y_test_fc, pred_rf_fc)
    mse_rf = mean_squared_error(y_test_fc, pred_rf_fc)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test_fc, pred_rf_fc)
    results.append({
        "Model Type": "ML Forecasting Model",
        "Model Name": "Random Forest Forecasting (Temporal + Lags)",
        "MAE": round(float(mae_rf), 4),
        "MSE": round(float(mse_rf), 2),
        "RMSE": round(float(rmse_rf), 4),
        "R2 Score": round(float(r2_rf), 6)
    })

    res_df = pd.DataFrame(results)
    return res_df, df_fc, rf_fc, (X_test_fc, y_test_fc, pred_rf_fc)


def run_forecasting_analysis(csv_path="datafile.csv", output_dir="results/forecasting"):
    """Runs forecasting analysis and saves outputs."""
    os.makedirs(output_dir, exist_ok=True)
    res_df, df_fc, model_fc, test_tuple = evaluate_forecasting_models(csv_path)
    
    out_file = os.path.join(output_dir, "forecasting_comparison.csv")
    res_df.to_csv(out_file, index=False)
    
    return res_df


if __name__ == "__main__":
    res = run_forecasting_analysis()
    print("=== FORECASTING COMPARISON ===")
    print(res.to_string(index=False))
    print("✅ Forecasting analysis executed successfully. Saved to results/forecasting/")
