"""
Smart Traffic Volume Prediction - Operational Segmented Error Analysis
Analyzes model residual errors across hour, day type, rush status,
traffic volume buckets, and months to identify systematic operational biases.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class SegmentedErrorAnalyzer:
    """Performs operational error breakdown across temporal and volume segments."""

    def __init__(self, model_path="saved_models/Random_Forest.pkl", csv_path="datafile.csv"):
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = joblib.load(model_path)
        self._prepare_data()

    def _prepare_data(self):
        """Loads clean dataset and prepares holdout test predictions."""
        df = pd.read_csv(self.csv_path).drop_duplicates()
        df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
        df_sorted = df.sort_values(by='date_time').reset_index(drop=True)

        df_sorted['hour'] = df_sorted['date_time'].dt.hour
        df_sorted['day'] = df_sorted['date_time'].dt.day
        df_sorted['month'] = df_sorted['date_time'].dt.month
        df_sorted['weekday'] = df_sorted['date_time'].dt.weekday
        df_sorted['is_rush'] = df_sorted['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)

        features = ['hour', 'day', 'month', 'weekday', 'is_rush']
        X = df_sorted[features]
        y = df_sorted['traffic_volume']

        # 85/15 Chronological split holdout set
        split_idx = int(len(df_sorted) * 0.85)
        self.test_df = df_sorted.iloc[split_idx:].copy()
        
        preds = self.model.predict(self.test_df[features])
        self.test_df['predicted_volume'] = preds
        self.test_df['error'] = self.test_df['traffic_volume'] - self.test_df['predicted_volume']
        self.test_df['abs_error'] = np.abs(self.test_df['error'])

    def analyze_segment(self, segment_col, segment_name=None):
        """Calculates MAE, RMSE, Mean Error (Bias), and Sample Count for a categorical/discrete column."""
        segment_name = segment_name or segment_col
        results = []
        
        grouped = self.test_df.groupby(segment_col)
        for val, group in grouped:
            y_true = group['traffic_volume']
            y_pred = group['predicted_volume']
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            bias = group['error'].mean()  # mean(actual - predicted)
            
            # Directional bias interpretation
            if bias > 50:
                bias_dir = "Underprediction (Actual > Pred)"
            elif bias < -50:
                bias_dir = "Overprediction (Pred > Actual)"
            else:
                bias_dir = "Unbiased (Near Zero)"

            results.append({
                "Segment Type": segment_name,
                "Segment Value": str(val),
                "Sample Count (N)": len(group),
                "MAE": round(float(mae), 2),
                "RMSE": round(float(rmse), 2),
                "Mean Bias Error": round(float(bias), 2),
                "Bias Direction": bias_dir
            })

        return pd.DataFrame(results)

    def analyze_volume_buckets(self):
        """Segment error analysis by Traffic Volume Buckets (<2000, 2000-5000, >=5000)."""
        temp_df = self.test_df.copy()
        temp_df['volume_bucket'] = pd.cut(
            temp_df['traffic_volume'],
            bins=[-1, 2000, 5000, 10000],
            labels=['Low (<2000)', 'Moderate (2000-5000)', 'Heavy (>=5000)']
        )
        
        results = []
        for bucket, group in temp_df.groupby('volume_bucket'):
            y_true = group['traffic_volume']
            y_pred = group['predicted_volume']
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            bias = group['error'].mean()

            if bias > 50:
                bias_dir = "Underprediction (Actual > Pred)"
            elif bias < -50:
                bias_dir = "Overprediction (Pred > Actual)"
            else:
                bias_dir = "Unbiased (Near Zero)"

            results.append({
                "Volume Bucket": str(bucket),
                "Sample Count (N)": len(group),
                "MAE": round(float(mae), 2),
                "RMSE": round(float(rmse), 2),
                "Mean Bias Error": round(float(bias), 2),
                "Bias Direction": bias_dir
            })

        return pd.DataFrame(results)

    def run_full_error_analysis(self, output_dir="results/error_analysis"):
        """Runs all segmented error analyses and saves CSV reports."""
        os.makedirs(output_dir, exist_ok=True)
        
        hourly_err = self.analyze_segment('hour', 'Hour of Day')
        weekday_err = self.analyze_segment('weekday', 'Day of Week')
        rush_err = self.analyze_segment('is_rush', 'Rush Hour Status')
        monthly_err = self.analyze_segment('month', 'Month of Year')
        bucket_err = self.analyze_volume_buckets()

        hourly_err.to_csv(os.path.join(output_dir, "error_by_hour.csv"), index=False)
        weekday_err.to_csv(os.path.join(output_dir, "error_by_weekday.csv"), index=False)
        rush_err.to_csv(os.path.join(output_dir, "error_by_rush.csv"), index=False)
        monthly_err.to_csv(os.path.join(output_dir, "error_by_month.csv"), index=False)
        bucket_err.to_csv(os.path.join(output_dir, "error_by_volume_bucket.csv"), index=False)

        return {
            "hourly": hourly_err,
            "weekday": weekday_err,
            "rush": rush_err,
            "monthly": monthly_err,
            "bucket": bucket_err
        }


def run_error_analysis():
    analyzer = SegmentedErrorAnalyzer()
    return analyzer.run_full_error_analysis()


if __name__ == "__main__":
    res = run_error_analysis()
    print("=== OPERATIONAL ERROR BREAKDOWN BY VOLUME BUCKET ===")
    print(res["bucket"].to_string(index=False))
    print("✅ Operational Error Analysis executed successfully. Saved to results/error_analysis/")
