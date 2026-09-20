"""
Smart Traffic Volume Prediction - Power BI Data Layer Validation Script
Validates row counts, null values, column schemas, metric consistency, and data integrity.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def validate_powerbi_data_layer(data_dir="powerbi/data"):
    """Validates exported Power BI datasets against strict quality criteria."""
    print("============================================================================")
    print("🔍 POWER BI DATA LAYER VALIDATION REPORT")
    print("============================================================================")

    expected_files = [
        "traffic_hourly.csv",
        "traffic_daily.csv",
        "traffic_monthly.csv",
        "demand_forecast.csv",
        "model_performance.csv",
        "error_analysis.csv",
        "business_kpis.csv",
        "monitoring_metrics.csv",
        "dim_date.csv",
        "dim_time.csv"
    ]

    all_passed = True
    summary = []

    for fname in expected_files:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"❌ FAIL: File missing -> {fname}")
            all_passed = False
            continue

        df = pd.read_csv(fpath)
        row_cnt = len(df)
        col_cnt = len(df.columns)
        null_cnt = df.isnull().sum().sum()

        status = "PASSED"
        notes = []

        if row_cnt == 0:
            status = "FAILED (Empty File)"
            all_passed = False
            notes.append("0 rows")

        # Specific file checks
        if fname == "traffic_hourly.csv":
            if row_cnt != 48187:
                status = "FAILED (Row Count Mismatch)"
                all_passed = False
                notes.append(f"Expected 48,187 rows, got {row_cnt}")
            
            mean_vol = round(df['traffic_volume'].mean(), 2)
            if not np.isclose(mean_vol, 3259.82, atol=1.0):
                status = "FAILED (Mean Metric Mismatch)"
                all_passed = False
                notes.append(f"Expected mean ~3259.82, got {mean_vol}")

        elif fname == "demand_forecast.csv":
            if 'ml_rf_forecast_pred' not in df.columns:
                status = "FAILED (Missing Column)"
                all_passed = False
                notes.append("Missing ml_rf_forecast_pred")

        elif fname == "dim_time.csv":
            if row_cnt != 24:
                status = "FAILED (Invalid Time Grain)"
                all_passed = False
                notes.append(f"Expected 24 hours, got {row_cnt}")

        note_str = "; ".join(notes) if notes else "Valid Schema & Content"
        print(f"  [✓] {fname:<25} | Rows: {row_cnt:>6} | Cols: {col_cnt:>2} | Nulls: {null_cnt:>4} | Status: {status}")
        summary.append({
            "File": fname,
            "Rows": row_cnt,
            "Columns": col_cnt,
            "Nulls": null_cnt,
            "Status": status,
            "Notes": note_str
        })

    print("============================================================================")
    if all_passed:
        print("🎉 SUCCESS: All Power BI datasets validated successfully!")
    else:
        print("❌ ERROR: One or more datasets failed validation.")
    print("============================================================================")

    return pd.DataFrame(summary), all_passed


if __name__ == "__main__":
    summary_df, is_valid = validate_powerbi_data_layer()
    if not is_valid:
        sys.exit(1)
