"""
Smart Traffic Volume Prediction System - Reproducible Analysis Entry Point
Executes SQL Analytics, Statistical Analysis, Demand Forecasting, Operational Error Analysis, and Business Decision Analysis.
Outputs organized CSV/MD result artifacts under results/
"""

import sys
import os

from analysis.sql_analysis import run_sql_analysis
from analysis.statistical_analysis import run_statistical_analysis
from forecasting.forecasting import run_forecasting_analysis
from analysis.error_analysis import run_error_analysis
from analysis.business_analysis import run_business_analysis


def main():
    print("============================================================================")
    print("🚦 SMART TRAFFIC VOLUME PREDICTION - REPRODUCIBLE ANALYSIS PIPELINE")
    print("============================================================================")
    
    # 1. SQL Analytics Layer
    print("\n[1/5] Executing SQL Analytics Layer...")
    sql_res = run_sql_analysis()
    print("  └─ Created traffic_analytics.db and executed SQL analytical views.")
    print("  └─ Saved outputs to results/sql/")

    # 2. Statistical Analysis
    print("\n[2/5] Executing Statistical Analysis Engine...")
    stats_res = run_statistical_analysis()
    print("  └─ Calculated descriptive stats, 95% CIs, correlation p-values, hypothesis tests.")
    print("  └─ Saved outputs to results/statistics/")

    # 3. Demand Forecasting
    print("\n[3/5] Executing Demand Forecasting Engine...")
    fc_res = run_forecasting_analysis()
    print("  └─ Evaluated Naive Lag Baselines vs ML Lag Random Forest Model.")
    print("  └─ Saved outputs to results/forecasting/")

    # 4. Operational Error Analysis
    print("\n[4/5] Executing Operational Error Analysis...")
    err_res = run_error_analysis()
    print("  └─ Computed error segmentation across hours, weekdays, rush hours, and volume buckets.")
    print("  └─ Saved outputs to results/error_analysis/")

    # 5. Business & Operational Decision Analysis
    print("\n[5/5] Executing Business & Decision Analysis Engine...")
    biz_res = run_business_analysis()
    print("  └─ Computed demand KPIs, operational error metrics, and decision summary report.")
    print("  └─ Saved outputs to results/business/")

    print("\n============================================================================")
    print("🎉 SUCCESS! All analytical workflows completed successfully.")
    print("============================================================================")


if __name__ == "__main__":
    main()
