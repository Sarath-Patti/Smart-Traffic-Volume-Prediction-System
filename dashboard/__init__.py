"""
Traffic Intelligence Dashboard Package
Modular browser-accessible analytics dashboard for Smart Traffic Volume Prediction System.
"""

from dashboard.data_loader import load_dataset, load_all_datasets
from dashboard.metrics import calculate_executive_kpis, calculate_forecast_kpis, calculate_error_kpis
from dashboard.charts import (
    plot_hourly_trend, plot_volume_distribution, plot_diurnal_profile,
    plot_weekday_weekend_bar, plot_hour_weekday_heatmap, plot_monthly_trend,
    plot_actual_vs_forecast, plot_forecast_error_over_time, plot_model_comparison,
    plot_residual_bias_by_segment, plot_feature_drift_bar
)
from dashboard.renderers import render_traffic_intelligence_dashboard

__all__ = [
    "load_dataset", "load_all_datasets", "calculate_executive_kpis",
    "calculate_forecast_kpis", "calculate_error_kpis", "plot_hourly_trend",
    "plot_volume_distribution", "plot_diurnal_profile", "plot_weekday_weekend_bar",
    "plot_hour_weekday_heatmap", "plot_monthly_trend", "plot_actual_vs_forecast",
    "plot_forecast_error_over_time", "plot_model_comparison",
    "plot_residual_bias_by_segment", "plot_feature_drift_bar",
    "render_traffic_intelligence_dashboard"
]
