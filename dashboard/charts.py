"""
Traffic Intelligence Dashboard - Plotly Charts Module
Generates interactive, professional Data Science / business analytics visuals.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Dark/Sleek theme template customization
PLOTLY_TEMPLATE = "plotly_dark"


def plot_hourly_trend(df_hourly):
    """Generates interactive line chart of traffic volume over date_time."""
    df_sample = df_hourly.copy()
    if len(df_sample) > 5000:
        df_sample = df_sample.sample(n=5000, random_state=42).sort_values("date_time")

    fig = px.line(
        df_sample,
        x="date_time",
        y="traffic_volume",
        color="volume_bucket",
        labels={"date_time": "Timestamp", "traffic_volume": "Traffic Volume (veh/hr)", "volume_bucket": "Volume Tier"},
        title="Hourly Traffic Volume Timeline (veh/hr)",
        template=PLOTLY_TEMPLATE
    )
    fig.update_layout(xaxis_rangeslider_visible=True, legend_title="Volume Tier")
    return fig


def plot_volume_distribution(df_hourly):
    """Generates donut chart of traffic volume bucket distribution."""
    bucket_counts = df_hourly["volume_bucket"].value_counts().reset_index()
    bucket_counts.columns = ["volume_bucket", "count"]

    fig = px.pie(
        bucket_counts,
        names="volume_bucket",
        values="count",
        hole=0.4,
        title="Traffic Demand Category Breakdown",
        color="volume_bucket",
        color_discrete_map={
            "Low (<2000)": "#2CA02C",
            "Moderate (2000-4999)": "#FF7F0E",
            "Heavy (>=5000)": "#D62728"
        },
        template=PLOTLY_TEMPLATE
    )
    fig.update_traces(textinfo="percent+label")
    return fig


def plot_diurnal_profile(df_hourly):
    """Generates diurnal 24-hour profile comparing Weekdays vs Weekends."""
    df_hourly['day_type'] = df_hourly['is_weekday'].apply(lambda x: 'Weekday' if x == 1 else 'Weekend')
    diurnal = df_hourly.groupby(['hour', 'day_type'])['traffic_volume'].mean().reset_index()

    fig = px.line(
        diurnal,
        x="hour",
        y="traffic_volume",
        color="day_type",
        markers=True,
        labels={"hour": "Hour of Day (0-23)", "traffic_volume": "Average Traffic Volume (veh/hr)", "day_type": "Day Type"},
        title="Diurnal Hourly Traffic Profile: Weekday vs. Weekend",
        color_discrete_map={"Weekday": "#1F77B4", "Weekend": "#FF7F0E"},
        template=PLOTLY_TEMPLATE
    )
    fig.update_xaxes(dtick=1)
    return fig


def plot_weekday_weekend_bar(df_hourly):
    """Generates bar chart of average volume across days of week."""
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df_hourly.groupby('weekday_name')['traffic_volume'].mean().reindex(weekday_order).reset_index()

    fig = px.bar(
        daily_avg,
        x="weekday_name",
        y="traffic_volume",
        color="traffic_volume",
        labels={"weekday_name": "Day of Week", "traffic_volume": "Average Traffic Volume (veh/hr)"},
        title="Average Traffic Volume by Day of Week",
        color_continuous_scale="Blues",
        template=PLOTLY_TEMPLATE
    )
    return fig


def plot_hour_weekday_heatmap(df_hourly):
    """Generates Hour x Weekday heatmap of average traffic volume."""
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_df = df_hourly.pivot_table(index='hour', columns='weekday_name', values='traffic_volume', aggfunc='mean')
    valid_cols = [c for c in weekday_order if c in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=valid_cols)

    fig = px.imshow(
        pivot_df,
        labels=dict(x="Day of Week", y="Hour of Day", color="Avg Volume (veh/hr)"),
        x=valid_cols,
        y=list(pivot_df.index),
        color_continuous_scale="Viridis",
        title="Traffic Density Heatmap (Hour of Day × Day of Week)",
        template=PLOTLY_TEMPLATE
    )
    fig.update_yaxes(dtick=2)
    return fig


def plot_monthly_trend(df_monthly):
    """Generates bar chart of monthly average traffic volume."""
    fig = px.bar(
        df_monthly,
        x="year_month",
        y="avg_traffic_volume",
        color="avg_traffic_volume",
        labels={"year_month": "Year-Month", "avg_traffic_volume": "Average Traffic Volume (veh/hr)"},
        title="Monthly Average Traffic Volume Trend (2012-2018)",
        color_continuous_scale="Viridis",
        template=PLOTLY_TEMPLATE
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_actual_vs_forecast(df_forecast):
    """Generates actual vs forecast time-series comparison chart."""
    df_sample = df_forecast.head(300).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sample['date_time'], y=df_sample['actual_traffic_volume'],
        mode='lines', name='Actual Volume', line=dict(color='#2CA02C', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_sample['date_time'], y=df_sample['ml_rf_forecast_pred'],
        mode='lines', name='ML Lag RF Forecast', line=dict(color='#1F77B4', width=2, dash='solid')
    ))
    fig.add_trace(go.Scatter(
        x=df_sample['date_time'], y=df_sample['naive_lag1h_pred'],
        mode='lines', name='Naive Lag-1h Baseline', line=dict(color='#FF7F0E', width=1.5, dash='dot')
    ))

    fig.update_layout(
        title="1-Hour Ahead Demand Forecast: Actual vs. Predictions (veh/hr)",
        xaxis_title="Timestamp",
        yaxis_title="Traffic Volume (veh/hr)",
        template=PLOTLY_TEMPLATE
    )
    return fig


def plot_forecast_error_over_time(df_forecast):
    """Generates forecast error residual timeline chart."""
    df_sample = df_forecast.head(500).copy()

    fig = px.scatter(
        df_sample,
        x="date_time",
        y="ml_rf_error",
        color="ml_rf_abs_error",
        labels={"date_time": "Timestamp", "ml_rf_error": "Residual Error (Actual - Pred)", "ml_rf_abs_error": "Absolute Error"},
        title="Forecast Residual Error Timeline (veh/hr)",
        color_continuous_scale="Reds",
        template=PLOTLY_TEMPLATE
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Zero Error Baseline")
    return fig


def plot_model_comparison(df_model_perf):
    """Generates bar chart comparing MAE and RMSE across model strategies."""
    df_melt = df_model_perf.melt(
        id_vars=["model_name"],
        value_vars=["mae", "rmse"],
        var_name="Metric",
        value_name="Value"
    )
    df_melt["Metric"] = df_melt["Metric"].map({"mae": "MAE (veh/hr)", "rmse": "RMSE (veh/hr)"})

    fig = px.bar(
        df_melt,
        x="model_name",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Evaluation Metrics Comparison (MAE & RMSE)",
        labels={"model_name": "Model Architecture", "Value": "Error (veh/hr)"},
        template=PLOTLY_TEMPLATE
    )
    return fig


def plot_residual_bias_by_segment(df_error_analysis):
    """Generates bar chart of Mean Bias Error across operational segments."""
    fig = px.bar(
        df_error_analysis,
        x="segment_label",
        y="mean_bias_error",
        color="bias_direction",
        facet_col="segment_type",
        facet_col_wrap=2,
        labels={"segment_label": "Segment", "mean_bias_error": "Mean Bias Error (veh/hr)", "bias_direction": "Bias Category"},
        title="Segmented Model Residual Bias (Actual - Predicted)",
        template=PLOTLY_TEMPLATE
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    return fig


def plot_feature_drift_bar(df_monitoring):
    """Generates feature drift PSI bar chart with reference threshold lines."""
    df_feat = df_monitoring[df_monitoring["metric_type"] == "Feature Drift (PSI)"].copy()

    fig = px.bar(
        df_feat,
        x="feature_name",
        y="psi_score",
        color="drift_status",
        labels={"feature_name": "Feature", "psi_score": "Population Stability Index (PSI)", "drift_status": "Drift Status"},
        title="MLOps Feature Distribution Drift (PSI Metric)",
        color_discrete_map={
            "No Drift": "#2CA02C",
            "Moderate Shift": "#FF7F0E",
            "Significant Drift": "#D62728"
        },
        template=PLOTLY_TEMPLATE
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Moderate Shift Threshold (0.10)")
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Significant Drift Threshold (0.25)")
    return fig
