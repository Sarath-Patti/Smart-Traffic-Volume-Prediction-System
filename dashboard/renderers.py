"""
Traffic Intelligence Dashboard - Renderers Module
Renders the complete 5-section interactive browser analytics dashboard in Streamlit.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

from dashboard.data_loader import load_all_datasets
from dashboard.metrics import calculate_executive_kpis, calculate_forecast_kpis, calculate_error_kpis
from dashboard.charts import (
    plot_hourly_trend, plot_volume_distribution, plot_diurnal_profile,
    plot_weekday_weekend_bar, plot_hour_weekday_heatmap, plot_monthly_trend,
    plot_actual_vs_forecast, plot_forecast_error_over_time, plot_model_comparison,
    plot_residual_bias_by_segment, plot_feature_drift_bar
)


def render_traffic_intelligence_dashboard(data_dir="powerbi/data"):
    """Renders the Traffic Intelligence Dashboard in Streamlit."""
    st.markdown("## 🚦 Traffic Intelligence Dashboard")
    st.markdown("*Interactive Analytics, Time-Series Demand Forecasting, Operational Bias & MLOps Monitoring Subsystem*")
    
    # Load all 10 datasets safely
    try:
        data = load_all_datasets(data_dir=data_dir)
    except Exception as e:
        st.error(f"⚠️ Error loading dashboard data layer from '{data_dir}': {e}")
        st.info("Ensure all 10 CSV datasets are generated. Run `python3 powerbi/export_powerbi_data.py`.")
        return

    df_hourly = data["traffic_hourly"]
    df_daily = data["traffic_daily"]
    df_monthly = data["traffic_monthly"]
    df_forecast = data["demand_forecast"]
    df_model_perf = data["model_performance"]
    df_error = data["error_analysis"]
    df_biz = data["business_kpis"]
    df_monitoring = data["monitoring_metrics"]
    dim_date = data["dim_date"]
    dim_time = data["dim_time"]

    # -------------------------------------------------------------------------
    # DATA & MODEL CONTEXT EXPANDER
    # -------------------------------------------------------------------------
    with st.expander("ℹ️ Data & Model Context Summary", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Dataset Parameters**")
            st.write(f"- Total Historical Records: `{len(df_hourly):,}`")
            st.write(f"- Date Range: `{df_hourly['date'].min()}` to `{df_hourly['date'].max()}`")
            st.write(f"- Target Variable: `traffic_volume` (veh/hr)")
        with c2:
            st.markdown("**Production Model Context**")
            st.write("- Model: `RandomForestRegressor` (200 trees, depth 10)")
            st.write("- Core Features: `['hour', 'day', 'month', 'weekday', 'is_rush']`")
            st.write("- Feature Order: Maintained strictly unchanged")
        with c3:
            st.markdown("**Forecasting & Monitoring Context**")
            st.write("- Short-term Lags: `lag_1h, lag_24h, lag_168h`")
            st.write("- Rolling Features: `rolling_mean_24h, rolling_std_24h`")
            st.write("- PSI Drift Thresholds: `0.10` (Shift), `0.25` (Drift)")

    # -------------------------------------------------------------------------
    # SIDEBAR FILTERS
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### 🎛️ Dashboard Slicers")
    
    # Date filter
    min_date = pd.to_datetime(dim_date["date"].min()).date()
    max_date = pd.to_datetime(dim_date["date"].max()).date()
    date_range = st.sidebar.date_input("Select Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # Rush status filter
    rush_option = st.sidebar.radio("Rush Hour Filter:", ["All Hours", "Rush Hours Only", "Non-Rush Hours Only"], horizontal=True)

    # Day type filter
    day_option = st.sidebar.selectbox("Day Type Filter:", ["All Days", "Weekdays Only (Mon-Fri)", "Weekends Only (Sat-Sun)"])

    # Apply filters to df_hourly
    filtered_hourly = df_hourly.copy()
    if len(date_range) == 2:
        start_d, end_d = date_range
        start_dt = pd.to_datetime(start_d)
        end_dt = pd.to_datetime(end_d)
        filtered_hourly = filtered_hourly[(filtered_hourly["date"] >= start_dt) & (filtered_hourly["date"] <= end_dt)]

    if rush_option == "Rush Hours Only":
        filtered_hourly = filtered_hourly[filtered_hourly["is_rush"] == 1]
    elif rush_option == "Non-Rush Hours Only":
        filtered_hourly = filtered_hourly[filtered_hourly["is_rush"] == 0]

    if day_option == "Weekdays Only (Mon-Fri)":
        filtered_hourly = filtered_hourly[filtered_hourly["is_weekday"] == 1]
    elif day_option == "Weekends Only (Sat-Sun)":
        filtered_hourly = filtered_hourly[filtered_hourly["is_weekday"] == 0]

    if filtered_hourly.empty:
        st.warning("⚠️ No data available for the selected sidebar filter combination.")
        return

    # Check if active filters modify the hourly dataset
    is_filtered = (len(filtered_hourly) < len(df_hourly)) or (rush_option != "All Hours") or (day_option != "All Days")

    # -------------------------------------------------------------------------
    # MAIN DASHBOARD TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Overview",
        "📈 Demand Patterns",
        "⏱️ Demand Forecasting",
        "🧪 Model & Error Analysis",
        "🛡️ Monitoring"
    ])

    # =========================================================================
    # TAB 1: EXECUTIVE OVERVIEW
    # =========================================================================
    with tab1:
        st.markdown("### 📊 Executive Overview")
        st.caption("High-level traffic volume throughput baselines, demand category breakdown, and key operational metrics.")
        
        exec_kpis = calculate_executive_kpis(filtered_hourly, is_filtered=is_filtered)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Average Traffic Volume", f"{exec_kpis['avg_traffic']:,} veh/hr")
        with col2:
            st.metric("Median Traffic Volume", f"{exec_kpis['median_traffic']:,} veh/hr")
        with col3:
            st.metric("Peak Hourly Volume", f"{exec_kpis['peak_traffic']:,} veh/hr")
        
        rush_label = "Rush vs Non-Rush Shift (Filtered Subset)" if exec_kpis["is_subset"] else "Rush vs Non-Rush Shift (Overall Baseline)"
        weekday_label = "Weekday vs Weekend Shift (Filtered Subset)" if exec_kpis["is_subset"] else "Weekday vs Weekend Shift (Overall Baseline)"
        
        with col4:
            st.metric(rush_label, f"+{exec_kpis['rush_diff']:,} veh/hr", delta=f"+{exec_kpis['rush_pct_shift']}%")
        with col5:
            st.metric(weekday_label, f"+{exec_kpis['weekday_diff']:,} veh/hr", delta=f"+{exec_kpis['weekday_pct_shift']}%")

        st.markdown("---")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig_trend = plot_hourly_trend(filtered_hourly)
            st.plotly_chart(fig_trend, use_container_width=True)
        with col_right:
            fig_pie = plot_volume_distribution(filtered_hourly)
            st.plotly_chart(fig_pie, use_container_width=True)

    # =========================================================================
    # TAB 2: DEMAND PATTERNS
    # =========================================================================
    with tab2:
        st.markdown("### 📈 Traffic Demand Patterns")
        st.caption("Diurnal hourly traffic profiles, weekday vs. weekend distributions, and density heatmaps.")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_diurnal = plot_diurnal_profile(filtered_hourly)
            st.plotly_chart(fig_diurnal, use_container_width=True)
        with col_d2:
            fig_dow = plot_weekday_weekend_bar(filtered_hourly)
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown("---")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_heatmap = plot_hour_weekday_heatmap(filtered_hourly)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        with col_h2:
            fig_month = plot_monthly_trend(df_monthly)
            st.plotly_chart(fig_month, use_container_width=True)

    # =========================================================================
    # TAB 3: DEMAND FORECASTING
    # =========================================================================
    with tab3:
        st.markdown("### ⏱️ Time-Series Demand Forecasting")
        st.caption("Short-term 1-hour ahead traffic demand forecasting benchmarks (ML Lag Random Forest vs Naive Prev-Hour Baseline).")

        fc_kpis = calculate_forecast_kpis(df_model_perf=df_model_perf, df_forecast=df_forecast)

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.metric("Forecast MAE (ML Lag RF)", f"{fc_kpis['ml_mae']} veh/hr")
        with fc2:
            st.metric("Forecast RMSE (ML Lag RF)", f"{fc_kpis['ml_rmse']} veh/hr")
        with fc3:
            st.metric("Forecast R² Score", f"{fc_kpis['ml_r2']}")
        with fc4:
            st.metric("Error Reduction vs Naive", f"-{fc_kpis['mae_reduction_pct']}%", delta=f"vs Naive {fc_kpis['naive_mae']} MAE")

        st.markdown("---")

        fig_fc = plot_actual_vs_forecast(df_forecast)
        st.plotly_chart(fig_fc, use_container_width=True)

        fig_err_time = plot_forecast_error_over_time(df_forecast)
        st.plotly_chart(fig_err_time, use_container_width=True)

    # =========================================================================
    # TAB 4: MODEL & ERROR ANALYSIS
    # =========================================================================
    with tab4:
        st.markdown("### 🧪 Model Performance & Operational Error Analysis")
        st.caption("Comparative model evaluation metrics and segmented residual error breakdown across operational dimensions.")

        st.markdown("#### 1. Architecture Performance Benchmarks")
        st.dataframe(df_model_perf, use_container_width=True)

        fig_comp = plot_model_comparison(df_model_perf)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 2. Segmented Residual Bias Breakdown")
        
        err_kpis = calculate_error_kpis(df_biz=df_biz, df_error=df_error, df_forecast=df_forecast)
        
        eb1, eb2, eb3 = st.columns(3)
        with eb1:
            st.metric("Underpredictions Count (%)", f"{err_kpis['underpred_count']:,} ({err_kpis['underpred_pct']}%)")
        with eb2:
            st.metric("Overpredictions Count (%)", f"{err_kpis['overpred_count']:,} ({err_kpis['overpred_pct']}%)")
        with eb3:
            st.metric("Heavy Volume (>=5000) Bias", f"{err_kpis['heavy_volume_bias']:+.2f} veh/hr", delta=f"MAE: {err_kpis['heavy_volume_mae']}, RMSE: {err_kpis['heavy_volume_rmse']}")

        fig_bias = plot_residual_bias_by_segment(df_error)
        st.plotly_chart(fig_bias, use_container_width=True)

    # =========================================================================
    # TAB 5: MONITORING
    # =========================================================================
    with tab5:
        st.markdown("### 🛡️ MLOps Model Drift & Health Monitoring")
        st.caption("Real-time Population Stability Index (PSI) feature drift evaluation and performance monitoring alert status.")

        mon_perf = df_monitoring.iloc[0]

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Monitored RMSE", f"{mon_perf['monitored_rmse']:.2f} veh/hr")
        with mcol2:
            st.metric("RMSE Alert Threshold", f"{mon_perf['rmse_threshold']:.2f} veh/hr")
        with mcol3:
            alert_status = "🚨 ALERT TRIGGERED" if mon_perf['alert_triggered'] else "✅ HEALTHY / STABLE"
            st.metric("Monitoring Alert Status", alert_status)
        with mcol4:
            pred_psi_row = df_monitoring[df_monitoring["metric_type"] == "Prediction Drift (PSI)"].iloc[0]
            st.metric("Prediction PSI Score", f"{pred_psi_row['psi_score']:.4f}", delta=pred_psi_row['drift_status'])

        if mon_perf['alert_triggered']:
            st.error(f"🚨 **ALERT**: {mon_perf['alert_message']}")
        else:
            st.success(f"✅ **STATUS**: {mon_perf['alert_message']}")

        st.markdown("---")

        fig_drift = plot_feature_drift_bar(df_monitoring)
        st.plotly_chart(fig_drift, use_container_width=True)

        st.markdown("#### MLOps Feature Drift Summary Table")
        st.dataframe(df_monitoring[df_monitoring["metric_type"] == "Feature Drift (PSI)"], use_container_width=True)
