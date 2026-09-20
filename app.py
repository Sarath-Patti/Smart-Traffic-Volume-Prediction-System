import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import hashlib

# ---------- Page Config ----------
st.set_page_config(
    page_title="Traffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ---------- UI Styling ----------
st.markdown("""
<style>
.stButton>button {
    border-radius: 10px;
    height: 42px;
    font-size: 16px;
}
[data-testid="stMetric"] {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
}
.center-box {
    max-width: 400px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------- DATABASE ----------
def connect_db():
    return sqlite3.connect("users.db")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = connect_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

# ---------- AUTH SCREEN ----------
def auth_screen():
    st.markdown('<div class="center-box">', unsafe_allow_html=True)

    st.title("🔐 Traffic Dashboard")
    st.markdown("### Login or Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # LOGIN
    with tab1:
        user = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login", key="login_btn"):
            if login_user(user, pwd):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # REGISTER
    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pwd = st.text_input("New Password", type="password", key="reg_pwd")

        if st.button("Register", key="reg_btn"):
            if create_user(new_user, new_pwd):
                st.success("Account created! Now login.")
            else:
                st.error("User already exists")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MAIN APP ----------
def main_app():
    model = joblib.load("saved_models/Random_Forest.pkl")

    # Header
    col1, col2 = st.columns([8,1])

    with col1:
        st.title("🚦 Traffic Prediction Dashboard")

    with col2:
        if st.button("🚪 Logout"):
            st.warning("Logged out successfully")
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("### Smart traffic Volume prediction using Machine Learning 🚀")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Prediction", "📊 Analytics", "📁 Batch", "🛡️ Monitoring", "📈 DS Analysis", "🚦 Traffic Intelligence"
    ])

    # ===================== TAB 1 =====================
    with tab1:
        st.subheader("🔢 Input Features")

        col1, col2 = st.columns(2)

        with col1:
            hour = st.slider("⏰ Hour", 0, 23, 12)
            day = st.slider("📅 Day", 1, 31, 15)

        with col2:
            month = st.slider("🗓️ Month", 1, 12, 6)
            weekday = st.slider("📆 Weekday (0=Mon, 6=Sun)", 0, 6, 2)

        is_rush = st.toggle("🚦 Rush Hour")

        st.markdown("---")

        if st.button("🚀 Predict Now"):

            with st.spinner("Analyzing traffic patterns... ⏳"):
                time.sleep(0.5)

                features = np.array([[hour, day, month, weekday, int(is_rush)]])
                prediction = model.predict(features)[0]

            st.subheader("📊 Prediction Result")

            # 🔥 smoother animation
            placeholder = st.empty()
            for i in np.linspace(1, prediction, 50):
                placeholder.metric("🚗 Traffic Volume", round(i, 2))
                time.sleep(0.005)

            placeholder.metric("🚗 Traffic Volume", round(prediction, 2))

            if prediction < 2000:
                st.success("🟢 Smooth Traffic Flow")
            elif prediction < 5000:
                st.warning("🟡 Moderate Traffic")
            else:
                st.error("🔴 Heavy Traffic")

            # Log prediction to database
            try:
                from monitoring.monitor import PredictionLogger
                logger = PredictionLogger()
                df_log = pd.DataFrame([{
                    "hour": hour, "day": day, "month": month, "weekday": weekday, "is_rush": int(is_rush)
                }])
                logger.log_predictions(df_log, [prediction])
            except Exception as e:
                pass

    # ===================== TAB 2 =====================
    with tab2:
        st.subheader("📊 Traffic Analytics")

        if hasattr(model, "feature_importances_"):
            df_imp = pd.DataFrame({
                "Feature": ["Hour", "Day", "Month", "Weekday", "Rush"],
                "Importance": model.feature_importances_
            })

            fig = px.bar(df_imp, x="Feature", y="Importance", color="Importance", title="Random Forest Feature Importances")
            st.plotly_chart(fig, use_container_width=True)

        hours = list(range(24))
        sample = np.array([[h, 15, 6, 2, 1] for h in hours])
        preds = model.predict(sample)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=preds, mode='lines+markers', name="Predicted Volume"))
        fig.update_layout(title="Diurnal Prediction Curve (24 Hours)", xaxis_title="Hour of Day", yaxis_title="Predicted Volume")

        st.plotly_chart(fig, use_container_width=True)

    # ===================== TAB 3 =====================
    with tab3:
        st.subheader("📁 Batch Prediction")

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            data = pd.read_csv(file)

            if "date_time" in data.columns:
                data['date_time'] = pd.to_datetime(data['date_time'], dayfirst=True)
                data['hour'] = data['date_time'].dt.hour
                data['day'] = data['date_time'].dt.day
                data['month'] = data['date_time'].dt.month
                data['weekday'] = data['date_time'].dt.weekday
                data['is_rush'] = data['hour'].apply(
                    lambda x: 1 if x in [7,8,9,17,18,19] else 0
                )

            if {"hour", "day", "month", "weekday", "is_rush"}.issubset(data.columns):
                preds = model.predict(
                    data[["hour", "day", "month", "weekday", "is_rush"]]
                )
                data["Prediction"] = preds

                st.dataframe(data)

                st.download_button(
                    "⬇ Download Results",
                    data.to_csv(index=False),
                    "predictions.csv"
                )
            else:
                st.error("CSV must contain required columns")

    # ===================== TAB 4 =====================
    with tab4:
        st.subheader("🛡️ Model Drift & Performance Monitoring")
        st.info("ℹ️ **Historical Simulation / Demonstration Mode**: Real-time evaluation of data drift (PSI), prediction drift, and model performance metrics.")

        try:
            from monitoring.monitor import (
                FeatureDriftMonitor, PredictionDriftMonitor, PerformanceMonitor,
                ModelAlertManager, SyntheticDriftGenerator, generate_monitoring_report
            )
            import json

            ref_path = "monitoring/reference_data.csv"
            if not os.path.exists(ref_path):
                st.error("Reference data file missing.")
            else:
                ref_df = pd.read_csv(ref_path)

                sim_mode = st.radio(
                    "Select Monitoring Dataset Stream:",
                    ["Normal Historical Holdout (2018 Test Data)", "Synthetic Feature Drift Test (Simulated Peak Shift)"],
                    horizontal=True
                )

                # Prepare test batch
                df_full = pd.read_csv("datafile.csv").drop_duplicates()
                df_full['date_time'] = pd.to_datetime(df_full['date_time'], dayfirst=True)
                df_full['hour'] = df_full['date_time'].dt.hour
                df_full['day'] = df_full['date_time'].dt.day
                df_full['month'] = df_full['date_time'].dt.month
                df_full['weekday'] = df_full['date_time'].dt.weekday
                df_full['is_rush'] = df_full['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

                # Get holdout set
                split_idx = int(len(df_full) * 0.85)
                current_df = df_full.iloc[split_idx:].copy().sample(n=1000, random_state=42)

                if "Synthetic" in sim_mode:
                    current_df = SyntheticDriftGenerator.generate_drifted_sample(current_df, target_hour=8, force_rush=1)

                report = generate_monitoring_report(ref_df, current_df, model, rmse_threshold=550.0, dataset_label=sim_mode)

                # 1. ALERT BANNER
                st.markdown("### 1. Model Health & Alert Status")
                alert = report["alert"]
                if alert and alert["Alert Triggered"]:
                    st.error(f"🚨 **STATUS: {alert['Status']}**\n\n{alert['Message']}")
                elif alert:
                    st.success(f"✅ **STATUS: {alert['Status']}**\n\n{alert['Message']}")

                # 2. FEATURE DRIFT TABLE
                st.markdown("### 2. Feature Distribution Drift (PSI Metric)")
                st.caption("PSI Rule: <0.10 (No Drift / Stable), 0.10-0.25 (Moderate Shift), >=0.25 (Significant Drift)")
                drift_df = pd.DataFrame(report["feature_drift"])
                st.dataframe(drift_df, use_container_width=True)

                # 3. PREDICTION DRIFT & PERFORMANCE
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 3. Prediction Drift")
                    p_drift = report["prediction_drift"]
                    st.metric("Prediction PSI", p_drift["Prediction PSI"], delta=p_drift["Prediction Drift Status"])
                    st.write(f"**Reference Mean Volume**: {p_drift['Reference Mean']:.2f}")
                    st.write(f"**Current Mean Volume**: {p_drift['Current Mean']:.2f}")
                    st.write(f"**Wasserstein Distance**: {p_drift['Wasserstein Distance']:.2f}")

                with col2:
                    st.markdown("### 4. Model Performance (Evaluated)")
                    perf = report["performance_metrics"]
                    if perf:
                        st.metric("Monitored RMSE", perf["RMSE"])
                        st.metric("Monitored R² Score", perf["R2 Score"])
                        st.metric("Monitored MAE", perf["MAE"])
                    else:
                        st.warning("No ground-truth actuals available for performance calculation.")

                # 5. MODEL METADATA
                st.markdown("### 5. Model Version & Metadata")
                meta = report["model_metadata"]
                if meta:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.write(f"**Model Name**: {meta.get('model_name', 'Random Forest')}")
                        st.write(f"**Version**: {meta.get('model_version', 'v1.0.0')}")
                    with col_m2:
                        st.write(f"**Training Date**: {meta.get('training_date', '2026-09-16')}")
                        st.write(f"**Ref Dataset Size**: {meta.get('reference_dataset_size', 40958)} rows")
                    with col_m3:
                        st.write(f"**Features**: `{meta.get('features', [])}`")
                        st.write(f"**Hyperparameters**: `{meta.get('hyperparameters', {})}`")

        except Exception as e:
            st.error(f"Error initializing Monitoring System: {e}")

    # ===================== TAB 5 =====================
    with tab5:
        st.subheader("📈 Data Science & Advanced Analytics")
        st.info("ℹ️ **Advanced DS Module**: Explore SQL analytical queries, empirical statistical tests, time-series forecasting benchmarks, and segmented error analysis.")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "🗄️ SQL Analytics", "📐 Statistical Hypothesis Testing", "⏱️ Demand Forecasting", "🔍 Operational Error Analysis", "💼 Business KPIs & Decision Analysis"
        ])

        with sub_tab1:
            st.markdown("### 🗄️ SQLite Analytical Views & Queries")
            st.caption("Demonstrating relational SQL analytical queries with window functions, CTEs, and aggregations.")
            try:
                if os.path.exists("traffic_analytics.db"):
                    conn = sqlite3.connect("traffic_analytics.db")
                    views = [
                        "v_daily_traffic_summary", "v_hourly_traffic_summary",
                        "v_day_type_comparison", "v_rush_hour_summary",
                        "v_monthly_traffic_trends", "v_rolling_24h_traffic",
                        "v_peak_traffic_rankings", "v_traffic_volume_buckets",
                        "v_weather_impact_summary"
                    ]
                    selected_view = st.selectbox("Select SQL View:", views)
                    df_view = pd.read_sql_query(f"SELECT * FROM {selected_view} LIMIT 100", conn)
                    st.dataframe(df_view, use_container_width=True)
                    st.caption(f"Showing top rows from `{selected_view}` in SQLite `traffic_analytics.db`.")
                    conn.close()
                else:
                    st.warning("SQL Database `traffic_analytics.db` not found. Run `python run_analysis.py` to generate.")
            except Exception as e:
                st.error(f"Error loading SQL analytics: {e}")

        with sub_tab2:
            st.markdown("### 📐 Empirical Statistical Hypothesis Testing")
            try:
                from analysis.statistical_analysis import test_rush_vs_non_rush, test_weekday_vs_weekend, calculate_correlations_with_pvalues
                df_data = pd.read_csv("datafile.csv").drop_duplicates()

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 1. Rush vs Non-Rush Hour Test")
                    rush_res = test_rush_vs_non_rush(df_data)
                    cohen_rush = rush_res["Cohen's d Effect Size"]
                    st.write(f"**Rush Sample (N1)**: {rush_res['Rush Hour Sample Count (N1)']}")
                    st.write(f"**Rush Mean Volume**: {rush_res['Rush Hour Mean Volume']:.2f}")
                    st.write(f"**Rush Median Volume**: {rush_res['Rush Hour Median Volume']:.2f}")
                    st.write(f"**Rush 95% CI**: `{rush_res['Rush Hour Mean 95% CI']}`")
                    st.write(f"**Non-Rush Sample (N2)**: {rush_res['Non-Rush Hour Sample Count (N2)']}")
                    st.write(f"**Non-Rush Mean Volume**: {rush_res['Non-Rush Hour Mean Volume']:.2f}")
                    st.write(f"**Non-Rush Median Volume**: {rush_res['Non-Rush Hour Median Volume']:.2f}")
                    st.write(f"**Non-Rush 95% CI**: `{rush_res['Non-Rush Hour Mean 95% CI']}`")
                    st.write(f"**Welch's t-statistic**: {rush_res['Welch t-statistic']}")
                    st.write(f"**p-value**: {rush_res['Welch t p-value']}")
                    st.write(f"**Cohen's d (Effect Size)**: {cohen_rush}")
                    st.write(f"**Rank-Biserial Correlation**: {rush_res['Rank-Biserial Correlation (r_rb)']}")

                with col2:
                    st.markdown("#### 2. Weekday vs Weekend Test")
                    day_res = test_weekday_vs_weekend(df_data)
                    cohen_day = day_res["Cohen's d Effect Size"]
                    st.write(f"**Weekday Sample (N1)**: {day_res['Weekday Sample Count (N1)']}")
                    st.write(f"**Weekday Mean Volume**: {day_res['Weekday Mean Volume']:.2f}")
                    st.write(f"**Weekday Median Volume**: {day_res['Weekday Median Volume']:.2f}")
                    st.write(f"**Weekday 95% CI**: `{day_res['Weekday Mean 95% CI']}`")
                    st.write(f"**Weekend Sample (N2)**: {day_res['Weekend Sample Count (N2)']}")
                    st.write(f"**Weekend Mean Volume**: {day_res['Weekend Mean Volume']:.2f}")
                    st.write(f"**Weekend Median Volume**: {day_res['Weekend Median Volume']:.2f}")
                    st.write(f"**Weekend 95% CI**: `{day_res['Weekend Mean 95% CI']}`")
                    st.write(f"**Welch's t-statistic**: {day_res['Welch t-statistic']}")
                    st.write(f"**p-value**: {day_res['Welch t p-value']}")
                    st.write(f"**Cohen's d (Effect Size)**: {cohen_day}")
                    st.write(f"**Rank-Biserial Correlation**: {day_res['Rank-Biserial Correlation (r_rb)']}")

                st.markdown("#### 3. Correlation with Target (Traffic Volume)")
                corr_df = calculate_correlations_with_pvalues(df_data)
                st.dataframe(corr_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error running statistical analysis: {e}")

        with sub_tab3:
            st.markdown("### ⏱️ Time-Series Demand Forecasting")
            st.caption("Comparing Naive Lag Baselines against a Time-Aware Random Forest Forecasting Model (Strict chronologically engineered lag features, no data leakage).")
            try:
                if os.path.exists("results/forecasting/forecasting_comparison.csv"):
                    df_fc = pd.read_csv("results/forecasting/forecasting_comparison.csv")
                    st.dataframe(df_fc, use_container_width=True)

                    fig = px.bar(df_fc, x="Model", y="R2 Score", color="RMSE", title="Forecasting Model Comparison (R² Score)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Forecasting results missing. Run `python run_analysis.py` to generate.")
            except Exception as e:
                st.error(f"Error loading forecasting analysis: {e}")

        with sub_tab4:
            st.markdown("### 🔍 Segmented Operational Error Analysis")
            st.caption("Dissecting model residuals across operational dimensions (traffic volume buckets, hour of day, rush status) to detect bias and error patterns.")
            try:
                if os.path.exists("results/error_analysis/error_by_volume_bucket.csv"):
                    vol_err = pd.read_csv("results/error_analysis/error_by_volume_bucket.csv")
                    st.markdown("#### Performance by Volume Bucket")
                    st.dataframe(vol_err, use_container_width=True)

                    if os.path.exists("results/error_analysis/error_by_hour.csv"):
                        hr_err = pd.read_csv("results/error_analysis/error_by_hour.csv")
                        fig_err = px.line(hr_err, x="hour", y="MAE", title="Mean Absolute Error (MAE) by Hour of Day")
                        st.plotly_chart(fig_err, use_container_width=True)
                else:
                    st.warning("Error analysis results missing. Run `python run_analysis.py` to generate.")
            except Exception as e:
                st.error(f"Error loading operational error analysis: {e}")

        with sub_tab5:
            st.markdown("### 💼 Business KPIs & Operational Decision Analysis")
            st.caption("Reproducible operational metrics, demand profile KPIs, and decision analysis answering key operational questions.")
            try:
                from analysis.business_analysis import BusinessDecisionAnalyzer
                analyzer = BusinessDecisionAnalyzer(csv_path="datafile.csv", model_path="saved_models/Random_Forest.pkl")
                kpis = analyzer.generate_summary_report()
                
                d_kpis = kpis["demand_kpis"]
                m_kpis = kpis["model_kpis"]

                st.markdown("#### 1. Demand Profile KPIs")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Average Traffic Volume", f"{d_kpis['Average Traffic Volume (veh/hr)']} veh/hr")
                    st.metric("Median Traffic Volume", f"{d_kpis['Median Traffic Volume (veh/hr)']} veh/hr")
                with c2:
                    st.metric("Peak Hourly Volume", f"{d_kpis['Peak Single-Hour Volume (veh/hr)']} veh/hr")
                    st.metric("Rush vs Non-Rush Shift", d_kpis["Rush vs Non-Rush Demand Difference"])
                with c3:
                    st.metric("Weekday vs Weekend Shift", d_kpis["Weekday vs Weekend Demand Difference"])
                    st.write(f"**Top Peak Hours**: {d_kpis['Top Peak Demand Hours']}")

                st.markdown("#### 2. Operational Error & Model Bias KPIs")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Holdout Model MAE", f"{m_kpis['Chronological Holdout Model MAE']} veh/hr")
                    st.metric("Heavy Volume (>=5000) MAE", f"{m_kpis['Heavy-Volume (>=5000) MAE']} veh/hr")
                with mc2:
                    st.metric("Overall Mean Bias Error", m_kpis["Overall Mean Bias Error"])
                    st.metric("Heavy Volume Mean Bias", m_kpis["Heavy-Volume Mean Bias Error"])

                st.markdown("#### 3. Operational Decision Q&A Summary")
                if os.path.exists("results/business/business_summary.md"):
                    with open("results/business/business_summary.md", "r") as f:
                        st.markdown(f.read())
            except Exception as e:
                st.error(f"Error loading business decision analysis: {e}")

    # ===================== TAB 6 =====================
    with tab6:
        try:
            from dashboard.renderers import render_traffic_intelligence_dashboard
            render_traffic_intelligence_dashboard()
        except Exception as e:
            st.error(f"Error rendering Traffic Intelligence Dashboard: {e}")

# ---------- ROUTING ----------
if not st.session_state.logged_in:
    auth_screen()
else:
    main_app()

