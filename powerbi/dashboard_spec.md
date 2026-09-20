# 📊 Power BI Dashboard Build Specification

This document details the complete 5-page Power BI dashboard design specification, visual hierarchy, slicers, star schema relationships, and layout rules for rebuilding the interactive report in Power BI Desktop.

---

## 🏗️ Star Schema Data Model & Relationships

```
+-----------------------------------------------------------------------------------+
| TABLE RELATIONSHIPS (Power BI Relationship View)                                  |
+----------------------+--------------------+---------------------+-----------------+
| From Table           | From Column        | To Table            | To Column       |
+----------------------+--------------------+---------------------+-----------------+
| traffic_hourly       | date               | dim_date            | date            |
| traffic_hourly       | hour               | dim_time            | hour            |
| demand_forecast      | hour               | dim_time            | hour            |
+----------------------+--------------------+---------------------+-----------------+
```
* **Cardinality**: `1-to-Many` (Dimensions to Fact Tables)
* **Cross Filter Direction**: `Single` (Dimension filters Fact)

---

## 📄 Page 1: Traffic Overview

**Target Audience**: City Traffic Managers & Operations Analysts  
**Objective**: Monitor overall traffic volume baselines, diurnal hourly patterns, and day-type volume shifts.

### KPI Cards (Top Banner):
1. **Average Traffic**: `3,259.82 veh/hr` (DAX: `[Average Traffic Volume]`)
2. **Peak Traffic**: `7,280 veh/hr` (DAX: `[Peak Traffic Volume]`)
3. **Rush-Hour Average**: `4,780.89 veh/hr` (DAX: `[Rush Hour Average Volume]`)
4. **Weekday vs Weekend Shift**: `+70.86%` (DAX: `[Weekday Shift Pct]`)

### Visual Layout:
* **Visual 1 (Line Chart - Main Body)**: Hourly Traffic Diurnal Profile (X-Axis: `hour` from `dim_time`, Y-Axis: `[Average Traffic Volume]`, Legend: `is_weekday`). Shows clear bimodal weekday peaks vs smooth weekend curves.
* **Visual 2 (Bar Chart - Bottom Left)**: Monthly Traffic Volume Trend (X-Axis: `year_month`, Y-Axis: `[Average Traffic Volume]`).
* **Visual 3 (Clustered Column Chart - Bottom Right)**: Day of Week Traffic Profile (X-Axis: `weekday_name`, Y-Axis: `[Average Traffic Volume]`).
* **Slicers (Top Right Sidebar)**: Date Range (`dim_date[date]`), Year (`dim_date[year]`), Weather Main (`traffic_hourly[weather_main]`).

---

## ⏱️ Page 2: Demand & Forecast

**Target Audience**: Data Scientists & Forecasting Engineers  
**Objective**: Evaluate short-term 1-hour ahead ML time-series forecasts against naive lag baselines.

### KPI Cards (Top Banner):
1. **ML Forecast MAE**: `158.14 veh/hr` (DAX: `[Forecast MAE (ML)]`)
2. **ML Forecast RMSE**: `240.38 veh/hr` (DAX: `[Forecast RMSE (ML)]`)
3. **ML Forecast R²**: `0.9853` (DAX: `[ML Forecast R2]`)
4. **Error Reduction vs Baseline**: `-67.46%` (DAX: `[Forecast Error Reduction Pct]`)

### Visual Layout:
* **Visual 1 (Line Chart - Time Series Overlay)**: Actual vs Predicted Volume Over Time (X-Axis: `date_time`, Y-Axis: `actual_traffic_volume`, `ml_rf_forecast_pred`, `naive_lag1h_pred`). Demonstrates near-perfect ML tracking during rapid volume transitions.
* **Visual 2 (Scatter Plot - Forecast Error vs Volume)**: Forecast Error Distribution (X-Axis: `actual_traffic_volume`, Y-Axis: `ml_rf_error`).
* **Visual 3 (Column Chart - MAE by Hour)**: Forecast MAE by Hour of Day (X-Axis: `hour`, Y-Axis: `[Forecast MAE (ML)]`).
* **Slicers**: Date Range Slicer, Hour Slicer.

---

## 🧪 Page 3: Model Performance & Error Breakdown

**Target Audience**: ML Engineers & Operational Auditors  
**Objective**: Compare model architectures and dissect residual bias across operational segments.

### Top Summary Table:
* **Model Strategy Comparison Table** (from `model_performance.csv`): Columns: `model_name`, `features_used`, `mae`, `rmse`, `r2_score`.

### Visual Layout:
* **Visual 1 (Clustered Bar Chart)**: Model Architecture MAE & RMSE Comparison (X-Axis: `mae`/`rmse`, Y-Axis: `model_name`). Highlights Random Forest supremacy.
* **Visual 2 (Line Chart)**: Absolute Error (MAE) Across Hour of Day (X-Axis: `hour`, Y-Axis: `mae`). Shows peak error during 07:00-08:00 and 16:00-17:00 transition hours.
* **Visual 3 (Bar Chart)**: Mean Bias Error by Traffic Volume Bucket (X-Axis: `segment_label`, Y-Axis: `mean_bias_error`). Visualizes positive bias (+182.40 veh/hr) during heavy volume surges ($\ge 5000$).
* **Visual 4 (Card)**: Heavy Volume Bias Alert Banner (`+182.40 veh/hr (Underpredicting peak spikes)`).

---

## 💼 Page 4: Business / Operational Insights

**Target Audience**: Executive Decision Makers & Operations Supervisors  
**Objective**: Answer practical operational decision questions using empirical traffic KPIs.

### Visual Layout:
* **Matrix Visual (Center)**: Key Operational Decision Q&A (derived from `business_kpis.csv` and `business_summary.md`):
  * *When is demand highest?* -> Weekday rush hours (16:00-18:00 & 07:00-09:00), peaking in August and October.
  * *Where is forecasting hardest?* -> Transition hours (07:00, 16:00) and heavy volume surges ($\ge 5000$ veh/hr).
  * *Is rush-hour demand significantly higher?* -> Yes (+72.42% higher, Cohen's $d = 1.1594$, Very Large Effect).
  * *When does the model underpredict?* -> Heavy traffic surges ($\ge 5000$ veh/hr, mean bias = $+182.40$ veh/hr).
* **Donut Chart**: Traffic Volume Bucket Distribution (Low 37.5%, Moderate 37.4%, Heavy 25.1%).
* **Bar Chart**: Top 5 Peak Volume Hours per Year (`v_peak_traffic_rankings`).

---

## 🛡️ Page 5: MLOps Drift Monitoring

**Target Audience**: MLOps Engineers & System Administrators  
**Objective**: Monitor feature distribution drift (PSI), prediction drift, and model performance alerts.

### Top KPI Banner:
1. **Monitored RMSE**: `542.10 veh/hr` (DAX: `[Monitored RMSE]`)
2. **RMSE Threshold**: `550.00 veh/hr` (DAX: `[RMSE Threshold]`)
3. **Alert Status**: `✅ Performance Within Threshold`
4. **Prediction PSI**: `0.0214 (No Drift)`

### Visual Layout:
* **Table Visual (Main Body)**: Feature Drift PSI Table (from `monitoring_metrics.csv`): Columns: `feature_name`, `reference_mean`, `current_mean`, `psi_score`, `drift_status`.
* **Gauge Visual**: Monitored RMSE vs 550.0 Threshold (Target Value: `550`, Actual Value: `542.10`).
* **KPI Card (Dynamic Color)**: Drift Alert Status (Green if no alert, Red if alert triggered).
