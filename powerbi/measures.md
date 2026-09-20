# 📐 Power BI DAX Measures & Calculations

This document provides exact, production-ready DAX measure expressions for dynamic calculation of traffic KPIs, forecasting metrics, error metrics, and monitoring statuses in Microsoft Power BI.

---

## 📊 1. Traffic Volume & Demand Measures

```dax
// Average Traffic Volume
Average Traffic Volume = 
AVERAGE(traffic_hourly[traffic_volume])

// Total Traffic Volume
Total Traffic Volume = 
SUM(traffic_hourly[traffic_volume])

// Peak Traffic Volume
Peak Traffic Volume = 
MAX(traffic_hourly[traffic_volume])

// Rush Hour Average Volume
Rush Hour Average Volume = 
CALCULATE(
    AVERAGE(traffic_hourly[traffic_volume]),
    traffic_hourly[is_rush] = 1
)

// Non-Rush Hour Average Volume
Non-Rush Hour Average Volume = 
CALCULATE(
    AVERAGE(traffic_hourly[traffic_volume]),
    traffic_hourly[is_rush] = 0
)

// Rush Hour Volume Difference (%)
Rush Hour Shift Pct = 
VAR RushAvg = [Rush Hour Average Volume]
VAR NonRushAvg = [Non-Rush Hour Average Volume]
RETURN
DIVIDE(RushAvg - NonRushAvg, NonRushAvg, 0)

// Weekday Average Volume
Weekday Average Volume = 
CALCULATE(
    AVERAGE(traffic_hourly[traffic_volume]),
    traffic_hourly[is_weekday] = 1
)

// Weekend Average Volume
Weekend Average Volume = 
CALCULATE(
    AVERAGE(traffic_hourly[traffic_volume]),
    traffic_hourly[is_weekday] = 0
)

// Weekday Volume Difference (%)
Weekday Shift Pct = 
VAR WkdayAvg = [Weekday Average Volume]
VAR WkendAvg = [Weekend Average Volume]
RETURN
DIVIDE(WkdayAvg - WkendAvg, WkendAvg, 0)
```

---

## ⏱️ 2. Forecasting & Model Performance DAX Measures

```dax
// Actual Volume (Forecast Dataset)
Actual Volume (Forecast) = 
AVERAGE(demand_forecast[actual_traffic_volume])

// ML Forecast Predicted Volume
ML Forecast Volume = 
AVERAGE(demand_forecast[ml_rf_forecast_pred])

// Naive Prev-Hour Predicted Volume
Naive Prev-Hour Volume = 
AVERAGE(demand_forecast[naive_lag1h_pred])

// ML Forecast Mean Absolute Error (MAE)
Forecast MAE (ML) = 
AVERAGE(demand_forecast[ml_rf_abs_error])

// Naive Baseline MAE
Forecast MAE (Naive) = 
AVERAGE(demand_forecast[naive_lag1h_abs_error])

// ML Forecast Root Mean Squared Error (RMSE)
Forecast RMSE (ML) = 
SQRT(AVERAGE(demand_forecast[ml_rf_squared_error]))

// Forecast Error Reduction vs Naive (%)
Forecast Error Reduction Pct = 
VAR ML_MAE = [Forecast MAE (ML)]
VAR Naive_MAE = [Forecast MAE (Naive)]
RETURN
DIVIDE(ML_MAE - Naive_MAE, Naive_MAE, 0)

// Mean Bias Error (ML Forecast)
Forecast Mean Bias = 
AVERAGE(demand_forecast[ml_rf_error])
```

---

## 🔍 3. Operational Error & Model Bias Measures

```dax
// Overall Mean Bias Error (Holdout Set)
Holdout Mean Bias = 
AVERAGE(error_analysis[mean_bias_error])

// Heavy-Volume (>=5000) Mean Bias Error
Heavy Volume Mean Bias = 
CALCULATE(
    AVERAGE(error_analysis[mean_bias_error]),
    error_analysis[segment_type] = "Volume Bucket",
    error_analysis[segment_label] = "Heavy (>=5000)"
)

// Heavy-Volume (>=5000) MAE
Heavy Volume MAE = 
CALCULATE(
    AVERAGE(error_analysis[mae]),
    error_analysis[segment_type] = "Volume Bucket",
    error_analysis[segment_label] = "Heavy (>=5000)"
)
```

---

## 🛡️ 4. MLOps Monitoring DAX Measures

```dax
// Monitored RMSE
Monitored RMSE = 
MAX(monitoring_metrics[monitored_rmse])

// RMSE Threshold
RMSE Threshold = 
MAX(monitoring_metrics[rmse_threshold])

// Maximum Feature PSI Score
Max Feature PSI = 
CALCULATE(
    MAX(monitoring_metrics[psi_score]),
    monitoring_metrics[metric_type] = "Feature Drift (PSI)"
)

// Prediction PSI Score
Prediction PSI = 
CALCULATE(
    MAX(monitoring_metrics[psi_score]),
    monitoring_metrics[metric_type] = "Prediction Drift (PSI)"
)

// Drift Alert Indicator KPI Color
Alert Status Color = 
VAR IsAlert = MAX(monitoring_metrics[alert_triggered])
RETURN
IF(IsAlert = 1, "#D9534F", "#5CB85C")
```
