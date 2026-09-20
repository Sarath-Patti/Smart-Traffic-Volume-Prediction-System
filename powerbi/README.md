# 📊 Power BI Analytics & Reporting Layer

The `powerbi/` module provides a production-ready, star-schema data layer, complete DAX measures specification, dashboard architecture document, and data export/validation automation for Microsoft Power BI reporting.

---

## 📌 1. Architecture & Data Sources

The Power BI data layer is populated directly from the project's cleaned raw historical dataset (`datafile.csv`), machine learning forecasting models (`forecasting/forecasting.py`), operational error analysis (`analysis/error_analysis.py`), business decision analyzer (`analysis/business_analysis.py`), and MLOps drift monitor (`monitoring/monitor.py`).

### Data Refresh Procedure
To regenerate and re-validate all 10 Power BI CSV datasets cleanly, run:

```bash
# Export all 10 clean Power BI CSV datasets
python3 powerbi/export_powerbi_data.py

# Validate schema, row counts, null values, and metric integrity
python3 powerbi/validate_data.py
```

---

## 📁 2. Exported Datasets & Field Definitions

| File Name | Grain | Row Count | Primary Key / Join Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `traffic_hourly.csv` | 1 Hour | 48,187 | `date_time`, `date`, `hour` | Full hourly observation fact table |
| `traffic_daily.csv` | 1 Day | 1,860 | `date` | Pre-aggregated daily throughput summaries |
| `traffic_monthly.csv` | 1 Month | 63 | `year_month` | Monthly throughput & peak tracking |
| `demand_forecast.csv` | 1 Hour | 7,225 | `date_time` | Holdout test set actuals vs lag predictions |
| `model_performance.csv`| Model | 3 | `model_name` | Evaluation metrics (MAE, RMSE, $R^2$) |
| `error_analysis.csv` | Segment | 36 | `segment_type`, `segment_value` | Residual bias across volume/hour segments |
| `business_kpis.csv` | KPI | 24 | `kpi_name` | Summary business metrics |
| `monitoring_metrics.csv`| Feature | 6 | `feature_name` | MLOps drift (PSI) and alert statuses |
| `dim_date.csv` | 1 Day | 2,190 | `date` | Date dimension table (calendar attributes) |
| `dim_time.csv` | 1 Hour | 24 | `hour` | Time dimension table (hourly periods) |

---

## 📐 3. Star Schema Data Model Specification

```
                          ┌────────────────┐
                          │   DimDate      │
                          │ (dim_date.csv) │
                          └───────┬────────┘
                                  │ 1
                                  │
                                  │ *
┌────────────────┐ 1           * ┌┴────────────────┐ *           1 ┌────────────────┐
│   DimTime      ├───────────────┤  FactTraffic    ├───────────────┤  FactForecast  │
│ (dim_time.csv) │               │(traffic_hourly) │               │(demand_forecast│
└────────────────┘               └─────────────────┘               └────────────────┘
```

---

## 📄 4. Dashboard Page Specifications

* **Page 1: Traffic Overview** — High-level throughput, diurnal curves, weekday vs weekend volume shifts.
* **Page 2: Demand & Forecast** — Short-term forecasting evaluation (actual vs predicted, error distribution over time).
* **Page 3: Model Performance** — Comparative metrics across modeling strategies, error breakdown by hour/volume bucket.
* **Page 4: Business / Operational Insights** — Key demand profile KPIs, capacity surge underpredictions, operational priorities.
* **Page 5: MLOps Drift Monitoring** — Feature & prediction distribution drift (PSI), RMSE performance alerts.

*For complete DAX measures, see [powerbi/measures.md](file:///Users/sarathpatti/Desktop/Traffice-Volume/powerbi/measures.md).*  
*For full visual build documentation, see [powerbi/dashboard_spec.md](file:///Users/sarathpatti/Desktop/Traffice-Volume/powerbi/dashboard_spec.md).*

---

## ⚠️ 5. Assumptions & Limitations

1. **Environment Compatibility**: Power BI Desktop is a Windows-only desktop application. In Linux/macOS environments, the complete Power BI-ready datasets, star schema, DAX measures, and dashboard build guide are provided so the `.pbix` file can be built directly in Power BI Desktop on Windows.
2. **Observational Scope**: All Power BI visuals and measures reflect observational traffic patterns; causal business claims regarding revenue or staffing are not asserted.
