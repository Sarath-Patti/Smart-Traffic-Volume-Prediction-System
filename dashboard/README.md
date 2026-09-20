# 🚦 Traffic Intelligence Dashboard Module

A modular, interactive browser-based analytics dashboard built for the Smart Traffic Volume Prediction System.

---

## 🏗️ Module Architecture

```
dashboard/
├── __init__.py           # Package exports
├── data_loader.py       # Data loading, caching, and schema validation
├── metrics.py           # KPI calculations & unit formatting
├── charts.py            # Interactive Plotly chart builders
├── renderers.py         # Streamlit dashboard section renderers
└── README.md            # Dashboard documentation
```

---

## 📊 Dashboard Sections

1. **Executive Overview**: High-level traffic volume baselines, peak volumes, rush/weekday percentage shifts, timeline trends, and volume bucket distributions.
2. **Demand Patterns**: Diurnal hourly profiles, weekday vs weekend comparisons, hour × weekday density heatmaps, monthly throughput trends.
3. **Demand Forecasting**: 1-hour ahead ML Lag Random Forest forecasting benchmarks vs naive baselines, error timelines.
4. **Model Performance & Error Analysis**: Model architecture evaluations, segmented residual error bias across volume buckets and hours.
5. **MLOps Monitoring**: Population Stability Index (PSI) feature drift evaluation and RMSE performance alert statuses.

---

## 🛠️ Usage & Integration

The dashboard is integrated directly into the main Streamlit application (`app.py`) under the **Traffic Intelligence** navigation entry.

To run the application:
```bash
streamlit run app.py
```
