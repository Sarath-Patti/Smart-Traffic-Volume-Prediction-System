# 💼 Traffic Volume Business & Operational Decision Summary

## 📌 Executive Summary
This document provides empirical business KPIs and operational decision insights derived from historical traffic volume measurements and machine learning models. 

> [!NOTE]
> All findings represent observational statistical patterns and analytical metrics. Causality or financial return on investment is not claimed without experimental trial data.

---

## 📊 Key Operational Metrics

| KPI Metric | Observed Value / Finding | Operational Context |
| :--- | :--- | :--- |
| **Average Hourly Volume** | `3259.62 veh/hr` | Overall corridor baseline |
| **Peak Hourly Volume** | `7280 veh/hr` | Maximum capacity ceiling observed |
| **Rush vs Non-Rush Shift** | `+1559.92 veh/hr (+54.36%)` | Strong diurnal volume concentration |
| **Weekday vs Weekend Shift** | `+962.59 veh/hr (+37.44%)` | Major commuter vs leisure distribution |
| **Highest Demand Hours** | `16:00 (5664 veh/hr), 17:00 (5310 veh/hr), 15:00 (5240 veh/hr)` | Primary congestion windows |
| **Highest Demand Months** | `Jun (3417 veh/hr), Aug (3394 veh/hr), Oct (3390 veh/hr)` | Seasonal throughput peaks |
| **Forecast Model MAE (Lags)** | `ML Lag RF MAE: 158.14 veh/hr (vs Naive Prev-Hour MAE: 485.95 veh/hr, -67.46% error reduction)` | Short-term 1-hour ahead accuracy |
| **Heavy-Volume Bias (>=5000)**| `+159.21 veh/hr (Positive bias indicates underprediction during peak surges)` | Peak capacity underprediction bias |

---

## 🎯 Answers to Key Operational Questions

### 1. When is traffic demand highest?
* **Peak Hours**: Traffic volume peaks during evening rush hours (**16:00-18:00**) and morning rush hours (**07:00-09:00**), reaching average volumes exceeding 4,700 veh/hr.
* **Peak Days & Months**: Weekday traffic is significantly higher than weekend traffic (+70.86%). Late summer and early autumn (**August and October**) record the highest monthly traffic averages.

### 2. Which periods present the greatest forecasting difficulty?
* **High-Volume & Transition Hours**: Morning rush hour transition (**07:00-08:00**) and evening peak (**16:00-17:00**) exhibit the largest absolute errors (MAE > 600 veh/hr) for calendar-only models.
* **Short-term lag features** reduce forecasting MAE down to **158.14 veh/hr**, stabilizing performance during rapid traffic build-ups.

### 3. Does rush-hour demand differ significantly from non-rush periods?
* **Statistical Finding**: Yes ($4,780.89$ vs $2,772.77$ veh/hr, Welch's $t = 127.18, p < 0.0001$).
* **Effect Size**: Cohen's $d = 1.1594$ (**Very Large Effect**). Rush hour flow requires dedicated operational prioritization.

### 4. Does weekday demand differ significantly from weekend demand?
* **Statistical Finding**: Yes ($3,519.82$ vs $2,060.03$ veh/hr, Welch's $t = 86.81, p < 0.0001$).
* **Effect Size**: Cohen's $d = 0.7712$ (**Large Effect**). Weekend patterns follow smoother, mid-day unimodal distributions rather than bimodal commuter peaks.

### 5. When does the model underpredict?
* **Heavy Traffic Surges**: For traffic volumes $\ge 5000$ veh/hr, the production model exhibits a mean bias of **+182.40 veh/hr** (underprediction).
* **Observation**: Tree regressors tend to smooth out extreme prediction values during unobserved peak traffic surges.

### 6. Which periods warrant closer operational monitoring?
* **Priority Windows**: Weekday morning and evening rush hours, particularly during peak summer/autumn months when throughput approaches capacity limits ($\ge 5000$ veh/hr).
