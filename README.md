# 🚦 Smart Traffic Volume Prediction & Analytics System

An end-to-end Data Science and Machine Learning engineering project built for production-grade traffic volume prediction, time-series forecasting, statistical hypothesis testing, SQL relational analytics, model drift monitoring, operational error breakdown, and business decision analysis.

---

## 📌 1. Problem Statement

Urban traffic congestion creates severe economic loss, increased fuel consumption, and environmental degradation. Accurate hourly traffic volume prediction empowers smart city traffic management systems to optimize signal timing, manage peak loads, and plan highway infrastructure.

This project delivers a multi-stage Data Science platform that covers the full lifecycle:
1. **Relational Analytics**: SQL window functions and analytical views.
2. **Statistical Inference**: Parametric and non-parametric hypothesis testing with effect size estimation.
3. **Machine Learning & Time-Aware Validation**: Random Forest baseline model evaluated under both random holdout and strict chronological splits.
4. **Demand Forecasting**: Short-term time-series forecasting using chronologically engineered lag and rolling window features.
5. **Model Monitoring**: Real-time distribution shift (Population Stability Index - PSI) and performance drift detection.
6. **Operational Error Breakdown**: Segmented residual analysis to identify structural prediction bias across peak and off-peak periods.
7. **Business & Decision Analysis**: Empirical business KPIs and decision Q&A report answering operational management questions.

---

## 📊 2. Data Overview & Data Hygiene

The project analyzes historical hourly traffic measurements from the Interstate 94 (I-94) westbound corridor in Minneapolis-St. Paul, Minnesota (October 2012 to September 2018).

* **Total Records**: 48,204 raw records -> **48,187 unique records** (17 exact duplicate rows removed).
* **Target Variable**: `traffic_volume` (Continuous, hourly vehicle count; Mean: 3259.8, Std: 1986.9, Min: 0, Max: 7280).
* **Features**:
  * Datetime: `date_time` (Hourly timestamps)
  * Weather Features: `temp` (Kelvin), `rain_1h` (mm), `snow_1h` (mm), `clouds_all` (%), `weather_main`, `weather_description`
  * Missing Values: `holiday` contains 48,126 missing entries (**99.87% missing**). Because it lacks complete temporal coverage, it is excluded from model feature sets.

---

## 📐 3. EDA & Statistical Analysis

Rigorous exploratory and statistical analyses were performed (`analysis/statistical_analysis.py`) to quantify temporal patterns:

### Summary Statistics
| Metric | Traffic Volume (vehicles/hr) |
| :--- | :--- |
| **Mean** | 3259.82 |
| **Median** | 3380.00 |
| **Std Dev** | 1986.94 |
| **95% Confidence Interval for Mean** | [3242.10, 3277.55] |
| **Skewness** | -0.0717 (Symmetric) |
| **Kurtosis** | -1.1969 (Platykurtic / Bimodal distribution) |

### Hypothesis Testing Results
1. **Rush Hour vs. Non-Rush Hour**:
   * **Null Hypothesis ($H_0$)**: $\mu_{\text{rush}} = \mu_{\text{non-rush}}$
   * **Alternative Hypothesis ($H_a$)**: $\mu_{\text{rush}} \neq \mu_{\text{non-rush}}$
   * **Tests Used**: Welch's $t$-test (unadjusted for unequal variances) & Mann-Whitney U (non-parametric rank test)
   * **Sample Sizes**: $N_1 = 12,057$ (Rush) vs $N_2 = 36,130$ (Non-Rush)
   * **Group Statistics**:
     * Rush: Mean = $4780.89$ veh/hr, Median = $4875.00$ veh/hr, 95% CI = $[4756.24, 4805.54]$
     * Non-Rush: Mean = $2752.12$ veh/hr, Median = $2579.50$ veh/hr, 95% CI = $[2732.50, 2771.74]$
   * **Mean Difference**: $+2028.77$ veh/hr (95% CI = $[1999.70, 2057.84]$)
   * **Test Statistics**: Welch $t = 127.18, p < 0.0001$; Mann-Whitney $U = 3.65 \times 10^8, p < 0.0001$
   * **Effect Sizes**: Cohen's $d = 1.1594$ (**Very Large Effect**), Rank-Biserial $r_{rb} = 0.6728$
   * **Interpretation**: Reject $H_0$. Observed rush-hour volume is significantly higher than non-rush volume (Cohen's $d = 1.1594$). *Note: Observational correlation, not direct causality.*

2. **Weekday vs. Weekend**:
   * **Null Hypothesis ($H_0$)**: $\mu_{\text{weekday}} = \mu_{\text{weekend}}$
   * **Alternative Hypothesis ($H_a$)**: $\mu_{\text{weekday}} \neq \mu_{\text{weekend}}$
   * **Tests Used**: Welch's $t$-test & Mann-Whitney U
   * **Sample Sizes**: $N_1 = 34,443$ (Weekday) vs $N_2 = 13,744$ (Weekend)
   * **Group Statistics**:
     * Weekday: Mean = $3519.82$ veh/hr, Median = $3739.00$ veh/hr, 95% CI = $[3499.11, 3540.53]$
     * Weekend: Mean = $2608.20$ veh/hr, Median = $2460.00$ veh/hr, 95% CI = $[2577.80, 2638.60]$
   * **Mean Difference**: $+911.62$ veh/hr (95% CI = $[876.50, 946.74]$)
   * **Test Statistics**: Welch $t = 86.81, p < 0.0001$; Mann-Whitney $U = 3.05 \times 10^8, p < 0.0001$
   * **Effect Sizes**: Cohen's $d = 0.7712$ (**Large Effect**), Rank-Biserial $r_{rb} = 0.2882$
   * **Interpretation**: Reject $H_0$. Observed weekday volume is significantly higher than weekend volume (Cohen's $d = 0.7712$). *Note: Observational correlation, not direct causality.*

3. **Weather Condition Differences (ANOVA & Kruskal-Wallis)**:
   * **ANOVA $F$-statistic**: $21.96$, $p < 0.0001$.
   * **Kruskal-Wallis $H$-statistic**: $216.71$, $p < 0.0001$.
   * *Interpretation*: Traffic volume exhibits statistically significant variance across major weather categories (e.g., Squall/Snow vs. Clear/Clouds).

---

## 🗄️ 4. SQL Analytics Layer

A production-grade SQLite analytical database (`traffic_analytics.db`) is automatically populated via `analysis.sql` and `analysis/sql_analysis.py`. It implements 9 relational views utilizing CTEs, window functions (`AVG() OVER`, `DENSE_RANK() OVER`), CASE expressions, and multi-level aggregations:

1. `v_daily_traffic_summary`: Daily traffic volume aggregations, min, max, std dev.
2. `v_hourly_traffic_summary`: Diurnal hourly profile across the entire multi-year timeline.
3. `v_day_type_comparison`: Comparative metrics between weekdays and weekends.
4. `v_rush_hour_summary`: Granular peak vs off-peak flow metrics.
5. `v_monthly_traffic_trends`: Year-over-year monthly trends.
6. `v_rolling_24h_traffic`: 24-hour moving averages and moving standard deviations using SQL window frame specifications (`ROWS BETWEEN 23 PRECEDING AND CURRENT ROW`).
7. `v_peak_traffic_rankings`: Windows ranking (`DENSE_RANK()`) of the highest volume hours per year.
8. `v_traffic_volume_buckets`: Volume distribution segmented into Low (<2000), Moderate (2000-5000), and Heavy (>=5000) flow buckets.
9. `v_weather_impact_summary`: Traffic volume aggregations grouped by primary weather main category.

---

## ⚙️ 5. Feature Engineering

### Production Model Features (5 Core Features)
The production prediction pipeline (`saved_models/Random_Forest.pkl`) strictly uses 5 temporal features derived from `date_time`:
* `hour`: Hour of day (0 to 23)
* `day`: Day of month (1 to 31)
* `month`: Month of year (1 to 12)
* `weekday`: Day of week (0 = Monday, 6 = Sunday)
* `is_rush`: Binary indicator (1 if hour in [7, 8, 9, 17, 18, 19], else 0)

### Time-Series Forecasting Features
For the demand forecasting module (`forecasting/forecasting.py`), strict chronological lag and rolling window features are engineered using `shift(1)` discipline to eliminate lookahead leakage:
* `lag_1h`: Previous hour's traffic volume ($y_{t-1}$)
* `lag_24h`: Same hour previous day's volume ($y_{t-24}$)
* `lag_168h`: Same hour previous week's volume ($y_{t-168}$)
* `rolling_mean_24h`: 24-hour moving average of past observations
* `rolling_std_24h`: 24-hour moving standard deviation of past observations
* `ewma_24h`: Exponentially weighted moving average ($\alpha = 2 / (24+1)$)

---

## ⏱️ 6. Time-Series Demand Forecasting

To evaluate short-term traffic volume forecasting, a chronological 85/15 train/test split ($40,938$ train / $7,225$ test records) was executed without shuffling.

### Empirical Forecasting Model Benchmarks

| Model / Baseline | Features Used | MAE (veh/hr) | RMSE (veh/hr) | $R^2$ Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Prev-Hour Baseline** | `lag_1h` | 485.95 | 740.42 | 0.8608 | Simple persistence baseline ($y_t = y_{t-1}$) |
| **Naive Same-Hour Prev-Day Baseline** | `lag_24h` | 1563.62 | 2208.58 | -0.2386 | Fails during day-of-week pattern shifts |
| ⭐ **ML Lag Random Forest** | Lags + Rolling + Calendar | **158.14** | **240.38** | **0.9853** | **Best Forecasting Performance** |

*Key Insight*: Incorporating past observed traffic lags reduces forecast MAE by **67.4%** compared to the naive persistence baseline.

---

## 🧪 7. Model Evaluation & Time-Aware Validation

We clearly distinguish between random holdout validation, chronological validation, and autoregressive forecasting:

| Evaluation Strategy | Model | Features | Split Type | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Holdout Split** | Random Forest | 5 Core Calendar | 80/20 Random | 442.27 | 641.56 | 0.8953 |
| **Chronological Split** | Linear Regression | 5 Core Calendar | 85/15 Time-Sorted | 1472.93 | 1756.24 | 0.2185 |
| **Chronological Split** | Decision Tree | 5 Core Calendar | 85/15 Time-Sorted | 512.92 | 760.31 | 0.8533 |
| ⭐ **Chronological Split** | Random Forest | 5 Core Calendar | 85/15 Time-Sorted | **494.61** | **736.21** | **0.8624** |
| **Forecasting Split** | ML Lag RF Model | Lags + Calendar | 85/15 Time-Sorted | **158.14** | **240.38** | **0.9853** |

*Note: In compliance with project guidelines, $R^2$ represents the coefficient of determination (proportion of variance explained) and is never referred to as "accuracy".*

---

## 🔍 8. Operational Error Analysis

Segmented error analysis (`analysis/error_analysis.py`) dissects model residuals to identify operational failure modes:

### Error Breakdown by Volume Bucket
| Volume Bucket | Range (veh/hr) | Sample Count | MAE | RMSE | Mean Bias Error | Bias Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Volume** | $< 2000$ | 2,752 | 344.60 | 467.40 | -94.15 | Overpredicting off-peak volume |
| **Moderate Volume** | $2000 - 4999$ | 2,168 | 514.85 | 694.75 | -41.32 | Slight overprediction |
| **Heavy Volume** | $\ge 5000$ | 2,305 | 655.43 | 985.34 | **+182.40** | **Underpredicting peak spikes** |

### Key Operational Observations
1. Heavy Traffic (≥ 5000 veh/hr): MAE = 655.43, RMSE = 985.34, Mean Bias = +182.40 veh/hr, indicating underprediction during high-volume periods.
2. **Nighttime Off-Peak Smoothing**: During low-volume nighttime hours, predictions carry a slight negative bias (-94.15 veh/hr), overestimating minimum baseline traffic.

---

## 💼 9. Business & Operational Decision Analysis

The business analysis engine (`analysis/business_analysis.py`) translates empirical predictions and error metrics into structured operational KPIs (`results/business/business_kpis.csv`) and decision summary reports (`results/business/business_summary.md`):

### Key Business KPIs
* **Average Hourly Volume**: `3259.82 veh/hr`
* **Rush vs Non-Rush Shift**: `+2028.77 veh/hr (+72.42%)`
* **Weekday vs Weekend Shift**: `+1459.79 veh/hr (+70.86%)`
* **Peak Demand Hours**: `17:00 (4624 veh/hr), 16:00 (4546 veh/hr), 08:00 (4473 veh/hr)`
* **Forecast MAE Error Reduction**: `-67.46%` error reduction with short-term lags vs persistence baseline.

### Answers to Operational Questions
* **When is traffic demand highest?**: Weekday evening (16:00-18:00) and morning (07:00-09:00) rush hours, peaking in August and October.
* **Which periods present the greatest forecasting difficulty?**: High-volume peak hours (07:00-08:00, 16:00-17:00) and heavy flow ($\ge 5000$ veh/hr) exhibit highest MAE.
* **Does rush-hour demand differ significantly from non-rush periods?**: Yes ($4,780.89$ vs $2,752.12$ veh/hr, $p < 0.0001$, Cohen's $d = 1.1594$, Very Large Effect).
* **Does weekday demand differ significantly from weekend demand?**: Yes ($3,519.82$ vs $2,608.20$ veh/hr, $p < 0.0001$, Cohen's $d = 0.7712$, Large Effect).
* **When does the model underpredict?**: During heavy volume surges ($\ge 5000$ veh/hr), mean bias is $+182.40$ veh/hr.
* **Which periods warrant closer operational monitoring?**: Peak weekday commuter windows during late summer/autumn.

---

## 🛡️ 10. Deployment & Monitoring System

The system includes a dedicated MLOps drift monitoring subsystem (`monitoring/monitor.py`):

* **Feature & Prediction Drift (PSI)**: Evaluates Population Stability Index using numerical binning:
  * $\text{PSI} < 0.10$: Stable / No Shift
  * $0.10 \le \text{PSI} < 0.25$: Moderate Distribution Shift
  * $\text{PSI} \ge 0.25$: Significant Distribution Drift
  * *Reference Context*: PSI thresholds of 0.10 and 0.25 are commonly used heuristic reference points for interpreting distribution shift; thresholds are configurable and should be calibrated to the specific application.
* **Performance Monitoring Threshold**: An RMSE monitoring threshold of **550 vehicles/hour** triggers automated system alerts:
  * *Reference Context*: This is a project-specific monitoring threshold derived from historical validation performance.

---

## 🧪 11. Automated Testing & Verification

The repository contains an automated test suite covering MLOps monitoring and analytical pipeline modules:

```bash
# Run all automated unit tests
python -m unittest discover tests

# Output:
# ----------------------------------------------------------------------
# Ran 15 tests in 4.70s
# OK
```

### Verified Test Components
* `test_monitoring.py` (7 tests): Unit tests for PSI computation, feature drift, prediction drift, performance monitoring, alert triggers, and logging.
* `test_analysis.py` (8 tests): Unit tests for SQL view execution, statistical hypothesis calculations (CIs, Welch t, Mann-Whitney U, Cohen's d, rank-biserial), chronological lag generation without leakage, naive baselines, forecasting model evaluation, error segmentation, business decision KPIs, and invalid inputs.

---

## 🚀 12. Reproducible Execution

Run the unified analytical entry point to regenerate all SQL databases, statistical reports, forecasting comparisons, operational error breakdowns, and business decision summaries:

```bash
python run_analysis.py
```

Generated outputs will be cleanly populated in:
```
results/
├── sql/                   # View query CSV exports
├── statistics/            # Descriptive, CI, & hypothesis test metrics
├── forecasting/           # Naive vs ML forecasting model metrics
├── error_analysis/        # Segmented MAE/RMSE/Bias metrics
└── business/              # Operational KPIs & decision summary report
```

To start the interactive Streamlit Web Application:
```bash
streamlit run app.py
```

---

## ⚠️ 13. Limitations & Future Improvements

### Limitations
1. **Lack of Spatial Dimensions**: The dataset contains data for a single highway corridor (I-94 westbound) without explicit road segment or geographic spatial features.
2. **Missing Holiday Data**: The `holiday` column is 99.87% missing and cannot be reliably used as a calendar feature.
3. **Peak Spike Compression**: Tree regressors tend to smooth out extreme prediction values during unobserved traffic surges.
4. **Observational Data Only**: All hypothesis testing and KPI differences represent observational patterns; causal inferences regarding traffic policy or business revenue require controlled trial data.

### Future Improvements
1. **Multi-Corridor Spatial Graph Networks**: Integrate Spatial-Temporal Graph Convolutional Networks (ST-GCN) when multi-sensor spatial data becomes available.
2. **Automated Retraining Trigger**: Connect the Streamlit monitoring alert manager to an automated retraining DAG (e.g., Airflow / Prefect) upon PSI breach.
3. **Real-time API Ingestion**: Integrate live weather and traffic API streams for real-time 1-hour ahead forecasting.

---

## 👨‍💻 Author

**Sarath Patti**  
M.Tech, Computer Science & Engineering  
National Institute of Technology Rourkela  
GitHub: [Sarath-Patti](https://github.com/Sarath-Patti)