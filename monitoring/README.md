# 🛡️ Traffic AI Model Monitoring System

An production-grade ML Model Monitoring System for the **Smart Traffic Volume Prediction System**, demonstrating an end-to-end Data Science lifecycle (`DATA -> EDA -> MODELING -> VALIDATION -> DEPLOYMENT -> MONITORING`).

---

## 📌 1. Why Model Monitoring is Essential

Machine learning models deployed in production operate in dynamic environments. Over time, underlying data distributions can shift due to changing real-world conditions (e.g. seasonal traffic patterns, new road infrastructure, or extreme weather). 

Model monitoring ensures:
1. **Early Risk Detection**: Detect distribution shifts before they cause severe performance drops.
2. **Data Quality Assurance**: Identify missing fields, out-of-bounds inputs, or pipeline corruptions.
3. **Auditability & Compliance**: Maintain immutable prediction logs for historical review.
4. **Triggering Model Retraining**: Inform engineering teams when a model needs recalibration or retraining.

---

## 📊 2. Reference Dataset & Baseline

The reference dataset (`monitoring/reference_data.csv`) is derived strictly from the baseline training set ($40,958$ clean observations spanning `2012-10-02` to `2018-01-25`). 

It establishes true baseline statistical distributions for:
- **`hour`**: Hour of day ($0 - 23$)
- **`day`**: Day of month ($1 - 31$)
- **`month`**: Month ($1 - 12$)
- **`weekday`**: Day of week ($0 = \text{Mon}, 6 = \text{Sun}$)
- **`is_rush`**: Rush hour flag ($1$ if hour in $[7, 8, 9, 17, 18, 19]$ else $0$)
- **`traffic_volume`**: Baseline target throughput ($0 - 7,280$ vehicles/hr)

---

## 📈 3. Data Drift Monitoring (PSI Metric)

Feature distribution drift is evaluated using the **Population Stability Index (PSI)**:

$$\text{PSI} = \sum_{i=1}^{B} (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)$$

Where $P_i$ is the baseline percentage in bin $i$, and $Q_i$ is the monitored batch percentage in bin $i$.

### PSI Thresholds & Heuristic Reference Points:
PSI thresholds of 0.10 and 0.25 are commonly used heuristic reference points for interpreting distribution shift. These thresholds are configurable and should be calibrated to the specific production application and data characteristics:
- **$\text{PSI} < 0.10$**: **No Drift / Stable (Green)** $\rightarrow$ Feature distribution is consistent with baseline.
- **$0.10 \le \text{PSI} < 0.25$**: **Moderate Shift (Yellow)** $\rightarrow$ Slight distribution shift detected; inspect input stream.
- **$\text{PSI} \ge 0.25$**: **Significant Drift (Red)** $\rightarrow$ Severe distribution change; flag feature for investigation or model retraining.

---

## 🔮 4. Prediction Drift Monitoring

Prediction drift compares the model's output distribution on current incoming batches against its output distribution on the reference dataset. 

It tracks:
- **Mean & Median Shift**
- **Standard Deviation & Quantile Spread (25%, 75%)**
- **Prediction PSI & Wasserstein Distance**

*Key Advantage*: Prediction drift can be calculated **immediately** in real-time without waiting for ground-truth actual labels to become available.

---

## 🎯 5. Model Performance Monitoring

When actual ground-truth traffic volumes become available (e.g. from toll gate sensors recorded post-hoc), actual model performance is evaluated using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination ($R^2$)**

---

## ⚡ 6. Drift vs. Performance Degradation

| Aspect | Data / Prediction Drift | Performance Degradation |
| :--- | :--- | :--- |
| **Requires Labels?** | **No** (Unsupervised) | **Yes** (Supervised) |
| **Evaluation Timing** | Real-time / Immediate | Delayed (when actuals arrive) |
| **What it Measures** | Changes in input or output shapes | Accuracy drops ($\text{RMSE} > \text{Threshold}$) |
| **Action** | Preemptive investigation | Alerting & Triggering Model Retraining |

---

## 🚨 7. Model Alerting & Retraining Triggers

### Configurable Performance Threshold:
- **Monitored Performance Threshold**: Project-specific monitoring threshold derived from the historical validation performance of this model ($\text{RMSE} \le 550.0$ vehicles/hour, calibrated above validation bounds: baseline test $\text{RMSE} = 434.14$, time-aware test $\text{RMSE} = 486.77$).
- **Alert Statuses**:
  - `Performance Within Threshold`: Monitored $\text{RMSE} \le 550.0$.
  - `Performance Degradation Detected`: Monitored $\text{RMSE} > 550.0$.

### Retraining Trigger Concept:
Monitoring follows an **Observe and Report** philosophy. The system does **NOT** retrain automatically or overwrite production models. When an alert triggers, an alert payload is dispatched to data engineers to evaluate candidate datasets, perform temporal validation, and safely promote a new model version.

---

## 🧪 8. Historical Simulation vs. Live Production Monitoring

In this project:
- Historical holdout datasets (e.g., 2018 test window) and synthetic shift generators are explicitly labeled as **"Historical Simulation / Demonstration"**.
- This provides an authentic software demonstration of monitoring functionality without falsely representing historical logs as live API telemetry.
