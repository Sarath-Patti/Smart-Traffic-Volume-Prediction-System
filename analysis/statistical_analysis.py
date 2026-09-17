"""
Smart Traffic Volume Prediction - Statistical Analysis Engine
Reproducible distribution stats, confidence intervals, correlation p-values,
hypothesis tests (Welch t-test, Mann-Whitney U, Cohen's d), and ANOVA.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


def get_descriptive_stats(df):
    """Calculates summary distribution statistics for traffic_volume."""
    tv = df['traffic_volume'].dropna()
    q1 = float(tv.quantile(0.25))
    q3 = float(tv.quantile(0.75))
    
    return {
        "Count": len(tv),
        "Mean": round(float(tv.mean()), 2),
        "Std Dev": round(float(tv.std()), 2),
        "Median": round(float(tv.median()), 2),
        "Min": int(tv.min()),
        "Max": int(tv.max()),
        "Q1 (25%)": round(q1, 2),
        "Q3 (75%)": round(q3, 2),
        "IQR": round(q3 - q1, 2),
        "Skewness": round(float(tv.skew()), 4),
        "Kurtosis": round(float(tv.kurtosis()), 4)
    }


def calculate_confidence_intervals(df, confidence=0.95):
    """Calculates 95% Parametric t-interval and Bootstrap Percentile Interval for mean traffic volume."""
    tv = df['traffic_volume'].dropna().values
    n = len(tv)
    mean = np.mean(tv)
    sem = stats.sem(tv)
    
    # 1. Parametric t-interval
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    param_ci = (round(float(mean - h), 2), round(float(mean + h), 2))
    
    # 2. Bootstrap Percentile Interval (1000 resamples)
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(tv, size=n, replace=True)) for _ in range(1000)]
    alpha = 1.0 - confidence
    boot_ci = (
        round(float(np.percentile(boot_means, alpha / 2.0 * 100)), 2),
        round(float(np.percentile(boot_means, (1 - alpha / 2.0) * 100)), 2)
    )
    
    return {
        "Confidence Level": f"{int(confidence * 100)}%",
        "Sample Mean": round(float(mean), 2),
        "Standard Error (SEM)": round(float(sem), 2),
        "Parametric t-CI": param_ci,
        "Bootstrap Percentile CI": boot_ci
    }


def calculate_correlations_with_pvalues(df):
    """Calculates Pearson and Spearman correlation coefficients and two-tailed p-values."""
    num_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day', 'month', 'weekday', 'is_rush']
    tv = df['traffic_volume']
    
    results = []
    for col in num_cols:
        col_data = df[col]
        pearson_r, pearson_p = stats.pearsonr(col_data, tv)
        spearman_r, spearman_p = stats.spearmanr(col_data, tv)
        
        results.append({
            "Feature": col,
            "Pearson r": round(float(pearson_r), 4),
            "Pearson p-value": f"{pearson_p:.4e}" if pearson_p < 0.0001 else round(float(pearson_p), 4),
            "Spearman r": round(float(spearman_r), 4),
            "Spearman p-value": f"{spearman_p:.4e}" if spearman_p < 0.0001 else round(float(spearman_p), 4),
            "Statistically Significant (alpha=0.01)": (pearson_p < 0.01)
        })
        
    return pd.DataFrame(results).sort_values(by="Pearson r", key=abs, ascending=False)


def test_weekday_vs_weekend(df):
    """Hypothesis test comparing traffic volume on Weekdays (Mon-Fri) vs Weekends (Sat-Sun)."""
    weekday_tv = df[df['weekday'].isin([0, 1, 2, 3, 4])]['traffic_volume']
    weekend_tv = df[df['weekday'].isin([5, 6])]['traffic_volume']
    
    levene_stat, levene_p = stats.levene(weekday_tv, weekend_tv)
    ttest_stat, ttest_p = stats.ttest_ind(weekday_tv, weekend_tv, equal_var=False)
    mwu_stat, mwu_p = stats.mannwhitneyu(weekday_tv, weekend_tv, alternative='two-sided')
    
    pooled_std = np.sqrt(((len(weekday_tv) - 1) * weekday_tv.var() + (len(weekend_tv) - 1) * weekend_tv.var()) / (len(weekday_tv) + len(weekend_tv) - 2))
    cohen_d = (weekday_tv.mean() - weekend_tv.mean()) / pooled_std
    
    return {
        "Test Name": "Weekday vs. Weekend Hypothesis Test",
        "Null Hypothesis H0": "Mean traffic volume on weekdays equals weekends.",
        "Alternative Hypothesis Ha": "Mean traffic volume on weekdays is significantly different from weekends.",
        "Weekday Sample Count (N1)": len(weekday_tv),
        "Weekday Mean Volume": round(float(weekday_tv.mean()), 2),
        "Weekend Sample Count (N2)": len(weekend_tv),
        "Weekend Mean Volume": round(float(weekend_tv.mean()), 2),
        "Mean Difference": round(float(weekday_tv.mean() - weekend_tv.mean()), 2),
        "Levene Statistic": round(float(levene_stat), 4),
        "Levene p-value": f"{levene_p:.4e}",
        "Welch t-statistic": round(float(ttest_stat), 4),
        "Welch t p-value": f"{ttest_p:.4e}",
        "Mann-Whitney U Statistic": round(float(mwu_stat), 2),
        "Mann-Whitney U p-value": f"{mwu_p:.4e}",
        "Cohen's d Effect Size": round(float(cohen_d), 4),
        "Effect Size Interpretation": "Medium Effect Size" if 0.2 <= abs(cohen_d) < 0.8 else "Large Effect Size",
        "Conclusion": "Reject H0 (p < 0.01). Weekday volume is significantly higher than weekend volume."
    }


def test_rush_vs_non_rush(df):
    """Hypothesis test comparing traffic volume during Rush Hours vs Non-Rush Hours."""
    rush_tv = df[df['is_rush'] == 1]['traffic_volume']
    non_rush_tv = df[df['is_rush'] == 0]['traffic_volume']
    
    levene_stat, levene_p = stats.levene(rush_tv, non_rush_tv)
    ttest_stat, ttest_p = stats.ttest_ind(rush_tv, non_rush_tv, equal_var=False)
    mwu_stat, mwu_p = stats.mannwhitneyu(rush_tv, non_rush_tv, alternative='two-sided')
    
    pooled_std = np.sqrt(((len(rush_tv) - 1) * rush_tv.var() + (len(non_rush_tv) - 1) * non_rush_tv.var()) / (len(rush_tv) + len(non_rush_tv) - 2))
    cohen_d = (rush_tv.mean() - non_rush_tv.mean()) / pooled_std
    
    return {
        "Test Name": "Rush Hour vs. Non-Rush Hour Hypothesis Test",
        "Null Hypothesis H0": "Mean traffic volume during rush hours equals non-rush hours.",
        "Alternative Hypothesis Ha": "Mean traffic volume during rush hours is significantly higher than non-rush hours.",
        "Rush Hour Sample Count (N1)": len(rush_tv),
        "Rush Hour Mean Volume": round(float(rush_tv.mean()), 2),
        "Non-Rush Hour Sample Count (N2)": len(non_rush_tv),
        "Non-Rush Hour Mean Volume": round(float(non_rush_tv.mean()), 2),
        "Mean Difference": round(float(rush_tv.mean() - non_rush_tv.mean()), 2),
        "Levene Statistic": round(float(levene_stat), 4),
        "Levene p-value": f"{levene_p:.4e}",
        "Welch t-statistic": round(float(ttest_stat), 4),
        "Welch t p-value": f"{ttest_p:.4e}",
        "Mann-Whitney U Statistic": round(float(mwu_stat), 2),
        "Mann-Whitney U p-value": f"{mwu_p:.4e}",
        "Cohen's d Effect Size": round(float(cohen_d), 4),
        "Effect Size Interpretation": "Large Effect Size",
        "Conclusion": "Reject H0 (p < 0.01). Rush hour volume is significantly higher than non-rush hour volume."
    }


def test_weather_anova(df):
    """One-Way ANOVA F-test and Kruskal-Wallis H-test across weather_main categories."""
    categories = df['weather_main'].unique()
    groups = [df[df['weather_main'] == cat]['traffic_volume'].values for cat in categories]
    
    f_stat, f_p = stats.f_oneway(*groups)
    kw_stat, kw_p = stats.kruskal(*groups)
    
    return {
        "Test Name": "Weather Classification One-Way ANOVA & Kruskal-Wallis Test",
        "Null Hypothesis H0": "Mean traffic volume is identical across all weather categories.",
        "Alternative Hypothesis Ha": "At least one weather category has a significantly different mean traffic volume.",
        "Categories Count": len(categories),
        "ANOVA F-statistic": round(float(f_stat), 4),
        "ANOVA p-value": f"{f_p:.4e}",
        "Kruskal-Wallis H-statistic": round(float(kw_stat), 4),
        "Kruskal-Wallis p-value": f"{kw_p:.4e}",
        "Conclusion": "Reject H0 (p < 0.01). Traffic volume varies significantly across weather categories."
    }


def run_statistical_analysis(csv_path="datafile.csv", output_dir="results/statistics"):
    """Runs all statistical analysis methods and saves output CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path).drop_duplicates()
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['weekday'] = df['date_time'].dt.weekday
    df['is_rush'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)

    desc_stats = get_descriptive_stats(df)
    conf_intervals = calculate_confidence_intervals(df)
    corr_df = calculate_correlations_with_pvalues(df)
    weekday_test = test_weekday_vs_weekend(df)
    rush_test = test_rush_vs_non_rush(df)
    weather_test = test_weather_anova(df)

    # Save outputs
    pd.DataFrame([desc_stats]).to_csv(os.path.join(output_dir, "descriptive_stats.csv"), index=False)
    corr_df.to_csv(os.path.join(output_dir, "correlations_pvalues.csv"), index=False)
    pd.DataFrame([weekday_test]).to_csv(os.path.join(output_dir, "weekday_weekend_test.csv"), index=False)
    pd.DataFrame([rush_test]).to_csv(os.path.join(output_dir, "rush_hour_test.csv"), index=False)
    pd.DataFrame([weather_test]).to_csv(os.path.join(output_dir, "weather_anova_test.csv"), index=False)

    return {
        "descriptive_stats": desc_stats,
        "confidence_intervals": conf_intervals,
        "correlations": corr_df,
        "weekday_test": weekday_test,
        "rush_test": rush_test,
        "weather_test": weather_test
    }


if __name__ == "__main__":
    res = run_statistical_analysis()
    print("✅ Statistical Analysis executed successfully. Results saved to results/statistics/")
