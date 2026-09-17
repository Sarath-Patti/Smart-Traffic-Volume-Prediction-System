-- ============================================================================
-- Smart Traffic Volume Prediction System - SQL Analytics Layer
-- Author: Data Science Team
-- Purpose: Analytical queries, aggregations, window functions, and rankings
-- ============================================================================

-- 1. Daily Traffic Summary
-- Aggregates total, average, minimum, and maximum traffic volume by date
CREATE VIEW IF NOT EXISTS v_daily_traffic_summary AS
SELECT 
    DATE(date_time) AS date_val,
    COUNT(*) AS record_count,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_volume,
    SUM(traffic_volume) AS total_traffic_volume,
    MIN(traffic_volume) AS min_traffic_volume,
    MAX(traffic_volume) AS max_traffic_volume
FROM traffic_data
GROUP BY DATE(date_time)
ORDER BY date_val;

-- 2. Hourly Traffic Distribution
-- Analyzes traffic throughput patterns across hours of the day (0-23)
CREATE VIEW IF NOT EXISTS v_hourly_traffic_summary AS
SELECT 
    hour,
    COUNT(*) AS record_count,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_volume,
    MIN(traffic_volume) AS min_traffic_volume,
    MAX(traffic_volume) AS max_traffic_volume
FROM traffic_data
GROUP BY hour
ORDER BY hour;

-- 3. Weekday vs Weekend Traffic Breakdown
-- Categorizes days into Weekday (Mon-Fri) and Weekend (Sat-Sun)
CREATE VIEW IF NOT EXISTS v_day_type_comparison AS
SELECT 
    CASE 
        WHEN weekday IN (5, 6) THEN 'Weekend'
        ELSE 'Weekday'
    END AS day_type,
    COUNT(*) AS record_count,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_volume,
    MIN(traffic_volume) AS min_traffic_volume,
    MAX(traffic_volume) AS max_traffic_volume
FROM traffic_data
GROUP BY day_type;

-- 4. Rush-Hour vs Non-Rush-Hour Analysis
-- Aggregates metrics comparing designated peak rush hours against non-rush hours
CREATE VIEW IF NOT EXISTS v_rush_hour_summary AS
SELECT 
    CASE 
        WHEN is_rush = 1 THEN 'Rush Hour'
        ELSE 'Non-Rush Hour'
    END AS rush_status,
    COUNT(*) AS record_count,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_volume,
    MIN(traffic_volume) AS min_traffic_volume,
    MAX(traffic_volume) AS max_traffic_volume
FROM traffic_data
GROUP BY rush_status;

-- 5. Monthly Traffic Trends
-- Aggregates volume across calendar months (1-12) to identify seasonal shifts
CREATE VIEW IF NOT EXISTS v_monthly_traffic_trends AS
SELECT 
    month,
    COUNT(*) AS record_count,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_volume,
    SUM(traffic_volume) AS total_traffic_volume
FROM traffic_data
GROUP BY month
ORDER BY month;

-- 6. Rolling 24-Hour Traffic Volume (SQL Window Function)
-- Calculates 24-hour moving average over sequential timestamps
CREATE VIEW IF NOT EXISTS v_rolling_24h_traffic AS
SELECT 
    date_time,
    hour,
    weekday,
    traffic_volume,
    ROUND(AVG(traffic_volume) OVER (
        ORDER BY date_time 
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_24h_avg
FROM traffic_data;

-- 7. Peak Traffic Period Rankings (SQL Window Function)
-- Ranks timestamps by traffic volume throughput using RANK() OVER()
CREATE VIEW IF NOT EXISTS v_peak_traffic_rankings AS
SELECT 
    date_time,
    hour,
    weekday,
    month,
    traffic_volume,
    RANK() OVER (ORDER BY traffic_volume DESC) AS volume_rank
FROM traffic_data;

-- 8. Traffic Volume Segmentation Buckets
-- Groups observations into Low (<2000), Moderate (2000-5000), and Heavy (>=5000)
CREATE VIEW IF NOT EXISTS v_traffic_volume_buckets AS
SELECT 
    CASE 
        WHEN traffic_volume < 2000 THEN 'Low (<2000)'
        WHEN traffic_volume < 5000 THEN 'Moderate (2000-5000)'
        ELSE 'Heavy (>=5000)'
    END AS volume_bucket,
    COUNT(*) AS record_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM traffic_data), 2) AS percentage
FROM traffic_data
GROUP BY volume_bucket;

-- 9. Weather Impact Aggregation
-- Analyzes traffic throughput grouped by primary weather classification
CREATE VIEW IF NOT EXISTS v_weather_impact_summary AS
SELECT 
    weather_main,
    COUNT(*) AS record_count,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_volume,
    ROUND(AVG(temp), 2) AS avg_temp_kelvin,
    ROUND(AVG(rain_1h), 4) AS avg_rain_mm
FROM traffic_data
GROUP BY weather_main
ORDER BY avg_traffic_volume DESC;
