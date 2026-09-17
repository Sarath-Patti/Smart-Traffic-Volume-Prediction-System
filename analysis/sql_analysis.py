"""
Smart Traffic Volume Prediction - SQL Analytics Python Utility
Executes analytical SQL queries against SQLite database traffic_analytics.db
"""

import os
import sqlite3
import pandas as pd


class SQLAnalyticsEngine:
    """Manages SQLite analytical table creation, SQL view execution, and query results."""

    def __init__(self, db_path="traffic_analytics.db", csv_path="datafile.csv"):
        self.db_path = db_path
        self.csv_path = csv_path
        self._setup_database()

    def _setup_database(self):
        """Loads clean dataset into SQLite table traffic_data and executes analysis.sql."""
        conn = sqlite3.connect(self.db_path)
        
        # Load and clean dataset
        df = pd.read_csv(self.csv_path).drop_duplicates()
        df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
        df_sorted = df.sort_values(by='date_time').reset_index(drop=True)

        df_sorted['hour'] = df_sorted['date_time'].dt.hour
        df_sorted['day'] = df_sorted['date_time'].dt.day
        df_sorted['month'] = df_sorted['date_time'].dt.month
        df_sorted['weekday'] = df_sorted['date_time'].dt.weekday
        df_sorted['is_rush'] = df_sorted['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)

        # Convert date_time to string ISO format for SQLite storage
        df_sorted['date_time'] = df_sorted['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Write to SQLite
        df_sorted.to_sql("traffic_data", conn, if_exists="replace", index=False)

        # Execute analysis.sql script if present
        sql_file = "analysis.sql"
        if os.path.exists(sql_file):
            with open(sql_file, "r") as f:
                sql_script = f.read()
            conn.executescript(sql_script)

        conn.commit()
        conn.close()

    def get_query_result(self, query_or_view):
        """Executes a SELECT query or queries a view and returns a DataFrame."""
        conn = sqlite3.connect(self.db_path)
        if not query_or_view.strip().upper().startswith("SELECT"):
            query = f"SELECT * FROM {query_or_view}"
        else:
            query = query_or_view
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def run_all_sql_analytics(self, output_dir="results/sql"):
        """Executes all analytical views and saves CSV summaries to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        
        views = [
            "v_daily_traffic_summary",
            "v_hourly_traffic_summary",
            "v_day_type_comparison",
            "v_rush_hour_summary",
            "v_monthly_traffic_trends",
            "v_traffic_volume_buckets",
            "v_weather_impact_summary"
        ]

        results = {}
        for view in views:
            df_res = self.get_query_result(view)
            results[view] = df_res
            out_file = os.path.join(output_dir, f"{view}.csv")
            df_res.to_csv(out_file, index=False)

        # Top 10 Peak Traffic Periods
        df_peaks = self.get_query_result("SELECT * FROM v_peak_traffic_rankings LIMIT 10")
        results["v_peak_traffic_rankings_top10"] = df_peaks
        df_peaks.to_csv(os.path.join(output_dir, "v_peak_traffic_rankings_top10.csv"), index=False)

        return results


def run_sql_analysis():
    engine = SQLAnalyticsEngine()
    return engine.run_all_sql_analytics()


if __name__ == "__main__":
    results = run_sql_analysis()
    print("✅ SQL Analytics executed successfully. Results saved to results/sql/")
