"""
CDR Data Preprocessing and Subscriber Behavioral Feature Engineering
Aggregates event-level Call Detail Records (CDRs) into subscriber-level behavioral metrics:
- Daily call count & duration distributions
- Internet traffic volume and usage patterns
- Peak-hour activity ratio (8:00 - 20:00) vs off-peak
- Weekend vs weekday usage balance
- International calling ratios and geographic entropy
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    setup_logger,
    load_config,
    ensure_directory,
    save_parquet,
    load_parquet,
    generate_synthetic_telecom_data
)

logger = setup_logger("preprocessing")

class CDRPreprocessor:
    """
    Processes raw event-level telecommunication streams into high-dimensional behavioral features.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.peak_start = self.config["cdr_schema"]["peak_start_hour"]
        self.peak_end = self.config["cdr_schema"]["peak_end_hour"]
        self.output_file = self.config["paths"]["customer_features_file"]

    def parse_timestamps(self, df: pd.DataFrame, time_col: str = "time_interval") -> pd.DataFrame:
        """
        Converts epoch timestamps into temporal components (hour, day, weekday, is_peak).
        """
        df = df.copy()
        # Telecom Italia CDR timestamps are in milliseconds
        if df[time_col].dtype in [np.int64, np.float64, int, float]:
            df["datetime"] = pd.to_datetime(df[time_col], unit="ms", errors="coerce")
        else:
            df["datetime"] = pd.to_datetime(df[time_col], errors="coerce")

        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_peak_hour"] = ((df["hour"] >= self.peak_start) & (df["hour"] < self.peak_end)).astype(int)
        return df

    def compute_spatial_entropy(self, group_series: pd.Series) -> float:
        """
        Calculates Shannon spatial entropy over grid cell visits.
        Higher entropy indicates higher mobility across different municipal sectors.
        """
        counts = group_series.value_counts(normalize=True)
        return -float(np.sum(counts * np.log2(counts + 1e-9)))

    def aggregate_event_stream(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw event interactions into customer-level behavioral summary.
        """
        logger.info(f"Aggregating {len(df_events):,} event records...")
        df_parsed = self.parse_timestamps(df_events)

        # Ensure required columns exist
        for col in ["sms_in", "sms_out", "call_in", "call_out", "internet_traffic"]:
            if col not in df_parsed.columns:
                df_parsed[col] = 0.0
            else:
                df_parsed[col] = df_parsed[col].fillna(0.0)

        df_parsed["total_calls"] = df_parsed["call_in"] + df_parsed["call_out"]
        df_parsed["total_sms"] = df_parsed["sms_in"] + df_parsed["sms_out"]
        df_parsed["is_international"] = (df_parsed["country_code"] != 39).astype(int)

        # Grouping by customer
        records = []
        for cust_id, group in df_parsed.groupby("customer_id"):
            num_days = max(1, group["datetime"].dt.date.nunique())
            tot_calls = group["total_calls"].sum()
            tot_sms = group["total_sms"].sum()
            tot_data = group["internet_traffic"].sum()
            tot_events = len(group)

            peak_events = group["is_peak_hour"].sum()
            weekend_events = group["is_weekend"].sum()
            intl_events = group["is_international"].sum()

            peak_ratio = round(peak_events / tot_events, 4) if tot_events > 0 else 0.5
            weekend_ratio = round(weekend_events / tot_events, 4) if tot_events > 0 else 0.28
            intl_ratio = round(intl_events / tot_events, 4) if tot_events > 0 else 0.05

            primary_sq = group["square_id"].mode().iloc[0] if not group["square_id"].empty else 1
            spatial_entropy = round(self.compute_spatial_entropy(group["square_id"]), 3)

            records.append({
                "customer_id": cust_id,
                "primary_square_id": int(primary_sq),
                "avg_daily_calls": round(tot_calls / num_days, 2),
                "avg_daily_sms": round(tot_sms / num_days, 2),
                "avg_daily_internet_mb": round(tot_data / num_days, 2),
                "peak_hour_activity_ratio": peak_ratio,
                "weekend_activity_ratio": weekend_ratio,
                "international_call_ratio": intl_ratio,
                "entropy_grid_dispersion": spatial_entropy,
                "observed_active_days": num_days
            })

        df_features = pd.DataFrame(records)
        logger.info(f"Aggregated into {len(df_features):,} subscriber feature rows.")
        return df_features

    def clean_and_impute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans distributions, clips extreme outlier values, and imputes missing columns.
        """
        logger.info("Cleaning feature distributions and clipping extreme outliers...")
        df_clean = df.copy()

        # Fill missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

        # Clip extreme outliers (99.5th percentile)
        for col in ["avg_daily_calls", "avg_daily_sms", "avg_daily_internet_mb"]:
            if col in df_clean.columns:
                upper_bound = df_clean[col].quantile(0.995)
                df_clean[col] = df_clean[col].clip(upper=upper_bound)

        return df_clean

    def build_pyspark_pipeline_spec(self) -> str:
        """
        Generates production PySpark batch aggregation code snippet for large-scale cluster execution.
        """
        spark_code = '''
# PySpark Batch Aggregator for 10M+ CDR Records
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("TelecomCDRPreprocess").getOrCreate()
raw_df = spark.read.option("header", "true").parquet("data/raw/cdr/*.parquet")

customer_daily = raw_df.withColumn("timestamp", (F.col("time_interval") / 1000).cast("timestamp")) \\
    .withColumn("hour", F.hour("timestamp")) \\
    .withColumn("is_peak", F.when((F.col("hour") >= 8) & (F.col("hour") < 20), 1).otherwise(0)) \\
    .withColumn("is_weekend", F.when(F.dayofweek("timestamp").isin([1, 7]), 1).otherwise(0)) \\
    .groupBy("customer_id") \\
    .agg(
        F.avg("call_out").alias("avg_daily_calls"),
        F.avg("internet_traffic").alias("avg_daily_internet_mb"),
        (F.sum("is_peak") / F.count("*")).alias("peak_hour_activity_ratio"),
        (F.sum("is_weekend") / F.count("*")).alias("weekend_activity_ratio")
    )
customer_daily.write.mode("overwrite").parquet("data/processed/spark_aggregated_features.parquet")
'''
        return spark_code

    def run(self) -> pd.DataFrame:
        """
        Executes feature preprocessing workflow from raw storage or synthetic generator.
        """
        raw_cust_file = os.path.join(self.config["paths"]["raw_cdr_dir"], "subscriber_profiles_raw.parquet")

        if os.path.exists(raw_cust_file):
            logger.info(f"Loading raw subscriber profiles from {raw_cust_file}...")
            df_customers = load_parquet(raw_cust_file)
        else:
            logger.info("Raw subscriber profile not detected. Generating integrated dataset...")
            df_customers, df_istat, df_cdr = generate_synthetic_telecom_data(num_subscribers=5000)

            # Aggregate CDR sample
            df_aggregated = self.aggregate_event_stream(df_cdr)
            # Merge aggregated CDR metrics where available, or keep synthetic behavioral features
            df_customers = self.clean_and_impute_features(df_customers)

        df_customers = self.clean_and_impute_features(df_customers)
        save_parquet(df_customers, self.output_file)
        logger.info(f"Preprocessing completed. Output saved to {self.output_file}")
        return df_customers

def main():
    preprocessor = CDRPreprocessor()
    df_features = preprocessor.run()
    logger.info(f"Feature engineering summary:\n{df_features.describe().round(2).transpose()[['mean', 'std', 'min', '50%', 'max']]}")

if __name__ == "__main__":
    main()
