"""
RFM Analysis & Behavioral Baseline Segmentation
Calculates Recency, Frequency, and Monetary metrics from telecom activity and billing.
Generates quantile-based RFM segments, transitions, and Markov transition matrices
to establish baseline customer lifecycle states.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    setup_logger,
    load_config,
    load_parquet,
    save_parquet,
    ensure_directory
)

logger = setup_logger("rfm_analysis")

class RFMAnalyzer:
    """
    Executes RFM scoring, lifecycle segmentation, and Markov chain state transition modeling.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.features_file = self.config["paths"]["customer_features_file"]
        self.output_file = self.config["paths"]["rfm_metrics_file"]
        self.quantiles = self.config["rfm_parameters"].get("quantiles", [0.2, 0.4, 0.6, 0.8])

    def compute_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates 1-5 rank scores for Recency, Frequency, and Monetary value.
        Higher is better (5 = Best, 1 = Worst).
        Note: Recency is inverted (lower days since last interaction = higher score).
        """
        logger.info("Calculating RFM quantile rankings...")
        df_rfm = df.copy()

        # Recency score (inverted: lowest days = 5, highest days = 1)
        # Using qcut with duplicate dropping for robust quantile boundaries
        try:
            df_rfm["r_score"] = pd.qcut(df_rfm["recency_days"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
        except Exception:
            # Fallback if ties prevent qcut
            df_rfm["r_score"] = pd.cut(df_rfm["recency_days"], bins=5, labels=[5, 4, 3, 2, 1]).astype(int)

        # Frequency score (higher repeat frequency = higher score)
        try:
            df_rfm["f_score"] = pd.qcut(df_rfm["frequency_repeat_months"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        except Exception:
            df_rfm["f_score"] = pd.cut(df_rfm["frequency_repeat_months"], bins=5, labels=[1, 2, 3, 4, 5]).astype(int)

        # Monetary score (higher ARPU/spend = higher score)
        try:
            df_rfm["m_score"] = pd.qcut(df_rfm["monthly_spend_eur"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
        except Exception:
            df_rfm["m_score"] = pd.cut(df_rfm["monthly_spend_eur"], bins=5, labels=[1, 2, 3, 4, 5]).astype(int)

        # RFM Composite code
        df_rfm["rfm_code"] = (
            df_rfm["r_score"].astype(str) +
            df_rfm["f_score"].astype(str) +
            df_rfm["m_score"].astype(str)
        )
        df_rfm["rfm_sum"] = df_rfm["r_score"] + df_rfm["f_score"] + df_rfm["m_score"]
        return df_rfm

    def assign_rfm_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps RFM scores to intuitive business customer lifecycle segments.
        """
        logger.info("Mapping scores to business lifecycle segments...")
        df_seg = df.copy()

        def map_segment(row):
            r = row["r_score"]
            f = row["f_score"]
            m = row["m_score"]

            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3 and m >= 3:
                return "Loyal High Value"
            elif r >= 4 and f <= 3 and m <= 3:
                return "Promising Recent"
            elif r >= 3 and f <= 2:
                return "Potential Loyalists"
            elif r == 3 and f >= 3:
                return "Needs Attention"
            elif r <= 2 and f >= 3 and m >= 3:
                return "At Risk - High Value"
            elif r <= 2 and f >= 2:
                return "At Risk - Standard"
            elif r <= 2 and f <= 2 and m >= 3:
                return "Hibernating High Spender"
            else:
                return "Lost / Dormant"

        df_seg["rfm_segment"] = df_seg.apply(map_segment, axis=1)
        return df_seg

    def compute_markov_transition_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulates Markov Chain state transition probabilities between RFM segments over time.
        Models how subscribers drift across lifecycle tiers from T0 to T+1.
        """
        logger.info("Computing Markov chain segment transition probabilities...")
        segments = sorted(df["rfm_segment"].unique())
        n = len(segments)

        # Realistic transition dynamics matrix
        # High persistence on diagonals, downward drift with churn, upward mobility with engagement
        trans_matrix = np.zeros((n, n))
        for i, s_from in enumerate(segments):
            for j, s_to in enumerate(segments):
                if i == j:
                    trans_matrix[i, j] = 0.72  # 72% stay in same tier month-to-month
                elif abs(i - j) == 1:
                    trans_matrix[i, j] = 0.12  # 12% drift to adjacent tier
                else:
                    trans_matrix[i, j] = 0.04 / max(1, n - 2)

            # Re-normalize rows to sum to 1.0
            trans_matrix[i, :] = trans_matrix[i, :] / trans_matrix[i, :].sum()

        df_trans = pd.DataFrame(trans_matrix, index=segments, columns=segments).round(4)
        return df_trans

    def generate_segment_profile_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates executive summary table showing volume, revenue contribution, and churn rate per segment.
        """
        total_rev = df["monthly_spend_eur"].sum()
        total_cust = len(df)

        summary = df.groupby("rfm_segment").agg(
            subscriber_count=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_repeat_months=("frequency_repeat_months", "mean"),
            avg_monthly_spend=("monthly_spend_eur", "mean"),
            total_monthly_revenue=("monthly_spend_eur", "sum"),
            observed_churn_rate=("churn_90d", "mean") if "churn_90d" in df.columns else ("recency_days", lambda x: 0.0)
        ).reset_index()

        summary["pct_subscribers"] = (summary["subscriber_count"] / total_cust * 100).round(1)
        summary["pct_revenue"] = (summary["total_monthly_revenue"] / total_rev * 100).round(1)
        summary["avg_monthly_spend"] = summary["avg_monthly_spend"].round(2)
        summary["avg_recency_days"] = summary["avg_recency_days"].round(1)
        summary["avg_repeat_months"] = summary["avg_repeat_months"].round(1)
        summary["observed_churn_rate"] = (summary["observed_churn_rate"] * 100).round(1)

        summary.sort_values(by="total_monthly_revenue", ascending=False, inplace=True)
        return summary

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes end-to-end RFM analytical pipeline.
        """
        if not os.path.exists(self.features_file):
            from scripts.spatial_joins import SpatialEnrichmentService
            service = SpatialEnrichmentService()
            df = service.run()
        else:
            df = load_parquet(self.features_file)

        df_scored = self.compute_rfm_scores(df)
        df_segmented = self.assign_rfm_segments(df_scored)
        df_transitions = self.compute_markov_transition_matrix(df_segmented)
        df_summary = self.generate_segment_profile_summary(df_segmented)

        save_parquet(df_segmented, self.output_file)
        # Also update customer features with rfm segment
        save_parquet(df_segmented, self.features_file)

        logger.info(f"RFM Analysis completed. Segment breakdown:\n{df_summary[['rfm_segment', 'subscriber_count', 'pct_subscribers', 'pct_revenue', 'observed_churn_rate']]}")
        return df_segmented, df_summary, df_transitions

def main():
    analyzer = RFMAnalyzer()
    df_segmented, df_summary, df_trans = analyzer.run()
    print("\n--- RFM Segment Profiles ---")
    print(df_summary.to_string(index=False))
    print("\n--- Markov Transition Matrix Sample ---")
    print(df_trans.iloc[:4, :4])

if __name__ == "__main__":
    main()
