"""
Spatial Joins and Socioeconomic Demographic Enrichment
Joins subscriber primary telecommunication grid cells with ISTAT census tract data,
integrating neighborhood disposable income, population density, unemployment rates,
and age demographic distributions.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    setup_logger,
    load_config,
    load_parquet,
    save_parquet,
    generate_synthetic_telecom_data
)

logger = setup_logger("spatial_joins")

class SpatialEnrichmentService:
    """
    Handles spatial alignment and feature joining between CDR grid cells and ISTAT demographic indicators.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.customer_features_path = self.config["paths"]["customer_features_file"]
        self.raw_istat_dir = self.config["paths"]["raw_istat_dir"]
        self.demographics_csv = os.path.join(self.raw_istat_dir, "istat_socioeconomic_indicators.csv")

    def load_istat_demographics(self) -> pd.DataFrame:
        """
        Loads ISTAT census dataset containing socioeconomic indicators per square_id.
        """
        if os.path.exists(self.demographics_csv):
            logger.info(f"Loading ISTAT indicators from {self.demographics_csv}...")
            df_istat = pd.read_csv(self.demographics_csv)
        else:
            logger.warning(f"ISTAT demographics file not found at {self.demographics_csv}. Generating synthetic indicators...")
            _, df_istat, _ = generate_synthetic_telecom_data(num_subscribers=1000)
            os.makedirs(self.raw_istat_dir, exist_ok=True)
            df_istat.to_csv(self.demographics_csv, index=False)

        return df_istat

    def calculate_grid_coordinates(self, square_id: int) -> Tuple[float, float]:
        """
        Converts Telecom Italia Milan square_id (1-10000, 100x100 grid) into simulated WGS84 coordinates.
        Milan grid bounds roughly: Lat 45.40 to 45.54, Lon 9.08 to 9.28.
        """
        grid_x = (square_id - 1) % 100
        grid_y = (square_id - 1) // 100

        lon = 9.08 + (grid_x / 100.0) * (9.28 - 9.08)
        lat = 45.40 + (grid_y / 100.0) * (45.54 - 45.40)
        return round(lat, 6), round(lon, 6)

    def attach_geographic_coordinates(self, df: pd.DataFrame, square_col: str = "primary_square_id") -> pd.DataFrame:
        """
        Appends estimated latitude and longitude coordinates for each customer's primary grid square.
        """
        logger.info("Computing geospatial centroid coordinates for telecom grid cells...")
        coords = [self.calculate_grid_coordinates(sq) for sq in df[square_col]]
        df_geo = df.copy()
        df_geo["centroid_latitude"] = [c[0] for c in coords]
        df_geo["centroid_longitude"] = [c[1] for c in coords]
        return df_geo

    def enrich_customers_with_demographics(
        self,
        df_customers: pd.DataFrame,
        df_istat: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Performs relational and spatial join between subscriber records and neighborhood ISTAT metrics.
        """
        logger.info("Joining customer behavioral records with ISTAT census indicators...")
        
        # Merge on square_id
        merged = pd.merge(
            df_customers,
            df_istat,
            left_on="primary_square_id",
            right_on="square_id",
            how="left"
        )

        # Impute missing demographics with regional medians if any square_id was unmapped
        demographic_cols = [
            "median_household_income_eur",
            "population_density_sqkm",
            "unemployment_rate_pct",
            "pct_age_under_25",
            "pct_age_25_64",
            "pct_age_over_65",
            "tertiary_education_rate_pct"
        ]

        for col in demographic_cols:
            if col in merged.columns:
                col_median = merged[col].median()
                if pd.isna(col_median):
                    col_median = 32000.0 if "income" in col else (8500.0 if "density" in col else 8.5)
                merged[col] = merged[col].fillna(col_median)

        # Derive composite socioeconomic index (wealth vs vulnerability)
        merged["affluence_index"] = (
            (merged["median_household_income_eur"] / 30000.0)
            + (merged["tertiary_education_rate_pct"] / 30.0)
            - (merged["unemployment_rate_pct"] / 8.0)
        ).round(3)

        if "square_id" in merged.columns and "primary_square_id" in merged.columns:
            merged.drop(columns=["square_id"], inplace=True)

        logger.info(f"Enrichment complete. Total columns: {merged.shape[1]}, records: {merged.shape[0]:,}")
        return merged

    def compute_spatial_churn_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes statistical correlations between neighborhood demographics and subscriber churn/spend.
        """
        corr_cols = [
            "monthly_spend_eur",
            "churn_90d",
            "median_household_income_eur",
            "population_density_sqkm",
            "unemployment_rate_pct",
            "affluence_index"
        ]
        available = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[available].corr()
        logger.info(f"Spatial demographic correlation with churn & spend:\n{corr_matrix[['churn_90d', 'monthly_spend_eur']].round(3)}")
        return corr_matrix

    def run(self) -> pd.DataFrame:
        """
        Runs the full spatial enrichment pipeline and persists the combined dataset.
        """
        if not os.path.exists(self.customer_features_path):
            logger.info(f"Features file not found at {self.customer_features_path}. Running preprocessing generator first...")
            from scripts.preprocessing import CDRPreprocessor
            preprocessor = CDRPreprocessor()
            df_customers = preprocessor.run()
        else:
            df_customers = load_parquet(self.customer_features_path)

        df_istat = self.load_istat_demographics()
        df_customers_geo = self.attach_geographic_coordinates(df_customers)
        df_enriched = self.enrich_customers_with_demographics(df_customers_geo, df_istat)

        self.compute_spatial_churn_correlations(df_enriched)
        save_parquet(df_enriched, self.customer_features_path)
        logger.info(f"Persisted spatially enriched customer dataset to {self.customer_features_path}")
        return df_enriched

def main():
    service = SpatialEnrichmentService()
    df_result = service.run()
    logger.info(f"Sample enriched subscriber records:\n{df_result[['customer_id', 'primary_square_id', 'median_household_income_eur', 'affluence_index', 'churn_90d']].head()}")

if __name__ == "__main__":
    main()
