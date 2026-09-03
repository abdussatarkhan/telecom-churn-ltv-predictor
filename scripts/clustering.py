"""
Behavioral Customer Clustering & Persona Profiling
Applies K-Means with Silhouette/Elbow analysis and Gaussian Mixture Models (GMM)
to multi-dimensional subscriber call/data/mobility patterns.
Produces distinct subscriber behavioral personas for targeted marketing.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    setup_logger,
    load_config,
    load_parquet,
    save_parquet,
    save_model_artifact,
    ensure_directory
)

logger = setup_logger("clustering")

class BehavioralClusterer:
    """
    Executes unsupervised behavioral segmentation using K-Means and Gaussian Mixture Models.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.features_file = self.config["paths"]["customer_features_file"]
        self.models_dir = ensure_directory(self.config["paths"]["models_dir"])
        self.cluster_features = self.config["clustering"].get("features", [
            "avg_daily_calls",
            "avg_daily_sms",
            "avg_daily_internet_mb",
            "peak_hour_activity_ratio",
            "weekend_activity_ratio",
            "international_call_ratio",
            "entropy_grid_dispersion",
            "median_household_income_eur"
        ])
        self.random_state = self.config["clustering"].get("kmeans", {}).get("random_state", 42)

    def prepare_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, List[str]]:
        """
        Extracts and standardizes numeric behavioral columns.
        """
        valid_cols = [c for c in self.cluster_features if c in df.columns]
        if not valid_cols:
            raise ValueError("None of the specified cluster features are present in the DataFrame.")

        logger.info(f"Using {len(valid_cols)} features for clustering: {valid_cols}")
        X_raw = df[valid_cols].copy()

        # Fill any missing values with median
        X_raw = X_raw.fillna(X_raw.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        return X_scaled, scaler, valid_cols

    def evaluate_optimal_k(self, X: np.ndarray, k_range: range = range(2, 9)) -> pd.DataFrame:
        """
        Computes Elbow (inertia), Silhouette Scores, and Davies-Bouldin indices across k values.
        """
        logger.info("Evaluating clustering performance across range of k...")
        metrics = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=15, max_iter=300)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels, sample_size=min(3000, len(X)))
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            inertia = km.inertia_

            metrics.append({
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil, 4),
                "davies_bouldin_score": round(db, 4),
                "calinski_harabasz": round(ch, 2)
            })

        df_metrics = pd.DataFrame(metrics)
        logger.info(f"K-Evaluation Summary:\n{df_metrics.to_string(index=False)}")
        return df_metrics

    def fit_kmeans(self, X: np.ndarray, k: int = 4) -> Tuple[KMeans, np.ndarray]:
        """
        Fits optimal K-Means model.
        """
        logger.info(f"Training K-Means model with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=25, max_iter=400)
        labels = kmeans.fit_predict(X)
        return kmeans, labels

    def fit_gmm(self, X: np.ndarray, n_components: int = 4) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
        """
        Fits Gaussian Mixture Model to calculate soft probabilistic cluster memberships.
        """
        logger.info(f"Fitting Gaussian Mixture Model with {n_components} components...")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=200
        )
        labels = gmm.fit_predict(X)
        probabilities = gmm.predict_proba(X)
        logger.info(f"GMM Log-Likelihood: {round(gmm.lower_bound_, 4)}, BIC: {round(gmm.bic(X), 2)}, AIC: {round(gmm.aic(X), 2)}")
        return gmm, labels, probabilities

    def interpret_clusters(self, df: pd.DataFrame, feature_cols: List[str], cluster_col: str = "kmeans_cluster") -> pd.DataFrame:
        """
        Generates descriptive profiles and assigns human-interpretable persona labels to clusters.
        """
        logger.info("Profiling clusters across behavioral dimensions...")
        profile = df.groupby(cluster_col)[feature_cols].mean()

        # Assign persona labels based on dominant feature characteristics
        persona_map = {}
        for cluster_id, row in profile.iterrows():
            if row.get("avg_daily_internet_mb", 0) > profile["avg_daily_internet_mb"].mean() * 1.2:
                persona_map[cluster_id] = "Digital Streamers & Data Power Users"
            elif row.get("avg_daily_calls", 0) > profile["avg_daily_calls"].mean() * 1.2:
                persona_map[cluster_id] = "Voice & Business Communicators"
            elif row.get("entropy_grid_dispersion", 0) > profile["entropy_grid_dispersion"].mean() * 1.1:
                persona_map[cluster_id] = "High-Mobility Commuters"
            else:
                persona_map[cluster_id] = "Light Utility / Standard Subscribers"

        df["cluster_persona"] = df[cluster_col].map(persona_map)

        profile["persona"] = [persona_map.get(i, f"Segment_{i}") for i in profile.index]
        profile["subscriber_count"] = df[cluster_col].value_counts().sort_index()
        profile["churn_rate_pct"] = (df.groupby(cluster_col)["churn_90d"].mean() * 100).round(1) if "churn_90d" in df.columns else 0.0

        return profile

    def run(self, optimal_k: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes complete clustering pipeline, serializes models, and enriches customer dataframe.
        """
        if not os.path.exists(self.features_file):
            from scripts.spatial_joins import SpatialEnrichmentService
            service = SpatialEnrichmentService()
            df = service.run()
        else:
            df = load_parquet(self.features_file)

        X_scaled, scaler, feature_cols = self.prepare_feature_matrix(df)

        # 1. Silhouette / Elbow Evaluation
        df_k_eval = self.evaluate_optimal_k(X_scaled, k_range=range(2, 8))

        # 2. Fit K-Means
        kmeans_model, km_labels = self.fit_kmeans(X_scaled, k=optimal_k)
        df["kmeans_cluster"] = km_labels

        # 3. Fit GMM
        gmm_model, gmm_labels, gmm_probs = self.fit_gmm(X_scaled, n_components=optimal_k)
        df["gmm_cluster"] = gmm_labels
        df["gmm_max_probability"] = gmm_probs.max(axis=1).round(4)

        # 4. Profile clusters
        profile_summary = self.interpret_clusters(df, feature_cols, cluster_col="kmeans_cluster")

        # 5. Persist models
        save_model_artifact(scaler, os.path.join(self.models_dir, "scaler_behavioral.joblib"))
        save_model_artifact(kmeans_model, os.path.join(self.models_dir, "kmeans_behavioral.joblib"))
        save_model_artifact(gmm_model, os.path.join(self.models_dir, "gmm_behavioral.joblib"))

        # 6. Save updated dataframe
        save_parquet(df, self.features_file)
        logger.info(f"Clustering complete. Persona Profiles:\n{profile_summary[['persona', 'subscriber_count', 'churn_rate_pct']].to_string()}")

        return df, profile_summary, df_k_eval

def main():
    clusterer = BehavioralClusterer()
    df, profiles, k_eval = clusterer.run(optimal_k=4)
    print("\n--- Cluster Personas Summary ---")
    print(profiles.to_string())

if __name__ == "__main__":
    main()
