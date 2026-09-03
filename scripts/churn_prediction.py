"""
Subscriber Churn Prediction with SMOTE Oversampling & Cost-Sensitive Optimization
Builds an imbalanced learning pipeline combining SMOTE with Random Forest Classifier.
Optimizes classification decision thresholds to minimize customer lifetime value destruction.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    setup_logger,
    load_config,
    load_parquet,
    save_parquet,
    save_model_artifact,
    calculate_cost_optimal_threshold,
    evaluate_binary_classification,
    ensure_directory
)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Try importing imbalanced-learn with fallback pipeline
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    from sklearn.pipeline import Pipeline as ImbPipeline

logger = setup_logger("churn_prediction")

class ChurnPredictor:
    """
    Trains and evaluates end-to-end subscriber churn prediction pipeline with SMOTE.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.features_file = self.config["paths"]["customer_features_file"]
        self.output_file = self.config["paths"]["churn_predictions_file"]
        self.models_dir = ensure_directory(self.config["paths"]["models_dir"])
        self.target_col = self.config["churn_prediction"].get("target_column", "churn_90d")
        self.random_state = self.config["churn_prediction"].get("random_state", 42)
        self.rf_params = self.config["churn_prediction"].get("rf_hyperparameters", {
            "n_estimators": 250,
            "max_depth": 14,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "n_jobs": -1
        })

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list, list]:
        """
        Partitions feature matrix into numeric and categorical predictors.
        """
        exclude_cols = [
            "customer_id", "churn_90d", "true_churn_probability", "square_id",
            "primary_square_id", "rfm_code", "rfm_sum"
        ]
        
        categorical_candidates = ["contract_type", "device_type", "rfm_segment", "cluster_persona", "clv_segment"]
        cat_cols = [c for c in categorical_candidates if c in df.columns]

        all_features = [c for c in df.columns if c not in exclude_cols]
        num_cols = [c for c in all_features if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]

        logger.info(f"Identified {len(num_cols)} numerical features and {len(cat_cols)} categorical features.")
        X = df[num_cols + cat_cols].copy()
        y = df[self.target_col].copy()

        return X, y, num_cols, cat_cols

    def build_pipeline(self, num_cols: list, cat_cols: list) -> ImbPipeline:
        """
        Constructs preprocessing, SMOTE resampler, and Random Forest classifier within an imblearn Pipeline.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ]
        )

        rf = RandomForestClassifier(
            n_estimators=self.rf_params.get("n_estimators", 200),
            max_depth=self.rf_params.get("max_depth", 14),
            min_samples_split=self.rf_params.get("min_samples_split", 5),
            min_samples_leaf=self.rf_params.get("min_samples_leaf", 2),
            random_state=self.random_state,
            n_jobs=-1
        )

        if IMBLEARN_AVAILABLE:
            smote_ratio = self.config["churn_prediction"].get("smote_sampling_strategy", 0.65)
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=self.random_state)
            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("sampler", smote),
                ("classifier", rf)
            ])
        else:
            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", rf)
            ])

        return pipeline

    def extract_feature_importances(self, pipeline: ImbPipeline, num_cols: list, cat_cols: list) -> pd.DataFrame:
        """
        Extracts and tabulates Gini feature importances after one-hot encoding.
        """
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        cat_encoder = preprocessor.named_transformers_["cat"]
        if hasattr(cat_encoder, "get_feature_names_out"):
            encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        else:
            encoded_cat_names = [f"cat_{i}" for i in range(len(cat_cols))]

        feature_names = num_cols + encoded_cat_names
        importances = classifier.feature_importances_

        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp

    def optimize_threshold_by_clv(
        self,
        y_test: np.ndarray,
        y_probs: np.ndarray,
        clv_values: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calibrates optimal classification cut-off threshold prioritizing high-CLV subscribers.
        High-value subscribers receive lower threshold to maximize recall (prevent costly churn).
        """
        logger.info("Conducting CLV-weighted decision threshold optimization...")
        thresholds = np.linspace(0.15, 0.85, 71)
        best_th = 0.50
        min_loss = float("inf")
        loss_curve = []

        cost_offer = self.config["churn_prediction"]["threshold_tuning"].get("cost_fp", 50.0)

        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            # False Positive: offered retention incentive unnecessarily
            fp_cost = np.sum((preds == 1) & (y_test == 0)) * cost_offer
            # False Negative: lost subscriber lifetime value
            fn_mask = (preds == 0) & (y_test == 1)
            fn_cost = np.sum(clv_values[fn_mask])

            total_cost = fp_cost + fn_cost
            loss_curve.append({"threshold": th, "total_cost": total_cost})
            if total_cost < min_loss:
                min_loss = total_cost
                best_th = th

        logger.info(f"Optimal economic threshold: {best_th:.3f} (Total business cost: EUR {min_loss:,.2f})")
        return {"optimal_threshold": best_th, "minimum_loss_eur": min_loss, "curve": loss_curve}

    def run(self) -> Tuple[ImbPipeline, pd.DataFrame, Dict[str, Any]]:
        """
        Executes end-to-end model training, threshold tuning, and prediction export.
        """
        if not os.path.exists(self.features_file):
            from scripts.clv_modeling import CLVPredictor
            predictor = CLVPredictor()
            predictor.run()

        df = load_parquet(self.features_file)
        X, y, num_cols, cat_cols = self.select_features(df)

        # Train/Test Split
        test_size = self.config["churn_prediction"].get("test_size", 0.25)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Training split: {len(X_train):,} samples, Test split: {len(X_test):,} samples. Churn rate: {y_train.mean():.2%}")

        # Build and fit pipeline
        pipeline = self.build_pipeline(num_cols, cat_cols)
        pipeline.fit(X_train, y_train)

        # Evaluate on Test set
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        y_test_pred_default = (y_test_prob >= 0.50).astype(int)

        # Evaluate baseline metrics
        metrics_default = evaluate_binary_classification(y_test.values, y_test_pred_default, y_test_prob)
        logger.info(f"Baseline Test Metrics (Threshold=0.50):\n{metrics_default}")

        # CLV-weighted threshold optimization
        test_clv = df.loc[idx_test, "predicted_clv_12m_eur"].values if "predicted_clv_12m_eur" in df.columns else np.full(len(idx_test), 400.0)
        th_optimization = self.optimize_threshold_by_clv(y_test.values, y_test_prob, test_clv)
        optimal_th = th_optimization["optimal_threshold"]

        # Optimized predictions
        y_test_pred_opt = (y_test_prob >= optimal_th).astype(int)
        metrics_opt = evaluate_binary_classification(y_test.values, y_test_pred_opt, y_test_prob)
        logger.info(f"Cost-Optimized Test Metrics (Threshold={optimal_th:.3f}):\n{metrics_opt}")

        # Feature importances
        df_importances = self.extract_feature_importances(pipeline, num_cols, cat_cols)
        logger.info(f"Top 10 Churn Drivers:\n{df_importances.head(10).to_string(index=False)}")

        # Full cohort inference
        all_probs = pipeline.predict_proba(X)[:, 1]
        all_preds = (all_probs >= optimal_th).astype(int)

        df_predictions = pd.DataFrame({
            "customer_id": df["customer_id"],
            "churn_probability": all_probs.round(4),
            "predicted_churn": all_preds,
            "actual_churn": y.values,
            "churn_risk_tier": pd.cut(
                all_probs,
                bins=[0.0, 0.25, 0.50, 0.75, 1.0],
                labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
            )
        })

        # Save artifacts
        save_model_artifact(pipeline, os.path.join(self.models_dir, "churn_rf_pipeline.joblib"))
        save_parquet(df_predictions, self.output_file)

        # Enrich master features table
        df["churn_probability"] = all_probs.round(4)
        df["churn_risk_tier"] = df_predictions["churn_risk_tier"]
        save_parquet(df, self.features_file)

        report = {
            "baseline_metrics": metrics_default,
            "optimized_metrics": metrics_opt,
            "optimal_threshold": optimal_th,
            "top_drivers": df_importances.head(10).to_dict(orient="records")
        }

        return pipeline, df_predictions, report

def main():
    predictor = ChurnPredictor()
    pipeline, df_preds, report = predictor.run()
    print("\n--- Model Performance Summary ---")
    print(f"ROC-AUC: {report['optimized_metrics']['roc_auc']:.4f}")
    print(f"PR-AUC:  {report['optimized_metrics']['pr_auc']:.4f}")
    print(f"Recall:  {report['optimized_metrics']['sensitivity_recall']:.4f}")
    print(f"F2-Score: {report['optimized_metrics']['f2_score']:.4f}")

if __name__ == "__main__":
    main()
