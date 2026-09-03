"""
Telecom Churn & Lifetime Value Predictor - Core Utilities
Provides centralized logging, configuration management, synthetic benchmark generation,
data persistence helpers, and statistical evaluation metrics.
"""

import os
import sys
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)

def setup_logger(name: str = "telecom_analytics", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a structured logger with standardized formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger("utils")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration from project root or provided path.
    """
    if config_path is None:
        # Default search path relative to current script or workspace
        candidates = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path is None or not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Falling back to default configuration dictionary.")
        return get_default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Fallback configuration dictionary if YAML is not accessible.
    """
    return {
        "paths": {
            "raw_cdr_dir": "data/raw/cdr",
            "raw_istat_dir": "data/raw/istat",
            "processed_dir": "data/processed",
            "external_dir": "data/external",
            "models_dir": "models",
            "customer_features_file": "data/processed/customer_features.parquet",
            "rfm_metrics_file": "data/processed/rfm_metrics.parquet",
            "clv_predictions_file": "data/processed/clv_predictions.parquet",
            "churn_predictions_file": "data/processed/churn_predictions.parquet"
        },
        "cdr_schema": {
            "peak_start_hour": 8,
            "peak_end_hour": 20
        },
        "clustering": {
            "optimal_k": 4,
            "random_state": 42
        },
        "churn_prediction": {
            "target_column": "churn_90d",
            "random_state": 42,
            "smote_sampling_strategy": 0.65
        }
    }

def ensure_directory(dir_path: str) -> str:
    """
    Ensures that a directory exists, creating parent directories if necessary.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def save_parquet(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves a DataFrame to Parquet format with snappy compression.
    """
    ensure_directory(str(Path(filepath).parent))
    df.to_parquet(filepath, index=False, engine="pyarrow", compression="snappy")
    logger.info(f"Saved {len(df):,} records to {filepath}")

def load_parquet(filepath: str) -> pd.DataFrame:
    """
    Loads a DataFrame from Parquet format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file does not exist at {filepath}")
    df = pd.read_parquet(filepath, engine="pyarrow")
    logger.info(f"Loaded {len(df):,} records from {filepath}")
    return df

def save_model_artifact(model: Any, filepath: str) -> None:
    """
    Serializes a trained machine learning model or pipeline using joblib.
    """
    ensure_directory(str(Path(filepath).parent))
    joblib.dump(model, filepath)
    logger.info(f"Persisted model artifact to {filepath}")

def load_model_artifact(filepath: str) -> Any:
    """
    Loads a serialized model artifact.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Loaded model artifact from {filepath}")
    return model

def calculate_cost_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    cost_fp: float = 50.0,
    cost_fn: float = 480.0
) -> Tuple[float, float]:
    """
    Finds classification probability threshold minimizing total business financial loss.
    Loss = False Positives * cost_fp (unnecessary retention cost) + False Negatives * cost_fn (lost subscriber CLV).
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    min_loss = float("inf")
    best_threshold = 0.50

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        total_loss = (fp * cost_fp) + (fn * cost_fn)
        if total_loss < min_loss:
            min_loss = total_loss
            best_threshold = th

    return best_threshold, min_loss

def evaluate_binary_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes comprehensive binary classification metrics for imbalanced telecom churn data.
    """
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = 0.0
            
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)
        metrics["sensitivity_recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return metrics

def generate_synthetic_telecom_data(
    num_subscribers: int = 5000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic synthetic CDR event stream, subscriber profiles, and ISTAT grid demographics
    mirroring the Telecom Italia Big Data Challenge Milan schema.
    Used for testing, demonstration, and offline execution.
    """
    np.random.seed(random_seed)
    logger.info(f"Generating synthetic telecom dataset for {num_subscribers:,} subscribers...")

    # 1. Grid Cells and ISTAT Demographics (Milan 10,000 grid simulation, central ~1,000 active cells)
    grid_cells = np.random.choice(range(1, 10001), size=800, replace=False)
    istat_records = []
    for g in grid_cells:
        dist_from_center = np.sqrt(((g % 100) - 50)**2 + ((g // 100) - 50)**2)
        income = max(18000.0, 48000.0 - dist_from_center * 450 + np.random.normal(0, 3000))
        pop_density = max(1200.0, 15000.0 - dist_from_center * 180 + np.random.normal(0, 1000))
        unemployment = min(18.5, max(4.0, 6.5 + dist_from_center * 0.15 + np.random.normal(0, 1)))
        istat_records.append({
            "square_id": int(g),
            "median_household_income_eur": round(income, 2),
            "population_density_sqkm": round(pop_density, 1),
            "unemployment_rate_pct": round(unemployment, 2),
            "pct_age_under_25": round(np.random.uniform(18.0, 26.0), 1),
            "pct_age_25_64": round(np.random.uniform(52.0, 62.0), 1),
            "pct_age_over_65": round(np.random.uniform(15.0, 28.0), 1),
            "tertiary_education_rate_pct": round(np.random.uniform(22.0, 58.0), 1)
        })
    df_istat = pd.DataFrame(istat_records)

    # 2. Customer Master Profiles
    customer_ids = [f"IT_TEL_{i:06d}" for i in range(1, num_subscribers + 1)]
    contract_types = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_subscribers, p=[0.55, 0.25, 0.20])
    tenures = np.random.exponential(scale=24, size=num_subscribers).clip(1, 72).astype(int)
    device_types = np.random.choice(["5G Flagship", "4G Smartphone", "Basic VoLTE", "IoT/M2M"], size=num_subscribers, p=[0.35, 0.50, 0.12, 0.03])
    primary_squares = np.random.choice(grid_cells, size=num_subscribers)
    
    # 3. Aggregated Behavioral Metrics (derived from simulated 10M CDR interactions)
    # Segments: Tech-heavy professionals, Students/Digital Natives, Traditional Voice/SMS, Light/Senior
    segment_assignment = np.random.choice(["Digital_Heavy", "Voice_Centric", "Balanced_Standard", "Dormant_Light"], size=num_subscribers, p=[0.30, 0.25, 0.30, 0.15])
    
    avg_calls = []
    avg_sms = []
    avg_data_mb = []
    peak_ratio = []
    weekend_ratio = []
    intl_ratio = []
    monetary_spend = []
    recency_days = []
    frequency_months = []
    
    for seg, tenure, contract in zip(segment_assignment, tenures, contract_types):
        if seg == "Digital_Heavy":
            c = np.random.gamma(4, 1.5)
            s = np.random.gamma(3, 2.0)
            d = np.random.gamma(15, 60.0) # ~900 MB daily average
            pr = np.random.beta(5, 3)
            wr = np.random.beta(3, 5)
            ir = np.random.beta(1.5, 8.0)
            base_arpu = 38.5 + (d / 80.0)
            rec = int(np.random.exponential(2).clip(0, 15))
            freq = int(min(tenure, np.random.poisson(11) + 1))
        elif seg == "Voice_Centric":
            c = np.random.gamma(12, 1.8)
            s = np.random.gamma(8, 1.5)
            d = np.random.gamma(4, 30.0)
            pr = np.random.beta(7, 2)
            wr = np.random.beta(2, 6)
            ir = np.random.beta(2.5, 6.0)
            base_arpu = 29.0 + (c * 0.8)
            rec = int(np.random.exponential(3).clip(0, 20))
            freq = int(min(tenure, np.random.poisson(10) + 1))
        elif seg == "Balanced_Standard":
            c = np.random.gamma(5, 1.2)
            s = np.random.gamma(5, 1.2)
            d = np.random.gamma(8, 45.0)
            pr = np.random.beta(4, 4)
            wr = np.random.beta(3, 4)
            ir = np.random.beta(1.0, 10.0)
            base_arpu = 22.0 + (d / 120.0)
            rec = int(np.random.exponential(5).clip(0, 30))
            freq = int(min(tenure, np.random.poisson(8) + 1))
        else: # Dormant_Light
            c = np.random.gamma(1, 0.8)
            s = np.random.gamma(1, 0.8)
            d = np.random.gamma(1.5, 15.0)
            pr = np.random.beta(2, 5)
            wr = np.random.beta(4, 3)
            ir = np.random.beta(0.5, 15.0)
            base_arpu = 9.50 + np.random.uniform(0, 4)
            rec = int(np.random.exponential(18).clip(0, 75))
            freq = int(min(tenure, np.random.poisson(3) + 1))
            
        avg_calls.append(round(c, 2))
        avg_sms.append(round(s, 2))
        avg_data_mb.append(round(d, 2))
        peak_ratio.append(round(pr, 3))
        weekend_ratio.append(round(wr, 3))
        intl_ratio.append(round(ir, 3))
        monetary_spend.append(round(base_arpu, 2))
        recency_days.append(rec)
        frequency_months.append(freq)

    # Churn probability generation correlated with month-to-month contracts, low frequency, high recency, high spend with poor voice stability
    churn_logits = (
        -1.8
        + 1.35 * (contract_types == "Month-to-month").astype(float)
        - 0.85 * (contract_types == "Two year").astype(float)
        - 0.04 * tenures
        + 0.05 * np.array(recency_days)
        - 0.12 * np.array(frequency_months)
        + 0.02 * np.array(monetary_spend)
        - 0.001 * np.array(avg_data_mb)
        + np.random.normal(0, 0.45, size=num_subscribers)
    )
    churn_probs = 1.0 / (1.0 + np.exp(-churn_logits))
    churn_labels = (churn_probs >= 0.48).astype(int)

    df_customers = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenures,
        "contract_type": contract_types,
        "device_type": device_types,
        "primary_square_id": primary_squares,
        "avg_daily_calls": avg_calls,
        "avg_daily_sms": avg_sms,
        "avg_daily_internet_mb": avg_data_mb,
        "peak_hour_activity_ratio": peak_ratio,
        "weekend_activity_ratio": weekend_ratio,
        "international_call_ratio": intl_ratio,
        "unique_contacts_monthly": np.random.poisson(24, size=num_subscribers).clip(2, 120),
        "entropy_grid_dispersion": np.random.beta(3, 2, size=num_subscribers).round(3),
        "monthly_spend_eur": monetary_spend,
        "recency_days": recency_days,
        "frequency_repeat_months": frequency_months,
        "churn_90d": churn_labels,
        "true_churn_probability": churn_probs.round(4)
    })

    # 4. Raw CDR interaction aggregate events simulation
    cdr_events = []
    sample_subscribers = customer_ids[:500]
    for cid in sample_subscribers:
        sq = np.random.choice(grid_cells)
        for day in range(1, 31):
            ts = 1383264000000 + (day * 86400000) + int(np.random.uniform(0, 86400000))
            cdr_events.append({
                "square_id": sq,
                "time_interval": ts,
                "country_code": 39 if np.random.rand() > 0.12 else int(np.random.choice([44, 33, 49, 34])),
                "sms_in": round(np.random.exponential(1.5), 3),
                "sms_out": round(np.random.exponential(1.2), 3),
                "call_in": round(np.random.exponential(2.8), 3),
                "call_out": round(np.random.exponential(2.5), 3),
                "internet_traffic": round(np.random.exponential(25.0), 3),
                "customer_id": cid
            })
    df_cdr_sample = pd.DataFrame(cdr_events)

    logger.info(f"Synthetic generation complete: {len(df_customers):,} customer profiles, {len(df_istat):,} grid zones, {len(df_cdr_sample):,} CDR sample events.")
    return df_customers, df_istat, df_cdr_sample
