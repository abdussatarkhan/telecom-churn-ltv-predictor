# Telecom Customer Churn & Lifetime Value (CLV) Engine

[![CI](https://github.com/abdussatarkhan/telecom-churn-ltv-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/abdussatarkhan/telecom-churn-ltv-predictor/actions)
[![XGBoost](https://img.shields.io/badge/XGBoost-Churn_ML-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/) [![Survival Analysis](https://img.shields.io/badge/Survival-Cox_PH-8338EC?style=for-the-badge)](https://lifelines.readthedocs.io/) [![Python](https://img.shields.io/badge/Python-LTV_Analytics-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **A dual-stage machine learning system combining Cox Proportional Hazards survival analysis for subscriber tenure estimation with XGBoost gradient boosting to predict churn risk and guide retention campaigns.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    Billing[Call Detail Records & Billing Logs] --> Feature[Tenure & Usage Feature Pipeline]
    Feature --> Hazard[Cox Proportional Hazards Survival Analysis]
    Feature --> XGB[XGBoost Churn Classifier]
    Hazard --> CLV[Customer Lifetime Value Valuation Matrix]
    XGB --> Retention[Uplift Retention Targeting Campaign]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `XGBoost` `Lifelines` `Pandas` `Scikit-Learn` `CLV`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![telecom-churn-ltv-predictor preview](images/churn_hazard_curves.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/telecom-churn-ltv-predictor.git
cd telecom-churn-ltv-predictor
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 🗺️ Roadmap & Upcoming Features

- [x] Cox Proportional Hazards customer tenure survival analysis
- [x] XGBoost churn risk classification and CLV matrix
- [ ] CausalML / EconML uplift modeling for retention targeting
- [ ] Automated high-risk customer retention discount generator
- [ ] Real-time CDR (Call Detail Record) streaming pipeline

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.