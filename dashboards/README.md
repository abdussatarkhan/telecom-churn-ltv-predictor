# Telecom Churn & LTV Predictor - Tableau & PowerBI Dashboard Specifications

This directory documents the visual analytics architecture, KPI definitions, and interactive dashboard wireframes supporting telecommunications marketing executives, churn management teams, and commercial finance.

---

## 1. Executive Dashboard Architecture

The business dashboard suite consists of three interconnected interactive views:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    TELECOM EXECUTIVE INTELLIGENCE SUITE                    │
├──────────────────────┬──────────────────────┬──────────────────────────────┤
│ 1. Executive Churn   │ 2. Spatiotemporal &  │ 3. Strategic Retention       │
│    & CLV Overview    │    Behavioral EDA    │    Campaign Simulator        │
└──────────────────────┴──────────────────────┴──────────────────────────────┘
```

---

## 2. Dashboard 1: Executive Churn & CLV Overview

### Primary Purpose
Provides C-level executives and commercial leaders with an immediate view of subscriber attrition risks, monetary value at risk, and forecasted lifetime value.

### Key Performance Indicators (KPI Cards)
1. **Total Active Subscribers**: `5,000` (Sample cohort) / `1.42M` (Production Milan Metropolitan)
2. **Portfolio Churn Rate (90-Day)**: `21.4%` (Target: `<18.0%`)
3. **Total 12-Month Forecasted CLV**: `€2,485,320`
4. **Immediate CLV at Risk**: `€532,180` (High & Critical Risk cohorts)
5. **Simulated Retention Program ROI**: `294.2%`

### Visual Components
- **Cross-Tabulation Matrix**: 4x4 interactive heatmap displaying customer volume and total monetary value across *Churn Risk Tiers* (Low, Medium, High, Critical) and *CLV Segments* (Platinum VIP, Gold High Value, Silver Core, Bronze Basic).
- **Geographic Milan Grid Map**: High-resolution choropleth map plotting Milan's 10,000 spatial cells, color-coded by churn concentration with overlay of average household income.
- **Contract Type & Tenure Distribution**: Grouped bar chart depicting tenure distribution partitioned by contract type (Month-to-Month, One Year, Two Year).

---

## 3. Dashboard 2: Spatiotemporal & Behavioral Intelligence

### Primary Purpose
Enables telecommunication network planners and segment marketers to dissect usage trends, peak vs. off-peak hours, and demographic correlation.

### Visual Components
- **Hourly Activity Profile (Dual Axis)**: Line chart tracking 24-hour Internet MB throughput alongside voice call volume, highlighting the official peak window (08:00 - 20:00).
- **Spatial Mobility Entropy Scatter**: Scatterplot of customer mobility entropy versus monthly spend, segmented by behavioral persona (*Digital Streamers*, *Voice Communicators*, *High-Mobility Commuters*, *Light Utility*).
- **Socioeconomic Demographics Correlation**: Multi-metric correlation matrix showing ISTAT census metrics (Median Household Income, Unemployment Rate, Tertiary Education) paired with churn probability.

---

## 4. Dashboard 3: Strategic Retention Campaign Simulator

### Primary Purpose
Operational workbench for customer retention specialists and CRM teams to simulate intervention costs, acceptances, and projected net financial returns.

### Interactive Parameters & What-If Sliders
- **VIP Offer Discount Rate**: Slider `[10% - 40%]`, default `25%`
- **Device Upgrade Subsidy Voucher**: Numeric input `[€30 - €150]`, default `€60`
- **Expected Offer Acceptance Rate**: Slider `[30% - 85%]`, default `65%`
- **Churn Mitigation Efficacy**: Slider `[20% - 60%]`, default `45%`

### Real-Time Financial Calculations
$$\text{Campaign Cost} = N_{\text{targeted}} \times \text{Unit Cost} \times \text{Take-up Rate}$$
$$\text{Gross Saved CLV} = \sum \text{CLV}_{\text{targeted}} \times \text{Churn Mitigation Efficacy}$$
$$\text{Net Business Benefit} = \text{Gross Saved CLV} - \text{Campaign Cost}$$
$$\text{Retention Campaign ROI (\%)} = \left(\frac{\text{Net Business Benefit}}{\text{Campaign Cost}}\right) \times 100\%$$

---

## 5. Tableau / PowerBI Data Connection Guide

1. **Data Source**: BigQuery or local processed Parquet files (`data/processed/customer_features.parquet`, `data/processed/retention_strategy_matrix.csv`).
2. **Refresh Frequency**: Daily batch refresh scheduled via Cloud Composer / Apache Airflow at 03:00 UTC.
3. **Row-Level Security (RLS)**: Filtered by regional operations manager permissions based on `dominant_square_id` and postal code sectors.
