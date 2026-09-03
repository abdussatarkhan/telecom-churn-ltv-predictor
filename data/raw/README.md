# Telecom Churn & LTV Predictor - Raw Data Ingestion Guide

This repository utilizes high-resolution telecommunications activity records and regional demographic indicators to build subscriber behavioral profiles, evaluate retention risks, and forecast 12-month Customer Lifetime Value (CLV).

---

## 1. Data Sources Overview

### A. Telecom Italia Big Data Challenge (Harvard Dataverse)
- **Repository**: Harvard Dataverse
- **Dataset Title**: *Open Data Trentino - Telecom Italia Big Data Challenge 2014*
- **Persistent DOI**: `doi:10.7910/DVN/0AGIX7`
- **Dataverse URL**: [Harvard Dataverse - Telecom Italia](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0AGIX7)
- **Coverage Area**: Metropolitan City of Milan (Grid of 10,000 square cells, 235m x 235m each) & Autonomous Province of Trento.
- **Temporal Resolution**: 10-minute intervals over November and December 2013 (~10M+ records per city slice).
- **Core Variables in CDR Stream**:
  - `square_id`: Unique integer identifier for spatial grid cell (1 to 10,000 in Milan).
  - `time_interval`: Epoch timestamp (milliseconds) representing the start of the 10-minute observation window.
  - `country_code`: Calling code (e.g., 39 for Italy, 44 for UK, 33 for France, etc.).
  - `sms_in`: Incoming SMS activity index.
  - `sms_out`: Outgoing SMS activity index.
  - `call_in`: Incoming call activity volume/duration index.
  - `call_out`: Outgoing call activity volume/duration index.
  - `internet_traffic`: Packet data traffic index (megabytes proportional).

### B. ISTAT Italian National Institute of Statistics
- **Portal**: [ISTAT Dati Censuari e Territoriali](https://www.istat.it/)
- **Demographic Dimensions**:
  - Census Sections (Basi territoriali e sezioni di censimento 2011 / 2021).
  - Spatial Boundaries: WGS84 EPSG:4326 & UTM Zone 32N EPSG:32632 shapefiles for Milan.
  - Key Socioeconomic Attributes:
    - `median_household_income_eur`: Estimated disposable income by administrative zone.
    - `population_density_sqkm`: Resident population per square kilometer.
    - `pct_age_under_25`, `pct_age_25_64`, `pct_age_over_65`: Age tier breakdown.
    - `unemployment_rate_pct`: Municipal labor survey estimates.
    - `tertiary_education_rate_pct`: Proportion of university graduates.

### C. IBM Telco Customer Churn Reference Schema
- Standard industry subscription features including tenure, contract type (Month-to-Month, One Year, Two Year), payment method, billing amounts, and churn indicators mapped onto behavioral usage segments.

---

## 2. Automated Download and Setup

Run the automated ingestion script to download sample partitions from Harvard Dataverse and ISTAT portals:

```bash
python scripts/data_collection.py --download-samples --city milano
```

To fetch full partitions directly via the Dataverse API:
```bash
python scripts/data_collection.py --doi "10.7910/DVN/0AGIX7" --output-dir data/raw/cdr/
```

If you operate in an offline or sandbox environment without internet connectivity, `scripts/data_collection.py` automatically generates a statistically representative synthetic CDR cohort (100,000+ subscriber-level activity events with spatio-temporal realism) matching the exact schema and distributions of the Telecom Italia dataset.

---

## 3. Directory Layout Expected

```
data/
├── raw/
│   ├── cdr/
│   │   ├── sms-call-internet-mi-2013-11-01.txt.gz
│   │   ├── sms-call-internet-mi-2013-11-02.txt.gz
│   │   └── ...
│   ├── istat/
│   │   ├── milano_census_2011.geojson
│   │   └── istat_socioeconomic_indicators.csv
│   └── README.md
├── processed/
│   ├── customer_features.parquet
│   ├── rfm_metrics.parquet
│   ├── clv_predictions.parquet
│   └── churn_predictions.parquet
└── external/
    └── milano_grid_geometry.geojson
```

---

## 4. Citation and Terms of Use

- *Barlacchi, G., Perentis, C., Mehrotra, A., Musolesi, M., & Lepri, B. (2015). A multi-source dataset of urban life in the city of Milan and the Province of Trentino. Scientific Data 2, 150055.*
- Dataverse DOI: [https://doi.org/10.7910/DVN/0AGIX7](https://doi.org/10.7910/DVN/0AGIX7)
- Telecom Italia Open Data License & CC-BY 4.0 International.
