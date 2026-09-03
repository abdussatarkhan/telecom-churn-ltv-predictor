# %% [markdown]
# # Notebook 01: Telecom Italia CDR & ISTAT Demographic Ingestion
# 
# **Project:** Subscriber Churn & Lifetime Value Predictor for Telecom  
# **Objective:** Ingest Call Detail Records (CDRs) from Harvard Dataverse, process spatial grid indexes for the City of Milan, and join municipal census socioeconomic indicators from ISTAT.

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in system path
sys.path.append(os.path.abspath(".."))
from scripts.utils import load_config, setup_logger, generate_synthetic_telecom_data, load_parquet, save_parquet
from scripts.data_collection import DataCollectionManager
from scripts.preprocessing import CDRPreprocessor
from scripts.spatial_joins import SpatialEnrichmentService

sns.set_theme(style="whitegrid", palette="muted")
logger = setup_logger("notebook_01")
config = load_config("../config/config.yaml")

# %% [markdown]
# ### 1. Ingestion Pipeline Execution
# Trigger the data ingestion pipeline to pull or instantiate Telecom Italia CDR records and ISTAT census indicators.

# %%
collector = DataCollectionManager(config_path="../config/config.yaml")
collector.execute_collection_pipeline(use_synthetic_fallback=True)

# %% [markdown]
# ### 2. Raw CDR Stream Inspection
# Let's inspect the raw structure of the telecommunications activity stream.
# Each record represents aggregate telecommunications volume across a 10-minute time window in a 235m x 235m grid square in Milan.

# %%
df_cust, df_istat, df_cdr = generate_synthetic_telecom_data(num_subscribers=2000, random_seed=42)
print("CDR Event Records Sample:")
display(df_cdr.head(8)) if "display" in dir() else print(df_cdr.head(8))

# %%
print(f"CDR Schema and Data Types:\n{df_cdr.dtypes}")
print(f"\nTotal Records: {len(df_cdr):,}")
print(f"Unique Grid Squares Represented: {df_cdr['square_id'].nunique():,}")
print(f"Unique Country Codes: {df_cdr['country_code'].unique().tolist()}")

# %% [markdown]
# ### 3. Temporal Distribution of Telecommunication Volume
# Parse timestamps from milliseconds to datetime and examine hourly activity peaks.

# %%
df_cdr_parsed = df_cdr.copy()
df_cdr_parsed["datetime"] = pd.to_datetime(df_cdr_parsed["time_interval"], unit="ms")
df_cdr_parsed["hour"] = df_cdr_parsed["datetime"].dt.hour

hourly_traffic = df_cdr_parsed.groupby("hour")[["call_in", "call_out", "internet_traffic"]].mean()

plt.figure(figsize=(12, 5))
plt.plot(hourly_traffic.index, hourly_traffic["internet_traffic"], label="Internet Traffic (MB Index)", color="#2563EB", lw=2.5, marker="o")
plt.plot(hourly_traffic.index, hourly_traffic["call_out"] * 5, label="Outgoing Calls (Scaled x5)", color="#DC2626", lw=2, linestyle="--")
plt.title("Telecom Activity Distribution by Hour of Day (Milan Grid)", fontsize=14, fontweight="bold")
plt.xlabel("Hour of Day (0 - 23)")
plt.ylabel("Average Activity Index")
plt.axvspan(8, 20, color="#FEF3C7", alpha=0.4, label="Peak Window (08:00 - 20:00)")
plt.legend(frameon=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. ISTAT Socioeconomic Indicators Ingestion
# Examine census variables: household income, population density, unemployment, and age distributions.

# %%
print("ISTAT Municipal Census Indicators Sample:")
display(df_istat.head()) if "display" in dir() else print(df_istat.head())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_istat["median_household_income_eur"], kde=True, ax=axes[0], color="#059669", bins=30)
axes[0].set_title("Median Household Income Distribution (€)", fontweight="bold")
axes[0].set_xlabel("Annual Income (€)")

sns.scatterplot(
    data=df_istat,
    x="population_density_sqkm",
    y="median_household_income_eur",
    hue="unemployment_rate_pct",
    palette="viridis",
    ax=axes[1],
    alpha=0.7
)
axes[1].set_title("Density vs Income by Milan Grid Cell", fontweight="bold")
axes[1].set_xlabel("Population Density (people / km²)")
axes[1].set_ylabel("Median Household Income (€)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Spatial Join and Feature Aggregation
# Run the preprocessing and spatial enrichment services to compile the master analytical dataset.

# %%
preprocessor = CDRPreprocessor(config_path="../config/config.yaml")
df_preprocessed = preprocessor.run()

spatial_service = SpatialEnrichmentService(config_path="../config/config.yaml")
df_master = spatial_service.run()

print(f"Master Enriched Customer Table Shape: {df_master.shape}")
print("Sample Columns:", df_master.columns.tolist()[:10])
df_master.head(3)
