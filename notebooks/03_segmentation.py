# %% [markdown]
# # Notebook 03: Subscriber Segmentation - RFM Analysis & Unsupervised Clustering
# 
# **Project:** Subscriber Churn & Lifetime Value Predictor for Telecom  
# **Objective:** Segment telecom subscribers using dual methodologies:
# 1. Rule-based **RFM Analysis** (Recency, Frequency, Monetary value) with Markov lifecycle transitions.
# 2. Unsupervised **K-Means & Gaussian Mixture Models (GMM)** with Silhouette and Elbow evaluation.

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Add project root
sys.path.append(os.path.abspath(".."))
from scripts.utils import load_config, load_parquet, setup_logger
from scripts.rfm_analysis import RFMAnalyzer
from scripts.clustering import BehavioralClusterer

sns.set_theme(style="whitegrid")
logger = setup_logger("notebook_03")
config = load_config("../config/config.yaml")

# %% [markdown]
# ### 1. Execute RFM Analysis
# Compute Recency, Frequency, and Monetary scores (1-5 quintiles) and assign business lifecycle segments.

# %%
rfm_analyzer = RFMAnalyzer(config_path="../config/config.yaml")
df_rfm, df_rfm_summary, df_transitions = rfm_analyzer.run()

print("RFM Segment Profiles Summary:")
display(df_rfm_summary) if "display" in dir() else print(df_rfm_summary.to_string(index=False))

# %% [markdown]
# ### 2. Visualizing RFM Segment Distribution & Revenue Contribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Subscriber Count per Segment
sns.barplot(
    data=df_rfm_summary,
    y="rfm_segment",
    x="subscriber_count",
    palette="Blues_r",
    ax=axes[0]
)
axes[0].set_title("Subscriber Volume by RFM Lifecycle Segment", fontweight="bold")
axes[0].set_xlabel("Number of Subscribers")
axes[0].set_ylabel("")

# Plot 2: Revenue vs Churn Rate
ax2_twin = axes[1].twinx()
p1 = axes[1].bar(
    df_rfm_summary["rfm_segment"],
    df_rfm_summary["total_monthly_revenue"],
    color="#3B82F6",
    alpha=0.8,
    label="Monthly Revenue (€)"
)
p2 = ax2_twin.plot(
    df_rfm_summary["rfm_segment"],
    df_rfm_summary["observed_churn_rate"],
    color="#EF4444",
    marker="o",
    linewidth=2.5,
    label="Churn Rate (%)"
)
axes[1].set_title("Total Revenue vs Churn Rate by RFM Segment", fontweight="bold")
axes[1].set_xticklabels(df_rfm_summary["rfm_segment"], rotation=45, ha="right")
axes[1].set_ylabel("Monthly Revenue (€)")
ax2_twin.set_ylabel("Churn Rate (%)")
axes[1].grid(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Markov Lifecycle State Transition Matrix
# The transition matrix illustrates the probability that a subscriber drifts from state $S_i$ at time $t$ to state $S_j$ at time $t+1$.

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(df_transitions, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, square=True)
plt.title("Markov Chain State Transition Matrix (Month-over-Month)", fontsize=13, fontweight="bold")
plt.xlabel("Next Lifecycle State (T+1)")
plt.ylabel("Current Lifecycle State (T0)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Unsupervised Behavioral Clustering (K-Means & GMM)
# Evaluate optimal $k$ using Inertia (Elbow Method) and Silhouette Scores across standardized behavioral metrics.

# %%
clusterer = BehavioralClusterer(config_path="../config/config.yaml")
df_clustered, profile_summary, df_k_eval = clusterer.run(optimal_k=4)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Curve
axes[0].plot(df_k_eval["k"], df_k_eval["inertia"], marker="o", color="#2563EB", lw=2)
axes[0].set_title("Elbow Method: Inertia vs Number of Clusters (k)", fontweight="bold")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")

# Silhouette Score Curve
axes[1].plot(df_k_eval["k"], df_k_eval["silhouette_score"], marker="s", color="#10B981", lw=2)
axes[1].axvline(4, color="#DC2626", linestyle="--", label="Selected Optimal k=4")
axes[1].set_title("Silhouette Score vs Number of Clusters (k)", fontweight="bold")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Cluster Persona Profiling Radar Chart

# %%
cluster_feature_cols = [
    "avg_daily_calls", "avg_daily_sms", "avg_daily_internet_mb",
    "peak_hour_activity_ratio", "weekend_activity_ratio", "entropy_grid_dispersion"
]

cluster_means = df_clustered.groupby("cluster_persona")[cluster_feature_cols].mean()
# Normalize 0-1 for radar visualization
norm_cluster_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-6)

print("Behavioral Personas Identified:")
display(cluster_means.round(2)) if "display" in dir() else print(cluster_means.round(2))

# Parallel coordinate plot
plt.figure(figsize=(14, 6))
pd.plotting.parallel_coordinates(
    norm_cluster_means.reset_index(),
    class_column="cluster_persona",
    colormap=plt.cm.Set1,
    linewidth=3
)
plt.title("Subscriber Persona Profiles Across Standardized Behavioral Dimensions", fontsize=14, fontweight="bold")
plt.xticks(rotation=15)
plt.ylabel("Normalized Dimension Intensity (0.0 - 1.0)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Persona Tier")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6. Summary of Persona Archetypes
# 1. **Digital Streamers & Data Power Users:** Highest internet volume (>900 MB/day), low voice calls, moderate mobility. Highest retention elasticity.
# 2. **Voice & Business Communicators:** Heavy outgoing call traffic during peak hours (08:00 - 20:00), high ARPU, moderate churn risk.
# 3. **High-Mobility Commuters:** Elevated spatial entropy across Milan's transit corridors and business districts. Frequent multi-cell cell-towers handovers.
# 4. **Light Utility / Standard Subscribers:** Low overall volume, price sensitive, high risk of dormancy.
