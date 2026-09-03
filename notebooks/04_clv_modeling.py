# %% [markdown]
# # Notebook 04: Customer Lifetime Value (CLV) Forecasting
# 
# **Project:** Subscriber Churn & Lifetime Value Predictor for Telecom  
# **Objective:** Calibrate probabilistic models to forecast 12-month forward subscriber revenue:
# 1. **BG/NBD Model:** Predicts future transaction frequency and individual $P(\text{Alive})$.
# 2. **Gamma-Gamma Submodel:** Predicts expected monetary spend per transaction.
# 3. **Net Present Value (NPV):** Discounts cash flows to compute 12-month forward CLV.

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.path.abspath(".."))
from scripts.utils import load_config, load_parquet, setup_logger
from scripts.clv_modeling import CLVPredictor

sns.set_theme(style="whitegrid")
logger = setup_logger("notebook_04")
config = load_config("../config/config.yaml")

# %% [markdown]
# ### 1. Calibrate BG/NBD and Gamma-Gamma Models
# Execute the CLV estimation pipeline across the subscriber portfolio.

# %%
clv_service = CLVPredictor(config_path="../config/config.yaml")
df_clv, bg_model, gg_model = clv_service.run()

print("Sample CLV Predictions:")
display(df_clv.head(8)) if "display" in dir() else print(df_clv.head(8))

# %% [markdown]
# ### 2. Visualizing Probability of Being Alive: $P(\text{Alive})$
# Subscribers with high repeat frequency and recent engagement maintain near 1.0 probability of life.
# As inactivity duration (recency gap) increases, $P(\text{Alive})$ rapidly degrades towards zero.

# %%
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df_clv["recency"],
    df_clv["frequency"],
    c=df_clv["prob_alive"],
    cmap="RdYlGn",
    alpha=0.6,
    edgecolors="none",
    s=35
)
plt.title("BG/NBD Customer Latency Map: P(Alive) vs Recency & Frequency", fontsize=14, fontweight="bold")
plt.xlabel("Recency (Months between first and latest transaction)")
plt.ylabel("Frequency (Number of repeat billing cycles)")
cbar = plt.colorbar(scatter)
cbar.set_label("Estimated P(Alive)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Expected 12-Month Transactions Distribution
# Compare anticipated billing events across different current engagement tiers.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(df_clv["expected_transactions_12m"], bins=25, kde=True, ax=axes[0], color="#2563EB")
axes[0].set_title("Distribution of Expected 12-Month Transactions", fontweight="bold")
axes[0].set_xlabel("Expected Transactions (0 - 12 Months)")
axes[0].axvline(df_clv["expected_transactions_12m"].mean(), color="#DC2626", linestyle="--", label=f"Mean: {df_clv['expected_transactions_12m'].mean():.1f}")
axes[0].legend()

sns.scatterplot(
    data=df_clv,
    x="expected_transactions_12m",
    y="expected_avg_spend_eur",
    hue="clv_segment",
    palette="viridis",
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title("Expected Transactions vs Average Spend by CLV Tier", fontweight="bold")
axes[1].set_xlabel("Expected 12M Transactions")
axes[1].set_ylabel("Expected Avg Monthly Spend (€)")
axes[1].legend(title="CLV Segment", loc="upper left")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. 12-Month Discounted CLV Portfolio Stratification
# Breakdown of forecasted net present value across customer segments.

# %%
clv_summary = df_clv.groupby("clv_segment").agg(
    subscribers=("customer_id", "count"),
    avg_clv_eur=("predicted_clv_12m_eur", "mean"),
    min_clv_eur=("predicted_clv_12m_eur", "min"),
    max_clv_eur=("predicted_clv_12m_eur", "max"),
    total_portfolio_clv=("predicted_clv_12m_eur", "sum")
).reset_index()

total_val = clv_summary["total_portfolio_clv"].sum()
clv_summary["revenue_share_pct"] = (clv_summary["total_portfolio_clv"] / total_val * 100).round(1)
clv_summary["avg_clv_eur"] = clv_summary["avg_clv_eur"].round(2)
clv_summary["total_portfolio_clv"] = clv_summary["total_portfolio_clv"].round(2)

print("Portfolio CLV Stratification:")
display(clv_summary) if "display" in dir() else print(clv_summary.to_string(index=False))

# %% [markdown]
# ### 5. Pareto Revenue Concentration (Lorenz Curve)
# Calculate the cumulative percentage of subscribers against cumulative forecasted CLV.

# %%
sorted_clv = df_clv["predicted_clv_12m_eur"].sort_values(ascending=False).values
cum_clv = np.cumsum(sorted_clv) / np.sum(sorted_clv)
cum_subs = np.linspace(0, 1, len(sorted_clv))

plt.figure(figsize=(9, 6))
plt.plot(cum_subs * 100, cum_clv * 100, color="#7C3AED", lw=3, label="Subscriber Portfolio Lorenz Curve")
plt.plot([0, 100], [0, 100], color="#9CA3AF", linestyle="--", label="Line of Perfect Equality")

# Highlight 20% mark
idx_20 = int(len(sorted_clv) * 0.20)
val_at_20 = cum_clv[idx_20] * 100
plt.scatter([20], [val_at_20], color="#DC2626", s=80, zorder=5)
plt.annotate(
    f"Top 20% generate {val_at_20:.1f}% of total CLV",
    xy=(20, val_at_20),
    xytext=(32, val_at_20 - 8),
    arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
    fontsize=11,
    fontweight="bold"
)

plt.title("Pareto Concentration of Customer Lifetime Value", fontsize=14, fontweight="bold")
plt.xlabel("Cumulative % of Subscribers (Ranked by CLV)")
plt.ylabel("Cumulative % of 12-Month Predicted CLV")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
