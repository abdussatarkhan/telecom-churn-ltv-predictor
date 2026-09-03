"""
Strategic Retention Engine & Financial ROI Simulator
Cross-tabulates subscriber Churn Risk against Customer Lifetime Value (CLV) tiers.
Designs targeted retention offers and simulates campaign financial return on investment (ROI).
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
    ensure_directory
)

logger = setup_logger("retention_strategy")

class RetentionStrategist:
    """
    Formulates data-driven retention campaigns and conducts portfolio-level ROI simulations.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.features_file = self.config["paths"]["customer_features_file"]
        self.matrix_output_file = self.config["paths"]["retention_matrix_file"]
        self.reports_dir = ensure_directory(self.config["paths"]["reports_dir"])
        self.tier_params = self.config.get("retention_strategy", {}).get("tiers", {})

    def build_cross_tabulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates 4x4 decision grid cross-tabulating Churn Risk Tier against CLV Tier.
        """
        logger.info("Constructing Churn Risk x CLV Segment cross-tabulation...")
        if "churn_risk_tier" not in df.columns or "clv_segment" not in df.columns:
            raise ValueError("Required segmentation columns ('churn_risk_tier', 'clv_segment') not present in data.")

        # Reorder categories logically
        risk_order = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
        clv_order = [
            "Platinum VIP (Top 5%)",
            "Gold High Value (Top 20%)",
            "Silver Core Value (Middle 30%)",
            "Bronze Basic (Bottom 50%)"
        ]

        df_filtered = df[df["churn_risk_tier"].isin(risk_order) & df["clv_segment"].isin(clv_order)]
        crosstab_count = pd.crosstab(
            df_filtered["churn_risk_tier"],
            df_filtered["clv_segment"]
        ).reindex(index=risk_order, columns=clv_order, fill_value=0)

        logger.info(f"Cross-tabulation (Subscriber counts):\n{crosstab_count}")
        return crosstab_count

    def design_retention_rules(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Defines tailored intervention actions, unit costs, and expected success rates per matrix cell.
        """
        rules = {
            ("Critical Risk", "Platinum VIP (Top 5%)"): {
                "action": "VIP Concierge Direct Call + 30% Contract Discount + Free 5G Roaming Pass",
                "cost_eur": 75.0,
                "takeup_rate": 0.75,
                "success_rate": 0.50,
                "priority": "P0 - Emergency"
            },
            ("High Risk", "Platinum VIP (Top 5%)"): {
                "action": "Dedicated Account Manager Review + Device Upgrade Voucher (€120)",
                "cost_eur": 60.0,
                "takeup_rate": 0.70,
                "success_rate": 0.45,
                "priority": "P0 - Critical"
            },
            ("Critical Risk", "Gold High Value (Top 20%)"): {
                "action": "Proactive Loyalty Bill Credit (€25) + Data Allowance Doubled for 6 Mos",
                "cost_eur": 40.0,
                "takeup_rate": 0.65,
                "success_rate": 0.42,
                "priority": "P1 - High"
            },
            ("High Risk", "Gold High Value (Top 20%)"): {
                "action": "Streaming Bundle Inclusion (DAZN/Netflix) + Speed Boost",
                "cost_eur": 30.0,
                "takeup_rate": 0.60,
                "success_rate": 0.38,
                "priority": "P1 - High"
            },
            ("Critical Risk", "Silver Core Value (Middle 30%)"): {
                "action": "Automated SMS/In-App Discount (15% for 3 Months)",
                "cost_eur": 18.0,
                "takeup_rate": 0.50,
                "success_rate": 0.30,
                "priority": "P2 - Standard"
            },
            ("High Risk", "Silver Core Value (Middle 30%)"): {
                "action": "Digital Retention Offer: Bonus 20GB Data Package",
                "cost_eur": 10.0,
                "takeup_rate": 0.45,
                "success_rate": 0.25,
                "priority": "P2 - Standard"
            },
            ("Critical Risk", "Bronze Basic (Bottom 50%)"): {
                "action": "Automated Push Notification + Tariff Optimizer Recommendation",
                "cost_eur": 3.0,
                "takeup_rate": 0.35,
                "success_rate": 0.18,
                "priority": "P3 - Automated"
            },
            ("High Risk", "Bronze Basic (Bottom 50%)"): {
                "action": "Automated Self-Service Loyalty Points Nudge",
                "cost_eur": 1.5,
                "takeup_rate": 0.25,
                "success_rate": 0.12,
                "priority": "P3 - Automated"
            },
            # Medium and Low Risk cells receive proactive engagement or organic management
            ("Medium Risk", "Platinum VIP (Top 5%)"): {
                "action": "VIP Anniversary Reward + Free International Calling Credits",
                "cost_eur": 25.0,
                "takeup_rate": 0.55,
                "success_rate": 0.25,
                "priority": "P2 - Proactive"
            },
            ("Medium Risk", "Gold High Value (Top 20%)"): {
                "action": "App Appreciation Perk (Discount partner voucher)",
                "cost_eur": 12.0,
                "takeup_rate": 0.40,
                "success_rate": 0.20,
                "priority": "P3 - Proactive"
            }
        }
        return rules

    def simulate_campaign_roi(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Simulates financial expenditures, gross saved CLV, net value preserved, and ROI percentage.
        """
        logger.info("Simulating retention campaign financial economics and ROI...")
        rules = self.design_retention_rules()
        simulation_records = []

        total_campaign_cost = 0.0
        total_clv_at_risk = 0.0
        total_clv_saved = 0.0
        total_subscribers_targeted = 0

        for (risk_tier, clv_tier), rule in rules.items():
            subset = df[(df["churn_risk_tier"] == risk_tier) & (df["clv_segment"] == clv_tier)]
            count = len(subset)
            if count == 0:
                continue

            clv_pool = subset["predicted_clv_12m_eur"].sum()
            avg_clv = subset["predicted_clv_12m_eur"].mean()

            cost_per_sub = rule["cost_eur"]
            takeup = rule["takeup_rate"]
            success = rule["success_rate"]

            campaign_cost = count * cost_per_sub * takeup
            saved_clv = clv_pool * success
            net_benefit = saved_clv - campaign_cost
            cell_roi = (net_benefit / campaign_cost * 100.0) if campaign_cost > 0 else 0.0

            total_subscribers_targeted += count
            total_campaign_cost += campaign_cost
            total_clv_at_risk += clv_pool
            total_clv_saved += saved_clv

            simulation_records.append({
                "churn_risk_tier": risk_tier,
                "clv_segment": clv_tier,
                "targeted_subscribers": count,
                "priority": rule["priority"],
                "prescribed_intervention": rule["action"],
                "unit_cost_eur": cost_per_sub,
                "campaign_cost_eur": round(campaign_cost, 2),
                "gross_clv_at_risk_eur": round(clv_pool, 2),
                "saved_clv_eur": round(saved_clv, 2),
                "net_profit_benefit_eur": round(net_benefit, 2),
                "roi_percentage": round(cell_roi, 1)
            })

        df_sim = pd.DataFrame(simulation_records)
        df_sim.sort_values(by="net_profit_benefit_eur", ascending=False, inplace=True)

        overall_net_benefit = total_clv_saved - total_campaign_cost
        overall_roi = (overall_net_benefit / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0

        portfolio_summary = {
            "total_targeted_subscribers": total_subscribers_targeted,
            "total_campaign_cost_eur": round(total_campaign_cost, 2),
            "total_clv_at_risk_eur": round(total_clv_at_risk, 2),
            "total_clv_saved_eur": round(total_clv_saved, 2),
            "net_financial_benefit_eur": round(overall_net_benefit, 2),
            "portfolio_roi_percentage": round(overall_roi, 1)
        }

        return df_sim, portfolio_summary

    def run(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Executes retention strategy matrix formulation and simulation exports.
        """
        if not os.path.exists(self.features_file):
            from scripts.churn_prediction import ChurnPredictor
            predictor = ChurnPredictor()
            predictor.run()

        df = load_parquet(self.features_file)
        crosstab = self.build_cross_tabulation(df)
        df_sim, portfolio_summary = self.simulate_campaign_roi(df)

        # Export matrix
        df_sim.to_csv(self.matrix_output_file, index=False)
        logger.info(f"Saved retention strategy simulation to {self.matrix_output_file}")

        logger.info(f"""
===========================================================
RECURRING RETENTION CAMPAIGN ROI SUMMARY
===========================================================
Targeted Subscribers:       {portfolio_summary['total_targeted_subscribers']:,}
Total Campaign Investment:  EUR {portfolio_summary['total_campaign_cost_eur']:,.2f}
Total 12M CLV at Risk:      EUR {portfolio_summary['total_clv_at_risk_eur']:,.2f}
Total Preserved 12M CLV:    EUR {portfolio_summary['total_clv_saved_eur']:,.2f}
Net Preserved Business Val: EUR {portfolio_summary['net_financial_benefit_eur']:,.2f}
Return on Investment (ROI): {portfolio_summary['portfolio_roi_percentage']:.1f}%
===========================================================
        """)

        return df_sim, portfolio_summary

def main():
    strategist = RetentionStrategist()
    df_sim, summary = strategist.run()
    print("\n--- Top Campaign Allocations by Net Financial Benefit ---")
    print(df_sim[["churn_risk_tier", "clv_segment", "campaign_cost_eur", "saved_clv_eur", "net_profit_benefit_eur", "roi_percentage"]].head(6).to_string(index=False))

if __name__ == "__main__":
    main()
