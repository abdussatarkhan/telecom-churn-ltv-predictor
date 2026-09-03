-- ==============================================================================
-- Telecom Churn & LTV Predictor - Production Google BigQuery SQL Suite
-- Domain: Telecommunications (Telecom Italia CDR + ISTAT Demographics)
-- Author: Telecom Data Science & Engineering
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Daily CDR Event Aggregation & Traffic Metrics
-- Summarizes 10-minute micro-intervals into daily subscriber activity metrics.
-- Partitioned by date and clustered by square_id for optimal query performance.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE TABLE `telecom-analytics-prod.telecom_italia_cdr.customer_daily_activity`
PARTITION BY activity_date
CLUSTER BY primary_square_id
AS
WITH parsed_events AS (
  SELECT
    customer_id,
    square_id AS primary_square_id,
    country_code,
    TIMESTAMP_MILLIS(time_interval) AS event_timestamp,
    DATE(TIMESTAMP_MILLIS(time_interval)) AS activity_date,
    EXTRACT(HOUR FROM TIMESTAMP_MILLIS(time_interval)) AS event_hour,
    EXTRACT(DAYOFWEEK FROM TIMESTAMP_MILLIS(time_interval)) AS day_of_week,
    COALESCE(call_in, 0.0) + COALESCE(call_out, 0.0) AS total_call_volume,
    COALESCE(sms_in, 0.0) + COALESCE(sms_out, 0.0) AS total_sms_volume,
    COALESCE(internet_traffic, 0.0) AS internet_traffic_mb
  FROM
    `telecom-analytics-prod.telecom_italia_cdr.raw_cdr_stream`
  WHERE
    time_interval >= 1383264000000 -- Nov 1, 2013
)
SELECT
  customer_id,
  activity_date,
  primary_square_id,
  COUNT(1) AS total_event_records,
  SUM(total_call_volume) AS daily_calls,
  SUM(total_sms_volume) AS daily_sms,
  SUM(internet_traffic_mb) AS daily_internet_mb,
  -- Peak Hours defined as 08:00 to 20:00 local time
  SUM(CASE WHEN event_hour >= 8 AND event_hour < 20 THEN 1 ELSE 0 END) / COUNT(1) AS peak_hour_ratio,
  -- Weekend defined as Saturday (7) and Sunday (1)
  CASE WHEN day_of_week IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
  -- International activity proportion (non-Italy prefix != 39)
  SUM(CASE WHEN country_code != 39 THEN 1 ELSE 0 END) / COUNT(1) AS international_activity_ratio
FROM
  parsed_events
GROUP BY
  customer_id,
  activity_date,
  primary_square_id,
  day_of_week;

-- ------------------------------------------------------------------------------
-- 2. 90-Day Rolling Behavioral Feature Engineering
-- Aggregates daily telemetry into subscriber-level features for machine learning.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE TABLE `telecom-analytics-prod.telecom_italia_cdr.customer_behavioral_features`
AS
WITH aggregated_stats AS (
  SELECT
    customer_id,
    APPROX_TOP_COUNT(primary_square_id, 1)[OFFSET(0)].value AS dominant_square_id,
    COUNT(DISTINCT activity_date) AS active_days_count,
    ROUND(AVG(daily_calls), 2) AS avg_daily_calls,
    ROUND(AVG(daily_sms), 2) AS avg_daily_sms,
    ROUND(AVG(daily_internet_mb), 2) AS avg_daily_internet_mb,
    ROUND(AVG(peak_hour_ratio), 4) AS avg_peak_hour_ratio,
    ROUND(SUM(CASE WHEN is_weekend = 1 THEN 1 ELSE 0 END) / COUNT(1), 4) AS weekend_activity_ratio,
    ROUND(AVG(international_activity_ratio), 4) AS avg_international_ratio,
    MAX(activity_date) AS latest_activity_date,
    MIN(activity_date) AS first_activity_date
  FROM
    `telecom-analytics-prod.telecom_italia_cdr.customer_daily_activity`
  GROUP BY
    customer_id
),
mobility_entropy AS (
  SELECT
    customer_id,
    -- Shannon Entropy calculation over visited square_ids: - SUM(p * log2(p))
    ROUND(-1.0 * SUM(prob * (LOG(prob) / LOG(2))), 3) AS spatial_mobility_entropy
  FROM (
    SELECT
      customer_id,
      primary_square_id,
      COUNT(1) / SUM(COUNT(1)) OVER (PARTITION BY customer_id) AS prob
    FROM
      `telecom-analytics-prod.telecom_italia_cdr.customer_daily_activity`
    GROUP BY
      customer_id,
      primary_square_id
  )
  GROUP BY
    customer_id
)
SELECT
  a.*,
  COALESCE(m.spatial_mobility_entropy, 0.0) AS spatial_mobility_entropy,
  DATE_DIFF(DATE('2013-12-31'), a.latest_activity_date, DAY) AS recency_days,
  DATE_DIFF(a.latest_activity_date, a.first_activity_date, DAY) AS observed_tenure_days
FROM
  aggregated_stats a
LEFT JOIN
  mobility_entropy m ON a.customer_id = m.customer_id;

-- ------------------------------------------------------------------------------
-- 3. Spatial Enrichment: Joining Subscriber Grid Squares with ISTAT Demographics
-- Enriches customer behavior with neighborhood income, density, and age profile.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE TABLE `telecom-analytics-prod.telecom_italia_cdr.customer_enriched_demographics`
AS
SELECT
  c.*,
  i.median_household_income_eur,
  i.population_density_sqkm,
  i.unemployment_rate_pct,
  i.pct_age_under_25,
  i.pct_age_25_64,
  i.pct_age_over_65,
  i.tertiary_education_rate_pct,
  -- Affluence Composite Index
  ROUND(
    (i.median_household_income_eur / 30000.0) +
    (i.tertiary_education_rate_pct / 30.0) -
    (i.unemployment_rate_pct / 8.0),
    3
  ) AS neighborhood_affluence_index
FROM
  `telecom-analytics-prod.telecom_italia_cdr.customer_behavioral_features` c
LEFT JOIN
  `telecom-analytics-prod.telecom_italia_cdr.istat_census_demographics` i
ON
  c.dominant_square_id = i.square_id;

-- ------------------------------------------------------------------------------
-- 4. BigQuery RFM Scoring & Quantile Lifecycle Segmentation
-- Uses NTILE(5) window functions to assign standard 1 to 5 scores for R, F, and M.
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW `telecom-analytics-prod.telecom_italia_cdr.view_customer_rfm_segments`
AS
WITH ranked_rfm AS (
  SELECT
    customer_id,
    recency_days,
    active_days_count AS frequency_active_days,
    -- Estimated monthly spend derived from active tariff and data usage
    ROUND(18.50 + (avg_daily_internet_mb * 0.035) + (avg_daily_calls * 0.45), 2) AS monthly_spend_eur,
    -- Recency: 5 is most recent (lowest days), 1 is least recent (highest days)
    NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score_inverted,
    -- Frequency: 5 is highest activity
    NTILE(5) OVER (ORDER BY active_days_count ASC) AS f_score,
    -- Monetary: 5 is highest spend
    NTILE(5) OVER (ORDER BY (18.50 + (avg_daily_internet_mb * 0.035) + (avg_daily_calls * 0.45)) ASC) AS m_score
  FROM
    `telecom-analytics-prod.telecom_italia_cdr.customer_enriched_demographics`
),
scored_rfm AS (
  SELECT
    customer_id,
    recency_days,
    frequency_active_days,
    monthly_spend_eur,
    (6 - r_score_inverted) AS r_score, -- Invert so 5 = lowest recency_days
    f_score,
    m_score,
    CONCAT(CAST(6 - r_score_inverted AS STRING), CAST(f_score AS STRING), CAST(m_score AS STRING)) AS rfm_code
  FROM
    ranked_rfm
)
SELECT
  *,
  CASE
    WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
    WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal High Value'
    WHEN r_score >= 4 AND f_score <= 3 THEN 'Promising Recent'
    WHEN r_score >= 3 AND f_score <= 2 THEN 'Potential Loyalists'
    WHEN r_score == 3 AND f_score >= 3 THEN 'Needs Attention'
    WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk High Value'
    WHEN r_score <= 2 AND f_score >= 2 THEN 'At Risk Standard'
    WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3 THEN 'Hibernating High Spender'
    ELSE 'Lost / Dormant'
  END AS rfm_lifecycle_segment
FROM
  scored_rfm;

-- ------------------------------------------------------------------------------
-- 5. Churn Extraction for High-CLV Retention Campaigns (CRM Feed)
-- Exports top-priority subscribers exhibiting critical attrition indicators for automated outreach.
-- ------------------------------------------------------------------------------
SELECT
  c.customer_id,
  c.churn_probability,
  c.churn_risk_tier,
  clv.predicted_clv_12m_eur,
  clv.clv_segment,
  rfm.rfm_lifecycle_segment,
  c.avg_daily_internet_mb,
  c.dominant_square_id,
  -- Action recommendation logic
  CASE
    WHEN clv.clv_segment = 'Platinum VIP (Top 5%)' AND c.churn_risk_tier IN ('Critical Risk', 'High Risk')
      THEN 'Priority White Glove Outreach - Account Concierge + 25% Discount Voucher'
    WHEN clv.clv_segment = 'Gold High Value (Top 20%)' AND c.churn_risk_tier IN ('Critical Risk', 'High Risk')
      THEN 'Direct Retention SMS Offer - Free 5G Roaming Pass + Bill Credit €25'
    WHEN clv.clv_segment = 'Silver Core Value (Middle 30%)' AND c.churn_risk_tier IN ('Critical Risk', 'High Risk')
      THEN 'Automated Digital Nudge - App 15% Discount Voucher'
    ELSE 'Standard Lifecycle Newsletter / Low-Cost Push'
  END AS recommended_intervention,
  ROUND(clv.predicted_clv_12m_eur * c.churn_probability, 2) AS expected_monetary_loss_eur
FROM
  `telecom-analytics-prod.telecom_italia_cdr.customer_churn_scores` c
JOIN
  `telecom-analytics-prod.telecom_italia_cdr.customer_clv_forecasts` clv ON c.customer_id = clv.customer_id
JOIN
  `telecom-analytics-prod.telecom_italia_cdr.view_customer_rfm_segments` rfm ON c.customer_id = rfm.customer_id
WHERE
  c.churn_risk_tier IN ('Critical Risk', 'High Risk')
ORDER BY
  expected_monetary_loss_eur DESC
LIMIT 5000;
