-- Calculate the proportion of members in each LOB
-- WITH lob_counts AS (
--     SELECT 
--         lob,
--         COUNT(DISTINCT individual_id) AS member_count
--     FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
--     GROUP BY lob
-- ),
-- total_count AS (
--     SELECT SUM(member_count) AS total_members
--     FROM lob_counts
-- )
-- SELECT 
--     lc.lob,
--     lc.member_count,
--     tc.total_members,
--     ROUND(lc.member_count * 100.0 / tc.total_members, 2) AS proportion_pct
-- FROM lob_counts lc
-- CROSS JOIN total_count tc
-- ORDER BY lc.lob;

-- [{
--   "lob": "Commercial",
--   "member_count": "9705678",
--   "total_members": "15387635",
--   "proportion_pct": "63.07"
-- }, {
--   "lob": "Medicaid",
--   "member_count": "2323073",
--   "total_members": "15387635",
--   "proportion_pct": "15.1"
-- }, {
--   "lob": "Medicare",
--   "member_count": "3358884",
--   "total_members": "15387635",
--   "proportion_pct": "21.83"
-- }]

-- Round 5 Pretrain Data Prep: 30% Proportional Stratified Sampling by LOB
-- Source: edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending
-- Reproducible random sampling using FARM_FINGERPRINT with seed
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample` AS
(

WITH lob_stats AS (
    SELECT 
        lob,
        COUNT(DISTINCT individual_id) AS lob_count,
        SUM(COUNT(DISTINCT individual_id)) OVER () AS total_count
    FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
    WHERE dt_cnt >= 10
    GROUP BY lob
),
sample_sizes AS (
    SELECT 
        lob,
        lob_count,
        total_count,
        ROUND(lob_count * 1.0 / total_count, 4) AS proportion,
        -- 20% sampling per LOB to maintain proportions, 10% of the total members 
        CAST(ROUND(lob_count * 0.2) AS INT64) AS sample_size_per_lob
    FROM lob_stats
),
ranked_members AS (
    SELECT 
        individual_id,
        lob,
        -- Reproducible random ranking using FARM_FINGERPRINT with seed 42
        ROW_NUMBER() OVER (
            PARTITION BY lob 
            ORDER BY FARM_FINGERPRINT(CONCAT(CAST(individual_id AS STRING), '_seed_42'))
        ) AS rn
    FROM (
        SELECT DISTINCT individual_id, lob 
        FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending`
        WHERE dt_cnt >= 10
    ) distinct_members
),
sampled_member_ids AS (
    SELECT rm.individual_id, rm.lob
    FROM ranked_members rm
    INNER JOIN sample_sizes ss ON rm.lob = ss.lob
    WHERE rm.rn <= ss.sample_size_per_lob
)
-- Final output: Full table data for sampled members
SELECT 
    t.individual_id,
    t.lob,
    t.index_dt,
    t.gender_cd,
    t.age_in_months,
    t.cd,
    t.target,
    t.dt_cnt
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_ending` t
INNER JOIN sampled_member_ids sm 
    ON t.individual_id = sm.individual_id 
    AND t.lob = sm.lob
WHERE t.dt_cnt >= 10

)
-- 10% entire population
-- 1	Commercial 1067814	
-- 2	Medicare 569412	
-- 3	Medicaid 127372	

-- Final output: sampled member IDs
-- SELECT lob, COUNT(*) as sampled_count FROM sampled_members GROUP BY lob;


-- if dt_cnt >= 10; CAST(ROUND(lob_count * 0.4) AS INT64) AS sample_size_per_lob
-- Row	lob	lob_count	total_count	proportion	sample_size_per_lob
-- 1	Medicaid	636860	8822989	0.0722	254744
-- 2	Commercial	5339069	8822989	0.6051	2135628
-- 3	Medicare	2847060	8822989	0.3227	1138824

-- if dt_cnt >= 5; CAST(ROUND(lob_count * 0.2) AS INT64) AS sample_size_per_lob
-- Row	lob	lob_count	total_count	proportion	sample_size_per_lob
-- 1	Medicare	3094914	11589474	0.267	618983
-- 2	Medicaid	1406877	11589474	0.1214	281375
-- 3	Commercial	7087683	11589474	0.6116	1417537

-- if dt_cnt >= 5; CAST(ROUND(lob_count * 0.3) AS INT64) AS sample_size_per_lob
-- use these to create edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_20pct_sample
-- Row	lob	lob_count	total_count	proportion	sample_size_per_lob
-- 1	Medicare	3094914	11589474	0.267	928474
-- 2	Medicaid	1406877	11589474	0.1214	422063
-- 3	Commercial	7087683	11589474	0.6116	2126305


-- WITH sample_stats AS (
--   SELECT
--     individual_id,
--     day_str,
--     day_idx,
--     -- Total tokens (including repeats)
--     ARRAY_LENGTH(SPLIT(day_str, ',')) AS total_tokens,
--     -- Unique codes per day
--     (SELECT COUNT(DISTINCT code) 
--      FROM UNNEST(SPLIT(day_str, ',')) AS code 
--      WHERE code != '') AS unique_codes
--   FROM (
--     SELECT 
--       individual_id,
--       day_str,
--       day_idx
--     FROM edp-prod-storage.edp_ent_sdoheir_cns.a834793_Combined_All_LOB_o3_train_10pct_sample,
--     UNNEST(SPLIT(cd, '*')) AS day_str WITH OFFSET AS day_idx
--     WHERE day_str != ''
--   )
-- )

-- SELECT
--   -- Per-day statistics (this is what matters for predictions)
--   AVG(unique_codes) AS avg_unique_codes_per_day,
--   STDDEV(unique_codes) AS std_unique_codes_per_day,
--   APPROX_QUANTILES(unique_codes, 100)[OFFSET(50)] AS median_unique_codes_per_day,
--   MIN(unique_codes) AS min_unique_codes,
--   MAX(unique_codes) AS max_unique_codes,
  
--   -- Token statistics (for sequence length)
--   AVG(total_tokens) AS avg_tokens_per_day,
--   STDDEV(total_tokens) AS std_tokens_per_day,
  
--   -- Total samples
--   COUNT(*) AS total_day_samples
-- FROM sample_stats;