# Medicaid Clinical Transformer Data Preparation Pipeline for IP classification

This SQL pipeline prepares raw Medicaid claims data into a format suitable for training a **clinical transformer model** that performs **next-day prediction** of medical events. Let me walk you through each step.

---

## 🎯 High-Level Overview

The pipeline transforms raw claims data into **temporal sequences** with two vocabularies:

| Vocabulary | Purpose | Size | Granularity |
|------------|---------|------|-------------|
| **Input (cd)** | What the model sees | ~84k codes | Detailed (e.g., ICD `E11.65`) |
| **Target** | What the model predicts | ~5k codes | Grouped (e.g., ICD `E11`) |

**Training Task**: Given codes on day N → Predict codes on day N+1

---

## Step-by-Step Breakdown

### **STEP 1: Create Membership Base** (`a834793_Medicaid`)

```156:184:data_ingestion/new_ingestion/medicaid_data_prep.sql
DROP TABLE IF EXISTS `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`;
CREATE OR REPLACE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`
// ... options ...
AS
  SELECT
     asdb_member_key AS individual_id
     , ARRAY_AGG(CAST(asdb_elig_dt AS DATE) ORDER BY RAND() LIMIT 1)[SAFE_OFFSET(0)] AS index_dt
FROM 
     `edp-prod-hcbstorage.edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH`
WHERE 
    CAST(asdb_elig_dt AS DATE) BETWEEN PARSE_DATE("%Y%m%d", CAST("20230101" AS STRING)) 
                                   AND PARSE_DATE("%Y%m%d", CAST("20231231" AS STRING))
GROUP BY asdb_member_key
```

**What it does:**
- Identifies eligible Medicaid members from **2023 calendar year**
- Selects **ONE random eligibility month** per member as `index_dt` (prediction point)
- **Why random?** Prevents temporal bias and creates diverse training examples
- Output: `individual_id` + `index_dt` (one row per member)

---

### **STEP 2: Create Member Training Table** (`a834793_Medicaid_member_train_ending`)

```207:232:data_ingestion/new_ingestion/medicaid_data_prep.sql
CREATE TABLE `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_member_train_ending`
AS
    SELECT 
        individual_id
        , index_dt 
        , individual_id AS member_id
    FROM 
    `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid`
    GROUP BY individual_id, index_dt
```

**What it does:**
- Creates a clean reference table for joins
- Adds `member_id` alias for compatibility with claims tables
- Ensures one row per member (deduplication safety)

---

### **STEP 3: Extract Claims & Procedures** (`a834793_Medicaid_d1a_train_ending`)

This is the **core feature extraction step** with multiple data enrichments:

#### 3a. Base Claims Filtering (CTE: `clm`)

```271:309:data_ingestion/new_ingestion/medicaid_data_prep.sql
WITH clm AS (
    SELECT
        base.individual_id, base.index_dt, base.member_id
        , clm.claimid, clm.asdb_incurred_dt, clm.ip_paid_days_ct
        , clm.revcode, clm.location, clm.servcode
        , clm.asdb_svc_prov_key, clm.asdb_coe_id_dev
    FROM 
        `a834793_Medicaid_member_train_ending` AS base 
    LEFT JOIN
        `ASDB_CLM_DATA_STAGE` AS clm ON base.member_id = clm.asdb_member_key
    WHERE 
        clm.final_claim = 1                           -- Only final adjudicated
        AND TRIM(UPPER(clm.status_header)) = "PAID"   -- Paid claims only
        AND TRIM(UPPER(clm.status_detail)) NOT IN ("DENY", "DENIED")
        
        -- 36-month lookback with 90-day claims lag
        AND CAST(clm.asdb_incurred_dt AS DATE) > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)
        AND CAST(clm.asdb_incurred_dt AS DATE) < DATE_SUB(base.index_dt, INTERVAL 90 DAY)
)
```

**Key filtering logic:**
- **36-month lookback** from `index_dt`
- **90-day claims lag** to ensure claims are finalized before training
- Only **final, paid, non-denied** claims for quality

#### 3b. Feature Extraction (Main SELECT)

The main SELECT extracts and transforms **9 code types**:

| Field | Description | Example |
|-------|-------------|---------|
| `days_cnt` | Inpatient LOS (capped 0-11, 99=unknown) | `5` |
| `gender_cd` | 0=F, 1=M, 2=Other | `1` |
| `age_in_months` | Age at service date (capped 0-1439) | `456` |
| `revenue_cd` | UB-04 revenue code | `0250` |
| `hcfa_plc_srv_cd` | Place of service (derived if missing) | `21` (IP Hospital) |
| `provider_taxonomy_cd` | NPPES provider taxonomy | `207Q00000X` |
| `prcdr_cd` | CPT/HCPCS procedure | `99213` |
| `icd9_prcdr_cd` | ICD procedure code | `0RJD4ZZ` |
| `drg_cd` | DRG code | `470` |
| `prcdr_group_cd` | **NEW: Algorithmic procedure group** | `prcdr_group_992` |

#### 3c. Place of Service Derivation

When `location` is missing, derives POS from category of care:

```353:401:data_ingestion/new_ingestion/medicaid_data_prep.sql
, CASE WHEN TRIM(clm.location) IS NOT NULL ... THEN CAST(clm.location AS STRING)
    -- 11 = Office (Outpatient non-hospital ambulatory)
    WHEN coe.asdb_coe_general_type = "Outpatient" AND coe.asdb_coe_sub_cat = "Non Hospital"
        THEN CAST(11 AS STRING)
    -- 21 = Inpatient Hospital
    WHEN coe.asdb_coe_general_type = "Inpatient" AND coe.asdb_coe_sub_cat = "Hospital"
        THEN CAST(21 AS STRING)
    -- 23 = Emergency Room - Hospital
    WHEN coe.emis_cat = "Emergency" THEN CAST(23 AS STRING)
    ...
```

#### 3d. Algorithmic Procedure Grouping (NEW)

Creates unified procedure groups for **target vocabulary**:

```445:498:data_ingestion/new_ingestion/medicaid_data_prep.sql
, COALESCE(
    -- CPT/HCPCS procedure group
    CASE
      WHEN REGEXP_CONTAINS(clm.servcode, r'^\d{5}$')        -- 99213 → prcdr_group_992
          THEN CONCAT('prcdr_group_', SUBSTR(clm.servcode, 1, 3))
      WHEN REGEXP_CONTAINS(clm.servcode, r'^[A-Z]\d{4}$')   -- J3490 → prcdr_group_j3
          THEN CONCAT('prcdr_group_', LOWER(SUBSTR(clm.servcode, 1, 2)))
      WHEN REGEXP_CONTAINS(clm.servcode, r'^D\d{4}$')       -- D1110 → prcdr_group_d11
          THEN CONCAT('prcdr_group_', LOWER(SUBSTR(clm.servcode, 1, 3)))
      ...
    END,
    -- ICD procedure group (fallback)
    CASE
      WHEN REGEXP_CONTAINS(icd.icdpx1, r'^[0-9A-Z]{6}$')    -- 0RJD4Z → prcdr_group_0rj
          THEN CONCAT('prcdr_group_', LOWER(SUBSTR(icd.icdpx1, 1, 3)))
      ...
    END
  ) AS prcdr_group_cd
```

**Grouping logic:**
| Code Type | Example | Group |
|-----------|---------|-------|
| 5-digit CPT | `99213` | `prcdr_group_992` |
| HCPCS J-codes | `J3490` | `prcdr_group_j3` |
| Dental | `D1110` | `prcdr_group_d11` |
| ICD-10-PCS | `0RJD4ZZ` | `prcdr_group_0rj` |

#### 3e. Data Source JOINs

7 tables are joined to enrich claims:

```500:556:data_ingestion/new_ingestion/medicaid_data_prep.sql
LEFT JOIN ASDB_MEMBER AS member        -- Demographics (gender, DOB)
LEFT JOIN ASDB_SVC_PROV AS prv         -- Provider info (NPI)
LEFT JOIN ASDB_TYPE_OF_SERVICE AS coe  -- Place of service derivation
LEFT JOIN a834793_provider_db_x_walk   -- Provider specialty crosswalk
LEFT JOIN bigquery-public-data.nppes.npi_raw AS nppes  -- Provider taxonomy
LEFT JOIN ASDB_CLAIMICDPROCSUMMARY AS icd  -- ICD procedure codes
LEFT JOIN ASDB_DRG AS drg              -- DRG codes
```

---

### **STEP 4: Extract Diagnosis Codes** (`a834793_Medicaid_d1b_train_ending`)

```573:741:data_ingestion/new_ingestion/medicaid_data_prep.sql
WITH 
base AS (...),
p AS (
    -- Primary diagnosis: split on '.' to standardize format
    SELECT ..., SPLIT(TRIM(b.icddxpri), '.')[offset(0)] AS x_0
            , SPLIT(TRIM(b.icddxpri), '.')[safe_offset(1)] AS x_1
    FROM base INNER JOIN ASDB_CLAIMDIAGSUMMARY AS b ...
),
s1 AS (...),  -- Secondary diagnosis 1
s2 AS (...),  -- Secondary diagnosis 2  
s3 AS (...),  -- Secondary diagnosis 3
x1 AS (
    SELECT * FROM p UNION DISTINCT SELECT * FROM s1 UNION DISTINCT ...
)
SELECT ...
    , CASE WHEN REGEXP_CONTAINS(code, r'^[A-Z][0-9A-Z]{2}[\.\w]*$')
           THEN UPPER(code)  -- Valid ICD-10: E11.65, Z3A.39
           ELSE NULL         -- Filter legacy ICD-9: 250.00
      END AS icd9_dx_cd
FROM x1
```

**What it does:**
- Extracts **primary + 3 secondary** diagnosis codes per claim
- Standardizes format: `XXX.XX` (e.g., `E11.65`)
- Validates ICD-10 format (filters legacy ICD-9 codes like `250.00`)
- Creates one row per diagnosis per claim

---

### **STEP 5: Extract Medications** (`a834793_Medicaid_d1c_train_ending`)

```759:813:data_ingestion/new_ingestion/medicaid_data_prep.sql
SELECT 
    base.individual_id
    , CASE WHEN LENGTH(TRIM(rx.adjudicated_gpi_cd)) >= 4 
           THEN CONCAT('gpi', SUBSTR(TRIM(rx.adjudicated_gpi_cd), 1, 4))
           ELSE NULL END AS gpi4  -- e.g., 'gpi2210' (Statins)
FROM a834793_Medicaid_member_train_ending AS base 
INNER JOIN ASDB_RX_DATA_STAGE AS rx ON base.member_id = rx.asdb_member_key
WHERE 
    rx.disp_dt > DATE_SUB(base.index_dt, INTERVAL 36 MONTH)
    AND rx.disp_dt < base.index_dt
```

**What it does:**
- Extracts **GPI-4** codes (first 4 digits of Generic Product Identifier)
- GPI-4 identifies drug therapeutic class (e.g., `gpi2210` = Statins)
- Same 36-month lookback as medical claims
- Format includes 'gpi' prefix: `gpi2210`

---

### **STEP 6: Map Input Codes to Indices** (`a834793_Medicaid_o1_train_ending`)

```830:1128:data_ingestion/new_ingestion/medicaid_data_prep.sql
WITH 
root0 AS (
    SELECT individual_id, dt, gender_cd, age_in_months FROM d1a_train_ending
),
x0 AS (
    -- Map 1: Days count → index
    SELECT base.individual_id, base.dt, w2ind.ind
    FROM d1a LEFT JOIN a834793_member_w2ind AS w2ind
        ON CONCAT('days_cnt', CAST(days_cnt AS STRING)) = w2ind.cd
    
    UNION DISTINCT
    -- Map 2: Place of service → index
    SELECT ... ON CONCAT('hcfa_plc_srv_cd', hcfa_plc_srv_cd) = w2ind.cd
    
    UNION DISTINCT
    -- Map 3: Provider taxonomy → index
    SELECT ... ON CONCAT('provider_taxonomy_cd', provider_taxonomy_cd) = w2ind.cd
    
    UNION DISTINCT
    -- ... 6 more code types ...
    
    UNION DISTINCT
    -- Map 9: GPI medications → index
    SELECT ... FROM d1c ON CAST(gpi4 AS STRING) = w2ind.cd
),
x3 AS (
    -- Aggregate to comma-separated string (max 80 per day)
    SELECT individual_id, dt, STRING_AGG(CAST(ind AS STRING), ',') AS cd
    FROM x2 WHERE seqno <= 80
    GROUP BY individual_id, dt
)
SELECT root2.*, x3.cd FROM root2 INNER JOIN x3 ...
```

**What it does:**
1. **Maps 9 code types** to numeric indices using `w2ind` lookup (~84k codes)
2. **UNIONs** all code types into single stream
3. **Deduplicates** indices per member-day
4. **Aggregates** into comma-separated string (max 80 codes/day)
5. **Output**: One row per member per day with `cd = "15,42,103,7,..."`

**The 9 input code types:**
1. Days count (LOS)
2. Place of service
3. Provider taxonomy
4. ICD diagnosis
5. Revenue codes
6. CPT/HCPCS procedures
7. ICD procedures
8. DRG codes
9. GPI-4 medications

---

### **STEP 6b: Map Target Codes to Indices** (`a834793_Medicaid_o1_train_ending_target`)

```1153:1284:data_ingestion/new_ingestion/medicaid_data_prep.sql
-- Target Type 3: ICD Diagnosis (First 3 digits only)
SELECT ... ON CONCAT('icd9_dx_cd', SPLIT(icd9_dx_cd, '.')[SAFE_OFFSET(0)]) = w2ind.cd
-- Example: E11.65 → E11 (all Type 2 Diabetes)

-- Target Type 4: GPI Medications (First 2 digits only)
SELECT ... ON CONCAT('gpi', SUBSTR(REPLACE(gpi4, 'gpi', ''), 1, 2)) = w2ind.cd
-- Example: gpi2210 → gpi22 (all Insulins)

-- Target Type 5: Revenue Code (First 3 digits)
SELECT ... ON CONCAT('revenue_cd', SUBSTR(revenue_cd, 1, 3)) = w2ind.cd
-- Example: 0250 → 025 (all Pharmacy)
```

**What it does:**
- Maps **grouped codes** to target vocabulary (~5k codes)
- Uses `w2ind_target` lookup table (smaller vocabulary)

**Target grouping strategy:**

| Code Type | Input (cd) | Target | Example |
|-----------|------------|--------|---------|
| ICD Diagnosis | Full code | First 3 chars | `E11.65` → `E11` |
| GPI Medication | 4 digits | 2 digits | `gpi2210` → `gpi22` |
| Revenue Code | 4 digits | 3 digits | `0250` → `025` |
| Procedure | Individual | Group code | `99213` → `prcdr_group_992` |
| Provider Taxonomy | Full 10-char | First 4 chars | `207Q00000X` → `207Q` |
| DRG/POS/Days | Keep as-is | Keep as-is | `470` → `470` |

---

### **STEP 7: Create Temporal Sequences** (`a834793_Medicaid_o3_train_ending`)

```1303:1438:data_ingestion/new_ingestion/medicaid_data_prep.sql
WITH 
x1 AS (
    -- Assign sequence numbers (most recent 200 days)
    SELECT *, ROW_NUMBER() OVER (PARTITION BY individual_id ORDER BY dt DESC) AS seqno
    FROM o1_train_ending
),
x2 AS (SELECT * FROM x1 WHERE seqno <= 200),

y1 AS (...),  -- Same for target codes
y2 AS (...),

z1 AS (
    -- Join input and target by individual+date
    SELECT x3.*, y3.target FROM x3 LEFT JOIN y3 ON x3.individual_id = y3.individual_id AND x3.dt = y3.dt
),
z2 AS (
    -- Apply LEAD for next-day prediction
    SELECT ..., LEAD(target, 1) OVER (PARTITION BY individual_id ORDER BY dt ASC) AS target_next_day
    FROM z1
),
z5 AS (
    -- Aggregate into asterisk-separated sequences
    SELECT 
        individual_id
        , STRING_AGG(CAST(gender_cd AS STRING), '*' ORDER BY seqno) AS gender_cd
        , STRING_AGG(CAST(cd AS STRING), '*' ORDER BY seqno) AS cd
        , STRING_AGG(CAST(target AS STRING), '*' ORDER BY seqno) AS target
        , COUNT(*) AS dt_cnt
    FROM z4 GROUP BY individual_id
)
```

**What it does:**
1. Limits to **most recent 200 days** per member
2. **JOINs** input codes (cd) with target codes
3. **Applies LEAD** to shift targets by 1 day (next-day prediction)
4. **Aggregates** days into asterisk-separated sequences
5. **Filters** out last day (no next-day target available)

---

## Final Output Format

**Table: `a834793_Medicaid_o3_train_ending`** (ONE ROW PER MEMBER)

| Column | Format | Example |
|--------|--------|---------|
| `individual_id` | Integer | `123456789` |
| `index_dt` | Date | `2023-07-15` |
| `gender_cd` | `*`-separated | `1*1*1*0*0*...` |
| `age_in_months` | `*`-separated | `456*456*457*457*...` |
| `cd` | `*`-separated days, `,`-separated codes | `15,42,103*7,88*12,55,67*...` |
| `target` | `*`-separated days, `,`-separated targets | `101,23*67,45*23*...` |
| `dt_cnt` | Integer | `156` |

---

## Visual Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAW MEDICAID CLAIMS DATA                        │
│  ASDB_CLM_DATA_STAGE, ASDB_RX_DATA_STAGE, ASDB_MEMBER, etc.        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1-2: MEMBERSHIP BASE                                          │
│  • Select 2023 members                                               │
│  • ONE random index_dt per member                                    │
│  Output: individual_id, index_dt                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
            │   STEP 3      │ │   STEP 4      │ │   STEP 5      │
            │   d1a table   │ │   d1b table   │ │   d1c table   │
            │ ───────────── │ │ ───────────── │ │ ───────────── │
            │ • Procedures  │ │ • ICD Dx      │ │ • GPI-4       │
            │ • Revenue     │ │   codes       │ │   medications │
            │ • POS         │ │               │ │               │
            │ • Provider    │ │               │ │               │
            │ • DRG         │ │               │ │               │
            │ • Prcdr Group │ │               │ │               │
            └───────────────┘ └───────────────┘ └───────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: MAP TO INPUT INDICES (o1 table) - ~84k vocabulary          │
│  • 9 code types → UNION DISTINCT                                     │
│  • Map to w2ind indices                                              │
│  • Aggregate: 80 codes/day, comma-separated                          │
│  Output: individual_id, dt, cd="15,42,103,7,88,..."                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6b: MAP TO TARGET INDICES (o1_target table) - ~5k vocabulary  │
│  • 8 grouped code types                                              │
│  • Map to w2ind_target indices                                       │
│  Output: individual_id, dt, target="101,23,67,..."                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: CREATE TEMPORAL SEQUENCES (o3 table)                       │
│  • Limit to 200 most recent days                                     │
│  • Join input + target by date                                       │
│  • Apply LEAD for next-day prediction                                │
│  • Aggregate into "*"-separated sequences                            │
│                                                                      │
│  Output: ONE ROW PER MEMBER                                          │
│  cd = "15,42*7,88*12,55*..."        (what model sees)               │
│  target = "101*67*23*..."            (what model predicts)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Differences from Earlier IP Model Pipeline

| Aspect | IP Model (dev_archive) | Transformer Pipeline |
|--------|------------------------|---------------------|
| **Purpose** | Tabular features for ML | Sequence data for transformer |
| **Output** | 300+ aggregated features | Temporal code sequences |
| **Granularity** | Member-level summary | Day-level sequences |
| **Lookback** | 12-24 months | 36 months |
| **Code format** | Flags & counts | Index sequences |
| **Target** | Binary IP flag | Next-day codes |