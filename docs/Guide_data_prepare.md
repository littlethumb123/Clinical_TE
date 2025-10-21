# Transformer Data Preparation Guide for Member Claims

## Introduction

This guide explains how to prepare members' daily claims data for the clinical transformer model to generate patient embeddings. The transformer model requires data in a very specific format to capture temporal patterns in patient medical histories and generate meaningful embeddings.

## Overview: What the Transformer Needs

The transformer model processes patient medical histories as **time-ordered sequences** to understand patterns in healthcare utilization and outcomes. Think of it like reading a patient's medical story day by day, where each day contains various medical events (diagnoses, procedures, visits).

## Required Data Format

### Core Requirements

The transformer expects data in this exact structure:

```
Patient → Up to 200-day sequence → Up to 80 medical codes per day + demographics (gender, age)
```
**OutputExample**
```python
sample_record = {
    'individual_id': 'PATIENT_12345',
    'age_in_months': '454*454*455*...*460',  # 200 values
    'gender_cd': '0*0*0*...*0',              # 200 values  
    'cd': '**250,58667*401,450**...*',       # 200 days, * used to separate days, , to separate codes within a day
    'dt_cnt': 179
}
```

**Dimensions:**
- **Sequence Length**: Exactly 200 days
- **Codes per Day**: Maximum 80 medical codes
- **Demographics**: Age (in months) and gender for each day
- **Output**: 256-dimensional embedding vector per patient

### Why This Format?

1. **Fixed Length (200 days)**: Transformers need consistent input sizes for efficient batch processing
2. **Daily Granularity**: Captures the timing of medical events, which is crucial for understanding disease progression
3. **Multiple Codes per Day**: Patients often have multiple medical events on the same day (multiple diagnoses, procedures)
4. **Temporal Demographics**: Age changes over time, affecting medical needs and treatments

## Data Transformation Process: Medicaid Example

Take medicaid workflow as an example to walk through how to transform raw claims data into transformer-ready format. For full SQL and Py files, please refer to [here](https://github.com/cvs-legacy-source-code/medicaid-transformer-embeddings)

### Step 1: Extract Patient Population

**File**: `001_Extract_Membership_Index.sql`

```sql
-- Get eligible Medicaid members for analysis
SELECT DISTINCT 
       asdb_member_key,
       CAST(asdb_elig_dt AS date) AS index_dt,
       coa_population_category
FROM edp_hcb_mdcd_core_srcv.ASDB_ELIG_DATA_MBR_PER_MTH
WHERE CAST(asdb_elig_dt AS DATE) BETWEEN [analysis_period]
```

**Purpose**: Establishes the patient cohort and their "index dates" (the reference point for building their 200-day sequences).

### Step 2: Collect Claims Data with Temporal Context

**File**: `002_Generate_Embeddings_Features.sql`
- This SQL generates the following files
  - `medicaid_te_MEMBER_SCORE_ENDING` - Base member table
  - `medicaid_te_D1A_SCORE_ENDING` - Claims with demographics
  - `medicaid_te_D1B_SCORE_ENDING` - Diagnosis codes processing
  - `medicaid_te_O3_SCORE_ENDING` - Final transformer input format
```sql
-- Extract claims with 24-month lookback
SELECT
    base.individual_id,
    CAST(clm.asdb_incurred_dt AS DATE) AS dt,
    -- Demographics (change over time)
    CASE WHEN TRIM(member.gender) = "M" THEN 1
         WHEN TRIM(member.gender) = "F" THEN 0 
         ELSE 2 END AS gender_cd,
    DATE_DIFF(CAST(clm.asdb_incurred_dt AS DATE), 
              CAST(member.dob AS DATE), MONTH) AS age_in_months,
    -- Medical codes
    CASE WHEN TRIM(clm.revcode) = "" THEN NULL 
         ELSE CAST(TRIM(clm.revcode) AS NUMERIC) END AS revenue_cd,
    -- Additional codes: procedure codes, diagnosis codes, etc.
FROM claims_data clm
WHERE clm.dt >= DATE_SUB(base.index_dt, INTERVAL 24 MONTH)
  AND clm.dt < base.index_dt
```

**Key Transformations:**
- **Gender Encoding**: M=1, F=0, Other=2
- **Age Calculation**: Age in months at each claim date
- **Temporal Filtering**: 24-month lookback from index date
- **Code Standardization**: Clean and normalize medical codes

**Warning**: This is a large, complex SQL that may take 1-3 hours to complete.
### Step 3: Data Preprocessing Functions

**File**: `medicaid-asdb-claims-embedding_transformer_embeddings.py`

#### Function: `conv_cd(ipt)` - Medical Code Conversion

```python
def conv_cd(ipt):
    # Input: "code1,code2,code3*code4,code5*code6,code7,code8*..."
    # Each '*' separates days, each ',' separates codes within a day
    
    ipt = ipt.split('*')              # Split by days
    ipt = ipt[:len_dy]                # Take first 200 days (len_dy=200)
    ipt = ipt + (len_dy-len(ipt))*[''] # Pad to exactly 200 days
    
    # Convert each day's codes
    ipt = [dy.split(',') for dy in ipt]  # Split codes within each day
    ipt = [[int(cd) if cd!='' else 0 for cd in dy] for dy in ipt]  # Convert to integers
    ipt = [dy + (len_cd-len(dy))*[0] for dy in ipt]  # Pad each day to 80 codes
    
    return ipt  # Returns: [200 days][80 codes per day]
```

**Example Input/Output:**
```python
# Input string representing 3 days of medical codes:
input_string = "250,401,427*305,250*401,427,305,780"

# After processing:
# Day 1: [250, 401, 427, 0, 0, ..., 0]  # Padded to 80 codes
# Day 2: [305, 250, 0, 0, 0, ..., 0]    # Padded to 80 codes  
# Day 3: [401, 427, 305, 780, 0, ..., 0] # Padded to 80 codes
# ... (197 more days of all zeros)
```

#### Function: `conv_age_gender(ipt)` - Demographics Conversion

```python
def conv_age_gender(ipt):
    # Input: "age1*age2*age3*..." (age for each day)
    
    ipt = ipt.split('*')              # Split by days
    ipt = ipt[:len_dy]                # Take first 200 days
    ipt = [min(int(cd),1439) for cd in ipt]  # Cap age at 1439 months (~120 years)
    ipt = ipt + (len_dy-len(ipt))*[0] # Pad to 200 days
    
    return ipt  # Returns: [age_day1, age_day2, ..., age_day200]
```

#### Function: `prepare_tensor(batch, device)` - Final Tensor Preparation

```python
def prepare_tensor(batch, device):
    # Convert age data
    age_in_months = [conv_age_gender(ipt) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months).to(device)
    age_in_months = age_in_months.reshape(batch_size, len_dy, 1)  # [batch, 200, 1]
    
    # Convert gender data  
    gender_cd = [conv_age_gender(ipt) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd).to(device)
    gender_cd = gender_cd.reshape(batch_size, len_dy, 1)  # [batch, 200, 1]
    
    # Convert medical codes
    cd = [conv_cd(ipt) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd).to(device)  # [batch, 200, 80]
    
    # Concatenate all features
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)  # [batch, 200, 82]
    # Final shape: [batch_size, 200_days, 82_features]
    # Features: [age, gender, code1, code2, ..., code80]
    
    return dt_cnt, x
```

## Complete Data Flow Example

### Raw Medicaid Claims Data
```
Patient ID: 12345
Claims:
- 2023-01-15: Diagnosis=250.00 (Diabetes), Procedure=99213 (Office Visit)
- 2023-01-20: Diagnosis=401.9 (Hypertension), Revenue=0450 (Emergency Room)
- 2023-02-10: Diagnosis=250.00, Procedure=82947 (Glucose Test)
```

### Step 1: Temporal Sequencing
```sql
-- Create daily sequence from index date backwards
Index Date: 2023-07-15
Lookback Period: 2021-07-15 to 2023-07-15 (24 months)

Daily Records:
2023-07-14: [] (no claims)
2023-07-13: [] (no claims)
...
2023-02-10: [250, 82947] (diabetes, glucose test)
...
2023-01-20: [401, 450] (hypertension, ER visit)
2023-01-15: [250, 99213] (diabetes, office visit)
```

### Step 2: String Encoding
```python
# Encoded as strings for database storage
age_in_months: "360*360*360*361*361*..."  # Age for each day
gender_cd: "0*0*0*0*0*..."              # Female (0) for each day  
cd: "250,99213*401,450*250,82947*..."   # Medical codes by day
```

### Step 3: Tensor Conversion
```python
# Final tensor shape: [1, 200, 82]
# Day 1: [360, 0, 250, 99213, 0, 0, ..., 0]  # age=360mo, female, 2 codes + 78 zeros
# Day 2: [360, 0, 401, 450, 0, 0, ..., 0]    # age=360mo, female, 2 codes + 78 zeros
# Day 3: [360, 0, 250, 82947, 0, 0, ..., 0]  # age=360mo, female, 2 codes + 78 zeros
# ...
# Day 200: [387, 0, 0, 0, ..., 0]            # age=387mo, female, no codes
```

## User Implementation Guide

### For New Healthcare Datasets

#### 1. Data Collection Requirements

**Essential Data Elements:**
- Patient identifier
- Claim/encounter dates  
- Medical codes (diagnosis, procedure, revenue codes)
- Patient demographics (date of birth, gender)
- Lookback period: 24 months minimum

#### 2. Data Preparation Steps

**Step 1: Define Analysis Cohort**
```sql
-- Example template
SELECT DISTINCT 
    patient_id,
    analysis_date as index_dt
FROM your_patient_table
WHERE [your_eligibility_criteria]
```

**Step 2: Extract Temporal Claims**  
```sql
-- Example template
SELECT 
    patient_id,
    claim_date as dt,
    DATEDIFF(MONTH, date_of_birth, claim_date) as age_in_months,
    CASE WHEN gender = 'M' THEN 1 WHEN gender = 'F' THEN 0 ELSE 2 END as gender_cd,
    diagnosis_code,
    procedure_code,
    revenue_code
FROM your_claims_table 
WHERE claim_date >= DATEADD(MONTH, -24, analysis_date)
  AND claim_date < analysis_date
```

**Step 3: Aggregate by Day**
```sql
-- Group multiple codes per day into comma-separated strings
SELECT 
    patient_id,
    dt,
    age_in_months,
    gender_cd,
    STRING_AGG(CAST(diagnosis_code AS STRING), ',') as daily_codes
FROM claims_with_dates
GROUP BY patient_id, dt, age_in_months, gender_cd
ORDER BY patient_id, dt
```

**Step 4: Create Sequence Strings**
```sql
-- Aggregate days into '*' separated sequences
SELECT 
    patient_id,
    STRING_AGG(CAST(age_in_months AS STRING), '*' ORDER BY dt) as age_sequence,
    STRING_AGG(CAST(gender_cd AS STRING), '*' ORDER BY dt) as gender_sequence,
    STRING_AGG(daily_codes, '*' ORDER BY dt) as code_sequence
FROM daily_aggregated_claims
GROUP BY patient_id
```
