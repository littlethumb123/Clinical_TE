I'll create a comprehensive updated version of the documentation that clearly explains what we discovered, what's uncertain, and proposes solutions. Since you're in ask mode, I'll provide the complete file content for you to copy and apply:

```markdown
# Training Strategies for Clinical Transformer Embeddings

## Overview

This document describes the training strategy for the hierarchical clinical transformer model, clarifies discovered ambiguities in the original implementation, and proposes solutions for retraining.

---

## Architecture Summary

### Model Components

**Input Vocabulary**: 84,010 medical codes (Com_ME) or 98,041 (Medicaid)
- Includes ICD-9 diagnoses, CPT procedures, revenue codes, GPI medications, specialty codes
- All codes are mapped to integer indices via `HPT_CP_IP_V9_W2IND` or `MD_TE_V9_W2IND` tables
- Enables rich contextual understanding of patient medical history

**Output Vocabulary**: 2,767 target classes (Com_ME) or 2 classes (Medicaid)
- **Com_ME**: Multi-class prediction over subset of most common/important medical codes
- **Medicaid**: Binary classification (e.g., high-risk vs low-risk patient classification)
- Much smaller than input vocabulary for tractable prediction

**Why Different Vocabularies?**
- **Input (large)**: Model reads ALL codes in patient history for comprehensive context
- **Output (small)**: Model predicts only actionable/common outcomes
- Analogous to reading Shakespeare (large vocabulary) but writing tweets (constrained vocabulary)

### Model Architecture

```
Level 1: Daily Code Encoder
├─ Input: [batch, 200 days, 80 codes per day]
├─ Embedding: 84,010-dim vocab → 256-dim vectors
├─ Transformer: 1 layer, 4 heads (captures code co-occurrence)
├─ Aggregation: MaxPool across 80 codes
└─ Output: [batch, 200 days, 256-dim] daily representations

Level 2: Temporal Encoder  
├─ Input: [200 days, batch, 256-dim]
├─ Transformer: 6 layers, 16 heads (captures disease progression)
├─ Causal Masking: Day t can only attend to days 0...t
└─ Output: [200 days, batch, 256-dim] temporal representations

Output Layer
├─ Linear: 256-dim → target_cd_cnt classes
├─ Log-Softmax: Probability distribution over target codes
└─ Output: [batch, 200 days, target_cd_cnt] predictions
```

---

## Training Strategy: What We Know

### Data Flow (Confirmed from Code Analysis)

```python
# Training loop structure (from medicaid-asdb-claims-embedding_transformer_embeddings.py)
dt_cnt, x, y = prepare_tensor(batch, device)  # Returns 3 values
opt = model(x)                                  # Forward pass
opt = opt.reshape(batch_size * len_dy, target_cd_cnt)  # Flatten predictions
y = [item for sublist in y for item in sublist]        # Flatten targets
opt = torch.cat([opt[len_dy*i:len_dy*i+dt_cnt[i], :] for i in range(batch_size)], dim=0)
y = torch.tensor(y).to(device)
loss = criterion(opt, y)  # NLLLoss
```

### Key Discovery: Shape Requirements

**Critical Constraint Discovered:**
```python
# After processing:
opt.shape = [sum(dt_cnt), target_cd_cnt]  # One prediction vector per actual day
y.shape = [sum(dt_cnt)]                    # One target code per actual day

# For NLLLoss to work: len(y) MUST equal sum(dt_cnt)
# This means: ONE TARGET CODE PER DAY, not multiple codes per day!
```

### Inference Mode (Confirmed Working)

```python
# Scoring/embedding generation (from min_transformer.py lines 192-235)
dt_cnt, x = prepare_tensor(batch)  # Only 2 return values (no targets needed!)
opts = model(x)
intermedia_output = activation['transformer_encoder_dy']
patient_embedding = intermedia_output[dt_cnt[i], i, :]  # [256-dim] vector
```

**Why inference works but training fails:**
- ✓ Inference extracts embeddings (no loss computation, no target comparison)
- ✗ Training computes loss (requires valid targets in [0, target_cd_cnt) range)
- ✗ Current prepare_tensor implementation returns wrong target format

---

## Critical Issues: What's Missing

### Issue 1: Incomplete prepare_tensor Implementation ⚠️

**Evidence:**
```python
# min_transformer.py line 166 (scoring version):
def prepare_tensor(batch):
    # ... processes age, gender, codes ...
    return dt_cnt, x  # Only 2 return values!

# min_transformer.py line 175 (training expects):
dt_cnt, x, y = prepare_tensor(batch)  # Expects 3 values!
# This version DOES NOT EXIST in the codebase!
```

**Conclusion:** The training version of `prepare_tensor` that returns target labels `y` is **missing** from the available code.

### Issue 2: Unknown Target Selection Strategy 🤔

**What we know:**
- Shape requirement: ONE target code per day (not multiple)
- Each patient has dt_cnt days with actual data
- Each day can have up to 80 input codes
- Training selects ONE code per day as the prediction target

**What we DON'T know:**
- Which code is selected when a day has multiple codes?
- Is it the first code? Primary diagnosis? Random? Most frequent?
- How is this selection logic implemented?

**Possible strategies:**
1. **Primary code selection**: Always pick first non-zero code (likely primary diagnosis)
2. **Random sampling**: Randomly select one code per day (data augmentation)
3. **Importance-based**: Select based on medical hierarchy (ICD > CPT > Revenue)
4. **Frequency-based**: Select most common code type for training balance

### Issue 3: Unknown 84K → 2.7K Mapping 🤔

**For Commercial/Medicare (Com_ME) only:**

**What we know:**
- Input embedding: 84,010 codes
- Output prediction: 2,767 classes
- W2IND table maps raw codes → integer indices [0, 84009]
- Model outputs [batch, 200, 2767] predictions

**What we DON'T know:**
- Which 2,767 codes are the prediction targets?
- Are they the 2,767 most common codes?
- Is there a pre-computed mapping table?
- Are targets pre-mapped in the data or mapped at runtime?

**Possible scenarios:**

**Scenario A: Pre-computed Target Column**
```sql
-- DataFrame has separate column:
-- 'cd': Input codes in [0, 84009] for context
-- 'target_cd': Target codes in [0, 2766] for prediction
```

**Scenario B: Top-K Most Common Codes**
```python
# At preprocessing time:
code_counts = Counter(all_training_codes)
top_2767_codes = [code for code, _ in code_counts.most_common(2767)]
code_to_target = {code: idx for idx, code in enumerate(top_2767_codes)}
# Codes not in top-2767 are skipped during training
```

**Scenario C: Fixed Medical Code Set**
```python
# Pre-defined clinically important codes
# E.g., top diabetes, hypertension, heart disease codes
# Plus common procedures and medications
```

### Issue 4: Medicaid Binary Classification 📊

**What we know:**
- Input: 98,041 codes
- Output: 2 classes only
- Much simpler prediction task

**What we DON'T know:**
- What do the 2 classes represent?
  - High risk vs low risk?
  - Eligible vs ineligible for program?
  - Readmission vs no readmission?
- How are patient days labeled with binary class?
- Is this a different task than Com_ME entirely?

---

## Proposed Solutions

### Solution 1: Commercial/Medicare (2,767 Targets)

#### Option A: Pre-computed Target Column (Recommended if Available)

**Implementation:**
```python
def prepare_tensor_commercial_precomputed(batch, device, prediction_mode='same_day'):
    """
    Assumes DataFrame has 'target_cd' column with pre-mapped targets [0, 2766].
    
    Required DataFrame columns:
        - cd: Input codes [0, 84009] for embedding lookup
        - target_cd: Target codes [0, 2766] for loss computation
        - age_in_months, gender_cd, dt_cnt
    """
    batch_size = len(batch)
    len_dy = 200
    len_cd = 80
    
    # === INPUT PROCESSING (uses 'cd' column - full 84K vocab) ===
    age_in_months = [conv_age_gender(ipt, len_dy) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months, dtype=torch.long).to(device)
    age_in_months = age_in_months.reshape(batch_size, len_dy, 1)
    
    gender_cd = [conv_age_gender(ipt, len_dy) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd, dtype=torch.long).to(device)
    gender_cd = gender_cd.reshape(batch_size, len_dy, 1)
    
    cd_raw = [conv_cd(ipt, len_dy, len_cd) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd_raw, dtype=torch.long).to(device)
    
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    dt_cnt = batch['dt_cnt'].tolist()
    
    # === TARGET EXTRACTION (uses 'target_cd' column - 2.7K vocab) ===
    if 'target_cd' not in batch.columns:
        raise ValueError(
            "Commercial training requires 'target_cd' column!\n"
            "Run upstream SQL to create target_cd from cd using top-2767 mapping."
        )
    
    target_raw = [conv_cd(ipt, len_dy, len_cd) for ipt in batch['target_cd'].tolist()]
    
    y = []
    for i in range(batch_size):
        patient_targets = []
        
        # ONE target per day (based on train function shape requirements)
        for day_idx in range(dt_cnt[i]):
            day_target_codes = target_raw[i][day_idx]
            
            # Select first non-zero target code (primary code strategy)
            non_zero_targets = [code for code in day_target_codes if code != 0]
            
            if non_zero_targets:
                # Use first code as target
                patient_targets.append(non_zero_targets[0])
            else:
                # No valid target for this day - use padding class (0)
                patient_targets.append(0)
        
        y.append(patient_targets)
    
    return dt_cnt, x, y

# Verification after data load:
assert 'target_cd' in df_train.columns, "Missing target_cd column!"
print(f"✓ Found target_cd column")

# Check target ranges
sample_targets = df_train.iloc[0]['target_cd']
target_codes = [int(c) for c in sample_targets.replace('*', ',').split(',') if c and c != '0']
assert max(target_codes) < 2767, f"Target codes exceed 2767: {max(target_codes)}"
print(f"✓ Target codes in valid range [0, 2766]")
```

**Upstream SQL Required:**
```sql
-- Create target_cd column during data preprocessing
WITH code_mapping AS (
  -- Define top 2767 most common codes
  SELECT cd, ROW_NUMBER() OVER (ORDER BY code_count DESC) - 1 as target_idx
  FROM (
    SELECT cd, COUNT(*) as code_count
    FROM medical_codes_table
    GROUP BY cd
  )
  WHERE ROW_NUMBER() OVER (ORDER BY code_count DESC) <= 2767
)
SELECT 
  individual_id,
  cd,  -- Input codes (keep original for context)
  CASE 
    WHEN m.target_idx IS NOT NULL THEN m.target_idx
    ELSE 0  -- Map rare codes to padding class
  END as target_cd,
  age_in_months,
  gender_cd,
  dt_cnt
FROM patient_data p
LEFT JOIN code_mapping m ON p.cd = m.cd;
```

#### Option B: Runtime Top-K Mapping (If No Target Column)

**Implementation:**
```python
def create_code_mapping(training_data, target_cd_cnt=2767):
    """
    Build mapping from training data frequencies.
    Run ONCE before training.
    """
    from collections import Counter
    
    print("Building code→target mapping from training data...")
    all_codes = []
    
    for idx, row in training_data.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(training_data)} rows...")
        
        cd_string = row['cd']
        codes = cd_string.replace('*', ',').split(',')
        codes = [int(c) for c in codes if c and c != '0']
        all_codes.extend(codes)
    
    code_counts = Counter(all_codes)
    print(f"✓ Found {len(code_counts)} unique codes")
    
    # Select top 2767
    top_codes = [code for code, count in code_counts.most_common(target_cd_cnt)]
    code_to_target = {code: idx for idx, code in enumerate(top_codes)}
    
    # Calculate coverage
    total_occurrences = sum(code_counts.values())
    covered_occurrences = sum([code_counts[c] for c in top_codes])
    coverage = covered_occurrences / total_occurrences * 100
    
    print(f"✓ Top {target_cd_cnt} codes cover {coverage:.2f}% of all code occurrences")
    print(f"  Most common code: {top_codes[0]} ({code_counts[top_codes[0]]:,} occurrences)")
    print(f"  Least common in top-{target_cd_cnt}: {top_codes[-1]} ({code_counts[top_codes[-1]]:,} occurrences)")
    
    return code_to_target

def prepare_tensor_commercial_runtime(batch, device, code_to_target):
    """
    Runtime mapping using code_to_target dictionary.
    Codes not in top-2767 are skipped (no target created for that day).
    """
    batch_size = len(batch)
    len_dy = 200
    len_cd = 80
    
    # === INPUT PROCESSING (same as Option A) ===
    age_in_months = [conv_age_gender(ipt, len_dy) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months, dtype=torch.long).to(device)
    age_in_months = age_in_months.reshape(batch_size, len_dy, 1)
    
    gender_cd = [conv_age_gender(ipt, len_dy) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd, dtype=torch.long).to(device)
    gender_cd = gender_cd.reshape(batch_size, len_dy, 1)
    
    cd_raw = [conv_cd(ipt, len_dy, len_cd) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd_raw, dtype=torch.long).to(device)
    
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    dt_cnt = batch['dt_cnt'].tolist()
    
    # === TARGET EXTRACTION WITH MAPPING ===
    y = []
    skipped_days = 0
    total_days = sum(dt_cnt)
    
    for i in range(batch_size):
        patient_targets = []
        
        for day_idx in range(dt_cnt[i]):
            day_codes = cd_raw[i][day_idx]
            non_zero_codes = [code for code in day_codes if code != 0]
            
            # Try to find a code in top-2767
            target_found = False
            for code in non_zero_codes:
                if code in code_to_target:
                    patient_targets.append(code_to_target[code])
                    target_found = True
                    break  # Use first mappable code
            
            if not target_found:
                # All codes on this day are rare (not in top-2767)
                # Use padding class 0 or skip this day entirely
                patient_targets.append(0)  # Padding class
                skipped_days += 1
        
        y.append(patient_targets)
    
    skip_rate = skipped_days / total_days * 100 if total_days > 0 else 0
    if skip_rate > 30:
        print(f"⚠️  WARNING: {skip_rate:.1f}% of days have no mappable codes")
        print(f"   Your code_to_target mapping may be incomplete!")
    
    return dt_cnt, x, y

# Usage:
# 1. Build mapping ONCE
code_to_target = create_code_mapping(df_train, target_cd_cnt=2767)

# 2. Save for reproducibility
import pickle
with open('commercial_code_mapping.pkl', 'wb') as f:
    pickle.dump(code_to_target, f)

# 3. Use in training
dt_cnt, x, y = prepare_tensor_commercial_runtime(batch, device, code_to_target)
```

### Solution 2: Medicaid Binary Classification (2 Targets)

**Implementation:**
```python
def prepare_tensor_medicaid(batch, device):
    """
    Medicaid: Binary classification (2 classes).
    Assumes 'target_label' column with values [0, 1].
    
    Required DataFrame columns:
        - cd: Input codes [0, 98040] for embedding
        - target_label: Binary class [0, 1] for entire patient/sequence
        - age_in_months, gender_cd, dt_cnt
    
    Note: In Medicaid, target is likely patient-level (not day-level),
    but we repeat it for each day to match shape requirements.
    """
    batch_size = len(batch)
    len_dy = 200
    len_cd = 80
    
    # === INPUT PROCESSING (full 98K vocab) ===
    age_in_months = [conv_age_gender(ipt, len_dy) for ipt in batch['age_in_months'].tolist()]
    age_in_months = torch.tensor(age_in_months, dtype=torch.long).to(device)
    age_in_months = age_in_months.reshape(batch_size, len_dy, 1)
    
    gender_cd = [conv_age_gender(ipt, len_dy) for ipt in batch['gender_cd'].tolist()]
    gender_cd = torch.tensor(gender_cd, dtype=torch.long).to(device)
    gender_cd = gender_cd.reshape(batch_size, len_dy, 1)
    
    cd_raw = [conv_cd(ipt, len_dy, len_cd) for ipt in batch['cd'].tolist()]
    cd = torch.tensor(cd_raw, dtype=torch.long).to(device)
    
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    dt_cnt = batch['dt_cnt'].tolist()
    
    # === TARGET EXTRACTION (binary class per patient) ===
    if 'target_label' not in batch.columns:
        raise ValueError(
            "Medicaid training requires 'target_label' column!\n"
            "Expected: Binary labels [0, 1] for each patient."
        )
    
    y = []
    for i in range(batch_size):
        patient_label = int(batch.iloc[i]['target_label'])
        
        # Validate binary class
        assert patient_label in [0, 1], f"Invalid target_label: {patient_label}"
        
        # Repeat patient label for each actual day
        # (To match shape requirement: one target per day)
        patient_targets = [patient_label] * dt_cnt[i]
        y.append(patient_targets)
    
    return dt_cnt, x, y

# Verification:
assert 'target_label' in df_train.columns, "Missing target_label column!"
unique_labels = df_train['target_label'].unique()
assert set(unique_labels).issubset({0, 1}), f"Invalid labels: {unique_labels}"
print(f"✓ Found valid binary target_label column")
print(f"  Class distribution: {df_train['target_label'].value_counts().to_dict()}")
```

---

## Recommendations for Retraining

### Step 1: Investigate Your Data

```python
# Check what columns you have
print("Available columns:", df_train.columns.tolist())

# Check code ranges
sample_cd = df_train.iloc[0]['cd']
codes = [int(c) for c in sample_cd.replace('*', ',').split(',') if c and c != '0']
print(f"Code range in 'cd' column: [{min(codes)}, {max(codes)}]")

# For Commercial: Check if target_cd exists
if 'target_cd' in df_train.columns:
    print("✓ Found 'target_cd' column → Use Solution 1, Option A")
    sample_target = df_train.iloc[0]['target_cd']
    targets = [int(c) for c in sample_target.replace('*', ',').split(',') if c and c != '0']
    print(f"Target range: [{min(targets)}, {max(targets)}]")
else:
    print("✗ No 'target_cd' column → Use Solution 1, Option B (runtime mapping)")

# For Medicaid: Check if target_label exists
if 'target_label' in df_train.columns:
    print("✓ Found 'target_label' column → Use Solution 2")
```

### Step 2: Query the W2IND Mapping Table

```sql
-- Check the actual mapping range
SELECT 
    COUNT(DISTINCT ind) as num_indices,
    MIN(ind) as min_ind,
    MAX(ind) as max_ind,
    COUNT(DISTINCT cd) as num_codes
FROM `anbc-hcb-prod.clin_analytics_share_hcb_prod.HPT_CP_IP_V9_W2IND`;

-- If max_ind ≈ 2766: Mapping already creates target space!
-- If max_ind ≈ 84009: Mapping creates full input space, need additional target mapping
```

### Step 3: Choose Implementation

**For Commercial (2,767 targets):**
- **IF** you have `target_cd` column → Use Solution 1, Option A
- **ELSE** → Use Solution 1, Option B (build runtime mapping)

**For Medicaid (2 targets):**
- Must have `target_label` column → Use Solution 2
- If missing, consult with business team on what the 2 classes represent

### Step 4: Validate Before Full Training

```python
# Test prepare_tensor on small batch
test_batch = df_train.iloc[:16]
dt_cnt, x, y = prepare_tensor(test_batch, device)

# Verify shapes
print(f"dt_cnt: {dt_cnt}")
print(f"x shape: {x.shape}")  # Should be [16, 200, 82]
print(f"y structure: {[len(targets) for targets in y]}")  # Should match dt_cnt

# Verify target ranges
y_flat = [item for sublist in y for item in sublist]
assert max(y_flat) < target_cd_cnt, f"Target {max(y_flat)} >= {target_cd_cnt}!"
assert min(y_flat) >= 0, f"Target {min(y_flat)} < 0!"
print(f"✓ All targets in valid range [0, {target_cd_cnt})")

# Test with model
opt = model(x)
opt = opt.reshape(16 * 200, target_cd_cnt)
y_tensor = torch.tensor(y_flat).to(device)
loss = criterion(opt[:len(y_flat)], y_tensor)
print(f"✓ Loss computation successful: {loss.item():.4f}")
```

---

## Summary: What We Know vs. What's Missing

### ✅ Confirmed Facts

1. **Architecture**: Hierarchical 2-level transformer (daily → temporal encoding)
2. **Input vocabulary**: 84,010 codes (Com_ME) or 98,041 (Medicaid) - full medical ontology
3. **Output vocabulary**: 2,767 classes (Com_ME) or 2 classes (Medicaid) - constrained prediction space
4. **Training structure**: ONE target code per day (not multi-label per day)
5. **Loss function**: NLLLoss with shape requirement: opt=[sum(dt_cnt), target_cd_cnt], y=[sum(dt_cnt)]
6. **Inference**: Extracts 256-dim embeddings from temporal encoder (no targets needed)

### ❓ Unknown/Missing

1. **Target selection**: When multiple codes per day, which one is selected as target?
2. **Code mapping**: How are 84K input codes mapped to 2.7K target classes?
3. **Target definition**: Which specific 2,767 codes are the prediction targets?
4. **Medicaid labels**: What do the 2 binary classes represent?
5. **prepare_tensor implementation**: Training version that returns `y` is missing from codebase

### 🎯 Proposed Solutions

**For Commercial**: Implement either pre-computed target column OR runtime top-K mapping
**For Medicaid**: Implement binary patient-level classification with label repetition per day

Both solutions are mathematically sound and align with the confirmed training structure. The choice depends on available data columns and upstream preprocessing capabilities.

---



# Deep dive into the training and data transformation

## **🏥 COMPLETE DATA TRANSFORMATION PIPELINE**

---

### **STAGE 1: Raw Claims Data (BigQuery)**

```
Patient ID: 123456789
Claims History (24 months):

┌─────────────┬──────────────┬───────────────────────────────────┐
│ Date        │ Claim Type   │ Codes                             │
├─────────────┼──────────────┼───────────────────────────────────┤
│ 2024-01-15  │ Outpatient   │ ICD: 250.00 (Diabetes)           │
│             │              │ CPT: 99213 (Office Visit)        │
│             │              │ REV: 0450 (Emergency)            │
├─────────────┼──────────────┼───────────────────────────────────┤
│ 2024-01-16  │ Lab          │ CPT: 80053 (Blood Panel)         │
│             │              │ REV: 0300 (Lab)                  │
├─────────────┼──────────────┼───────────────────────────────────┤
│ 2024-01-17  │ Pharmacy     │ NDC: 00093-0058 (Metformin)      │
│             │              │ GPI: 2710 (Diabetes Meds)        │
├─────────────┼──────────────┼───────────────────────────────────┤
│ ...         │ ...          │ ...                               │
├─────────────┼──────────────┼───────────────────────────────────┤
│ 2024-06-10  │ Inpatient    │ ICD: 428.0 (Heart Failure)       │
│             │              │ CPT: 93000 (EKG)                 │
│             │              │ DRG: 291 (Heart Failure)         │
└─────────────┴──────────────┴───────────────────────────────────┘

Total: 150 days with claims
Demographics: Age 65, Male
```

---

### **STAGE 2: SQL Preprocessing (Code Mapping)**

**File: `5_a_prep_transformer_dec.sql`**

```sql
-- Map raw medical codes → integer indices [0, 84009]
-- Using HPT_CP_IP_V9_W2IND lookup table

SELECT 
    individual_id,
    dt,
    CASE WHEN w2ind.ind IS NULL THEN 0 
         ELSE w2ind.ind END AS ind
FROM raw_claims base
LEFT JOIN HPT_CP_IP_V9_W2IND w2ind
    ON concat('icd9_dx_cd', cast(base.icd9_dx_cd as string)) = w2ind.cd

-- Example mapping:
-- 'icd9_dx_cd250.00'  → ind: 15    (Diabetes)
-- 'cpt99213'          → ind: 156   (Office Visit)
-- 'revenue_cd0450'    → ind: 2134  (Emergency)
-- 'cpt80053'          → ind: 823   (Lab)
-- ...
```

**Output per day:**
```
Day 2024-01-15: indices = [15, 156, 2134]
Day 2024-01-16: indices = [823, 401]
Day 2024-01-17: indices = [5042, 2710]
...
Day 2024-06-10: indices = [89, 3421, 892]
```

**Aggregation to 200 days × 80 codes:**
```sql
-- Group codes by day, limit to 80 codes per day
-- Aggregate across 200-day window

SELECT 
    individual_id,
    STRING_AGG(CAST(ind AS STRING), ",") AS cd  -- Per day
FROM codes_per_day
WHERE seqno <= 80  -- Max 80 codes per day
GROUP BY individual_id, dt
```

---

### **STAGE 3: Final DataFrame Structure**

```python
df_train.columns = ['individual_id', 'age_in_months', 'gender_cd', 'cd', 'dt_cnt']

# Single patient row:
{
    'individual_id': 123456789,
    
    'age_in_months': '780*781*781*782*782*783*...',  # 150 values (age per day)
    # Format: "age_day0*age_day1*age_day2*..." (months)
    
    'gender_cd': '1*1*1*1*1*1*...',  # 150 values (gender per day, 1=Male)
    # Format: "gender_day0*gender_day1*..."
    
    'cd': '15,156,2134*823,401*5042,2710*...*89,3421,892',  # 150 days
    # Format: "day0_codes*day1_codes*day2_codes*..."
    # Each day: "code1,code2,code3,..."
    # Values in range [0, 84009]
    
    'dt_cnt': 150  # Number of actual days (rest padded to 200)
}

# Shape visualization:
# age_in_months: String → will become [200] array
# gender_cd:     String → will become [200] array  
# cd:            String → will become [200, 80] array
# dt_cnt:        Integer (scalar)
```

---

### **STAGE 4: prepare_tensor() Transformation**

```python
def prepare_tensor(batch):
    # batch_size = 16 patients
    # len_dy = 200 days
    # len_cd = 80 codes per day
    
    # === Step 4.1: Parse age_in_months ===
    age_in_months = [conv_age_gender(ipt) for ipt in batch['age_in_months'].tolist()]
    # Input:  ['780*781*781*...'] × 16 patients
    # Output: [[780, 781, 781, ..., 0, 0], [...], ...] × 16
    # Shape:  List[List[int]] → 16 × 200
    
    age_in_months = torch.tensor(age_in_months).to(device)
    # Shape: [16, 200]
    
    age_in_months = age_in_months.reshape(batch_size, len_dy, 1)
    # Shape: [16, 200, 1]  ← Add feature dimension
    
    # === Step 4.2: Parse gender_cd ===
    gender_cd = [conv_age_gender(ipt) for ipt in batch['gender_cd'].tolist()]
    # Shape: [16, 200, 1]  ← Same process
    
    # === Step 4.3: Parse medical codes ===
    cd = [conv_cd(ipt) for ipt in batch['cd'].tolist()]
    # Input:  ['15,156,2134*823,401*...'] × 16 patients
    # Output after conv_cd:
    # [
    #   [[15, 156, 2134, 0, 0, ..., 0],    # Day 0: 80 codes
    #    [823, 401, 0, 0, ..., 0],         # Day 1: 80 codes
    #    [5042, 2710, 0, 0, ..., 0],       # Day 2: 80 codes
    #    ...
    #    [89, 3421, 892, 0, ..., 0],       # Day 149: 80 codes
    #    [0, 0, 0, ..., 0],                # Day 150-199: padding
    #   ],
    #   [...],  # Patient 2
    #   ...
    # ]
    # Shape: List of 16 × [200 days × 80 codes]
    
    cd = torch.tensor(cd).to(device)
    # Shape: [16, 200, 80]
    
    # === Step 4.4: Concatenate features ===
    x = torch.cat([age_in_months, gender_cd, cd], dim=-1)
    # Concatenate along feature dimension:
    # [16, 200, 1] + [16, 200, 1] + [16, 200, 80]
    # Shape: [16, 200, 82]
    #        ↑   ↑    ↑
    #        │   │    └─ 1 age + 1 gender + 80 codes = 82 features
    #        │   └────── 200 days (max sequence length)
    #        └────────── 16 patients (batch size)
    
    dt_cnt = batch['dt_cnt'].tolist()
    # Shape: List[int] of length 16
    # Example: [150, 180, 200, 120, ...]
    
    return dt_cnt, x  # For inference
    # For training, would also return y (targets)
```

**Summary of prepare_tensor output:**
```
x shape:      [16, 200, 82]
              [batch, days, features]
dt_cnt shape: List of 16 integers
```

---

### **STAGE 5: Model Forward Pass - Embedding Layer**

```python
def forward(self, x):
    # Input shape: [16, 200, 82]
    gpu_batchsize = x.shape[0]  # 16
    
    # === Step 5.1: Extract features ===
    age_in_months = x[:, :, 0]  # [16, 200]
    gender_cd = x[:, :, 1]      # [16, 200]
    cd = x[:, :, 2:]            # [16, 200, 80]
    
    # === Step 5.2: Embed age ===
    # self.embedding_age_in_months = nn.Embedding(1440, 256)
    age_in_months = self.embedding_age_in_months(age_in_months)
    # Input:  [16, 200] with values in [0, 1439]
    # Output: [16, 200, 256]
    #         Each age value → 256-dim vector
    
    # === Step 5.3: Embed gender ===
    # self.embedding_gender_cd = nn.Embedding(4, 256)
    gender_cd = self.embedding_gender_cd(gender_cd)
    # Input:  [16, 200] with values in [0, 3]
    # Output: [16, 200, 256]
    
    # === Step 5.4: Embed medical codes ===
    # self.embedding_cd = nn.Embedding(84010, 256)
    cd = self.embedding_cd(cd)
    # Input:  [16, 200, 80] with values in [0, 84009]
    #         ↑   ↑    ↑
    #         │   │    └─ 80 codes per day
    #         │   └────── 200 days
    #         └────────── 16 patients
    #
    # Output: [16, 200, 80, 256]
    #         ↑   ↑    ↑   ↑
    #         │   │    │   └─ Each code → 256-dim embedding
    #         │   │    └───── 80 codes embedded
    #         │   └────────── 200 days
    #         └────────────── 16 patients
    
    # Visual example for one day:
    # Day 0 codes: [15, 156, 2134, 0, 0, ..., 0]
    #              ↓   ↓    ↓     ↓
    # Embedded:    [emb_15, emb_156, emb_2134, emb_0, ...]
    #              [256-d,  256-d,   256-d,    256-d, ...]
    
    # === Step 5.5: Residual connection (for later) ===
    cd_res = cd.sum(-2)
    # Sum across 80 codes dimension:
    # Input:  [16, 200, 80, 256]
    # Output: [16, 200, 256]
    # This creates a "bag of codes" representation per day
```

**Summary after embeddings:**
```
age_in_months: [16, 200, 256]
gender_cd:     [16, 200, 256]
cd:            [16, 200, 80, 256]  ← 80 embedded codes per day
cd_res:        [16, 200, 256]      ← Sum of 80 codes per day
```

---

### **STAGE 6: Daily Code Encoder (Level 1)**

```python
    # === Step 6.1: Reshape for daily encoding ===
    # Process all days in parallel (batch them together)
    cd = cd.reshape(gpu_batchsize * len_dy, len_cd, embedding_size)
    # Input:  [16, 200, 80, 256]
    # Output: [3200, 80, 256]
    #         ↑     ↑   ↑
    #         │     │   └─ 256-dim embeddings
    #         │     └───── 80 codes per day
    #         └─────────── 16×200 = 3200 days total
    
    cd = torch.swapaxes(cd, 0, 1)
    # Transformer expects: [sequence_length, batch, features]
    # Output: [80, 3200, 256]
    #         ↑   ↑     ↑
    #         │   │     └─ 256-dim embeddings
    #         │   └─────── 3200 days (as batch)
    #         └─────────── 80 codes (sequence length)
    
    # === Step 6.2: Daily transformer encoder ===
    # self.transformer_encoder_cd = TransformerEncoder(
    #     TransformerEncoderLayer(256, 4 heads, 256 FFN, 0 dropout), 
    #     1 layer
    # )
    cd = self.transformer_encoder_cd(cd)
    # Input:  [80, 3200, 256]
    # Output: [80, 3200, 256]  ← Same shape, but contextualized
    #
    # What this does: Each of 80 codes attends to other codes on SAME day
    # "Which codes co-occur? Diabetes + Office Visit + Lab = routine monitoring"
    
    # === Step 6.3: Max pooling across codes ===
    cd = cd.permute(1, 2, 0)
    # Rearrange for pooling: [3200, 256, 80]
    
    cd = nn.MaxPool1d(len_cd)(cd)
    # Pool across 80 codes dimension:
    # Input:  [3200, 256, 80]
    # Output: [3200, 256, 1]
    # Takes maximum activation across all 80 codes per dimension
    
    cd = cd.reshape(gpu_batchsize, len_dy, embedding_size)
    # Output: [16, 200, 256]
    #         ↑   ↑    ↑
    #         │   │    └─ Single 256-dim vector per day (aggregated from 80 codes)
    #         │   └────── 200 days
    #         └────────── 16 patients
    
    # NOW: Each day reduced from 80×256 codes → single 256-dim summary
```

**Visual of one patient's data after daily encoder:**
```
Before:  [200 days × 80 codes × 256 dim]  = 4,096,000 values
         Day 0: [code1_emb, code2_emb, ..., code80_emb]
         Day 1: [code1_emb, code2_emb, ..., code80_emb]
         ...

After:   [200 days × 256 dim] = 51,200 values
         Day 0: [daily_summary_256d]  ← Aggregation of all codes on day 0
         Day 1: [daily_summary_256d]
         ...
```

---

### **STAGE 7: Combine with Demographics**

```python
    # === Step 7.1: Combine all features ===
    cd = cd_res + cd + gender_cd + age_in_months
    # Element-wise addition:
    # cd_res:        [16, 200, 256]  ← Sum of 80 embedded codes
    # cd:            [16, 200, 256]  ← MaxPooled codes
    # gender_cd:     [16, 200, 256]  ← Gender embedding
    # age_in_months: [16, 200, 256]  ← Age embedding
    # Output:        [16, 200, 256]  ← Combined representation
    
    cd = self.mm(cd)     # GELU activation
    cd = self.norm(cd)   # LayerNorm
    # Output: [16, 200, 256]
    
    cd = torch.swapaxes(cd, 0, 1)
    # Prepare for temporal transformer: [sequence_length, batch, features]
    # Output: [200, 16, 256]
    #         ↑    ↑   ↑
    #         │    │   └─ 256-dim daily representation
    #         │    └───── 16 patients (batch)
    #         └────────── 200 days (sequence)
```

**Summary: Daily representations ready for temporal modeling**
```
Shape: [200, 16, 256]
       [days, patients, features]

Each element = one day's summary for one patient
Includes: codes + age + gender information
```

---

### **STAGE 8: Temporal Encoder (Level 2)**

```python
    # === Step 8.1: Generate causal mask ===
    mth_mask = self._generate_square_subsequent_mask(len_dy).to(device)
    # Creates [200, 200] mask:
    # [[0,   -inf, -inf, -inf, ...],   ← Day 0 can only see itself
    #  [0,    0,   -inf, -inf, ...],   ← Day 1 can see days 0-1
    #  [0,    0,    0,   -inf, ...],   ← Day 2 can see days 0-2
    #  ...
    #  [0,    0,    0,    0,   ...]]   ← Day 199 can see all days 0-199
    
    # === Step 8.2: Temporal transformer encoder ===
    # self.transformer_encoder_dy = TransformerEncoder(
    #     TransformerEncoderLayer(256, 16 heads, 512 FFN, 0.1 dropout),
    #     6 layers
    # )
    cd = self.transformer_encoder_dy(cd, mth_mask)
    # Input:  [200, 16, 256]
    # Output: [200, 16, 256]  ← Same shape, but with temporal context
    #
    # What this does:
    # - Layer 1: Learn short-term patterns (yesterday affects today)
    # - Layer 2: Learn medium-term patterns (last week affects today)
    # - Layer 3-6: Learn long-term patterns, disease progression
    #
    # Each day's representation now includes:
    # - Its own medical codes
    # - Context from ALL previous days (via attention)
    # - Temporal patterns (worsening condition, treatment effects)
    
    cd = torch.swapaxes(cd, 0, 1)
    # Back to batch-first: [16, 200, 256]
    
    cd = self.norm(cd)
    cd = self.dropout(cd)
    # Output: [16, 200, 256]
```

**Visual of attention at Day 100:**
```
Day 100's representation learns from:
  Day 0-99 context (via self-attention)
  ↓
  "Patient had diabetes (day 0), started treatment (day 5),
   blood sugar stabilized (day 20), routine monitoring (day 50-99),
   NOW on day 100: What's likely to happen?"
```

---

### **STAGE 9: Output Predictions**

```python
    # === Step 9.1: Project to target vocabulary ===
    # self.decoder_cd = nn.Linear(256, 2767)
    cd = self.decoder_cd(cd)
    # Input:  [16, 200, 256]
    # Output: [16, 200, 2767]
    #         ↑   ↑    ↑
    #         │   │    └─ 2767 classes (target vocabulary)
    #         │   └────── 200 days
    #         └────────── 16 patients
    #
    # cd[patient_i, day_j, class_k] = logit for class k on day j for patient i
    
    cd = F.log_softmax(cd, dim=-1)
    # Apply log softmax across 2767 classes:
    # Output: [16, 200, 2767]
    # cd[patient_i, day_j, :] = log probabilities over 2767 classes
    #
    # Example for patient 0, day 100:
    # cd[0, 100, :] = [log_prob_class_0, log_prob_class_1, ..., log_prob_class_2766]
    # Sum of exp(these values) = 1.0 (probability distribution)
    
    return cd
```

**Final model output:**
```
Shape: [16, 200, 2767]
       [patients, days, classes]

Interpretation:
- For each patient
  - For each day
    - Probability distribution over 2,767 target codes
    
Example: Patient 0, Day 100
  Class 15 (Diabetes):      log_prob = -0.5  → prob ≈ 0.60
  Class 42 (Hypertension):  log_prob = -1.8  → prob ≈ 0.17
  Class 89 (Heart Failure): log_prob = -3.2  → prob ≈ 0.04
  ...
  (2,767 total classes)
```

---

### **STAGE 10: Training - Loss Computation**

```python
def train(data, model, optimizer, criterion):
    # === Batch processing ===
    batch = data.iloc[idx*batch_size : idx*batch_size+batch_size, :]
    dt_cnt, x, y = prepare_tensor(batch, device)
    
    # dt_cnt: [150, 180, 200, 120, ...] (16 values)
    # x:      [16, 200, 82]
    # y:      [[target_day0, target_day1, ..., target_day149],  # Patient 0: 150 targets
    #          [target_day0, ..., target_day179],               # Patient 1: 180 targets
    #          ...]                                             # 16 nested lists
    
    # === Step 10.1: Forward pass ===
    opt = model(x)
    # Output: [16, 200, 2767]
    
    # === Step 10.2: Reshape predictions ===
    opt = opt.reshape(batch_size * len_dy, target_cd_cnt)
    # Input:  [16, 200, 2767]
    # Output: [3200, 2767]
    #         ↑     ↑
    #         │     └─ 2767 classes
    #         └─────── 16×200 = 3200 day-predictions (including padding)
    
    # === Step 10.3: Extract ACTUAL days (remove padding) ===
    opt = torch.cat([
        opt[len_dy*i : len_dy*i+dt_cnt[i], :]  # Extract days [0, dt_cnt[i]) for patient i
        for i in range(batch_size)
    ], dim=0)
    # Patient 0: days [0,   200) → extract [0,   150)  = 150 predictions
    # Patient 1: days [200, 400) → extract [200, 380)  = 180 predictions
    # Patient 2: days [400, 600) → extract [400, 600)  = 200 predictions
    # ...
    # Output: [sum(dt_cnt), 2767]
    #         [~2400, 2767] for this batch
    
    # === Step 10.4: Flatten targets ===
    y = [item for sublist in y for item in sublist]
    # Input:  [[150 codes], [180 codes], [200 codes], ...]
    # Output: [code, code, code, ...] (flat list of ~2400 codes)
    
    y = torch.tensor(y).to(device)
    # Output: [~2400]  ← ONE target code per day
    
    # === Step 10.5: COMPUTE LOSS ===
    loss = criterion(opt, y)
    # NLLLoss with:
    # - opt:  [~2400, 2767]  ← Predictions (log probabilities)
    # - y:    [~2400]        ← Target codes (integers)
    #
    # For each day i:
    #   loss += -opt[i, y[i]]  (negative log probability of true class)
    #
    # ⚠️ THIS IS WHERE ERROR HAPPENS IF y contains values >= 2767!
    #
    # NLLLoss internally does:
    # assert all(0 <= target < 2767 for target in y)
    # If y = [15, 42, 20585, ...]  ← 20585 >= 2767
    #   → CUDA assertion error!
```

---

## **🎯 COMPLETE SHAPE TRANSFORMATION SUMMARY**

```
Raw Data (SQL):
  Patient with 150 days of claims
  Each day: 1-80 medical codes

↓ SQL Preprocessing

DataFrame (one row per patient):
  cd: '15,156,2134*823,401*...'  (string)

↓ prepare_tensor()

Tensor Input:
  x: [16, 200, 82]
     [batch, days, features: age + gender + 80_codes]

↓ Embedding Layer

Embedded Features:
  age:      [16, 200, 256]
  gender:   [16, 200, 256]
  codes:    [16, 200, 80, 256]  ← 80 codes × 256-dim each
  
↓ Daily Encoder (Level 1)

Daily Representations:
  [16, 200, 256]  ← Each day: 1 vector (aggregated from 80 codes)

↓ Temporal Encoder (Level 2)

Contextualized Daily Representations:
  [16, 200, 256]  ← Each day includes temporal context

↓ Output Layer

Predictions:
  [16, 200, 2767]  ← Probability distribution over 2767 classes per day

↓ Training: Extract Actual Days

Filtered Predictions:
  [~2400, 2767]  ← Only actual days (dt_cnt), no padding

↓ Loss Computation

NLLLoss:
  predictions: [~2400, 2767]
  targets:     [~2400]  ← ONE code per day
  
  ⚠️ ERROR if targets contain values >= 2767!
```

---

## **🔑 KEY INSIGHTS**

1. **Input uses 84K vocabulary**: Codes embedded via `nn.Embedding(84010, 256)`
2. **Output uses 2.7K vocabulary**: Predictions via `nn.Linear(256, 2767)`
3. **One target per day**: NOT multi-label! Single code predicted per day
4. **The mismatch**: Input codes in [0, 84009], but model expects targets in [0, 2766]

**The fix requires**: Map input codes → target codes in [0, 2767) within `prepare_tensor()` when extracting targets `y`! 🎯
