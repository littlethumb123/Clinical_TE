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

## Next Actions

1. **Inspect your training data** to determine which columns exist
2. **Query W2IND tables** to understand existing code mappings
3. **Choose appropriate solution** based on data structure
4. **Implement and validate** on small batch before full training
5. **Document your choice** for future reference and reproducibility

The original implementation likely used one of these approaches, but without the complete training code, we cannot definitively determine which. Both proposed solutions will produce a working, trainable model consistent with the observed architecture and loss computation structure.
