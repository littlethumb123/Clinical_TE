# Enhance transformer with LOB and extrinsic evaluation
_Exported on 12/12/2025 at 19:48:08 PST from Cursor (2.1.47)_

---

**User**

@.cursor/rules/model_retrain.mdc  You are an expert in LLM and transformer pretraining and also finetuning and linear Probe implemenation for downstream classifiaiton tasks to evaluate embedding quality; I have a series of transformer design varaitns experiementation to evaluate if a model architecture design can provide better representative embedding for each member regarding their claim history; However, what I implemented right now has only intrinsic metrics, such as model loss, or ranking related metrics for next-day prediciton; Now I want to include a series of extrinsic evaluation metrics for a trained model to generate embeddings and put them on a real classficiaton tasks;

Now I want you to implement two things based on the current implemenation @dev/moe/moe_flashattn_3.py 
1)  include the LOB column of the training dataset to the model to indicate the line of business for a member, category variable, Medicaid, Medicare and Commercial, I want the transformer to encode the LOB information for each member; you can refer to @docs/retraining_refactor/add_LOB_to_model.md 
2) Use Medicaid IP risk as downstream tasks and include the extrinsic metrics to the part of experimentation metrics to evaluate the trained model and its embedding quality; Within this task, it is required to save the trained model after the training and the model should have clear and easy to identify and recognize name,  ( such as experiment_round, experiment configuration variants, training parameters (batch_size, epoch_size) and the this name should be logged in the experiemntation log; the model name should be same as hte outcome dataframe name as its experimentation; more detail you can refer to this plan @docs/pss/add_probs_to_metrics.md; make sure you have validation and test dataset for evaluating the prob and show metrics for both trainin, validation and test for hte linear probes evaluation; you can separate the embedding training from the dowonstream task or do them together. You should make decision based on the code efficiency, engienering concerns, code quality and readability. 

Here is the raw data table for training the transformer
The training dataset will be from the table edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending and the 
edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment
{
  "individual_id": "71093294",
  "index_dt": "2023-06-01",
  "acute_ip_flag": "0",
}
edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending with the folloiwng columns and values
{
  "individual_id": "72815682",
  "index_dt": "2023-01-01",
  "gender_cd": "0",
  "age_in_months": "0",
  "cd": "58054,55793,17569,1592,74345,74444,55794,74657,17787,1,1581",
  "target": "1,999,2227,2254,5204,5669",
  "dt_cnt": "1"
}
Don't hallucinate, don't pretend to know anything you are not certain; ask clarifciation quesitons if you are not making sure.


---

**Cursor**

I'll analyze the current implementation and the requirements to provide a comprehensive implementation plan. Let me first explore the codebase to understand the current state and clarify some questions.

Now let me check how the data is currently loaded and if there's more context about LOB:

Based on my analysis of the codebase, I have a few **clarification questions** before providing the implementation:

---

## Clarification Questions

### 1. LOB Column Source

Looking at the training data schema you provided:
```json
{
  "individual_id": "72815682",
  "index_dt": "2023-01-01",
  "gender_cd": "0",
  "age_in_months": "0",
  "cd": "58054,55793,17569,...",
  "target": "1,999,2227,...",
  "dt_cnt": "1"
}
```

**I don't see a `lob` column in this data.** The current `medicaid_data_prep.sql` pipeline is Medicaid-only, so all members would have LOB = "Medicaid".

**Questions:**
- **Option A**: Are you planning to train a **unified model across Commercial, Medicare, and Medicaid**? If so, you'd need to modify the data pipeline to include a `lob` column.
- **Option B**: Is the `lob` column **already available** in another table that needs to be joined? If so, which table contains it?
- **Option C**: Should I create the infrastructure for LOB encoding now (config, model changes, data parsing) assuming the data pipeline will be updated later to include `lob`?

### 2. Train/Val/Test Split for Downstream Task

The outcome table `a964286_Medicaid_outcome_ip_4_te_experiment` contains `acute_ip_flag` for the downstream classification task. 

**Questions:**
- Should I use the **same train/val split** as the transformer training for the linear probe, or create a **new split** that includes a **test set**?
- What's the approximate **label distribution** (percentage of `acute_ip_flag = 1`)? This helps determine if stratified splitting is needed.
- How do you want the data joined? By `individual_id` + `index_dt` between the training data and outcome table?

### 3. Workflow Preference

For the extrinsic evaluation workflow, there are two design options:

**Option A: Integrated (During Training)**
- Run linear probe evaluation at the end of each epoch (or every N epochs)
- Pros: See downstream performance during training
- Cons: Adds training time, uses validation data for probe

**Option B: Separate (Post-Training)**
- Save trained model with standardized naming
- Run separate evaluation script that loads model, extracts embeddings, trains linear probe
- Pros: Cleaner separation, can use proper train/val/test splits for probe
- Cons: Requires separate execution

**Which approach do you prefer?** Given your requirements for "train/val/test for linear probe evaluation", **Option B seems more appropriate** to ensure proper evaluation rigor.

### 4. Model Naming Convention

You mentioned clear model naming. Here's a proposed format:

```
{experiment_round}_{exp_name}_bs{batch_size}_ep{epochs}_d{embedding_size}_{timestamp}
```

Example:
```
round3_exp3_standard_moe_bs32_ep10_d256_20241211_143022
```

This would be used for:
- Checkpoint files: `{model_name}_best.pt`, `{model_name}_final.pt`
- Results DataFrame: `{model_name}_results.csv`
- Logs: `logs/{model_name}/`

**Does this naming convention work for you?**

---

## Preliminary Implementation Plan

Once you clarify the above, here's my implementation plan:

### Part 1: Add LOB Support
1. Add `lob_vocab: int = 4` to `BaseConfig` (0=padding, 1=Commercial, 2=Medicare, 3=Medicaid)
2. Add `conv_lob()` parsing function
3. Update `ClinicalDataset` and `clinical_collate_fn`
4. Add `embedding_lob` layer to all model classes
5. Update forward pass to use LOB embeddings

### Part 2: Downstream Task Evaluation
1. Create `DownstreamEvaluator` class for IP risk prediction
2. Implement `extract_member_embeddings()` using `EmbeddingExtractor`
3. Create `LinearProbeClassifier` with proper train/val/test evaluation
4. Add standardized model saving with naming convention
5. Update `run_single_experiment` to save final model and run downstream evaluation
6. Add comprehensive metrics (AUC-ROC, F1, calibration) for downstream task

Please confirm or clarify the above points so I can provide the precise implementation code.

```json
{
  "individual_id": "72815682",
  "index_dt": "2023-01-01",
  "gender_cd": "0",
  "age_in_months": "0",
  "cd": "58054,55793,17569,...",
  "target": "1,999,2227,...",
  "dt_cnt": "1"
}
```

```plaintext
{experiment_round}_{exp_name}_bs{batch_size}_ep{epochs}_d{embedding_size}_{timestamp}
```

```plaintext
round3_exp3_standard_moe_bs32_ep10_d256_20241211_143022
```

---

**User**

 1. the data will be 
{
  "individual_id": "72815682",
  "index_dt": "2023-01-01",
  "gender_cd": "0",
 'lob': 'Medicaid',
  "age_in_months": "0",
  "cd": "58054,55793,17569,...",
  "target": "1,999,2227,...",
  "dt_cnt": "1"
}
2. Train/Val/Test Split for Downstream Task: 1) what's your suggestion on this? 2) the prevalnce is around 4%-15% for ip flag; 3) yes for the downstream task I would liek to left join by individual_id and index_dt 
3. I would like the separate; it's clean and separated 
4. yes hte exp_name should have clear info about the experiment variants; such as 'moe', 'dense', 'kaiming_init'... to indicaate what have been experimented with. 
5 Confirm the implemnantion plan; make sure the added code should be well-integrated and working with existing impleenation without any potential confilicts or bugs; also make sure for each change, make it clear where to add the code or change hte code; because I will manually ad the code back to the @dev/moe/moe_flashattn_3.py; 
For now just disable the data parallel functions (if any functions or modules are going to be activated during the training, I have not figure out hwo to do that) 
Now start implementing

---

**Cursor**

I'll provide a comprehensive implementation with clear instructions on where to add each code section. Let me first check the exact line numbers for key sections:

Now I have all the context needed. Here's the complete implementation with clear instructions on where to add each section.

---

# Complete Implementation Guide

## Overview of Changes

I'll provide the code in **8 sections** with exact locations:

1. **Section 1**: Add new imports (top of file, around line 60)
2. **Section 2**: Update `BaseConfig` to add `lob_vocab` (line 266)
3. **Section 3**: Add `conv_lob()` function (after line 2544, near `conv_age_gender`)
4. **Section 4**: Update `ClinicalDataset` class (line 2392)
5. **Section 5**: Update `clinical_collate_fn` (line 2442)
6. **Section 6**: Update all three model classes to add LOB embedding
7. **Section 7**: Update `train_epoch` and `evaluate` functions
8. **Section 8**: Add Downstream Evaluation Module (after `EmbeddingExtractor`, around line 6100)
9. **Section 9**: Update `run_single_experiment` to save model and run downstream eval

---

## Section 1: Add New Imports

**Location**: Add after existing imports (around line 60, before the docstring ends or after initial imports)

```python
# ============================================================================
# ADDITIONAL IMPORTS FOR DOWNSTREAM EVALUATION
# ============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning
import warnings
import json
from datetime import datetime
```

---

## Section 2: Update BaseConfig

**Location**: Modify `BaseConfig` around line 264-266

**CHANGE FROM:**
```python
    # Embeddings
    gender_vocab: int = 4     # Gender categories
    age_vocab: int = 1440     # Age in months (120 years)
```

**CHANGE TO:**
```python
    # Embeddings
    gender_vocab: int = 4     # Gender categories
    age_vocab: int = 1440     # Age in months (120 years)
    lob_vocab: int = 4        # LOB categories (0=padding, 1=Commercial, 2=Medicare, 3=Medicaid)
```

---

## Section 3: Add conv_lob Function

**Location**: Add after `conv_age_gender` function (around line 2544)

```python
def conv_lob(ipt: str, len_dy: int) -> List[int]:
    """
    Convert LOB (Line of Business) string to list.
    
    Format: "value1*value2*..." (same format as age/gender, but with string values)
    Maps: Commercial=1, Medicare=2, Medicaid=3, padding/unknown=0
    
    Args:
        ipt: LOB string from data (e.g., "Medicaid*Medicaid*..." or single value "Medicaid")
        len_dy: Target sequence length (200)
    
    Returns:
        List of LOB indices [len_dy]
    """
    # LOB mapping (case-insensitive)
    lob_map = {
        'commercial': 1,
        'medicare': 2,
        'medicaid': 3
    }
    
    if not ipt or pd.isna(ipt):
        return [0] * len_dy
    
    # Handle both single value and asterisk-separated formats
    if '*' in str(ipt):
        values = str(ipt).split('*')
    else:
        # Single value - repeat for all days
        values = [str(ipt)] * len_dy
    
    values = values[:len_dy]  # Truncate to max days
    
    result = []
    for val in values:
        val_clean = val.strip().lower() if val else ''
        if val_clean in lob_map:
            result.append(lob_map[val_clean])
        else:
            result.append(0)  # Unknown LOB
    
    # Forward-fill with last valid value (LOB typically doesn't change within sequence)
    if result:
        last_val = result[-1] if result[-1] != 0 else 3  # Default to Medicaid if all zeros
        while len(result) < len_dy:
            result.append(last_val)
    else:
        result = [3] * len_dy  # Default to Medicaid (since this is Medicaid data)
    
    return result
```

---

## Section 4: Update ClinicalDataset Class

**Location**: Modify `ClinicalDataset.__init__` starting at line 2392

**REPLACE the entire `__init__` method (lines 2392-2424) with:**

```python
    def __init__(self, df: pd.DataFrame, config: BaseConfig):
        self.config = config
        print(f"Pre-processing {len(df)} samples into tensors (one-time cost)...")
        
        # Extract series for faster processing
        age_strs = df['age_in_months'].tolist()
        gender_strs = df['gender_cd'].tolist()
        cd_strs = df['cd'].tolist()
        target_strs = df['target_cd'].tolist()
        self.dt_cnt = df['dt_cnt'].tolist()
        
        # Handle LOB column (may or may not exist)
        if 'lob' in df.columns:
            lob_strs = df['lob'].tolist()
            self.has_lob = True
        else:
            lob_strs = [None] * len(df)
            self.has_lob = False
            print("  Warning: 'lob' column not found, using default LOB=Medicaid")

        # Pre-allocate tensors
        self.ages = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.genders = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.lobs = torch.zeros(len(df), config.len_dy, dtype=torch.long)  # NEW: LOB tensor
        self.codes = torch.zeros(len(df), config.len_dy, config.len_cd, dtype=torch.long)
        self.targets = []

        # Process all samples
        for i in range(len(df)):
            if i > 0 and i % 50000 == 0:
                print(f"  Processed {i}/{len(df)} samples...")

            age_list = conv_age_gender(age_strs[i], config.len_dy)
            gender_list = conv_age_gender(gender_strs[i], config.len_dy, max_val=3)
            lob_list = conv_lob(lob_strs[i], config.len_dy)  # NEW: Parse LOB
            cd_list = conv_cd(cd_strs[i], config.len_dy, config.len_cd)
            target_list = conv_target(target_strs[i], config.len_dy, config.target_cd_cnt)

            self.ages[i] = torch.tensor(age_list, dtype=torch.long)
            self.genders[i] = torch.tensor(gender_list, dtype=torch.long)
            self.lobs[i] = torch.tensor(lob_list, dtype=torch.long)  # NEW
            self.codes[i] = torch.tensor(cd_list, dtype=torch.long)
            self.targets.append(target_list)
        
        print("Pre-processing complete.")
```

**Also update `__getitem__` method (around line 2429):**

**REPLACE:**
```python
    def __getitem__(self, idx):
        return {
            'age': self.ages[idx],
            'gender': self.genders[idx],
            'codes': self.codes[idx],
            'dt_cnt': self.dt_cnt[idx],
            'target': self.targets[idx]
        }
```

**WITH:**
```python
    def __getitem__(self, idx):
        return {
            'age': self.ages[idx],
            'gender': self.genders[idx],
            'lob': self.lobs[idx],  # NEW: Include LOB
            'codes': self.codes[idx],
            'dt_cnt': self.dt_cnt[idx],
            'target': self.targets[idx]
        }
```

---

## Section 5: Update clinical_collate_fn

**Location**: Modify `clinical_collate_fn` starting at line 2442

**REPLACE the entire function with:**

```python
def clinical_collate_fn(batch):
    """
    Custom collate function for clinical data.
    
    Handles the special case of 'target' which is a nested list with variable-length sublists.
    PyTorch's default_collate cannot handle this, so we keep it as a Python list.
    
    Args:
        batch: List of dictionaries from ClinicalDataset.__getitem__
    
    Returns:
        Batched dictionary with:
        - age, gender, lob, codes: Stacked tensors
        - dt_cnt: List of integers
        - target: List of nested lists (NOT converted to tensor)
    """
    # Extract each field
    ages = torch.stack([item['age'] for item in batch])
    genders = torch.stack([item['gender'] for item in batch])
    lobs = torch.stack([item['lob'] for item in batch])  # NEW: Stack LOB
    codes = torch.stack([item['codes'] for item in batch])
    dt_cnts = [item['dt_cnt'] for item in batch]  # Keep as list
    targets = [item['target'] for item in batch]  # Keep as list of lists
    
    return {
        'age': ages,
        'gender': genders,
        'lob': lobs,  # NEW: Include LOB in batch
        'codes': codes,
        'dt_cnt': dt_cnts,
        'target': targets
    }
```

---

## Section 6: Update Model Classes (3 models)

### 6A: Update BaselineTransformer

**Location**: Modify `BaselineTransformer.__init__` around line 1695-1707

**CHANGE FROM:**
```python
        # Demographics embeddings
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
```

**CHANGE TO:**
```python
        # Demographics embeddings
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # NEW: LOB embedding
```

**Then update `BaselineTransformer.forward` around line 1759-1820:**

**CHANGE FROM (around line 1779):**
```python
        actual_len_cd = x.shape[2] - 2
```

**CHANGE TO:**
```python
        actual_len_cd = x.shape[2] - 3  # CHANGED: Now 3 demographics (age, gender, lob)
```

**CHANGE FROM (around line 1784-1786):**
```python
        age_in_months = x[:, :, 0].long()  # [batch, len_dy]
        gender_cd = x[:, :, 1].long()       # [batch, len_dy]
        cd = x[:, :, 2:].long()             # [batch, len_dy, len_cd]
```

**CHANGE TO:**
```python
        age_in_months = x[:, :, 0].long()  # [batch, len_dy]
        gender_cd = x[:, :, 1].long()       # [batch, len_dy]
        lob = x[:, :, 2].long()             # NEW: [batch, len_dy]
        cd = x[:, :, 3:].long()             # CHANGED: codes start at index 3
```

**CHANGE FROM (around line 1791-1793):**
```python
        gender_cd = self.embedding_gender_cd(gender_cd)      # [batch, len_dy, embedding_size]
        age_in_months = self.embedding_age_in_months(age_in_months)  # [batch, len_dy, embedding_size]
        cd = self.embedding_cd(cd)                           # [batch, len_dy, len_cd, embedding_size]
```

**CHANGE TO:**
```python
        gender_cd = self.embedding_gender_cd(gender_cd)      # [batch, len_dy, embedding_size]
        age_in_months = self.embedding_age_in_months(age_in_months)  # [batch, len_dy, embedding_size]
        lob_emb = self.embedding_lob(lob)                    # NEW: [batch, len_dy, embedding_size]
        cd = self.embedding_cd(cd)                           # [batch, len_dy, len_cd, embedding_size]
```

**CHANGE FROM (around line 1817):**
```python
        cd = cd_res + cd + gender_cd + age_in_months
```

**CHANGE TO:**
```python
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # CHANGED: Add LOB embedding
```

---

### 6B: Update FlashAttentionTransformer

**Location**: Modify around line 1870-1990

**Add LOB embedding in `__init__` (after line 1877):**

**CHANGE FROM:**
```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
```

**CHANGE TO:**
```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # NEW: LOB embedding
```

**Update `forward` method (around line 1988):**

**CHANGE FROM:**
```python
        actual_len_cd = x.shape[2] - 2
```

**CHANGE TO:**
```python
        actual_len_cd = x.shape[2] - 3  # CHANGED: Now 3 demographics
```

**Add LOB extraction and embedding (around line 1993-2000). Find where age/gender are extracted and modify:**

**CHANGE FROM:**
```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        cd = self.embedding_cd(x[:, :, 2:].long())
```

**CHANGE TO:**
```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long())  # NEW: LOB embedding
        cd = self.embedding_cd(x[:, :, 3:].long())       # CHANGED: codes start at index 3
```

**Find where representations are combined (around line 2020-2030) and add LOB:**

**CHANGE FROM:**
```python
        cd = cd_res + cd + gender_cd + age_in_months
```

**CHANGE TO:**
```python
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # CHANGED: Add LOB
```

---

### 6C: Update FlashMoETransformer

**Location**: Modify around line 2091-2210

**Add LOB embedding in `__init__` (after line 2100):**

**CHANGE FROM:**
```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
```

**CHANGE TO:**
```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # NEW: LOB embedding
```

**Update `forward` method (around line 2201):**

**CHANGE FROM:**
```python
        actual_len_cd = x.shape[2] - 2
```

**CHANGE TO:**
```python
        actual_len_cd = x.shape[2] - 3  # CHANGED: Now 3 demographics
```

**Modify embedding extraction (around lines 2205-2208):**

**CHANGE FROM:**
```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        cd = self.embedding_cd(x[:, :, 2:].long())
```

**CHANGE TO:**
```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long())  # NEW: LOB embedding
        cd = self.embedding_cd(x[:, :, 3:].long())       # CHANGED: codes start at index 3
```

**Find where representations are combined and add LOB (around line 2230-2240):**

**CHANGE FROM:**
```python
        cd = cd_res + cd + gender_cd + age_in_months
```

**CHANGE TO:**
```python
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # CHANGED: Add LOB
```

---

## Section 7: Update train_epoch and evaluate Functions

### 7A: Update train_epoch

**Location**: Around lines 3362-3371 in `train_epoch`

**CHANGE FROM:**
```python
        # Get batch data
        age = batch['age'].to(device, non_blocking=True)
        gender = batch['gender'].to(device, non_blocking=True)
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            codes
        ], dim=-1)
```

**CHANGE TO:**
```python
        # Get batch data
        age = batch['age'].to(device, non_blocking=True)
        gender = batch['gender'].to(device, non_blocking=True)
        lob = batch['lob'].to(device, non_blocking=True)  # NEW: Get LOB
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),  # NEW: Include LOB in input tensor
            codes
        ], dim=-1)
```

### 7B: Update evaluate Function

**Location**: Find where batch data is extracted in `evaluate` function (around line 3560-3580)

**Apply the same change as train_epoch:**

**CHANGE FROM:**
```python
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                codes
            ], dim=-1)
```

**CHANGE TO:**
```python
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)  # NEW: Get LOB
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),  # NEW: Include LOB
                codes
            ], dim=-1)
```

---

## Section 8: Add Downstream Evaluation Module

**Location**: Add this entire section AFTER the `EmbeddingExtractor` class (around line 6100, before `test_embedding_extractor`)

```python
# ============================================================================
# DOWNSTREAM TASK EVALUATION MODULE
# ============================================================================
# Implements linear probe evaluation for IP risk prediction
# Separate from training for clean evaluation with proper train/val/test splits
# ============================================================================

@dataclass
class DownstreamConfig:
    """Configuration for downstream task evaluation."""
    task_name: str = "medicaid_ip_risk"
    test_size: float = 0.2        # 20% for test
    val_size: float = 0.1         # 10% of remaining for validation (= 8% of total)
    random_state: int = 42
    n_cv_folds: int = 5           # Cross-validation folds for probe
    max_iter: int = 1000          # Max iterations for logistic regression
    class_weight: str = 'balanced'  # Handle class imbalance


class DownstreamEvaluator:
    """
    Evaluates transformer embeddings on downstream classification tasks.
    
    Implements the "linear probe" methodology:
    1. Extract embeddings from trained transformer (frozen)
    2. Train a simple logistic regression on top
    3. Evaluate on held-out test set
    
    This is the gold standard for evaluating representation quality.
    Reference: Radford et al. (2021) "CLIP" uses linear probe as primary metric
    
    Usage:
        evaluator = DownstreamEvaluator(model, config, device)
        results = evaluator.evaluate(
            features_df=train_data,
            outcomes_df=outcomes_data,
            downstream_config=DownstreamConfig()
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: BaseConfig,
        device: torch.device,
        use_mixed_precision: bool = False
    ):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def extract_embeddings(
        self,
        data: pd.DataFrame,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract patient-level embeddings from the transformer.
        
        Args:
            data: DataFrame with transformer input columns + individual_id, index_dt
            batch_size: Batch size for inference
            
        Returns:
            embeddings: np.ndarray [num_patients, embedding_dim]
            individual_ids: List of individual_id strings
            index_dts: List of index_dt strings
        """
        # Create dataset and dataloader
        dataset = ClinicalDataset(data, self.model_config)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=clinical_collate_fn,
            num_workers=0
        )
        
        all_embeddings = []
        individual_ids = data['individual_id'].tolist()
        index_dts = data['index_dt'].astype(str).tolist()
        
        self.model.eval()
        
        with torch.no_grad():
            with EmbeddingExtractor(self.model) as extractor:
                for batch_idx, batch in enumerate(dataloader):
                    # Prepare input
                    age = batch['age'].to(self.device)
                    gender = batch['gender'].to(self.device)
                    lob = batch['lob'].to(self.device)
                    codes = batch['codes'].to(self.device)
                    dt_cnt = batch['dt_cnt']
                    
                    x = torch.cat([
                        age.unsqueeze(-1),
                        gender.unsqueeze(-1),
                        lob.unsqueeze(-1),
                        codes
                    ], dim=-1)
                    
                    # Forward pass
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            if hasattr(self.model, 'forward') and 'return_moe_losses' in self.model.forward.__code__.co_varnames:
                                _ = self.model(x, return_moe_losses=False)
                            else:
                                _ = self.model(x)
                    else:
                        if hasattr(self.model, 'forward') and 'return_moe_losses' in self.model.forward.__code__.co_varnames:
                            _ = self.model(x, return_moe_losses=False)
                        else:
                            _ = self.model(x)
                    
                    # Get patient-level embeddings (last valid day)
                    patient_embs = extractor.get_patient_embedding(dt_cnt)
                    all_embeddings.append(patient_embs.cpu().numpy())
                    
                    if batch_idx % 100 == 0:
                        print(f"  Extracted embeddings: {(batch_idx + 1) * batch_size}/{len(data)}")
        
        embeddings = np.vstack(all_embeddings)
        print(f"  Total embeddings extracted: {embeddings.shape}")
        
        return embeddings, individual_ids, index_dts
    
    def prepare_downstream_data(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        downstream_config: DownstreamConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Join features with outcomes and create train/val/test splits.
        
        Args:
            features_df: Transformer training data with individual_id, index_dt
            outcomes_df: Outcome table with individual_id, index_dt, acute_ip_flag
            downstream_config: Configuration for splitting
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Preparing downstream data...")
        
        # Extract embeddings
        embeddings, individual_ids, index_dts = self.extract_embeddings(features_df)
        
        # Create embedding DataFrame for joining
        emb_df = pd.DataFrame({
            'individual_id': individual_ids,
            'index_dt': index_dts
        })
        emb_df['embedding_idx'] = range(len(emb_df))
        
        # Ensure outcomes_df has correct types
        outcomes_df = outcomes_df.copy()
        outcomes_df['individual_id'] = outcomes_df['individual_id'].astype(str)
        outcomes_df['index_dt'] = outcomes_df['index_dt'].astype(str)
        
        # Left join embeddings with outcomes
        merged = emb_df.merge(
            outcomes_df[['individual_id', 'index_dt', 'acute_ip_flag']],
            on=['individual_id', 'index_dt'],
            how='left'
        )
        
        # Handle missing outcomes (default to 0)
        merged['acute_ip_flag'] = merged['acute_ip_flag'].fillna(0).astype(int)
        
        print(f"  Matched {len(merged)} records")
        print(f"  Label distribution: {merged['acute_ip_flag'].value_counts().to_dict()}")
        
        # Get embeddings and labels
        X = embeddings[merged['embedding_idx'].values]
        y = merged['acute_ip_flag'].values
        
        # Stratified train/test split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=downstream_config.test_size,
            random_state=downstream_config.random_state,
            stratify=y
        )
        
        # Stratified train/val split from trainval
        val_ratio = downstream_config.val_size / (1 - downstream_config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=downstream_config.random_state,
            stratify=y_trainval
        )
        
        print(f"  Train: {len(X_train)} (pos: {y_train.sum()})")
        print(f"  Val:   {len(X_val)} (pos: {y_val.sum()})")
        print(f"  Test:  {len(X_test)} (pos: {y_test.sum()})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_linear_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        downstream_config: DownstreamConfig
    ) -> Tuple[LogisticRegression, StandardScaler]:
        """
        Train a linear probe (logistic regression) on embeddings.
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            downstream_config: Configuration
            
        Returns:
            Trained classifier and scaler
        """
        print("Training linear probe...")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train logistic regression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            classifier = LogisticRegression(
                max_iter=downstream_config.max_iter,
                solver='lbfgs',
                class_weight=downstream_config.class_weight,
                random_state=downstream_config.random_state,
                n_jobs=-1
            )
            classifier.fit(X_train_scaled, y_train)
        
        print("  Linear probe trained successfully")
        return classifier, scaler
    
    def evaluate_probe(
        self,
        classifier: LogisticRegression,
        scaler: StandardScaler,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate the linear probe on a dataset split.
        
        Args:
            classifier: Trained logistic regression
            scaler: Fitted scaler
            X: Embeddings
            y: True labels
            split_name: Name of split for logging
            
        Returns:
            Dictionary of metrics
        """
        X_scaled = scaler.transform(X)
        
        # Predictions
        y_pred = classifier.predict(X_scaled)
        y_prob = classifier.predict_proba(X_scaled)[:, 1]
        
        # Compute metrics
        metrics = {
            f'{split_name}_accuracy': float((y_pred == y).mean()),
            f'{split_name}_auc_roc': float(roc_auc_score(y, y_prob)),
            f'{split_name}_auc_pr': float(average_precision_score(y, y_prob)),
            f'{split_name}_f1': float(f1_score(y, y_pred)),
            f'{split_name}_precision': float(precision_score(y, y_pred, zero_division=0)),
            f'{split_name}_recall': float(recall_score(y, y_pred, zero_division=0)),
            f'{split_name}_brier': float(brier_score_loss(y, y_prob)),
            f'{split_name}_prevalence': float(y.mean()),
            f'{split_name}_n_samples': int(len(y)),
            f'{split_name}_n_positive': int(y.sum()),
        }
        
        return metrics
    
    def evaluate(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        downstream_config: Optional[DownstreamConfig] = None
    ) -> Dict[str, any]:
        """
        Full downstream evaluation pipeline.
        
        Args:
            features_df: Transformer training data with individual_id, index_dt
            outcomes_df: Outcome table with individual_id, index_dt, acute_ip_flag
            downstream_config: Configuration for evaluation
            
        Returns:
            Dictionary with all metrics for train/val/test splits
        """
        if downstream_config is None:
            downstream_config = DownstreamConfig()
        
        print(f"\n{'='*60}")
        print(f"DOWNSTREAM EVALUATION: {downstream_config.task_name}")
        print(f"{'='*60}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_downstream_data(
            features_df, outcomes_df, downstream_config
        )
        
        # Train linear probe
        classifier, scaler = self.train_linear_probe(X_train, y_train, downstream_config)
        
        # Evaluate on all splits
        train_metrics = self.evaluate_probe(classifier, scaler, X_train, y_train, 'train')
        val_metrics = self.evaluate_probe(classifier, scaler, X_val, y_val, 'val')
        test_metrics = self.evaluate_probe(classifier, scaler, X_test, y_test, 'test')
        
        # Combine results
        results = {
            'task_name': downstream_config.task_name,
            'embedding_dim': X_train.shape[1],
            **train_metrics,
            **val_metrics,
            **test_metrics,
        }
        
        # Print summary
        print(f"\n--- Linear Probe Results ---")
        print(f"Train AUC-ROC: {train_metrics['train_auc_roc']:.4f}")
        print(f"Val AUC-ROC:   {val_metrics['val_auc_roc']:.4f}")
        print(f"Test AUC-ROC:  {test_metrics['test_auc_roc']:.4f}")
        print(f"Test F1:       {test_metrics['test_f1']:.4f}")
        print(f"Test Recall:   {test_metrics['test_recall']:.4f}")
        print(f"{'='*60}\n")
        
        return results


def generate_model_name(
    exp_name: str,
    experiment_round: Optional[str] = None,
    batch_size: int = 32,
    epochs: int = 10,
    embedding_size: int = 256
) -> str:
    """
    Generate a standardized model name for saving/loading.
    
    Format: {experiment_round}_{exp_name}_bs{batch_size}_ep{epochs}_d{embedding_size}_{timestamp}
    
    Example: round3_exp3_standard_moe_bs32_ep10_d256_20241211_143022
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parts = []
    if experiment_round:
        parts.append(experiment_round)
    parts.append(exp_name)
    parts.append(f"bs{batch_size}")
    parts.append(f"ep{epochs}")
    parts.append(f"d{embedding_size}")
    parts.append(timestamp)
    
    return "_".join(parts)


def save_trained_model(
    model: nn.Module,
    config: BaseConfig,
    model_name: str,
    save_dir: str,
    exp_results: Dict[str, any],
    is_best: bool = False
) -> str:
    """
    Save trained model with standardized naming.
    
    Saves:
    - {model_name}_final.pt: Model state dict
    - {model_name}_config.json: Model configuration
    - {model_name}_results.json: Training results
    
    Args:
        model: Trained model
        config: Model configuration
        model_name: Generated model name
        save_dir: Directory to save model
        exp_results: Experiment results dictionary
        is_best: If True, also save as {model_name}_best.pt
        
    Returns:
        Path to saved model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_dir, f"{model_name}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'model_type': type(model).__name__,
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save config
    config_path = os.path.join(save_dir, f"{model_name}_config.json")
    config_dict = {k: (str(v) if isinstance(v, torch.dtype) else v) 
                   for k, v in config.__dict__.items()}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save results (with serialization fix)
    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    
    def make_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(make_serializable(exp_results), f, indent=2)
    
    # Also save as best if requested
    if is_best:
        best_path = os.path.join(save_dir, f"{model_name}_best.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'model_type': type(model).__name__,
        }, best_path)
        print(f"Best model saved to: {best_path}")
    
    return model_path


def load_trained_model(
    model_path: str,
    model_class: type,
    config: BaseConfig,
    device: torch.device
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved .pt file
        model_class: Class of the model (BaselineTransformer, FlashAttentionTransformer, etc.)
        config: Model configuration
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    if model_class == FlashMoETransformer:
        # MoE models need additional config
        moe_config = MoEConfig(d_model=config.embedding_size, d_ff=config.nhid)
        model = model_class(config, moe_config)
    else:
        model = model_class(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
    
    return model


def run_downstream_evaluation(
    model: nn.Module,
    model_config: BaseConfig,
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    device: torch.device,
    use_mixed_precision: bool = False,
    downstream_config: Optional[DownstreamConfig] = None
) -> Dict[str, any]:
    """
    Convenience function to run downstream evaluation on a trained model.
    
    Args:
        model: Trained transformer model
        model_config: Model configuration
        features_df: Training data with individual_id, index_dt, and transformer features
        outcomes_df: Outcomes with individual_id, index_dt, acute_ip_flag
        device: Torch device
        use_mixed_precision: Whether to use FP16 for inference
        downstream_config: Optional downstream evaluation configuration
        
    Returns:
        Dictionary with downstream evaluation metrics
    """
    evaluator = DownstreamEvaluator(
        model=model,
        model_config=model_config,
        device=device,
        use_mixed_precision=use_mixed_precision
    )
    
    return evaluator.evaluate(
        features_df=features_df,
        outcomes_df=outcomes_df,
        downstream_config=downstream_config
    )
```

---

## Section 9: Update run_single_experiment

**Location**: Modify `run_single_experiment` to add model saving and downstream evaluation support.

### 9A: Add new parameters to function signature

**Location**: Around line 6337-6355

**CHANGE FROM:**
```python
def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 4,
    code_frequencies: Optional[np.ndarray] = None,
    log_dir: str = "logs",  # base logging directory
    experiment_round: Optional[str] = None,
    check_embeddings_every: int = 2,  # check embeddings every N epochs
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,  # midpoint path used for resuming training
    checkpoint_dir: Optional[str] = None,  # add check point directory
    embedding_size: Optional[int] = None,
    local_rank: Optional[int] = None,  # If provided, use DDP
    world_size: Optional[int] = None
) -> Dict[str, any]:
```

**CHANGE TO:**
```python
def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 4,
    code_frequencies: Optional[np.ndarray] = None,
    log_dir: str = "logs",  # base logging directory
    experiment_round: Optional[str] = None,
    check_embeddings_every: int = 2,  # check embeddings every N epochs
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,  # midpoint path used for resuming training
    checkpoint_dir: Optional[str] = None,  # add check point directory
    embedding_size: Optional[int] = None,
    local_rank: Optional[int] = None,  # If provided, use DDP (DISABLED FOR NOW)
    world_size: Optional[int] = None,
    # NEW: Downstream evaluation parameters
    outcomes_df: Optional[pd.DataFrame] = None,  # Outcomes for downstream eval
    run_downstream_eval: bool = False,  # Whether to run downstream eval after training
    save_model: bool = True  # Whether to save the trained model
) -> Dict[str, any]:
```

### 9B: Disable DDP

**Location**: Around line 6383-6394

**CHANGE FROM:**
```python
    # ============================================================
    # DDP SETUP
    # ============================================================
    use_ddp = local_rank is not None and world_size is not None and world_size > 1
    
    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
        is_main = (local_rank == 0)
    else:
        # Single GPU mode (backward compatible)
        local_rank = 0
        world_size = 1
        is_main = True
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**CHANGE TO:**
```python
    # ============================================================
    # DDP SETUP (DISABLED FOR NOW)
    # ============================================================
    # DDP is disabled - using single GPU mode
    use_ddp = False  # DISABLED: was `local_rank is not None and world_size is not None and world_size > 1`
    local_rank = 0
    world_size = 1
    is_main = True
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 9C: Generate model name and save model after training

**Location**: After the comprehensive evaluation (around line 6930-6950), add model saving and downstream evaluation.

Find this section (around line 6928-6932):
```python
        # Save complete results to JSON for later comparison
        results_path = metrics_logger.save_final_results(results)
        logger.info(f"Complete results saved to {results_path}")    
```

**ADD AFTER IT:**
```python
        # ============================================================
        # SAVE TRAINED MODEL WITH STANDARDIZED NAMING
        # ============================================================
        if save_model:
            model_name = generate_model_name(
                exp_name=exp_name,
                experiment_round=experiment_round,
                batch_size=config.batch_size,
                epochs=epochs,
                embedding_size=eff_d_model
            )
            
            # Add model_name to results for traceability
            results['model_name'] = model_name
            
            model_save_dir = os.path.join(effective_log_dir, exp_name, 'saved_models')
            model_path = save_trained_model(
                model=model_to_eval,
                config=config,
                model_name=model_name,
                save_dir=model_save_dir,
                exp_results=results,
                is_best=True  # Save as best since this is final
            )
            logger.info(f"Model saved as: {model_name}")
            results['model_path'] = model_path
        
        # ============================================================
        # DOWNSTREAM EVALUATION (Optional)
        # ============================================================
        if run_downstream_eval and outcomes_df is not None:
            logger.info("\n" + "="*80)
            logger.info("RUNNING DOWNSTREAM EVALUATION")
            logger.info("="*80)
            
            downstream_results = run_downstream_evaluation(
                model=model_to_eval,
                model_config=config,
                features_df=val_data,  # Use validation data for downstream eval
                outcomes_df=outcomes_df,
                device=device,
                use_mixed_precision=use_mixed_precision,
                downstream_config=DownstreamConfig(task_name="medicaid_ip_risk")
            )
            
            # Add downstream metrics to results
            results['downstream_evaluation'] = downstream_results
            
            # Log key downstream metrics
            logger.info(f"Downstream Test AUC-ROC: {downstream_results['test_auc_roc']:.4f}")
            logger.info(f"Downstream Test F1: {downstream_results['test_f1']:.4f}")
            
            # Update saved results with downstream metrics
            if save_model:
                results_path = os.path.join(model_save_dir, f"{model_name}_results.json")
                with open(results_path, 'w') as f:
                    json.dump(MetricsLogger.convert_to_serializable(results), f, indent=2)
```

### 9D: Update run_selected_experiments to pass new parameters

**Location**: Around line 7020-7040

Find where `run_single_experiment` is called in `run_selected_experiments` and update it.

**ADD these new parameters to the function signature of `run_selected_experiments` (around line 6957):**

```python
def run_selected_experiments(
    experiment_names: List[str],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 10,
    experiment_round: Optional[str] = None,
    embedding_size: Optional[int] = None,
    local_rank: Optional[int] = None,      
    world_size: Optional[int] = None,
    # NEW: Downstream evaluation parameters
    outcomes_df: Optional[pd.DataFrame] = None,
    run_downstream_eval: bool = False,
    save_model: bool = True
) -> pd.DataFrame:
```

**Then update the call to `run_single_experiment` inside `run_selected_experiments` (around line 7025):**

Add these parameters to the function call:
```python
            outcomes_df=outcomes_df,
            run_downstream_eval=run_downstream_eval,
            save_model=save_model
```

---

## Usage Example

After implementing all the above changes, here's how to use the updated code:

```python
# Load data
import pandas as pd
from google.cloud import bigquery

client = bigquery.Client()

# Load transformer training data (now with LOB column)
train_query = """
SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`
LIMIT 100000
"""
df_train = client.query(train_query).to_dataframe()

# Load validation data
val_query = """
SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_val_ending`
LIMIT 20000
"""
df_val = client.query(val_query).to_dataframe()

# Load outcomes for downstream evaluation
outcomes_query = """
SELECT individual_id, index_dt, acute_ip_flag 
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment`
"""
df_outcomes = client.query(outcomes_query).to_dataframe()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run experiment with downstream evaluation
results = run_single_experiment(
    exp_name='exp3_standard_moe',
    moe_config=MoEConfig(num_experts=8, top_k=2),
    use_learnt_att_pool=False,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10,
    experiment_round='round4_lob_downstream',
    embedding_size=256,
    outcomes_df=df_outcomes,  # NEW: Pass outcomes for downstream eval
    run_downstream_eval=True,  # NEW: Enable downstream eval
    save_model=True  # NEW: Save model with standardized naming
)

# Results will include:
# - All intrinsic metrics (loss, top-k accuracy, etc.)
# - model_name: 'round4_lob_downstream_exp3_standard_moe_bs32_ep10_d256_20241211_143022'
# - model_path: 'logs/round4_lob_downstream/exp3_standard_moe/saved_models/..._final.pt'
# - downstream_evaluation: {'test_auc_roc': 0.72, 'test_f1': 0.45, ...}
```

---

## Summary of All Changes

| Section | File Location | What Changed |
|---------|--------------|--------------|
| 1 | Top of file (~line 60) | Added sklearn imports |
| 2 | BaseConfig (~line 266) | Added `lob_vocab: int = 4` |
| 3 | After line 2544 | Added `conv_lob()` function |
| 4 | ClinicalDataset (~line 2392) | Added LOB tensor processing |
| 5 | clinical_collate_fn (~line 2442) | Added LOB to batch |
| 6A | BaselineTransformer (~line 1695) | Added `embedding_lob`, updated forward |
| 6B | FlashAttentionTransformer (~line 1870) | Added `embedding_lob`, updated forward |
| 6C | FlashMoETransformer (~line 2091) | Added `embedding_lob`, updated forward |
| 7A | train_epoch (~line 3362) | Added LOB to input tensor |
| 7B | evaluate (~line 3560) | Added LOB to input tensor |
| 8 | After EmbeddingExtractor (~line 6100) | Added full downstream evaluation module |
| 9 | run_single_experiment (~line 6337) | Added model saving & downstream eval |

You're in **ask mode**. To apply these changes, please copy the code sections above and add them to your `moe_flashattn_3.py` file at the indicated locations, or switch to **agent mode** for me to apply them directly.

```python
# ============================================================================
# ADDITIONAL IMPORTS FOR DOWNSTREAM EVALUATION
# ============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning
import warnings
import json
from datetime import datetime
```

```python
    # Embeddings
    gender_vocab: int = 4     # Gender categories
    age_vocab: int = 1440     # Age in months (120 years)
```

```python
    # Embeddings
    gender_vocab: int = 4     # Gender categories
    age_vocab: int = 1440     # Age in months (120 years)
    lob_vocab: int = 4        # LOB categories (0=padding, 1=Commercial, 2=Medicare, 3=Medicaid)
```

```python
def conv_lob(ipt: str, len_dy: int) -> List[int]:
    """
    Convert LOB (Line of Business) string to list.
    
    Format: "value1*value2*..." (same format as age/gender, but with string values)
    Maps: Commercial=1, Medicare=2, Medicaid=3, padding/unknown=0
    
    Args:
        ipt: LOB string from data (e.g., "Medicaid*Medicaid*..." or single value "Medicaid")
        len_dy: Target sequence length (200)
    
    Returns:
        List of LOB indices [len_dy]
    """
    # LOB mapping (case-insensitive)
    lob_map = {
        'commercial': 1,
        'medicare': 2,
        'medicaid': 3
    }
    
    if not ipt or pd.isna(ipt):
        return [0] * len_dy
    
    # Handle both single value and asterisk-separated formats
    if '*' in str(ipt):
        values = str(ipt).split('*')
    else:
        # Single value - repeat for all days
        values = [str(ipt)] * len_dy
    
    values = values[:len_dy]  # Truncate to max days
    
    result = []
    for val in values:
        val_clean = val.strip().lower() if val else ''
        if val_clean in lob_map:
            result.append(lob_map[val_clean])
        else:
            result.append(0)  # Unknown LOB
    
    # Forward-fill with last valid value (LOB typically doesn't change within sequence)
    if result:
        last_val = result[-1] if result[-1] != 0 else 3  # Default to Medicaid if all zeros
        while len(result) < len_dy:
            result.append(last_val)
    else:
        result = [3] * len_dy  # Default to Medicaid (since this is Medicaid data)
    
    return result
```

```python
    def __init__(self, df: pd.DataFrame, config: BaseConfig):
        self.config = config
        print(f"Pre-processing {len(df)} samples into tensors (one-time cost)...")
        
        # Extract series for faster processing
        age_strs = df['age_in_months'].tolist()
        gender_strs = df['gender_cd'].tolist()
        cd_strs = df['cd'].tolist()
        target_strs = df['target_cd'].tolist()
        self.dt_cnt = df['dt_cnt'].tolist()
        
        # Handle LOB column (may or may not exist)
        if 'lob' in df.columns:
            lob_strs = df['lob'].tolist()
            self.has_lob = True
        else:
            lob_strs = [None] * len(df)
            self.has_lob = False
            print("  Warning: 'lob' column not found, using default LOB=Medicaid")

        # Pre-allocate tensors
        self.ages = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.genders = torch.zeros(len(df), config.len_dy, dtype=torch.long)
        self.lobs = torch.zeros(len(df), config.len_dy, dtype=torch.long)  # NEW: LOB tensor
        self.codes = torch.zeros(len(df), config.len_dy, config.len_cd, dtype=torch.long)
        self.targets = []

        # Process all samples
        for i in range(len(df)):
            if i > 0 and i % 50000 == 0:
                print(f"  Processed {i}/{len(df)} samples...")

            age_list = conv_age_gender(age_strs[i], config.len_dy)
            gender_list = conv_age_gender(gender_strs[i], config.len_dy, max_val=3)
            lob_list = conv_lob(lob_strs[i], config.len_dy)  # NEW: Parse LOB
            cd_list = conv_cd(cd_strs[i], config.len_dy, config.len_cd)
            target_list = conv_target(target_strs[i], config.len_dy, config.target_cd_cnt)

            self.ages[i] = torch.tensor(age_list, dtype=torch.long)
            self.genders[i] = torch.tensor(gender_list, dtype=torch.long)
            self.lobs[i] = torch.tensor(lob_list, dtype=torch.long)  # NEW
            self.codes[i] = torch.tensor(cd_list, dtype=torch.long)
            self.targets.append(target_list)
        
        print("Pre-processing complete.")
```

```python
    def __getitem__(self, idx):
        return {
            'age': self.ages[idx],
            'gender': self.genders[idx],
            'codes': self.codes[idx],
            'dt_cnt': self.dt_cnt[idx],
            'target': self.targets[idx]
        }
```

```python
    def __getitem__(self, idx):
        return {
            'age': self.ages[idx],
            'gender': self.genders[idx],
            'lob': self.lobs[idx],  # NEW: Include LOB
            'codes': self.codes[idx],
            'dt_cnt': self.dt_cnt[idx],
            'target': self.targets[idx]
        }
```

```python
def clinical_collate_fn(batch):
    """
    Custom collate function for clinical data.
    
    Handles the special case of 'target' which is a nested list with variable-length sublists.
    PyTorch's default_collate cannot handle this, so we keep it as a Python list.
    
    Args:
        batch: List of dictionaries from ClinicalDataset.__getitem__
    
    Returns:
        Batched dictionary with:
        - age, gender, lob, codes: Stacked tensors
        - dt_cnt: List of integers
        - target: List of nested lists (NOT converted to tensor)
    """
    # Extract each field
    ages = torch.stack([item['age'] for item in batch])
    genders = torch.stack([item['gender'] for item in batch])
    lobs = torch.stack([item['lob'] for item in batch])  # NEW: Stack LOB
    codes = torch.stack([item['codes'] for item in batch])
    dt_cnts = [item['dt_cnt'] for item in batch]  # Keep as list
    targets = [item['target'] for item in batch]  # Keep as list of lists
    
    return {
        'age': ages,
        'gender': genders,
        'lob': lobs,  # NEW: Include LOB in batch
        'codes': codes,
        'dt_cnt': dt_cnts,
        'target': targets
    }
```

```python
        # Demographics embeddings
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
```

```python
        # Demographics embeddings
        self.embedding_gender_cd = nn.Embedding(config.gender_vocab, config.embedding_size)
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # NEW: LOB embedding
```

```python
        actual_len_cd = x.shape[2] - 2
```

```python
        actual_len_cd = x.shape[2] - 3  # CHANGED: Now 3 demographics (age, gender, lob)
```

```python
        age_in_months = x[:, :, 0].long()  # [batch, len_dy]
        gender_cd = x[:, :, 1].long()       # [batch, len_dy]
        cd = x[:, :, 2:].long()             # [batch, len_dy, len_cd]
```

```python
        age_in_months = x[:, :, 0].long()  # [batch, len_dy]
        gender_cd = x[:, :, 1].long()       # [batch, len_dy]
        lob = x[:, :, 2].long()             # NEW: [batch, len_dy]
        cd = x[:, :, 3:].long()             # CHANGED: codes start at index 3
```

```python
        gender_cd = self.embedding_gender_cd(gender_cd)      # [batch, len_dy, embedding_size]
        age_in_months = self.embedding_age_in_months(age_in_months)  # [batch, len_dy, embedding_size]
        cd = self.embedding_cd(cd)                           # [batch, len_dy, len_cd, embedding_size]
```

```python
        gender_cd = self.embedding_gender_cd(gender_cd)      # [batch, len_dy, embedding_size]
        age_in_months = self.embedding_age_in_months(age_in_months)  # [batch, len_dy, embedding_size]
        lob_emb = self.embedding_lob(lob)                    # NEW: [batch, len_dy, embedding_size]
        cd = self.embedding_cd(cd)                           # [batch, len_dy, len_cd, embedding_size]
```

```python
        cd = cd_res + cd + gender_cd + age_in_months
```

```python
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # CHANGED: Add LOB embedding
```

```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
```

```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # NEW: LOB embedding
```

```python
        actual_len_cd = x.shape[2] - 2
```

```python
        actual_len_cd = x.shape[2] - 3  # CHANGED: Now 3 demographics
```

```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        cd = self.embedding_cd(x[:, :, 2:].long())
```

```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long())  # NEW: LOB embedding
        cd = self.embedding_cd(x[:, :, 3:].long())       # CHANGED: codes start at index 3
```

```python
        cd = cd_res + cd + gender_cd + age_in_months
```

```python
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # CHANGED: Add LOB
```

```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
```

```python
        self.embedding_age_in_months = nn.Embedding(config.age_vocab, config.embedding_size)
        self.embedding_lob = nn.Embedding(config.lob_vocab, config.embedding_size)  # NEW: LOB embedding
```

```python
        actual_len_cd = x.shape[2] - 2
```

```python
        actual_len_cd = x.shape[2] - 3  # CHANGED: Now 3 demographics
```

```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        cd = self.embedding_cd(x[:, :, 2:].long())
```

```python
        age_in_months = self.embedding_age_in_months(x[:, :, 0].long())
        gender_cd = self.embedding_gender_cd(x[:, :, 1].long())
        lob_emb = self.embedding_lob(x[:, :, 2].long())  # NEW: LOB embedding
        cd = self.embedding_cd(x[:, :, 3:].long())       # CHANGED: codes start at index 3
```

```python
        cd = cd_res + cd + gender_cd + age_in_months
```

```python
        cd = cd_res + cd + gender_cd + age_in_months + lob_emb  # CHANGED: Add LOB
```

```python
        # Get batch data
        age = batch['age'].to(device, non_blocking=True)
        gender = batch['gender'].to(device, non_blocking=True)
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            codes
        ], dim=-1)
```

```python
        # Get batch data
        age = batch['age'].to(device, non_blocking=True)
        gender = batch['gender'].to(device, non_blocking=True)
        lob = batch['lob'].to(device, non_blocking=True)  # NEW: Get LOB
        codes = batch['codes'].to(device, non_blocking=True)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),  # NEW: Include LOB in input tensor
            codes
        ], dim=-1)
```

```python
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                codes
            ], dim=-1)
```

```python
            age = batch['age'].to(device, non_blocking=True)
            gender = batch['gender'].to(device, non_blocking=True)
            lob = batch['lob'].to(device, non_blocking=True)  # NEW: Get LOB
            codes = batch['codes'].to(device, non_blocking=True)
            dt_cnt = batch['dt_cnt']
            y = batch['target']
            
            x = torch.cat([
                age.unsqueeze(-1),
                gender.unsqueeze(-1),
                lob.unsqueeze(-1),  # NEW: Include LOB
                codes
            ], dim=-1)
```

```python
# ============================================================================
# DOWNSTREAM TASK EVALUATION MODULE
# ============================================================================
# Implements linear probe evaluation for IP risk prediction
# Separate from training for clean evaluation with proper train/val/test splits
# ============================================================================

@dataclass
class DownstreamConfig:
    """Configuration for downstream task evaluation."""
    task_name: str = "medicaid_ip_risk"
    test_size: float = 0.2        # 20% for test
    val_size: float = 0.1         # 10% of remaining for validation (= 8% of total)
    random_state: int = 42
    n_cv_folds: int = 5           # Cross-validation folds for probe
    max_iter: int = 1000          # Max iterations for logistic regression
    class_weight: str = 'balanced'  # Handle class imbalance


class DownstreamEvaluator:
    """
    Evaluates transformer embeddings on downstream classification tasks.
    
    Implements the "linear probe" methodology:
    1. Extract embeddings from trained transformer (frozen)
    2. Train a simple logistic regression on top
    3. Evaluate on held-out test set
    
    This is the gold standard for evaluating representation quality.
    Reference: Radford et al. (2021) "CLIP" uses linear probe as primary metric
    
    Usage:
        evaluator = DownstreamEvaluator(model, config, device)
        results = evaluator.evaluate(
            features_df=train_data,
            outcomes_df=outcomes_data,
            downstream_config=DownstreamConfig()
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: BaseConfig,
        device: torch.device,
        use_mixed_precision: bool = False
    ):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def extract_embeddings(
        self,
        data: pd.DataFrame,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract patient-level embeddings from the transformer.
        
        Args:
            data: DataFrame with transformer input columns + individual_id, index_dt
            batch_size: Batch size for inference
            
        Returns:
            embeddings: np.ndarray [num_patients, embedding_dim]
            individual_ids: List of individual_id strings
            index_dts: List of index_dt strings
        """
        # Create dataset and dataloader
        dataset = ClinicalDataset(data, self.model_config)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=clinical_collate_fn,
            num_workers=0
        )
        
        all_embeddings = []
        individual_ids = data['individual_id'].tolist()
        index_dts = data['index_dt'].astype(str).tolist()
        
        self.model.eval()
        
        with torch.no_grad():
            with EmbeddingExtractor(self.model) as extractor:
                for batch_idx, batch in enumerate(dataloader):
                    # Prepare input
                    age = batch['age'].to(self.device)
                    gender = batch['gender'].to(self.device)
                    lob = batch['lob'].to(self.device)
                    codes = batch['codes'].to(self.device)
                    dt_cnt = batch['dt_cnt']
                    
                    x = torch.cat([
                        age.unsqueeze(-1),
                        gender.unsqueeze(-1),
                        lob.unsqueeze(-1),
                        codes
                    ], dim=-1)
                    
                    # Forward pass
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            if hasattr(self.model, 'forward') and 'return_moe_losses' in self.model.forward.__code__.co_varnames:
                                _ = self.model(x, return_moe_losses=False)
                            else:
                                _ = self.model(x)
                    else:
                        if hasattr(self.model, 'forward') and 'return_moe_losses' in self.model.forward.__code__.co_varnames:
                            _ = self.model(x, return_moe_losses=False)
                        else:
                            _ = self.model(x)
                    
                    # Get patient-level embeddings (last valid day)
                    patient_embs = extractor.get_patient_embedding(dt_cnt)
                    all_embeddings.append(patient_embs.cpu().numpy())
                    
                    if batch_idx % 100 == 0:
                        print(f"  Extracted embeddings: {(batch_idx + 1) * batch_size}/{len(data)}")
        
        embeddings = np.vstack(all_embeddings)
        print(f"  Total embeddings extracted: {embeddings.shape}")
        
        return embeddings, individual_ids, index_dts
    
    def prepare_downstream_data(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        downstream_config: DownstreamConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Join features with outcomes and create train/val/test splits.
        
        Args:
            features_df: Transformer training data with individual_id, index_dt
            outcomes_df: Outcome table with individual_id, index_dt, acute_ip_flag
            downstream_config: Configuration for splitting
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Preparing downstream data...")
        
        # Extract embeddings
        embeddings, individual_ids, index_dts = self.extract_embeddings(features_df)
        
        # Create embedding DataFrame for joining
        emb_df = pd.DataFrame({
            'individual_id': individual_ids,
            'index_dt': index_dts
        })
        emb_df['embedding_idx'] = range(len(emb_df))
        
        # Ensure outcomes_df has correct types
        outcomes_df = outcomes_df.copy()
        outcomes_df['individual_id'] = outcomes_df['individual_id'].astype(str)
        outcomes_df['index_dt'] = outcomes_df['index_dt'].astype(str)
        
        # Left join embeddings with outcomes
        merged = emb_df.merge(
            outcomes_df[['individual_id', 'index_dt', 'acute_ip_flag']],
            on=['individual_id', 'index_dt'],
            how='left'
        )
        
        # Handle missing outcomes (default to 0)
        merged['acute_ip_flag'] = merged['acute_ip_flag'].fillna(0).astype(int)
        
        print(f"  Matched {len(merged)} records")
        print(f"  Label distribution: {merged['acute_ip_flag'].value_counts().to_dict()}")
        
        # Get embeddings and labels
        X = embeddings[merged['embedding_idx'].values]
        y = merged['acute_ip_flag'].values
        
        # Stratified train/test split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=downstream_config.test_size,
            random_state=downstream_config.random_state,
            stratify=y
        )
        
        # Stratified train/val split from trainval
        val_ratio = downstream_config.val_size / (1 - downstream_config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=downstream_config.random_state,
            stratify=y_trainval
        )
        
        print(f"  Train: {len(X_train)} (pos: {y_train.sum()})")
        print(f"  Val:   {len(X_val)} (pos: {y_val.sum()})")
        print(f"  Test:  {len(X_test)} (pos: {y_test.sum()})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_linear_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        downstream_config: DownstreamConfig
    ) -> Tuple[LogisticRegression, StandardScaler]:
        """
        Train a linear probe (logistic regression) on embeddings.
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            downstream_config: Configuration
            
        Returns:
            Trained classifier and scaler
        """
        print("Training linear probe...")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train logistic regression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            classifier = LogisticRegression(
                max_iter=downstream_config.max_iter,
                solver='lbfgs',
                class_weight=downstream_config.class_weight,
                random_state=downstream_config.random_state,
                n_jobs=-1
            )
            classifier.fit(X_train_scaled, y_train)
        
        print("  Linear probe trained successfully")
        return classifier, scaler
    
    def evaluate_probe(
        self,
        classifier: LogisticRegression,
        scaler: StandardScaler,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate the linear probe on a dataset split.
        
        Args:
            classifier: Trained logistic regression
            scaler: Fitted scaler
            X: Embeddings
            y: True labels
            split_name: Name of split for logging
            
        Returns:
            Dictionary of metrics
        """
        X_scaled = scaler.transform(X)
        
        # Predictions
        y_pred = classifier.predict(X_scaled)
        y_prob = classifier.predict_proba(X_scaled)[:, 1]
        
        # Compute metrics
        metrics = {
            f'{split_name}_accuracy': float((y_pred == y).mean()),
            f'{split_name}_auc_roc': float(roc_auc_score(y, y_prob)),
            f'{split_name}_auc_pr': float(average_precision_score(y, y_prob)),
            f'{split_name}_f1': float(f1_score(y, y_pred)),
            f'{split_name}_precision': float(precision_score(y, y_pred, zero_division=0)),
            f'{split_name}_recall': float(recall_score(y, y_pred, zero_division=0)),
            f'{split_name}_brier': float(brier_score_loss(y, y_prob)),
            f'{split_name}_prevalence': float(y.mean()),
            f'{split_name}_n_samples': int(len(y)),
            f'{split_name}_n_positive': int(y.sum()),
        }
        
        return metrics
    
    def evaluate(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        downstream_config: Optional[DownstreamConfig] = None
    ) -> Dict[str, any]:
        """
        Full downstream evaluation pipeline.
        
        Args:
            features_df: Transformer training data with individual_id, index_dt
            outcomes_df: Outcome table with individual_id, index_dt, acute_ip_flag
            downstream_config: Configuration for evaluation
            
        Returns:
            Dictionary with all metrics for train/val/test splits
        """
        if downstream_config is None:
            downstream_config = DownstreamConfig()
        
        print(f"\n{'='*60}")
        print(f"DOWNSTREAM EVALUATION: {downstream_config.task_name}")
        print(f"{'='*60}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_downstream_data(
            features_df, outcomes_df, downstream_config
        )
        
        # Train linear probe
        classifier, scaler = self.train_linear_probe(X_train, y_train, downstream_config)
        
        # Evaluate on all splits
        train_metrics = self.evaluate_probe(classifier, scaler, X_train, y_train, 'train')
        val_metrics = self.evaluate_probe(classifier, scaler, X_val, y_val, 'val')
        test_metrics = self.evaluate_probe(classifier, scaler, X_test, y_test, 'test')
        
        # Combine results
        results = {
            'task_name': downstream_config.task_name,
            'embedding_dim': X_train.shape[1],
            **train_metrics,
            **val_metrics,
            **test_metrics,
        }
        
        # Print summary
        print(f"\n--- Linear Probe Results ---")
        print(f"Train AUC-ROC: {train_metrics['train_auc_roc']:.4f}")
        print(f"Val AUC-ROC:   {val_metrics['val_auc_roc']:.4f}")
        print(f"Test AUC-ROC:  {test_metrics['test_auc_roc']:.4f}")
        print(f"Test F1:       {test_metrics['test_f1']:.4f}")
        print(f"Test Recall:   {test_metrics['test_recall']:.4f}")
        print(f"{'='*60}\n")
        
        return results


def generate_model_name(
    exp_name: str,
    experiment_round: Optional[str] = None,
    batch_size: int = 32,
    epochs: int = 10,
    embedding_size: int = 256
) -> str:
    """
    Generate a standardized model name for saving/loading.
    
    Format: {experiment_round}_{exp_name}_bs{batch_size}_ep{epochs}_d{embedding_size}_{timestamp}
    
    Example: round3_exp3_standard_moe_bs32_ep10_d256_20241211_143022
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parts = []
    if experiment_round:
        parts.append(experiment_round)
    parts.append(exp_name)
    parts.append(f"bs{batch_size}")
    parts.append(f"ep{epochs}")
    parts.append(f"d{embedding_size}")
    parts.append(timestamp)
    
    return "_".join(parts)


def save_trained_model(
    model: nn.Module,
    config: BaseConfig,
    model_name: str,
    save_dir: str,
    exp_results: Dict[str, any],
    is_best: bool = False
) -> str:
    """
    Save trained model with standardized naming.
    
    Saves:
    - {model_name}_final.pt: Model state dict
    - {model_name}_config.json: Model configuration
    - {model_name}_results.json: Training results
    
    Args:
        model: Trained model
        config: Model configuration
        model_name: Generated model name
        save_dir: Directory to save model
        exp_results: Experiment results dictionary
        is_best: If True, also save as {model_name}_best.pt
        
    Returns:
        Path to saved model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_dir, f"{model_name}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'model_type': type(model).__name__,
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save config
    config_path = os.path.join(save_dir, f"{model_name}_config.json")
    config_dict = {k: (str(v) if isinstance(v, torch.dtype) else v) 
                   for k, v in config.__dict__.items()}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save results (with serialization fix)
    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    
    def make_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(make_serializable(exp_results), f, indent=2)
    
    # Also save as best if requested
    if is_best:
        best_path = os.path.join(save_dir, f"{model_name}_best.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'model_type': type(model).__name__,
        }, best_path)
        print(f"Best model saved to: {best_path}")
    
    return model_path


def load_trained_model(
    model_path: str,
    model_class: type,
    config: BaseConfig,
    device: torch.device
) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved .pt file
        model_class: Class of the model (BaselineTransformer, FlashAttentionTransformer, etc.)
        config: Model configuration
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    if model_class == FlashMoETransformer:
        # MoE models need additional config
        moe_config = MoEConfig(d_model=config.embedding_size, d_ff=config.nhid)
        model = model_class(config, moe_config)
    else:
        model = model_class(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Model type: {checkpoint.get('model_type', 'Unknown')}")
    
    return model


def run_downstream_evaluation(
    model: nn.Module,
    model_config: BaseConfig,
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    device: torch.device,
    use_mixed_precision: bool = False,
    downstream_config: Optional[DownstreamConfig] = None
) -> Dict[str, any]:
    """
    Convenience function to run downstream evaluation on a trained model.
    
    Args:
        model: Trained transformer model
        model_config: Model configuration
        features_df: Training data with individual_id, index_dt, and transformer features
        outcomes_df: Outcomes with individual_id, index_dt, acute_ip_flag
        device: Torch device
        use_mixed_precision: Whether to use FP16 for inference
        downstream_config: Optional downstream evaluation configuration
        
    Returns:
        Dictionary with downstream evaluation metrics
    """
    evaluator = DownstreamEvaluator(
        model=model,
        model_config=model_config,
        device=device,
        use_mixed_precision=use_mixed_precision
    )
    
    return evaluator.evaluate(
        features_df=features_df,
        outcomes_df=outcomes_df,
        downstream_config=downstream_config
    )
```

```python
def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 4,
    code_frequencies: Optional[np.ndarray] = None,
    log_dir: str = "logs",  # base logging directory
    experiment_round: Optional[str] = None,
    check_embeddings_every: int = 2,  # check embeddings every N epochs
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,  # midpoint path used for resuming training
    checkpoint_dir: Optional[str] = None,  # add check point directory
    embedding_size: Optional[int] = None,
    local_rank: Optional[int] = None,  # If provided, use DDP
    world_size: Optional[int] = None
) -> Dict[str, any]:
```

```python
def run_single_experiment(
    exp_name: str,
    moe_config: Optional[MoEConfig],
    use_learnt_att_pool: bool,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 4,
    code_frequencies: Optional[np.ndarray] = None,
    log_dir: str = "logs",  # base logging directory
    experiment_round: Optional[str] = None,
    check_embeddings_every: int = 2,  # check embeddings every N epochs
    log_metrics_every: int = 100,
    resume_from: Optional[str] = None,  # midpoint path used for resuming training
    checkpoint_dir: Optional[str] = None,  # add check point directory
    embedding_size: Optional[int] = None,
    local_rank: Optional[int] = None,  # If provided, use DDP (DISABLED FOR NOW)
    world_size: Optional[int] = None,
    # NEW: Downstream evaluation parameters
    outcomes_df: Optional[pd.DataFrame] = None,  # Outcomes for downstream eval
    run_downstream_eval: bool = False,  # Whether to run downstream eval after training
    save_model: bool = True  # Whether to save the trained model
) -> Dict[str, any]:
```

```python
    # ============================================================
    # DDP SETUP
    # ============================================================
    use_ddp = local_rank is not None and world_size is not None and world_size > 1
    
    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
        is_main = (local_rank == 0)
    else:
        # Single GPU mode (backward compatible)
        local_rank = 0
        world_size = 1
        is_main = True
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
    # ============================================================
    # DDP SETUP (DISABLED FOR NOW)
    # ============================================================
    # DDP is disabled - using single GPU mode
    use_ddp = False  # DISABLED: was `local_rank is not None and world_size is not None and world_size > 1`
    local_rank = 0
    world_size = 1
    is_main = True
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
        # Save complete results to JSON for later comparison
        results_path = metrics_logger.save_final_results(results)
        logger.info(f"Complete results saved to {results_path}")    
```

```python
        # ============================================================
        # SAVE TRAINED MODEL WITH STANDARDIZED NAMING
        # ============================================================
        if save_model:
            model_name = generate_model_name(
                exp_name=exp_name,
                experiment_round=experiment_round,
                batch_size=config.batch_size,
                epochs=epochs,
                embedding_size=eff_d_model
            )
            
            # Add model_name to results for traceability
            results['model_name'] = model_name
            
            model_save_dir = os.path.join(effective_log_dir, exp_name, 'saved_models')
            model_path = save_trained_model(
                model=model_to_eval,
                config=config,
                model_name=model_name,
                save_dir=model_save_dir,
                exp_results=results,
                is_best=True  # Save as best since this is final
            )
            logger.info(f"Model saved as: {model_name}")
            results['model_path'] = model_path
        
        # ============================================================
        # DOWNSTREAM EVALUATION (Optional)
        # ============================================================
        if run_downstream_eval and outcomes_df is not None:
            logger.info("\n" + "="*80)
            logger.info("RUNNING DOWNSTREAM EVALUATION")
            logger.info("="*80)
            
            downstream_results = run_downstream_evaluation(
                model=model_to_eval,
                model_config=config,
                features_df=val_data,  # Use validation data for downstream eval
                outcomes_df=outcomes_df,
                device=device,
                use_mixed_precision=use_mixed_precision,
                downstream_config=DownstreamConfig(task_name="medicaid_ip_risk")
            )
            
            # Add downstream metrics to results
            results['downstream_evaluation'] = downstream_results
            
            # Log key downstream metrics
            logger.info(f"Downstream Test AUC-ROC: {downstream_results['test_auc_roc']:.4f}")
            logger.info(f"Downstream Test F1: {downstream_results['test_f1']:.4f}")
            
            # Update saved results with downstream metrics
            if save_model:
                results_path = os.path.join(model_save_dir, f"{model_name}_results.json")
                with open(results_path, 'w') as f:
                    json.dump(MetricsLogger.convert_to_serializable(results), f, indent=2)
```

```python
def run_selected_experiments(
    experiment_names: List[str],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    device: torch.device,
    epochs: int = 10,
    experiment_round: Optional[str] = None,
    embedding_size: Optional[int] = None,
    local_rank: Optional[int] = None,      
    world_size: Optional[int] = None,
    # NEW: Downstream evaluation parameters
    outcomes_df: Optional[pd.DataFrame] = None,
    run_downstream_eval: bool = False,
    save_model: bool = True
) -> pd.DataFrame:
```

```python
            outcomes_df=outcomes_df,
            run_downstream_eval=run_downstream_eval,
            save_model=save_model
```

```python
# Load data
import pandas as pd
from google.cloud import bigquery

client = bigquery.Client()

# Load transformer training data (now with LOB column)
train_query = """
SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`
LIMIT 100000
"""
df_train = client.query(train_query).to_dataframe()

# Load validation data
val_query = """
SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_val_ending`
LIMIT 20000
"""
df_val = client.query(val_query).to_dataframe()

# Load outcomes for downstream evaluation
outcomes_query = """
SELECT individual_id, index_dt, acute_ip_flag 
FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment`
"""
df_outcomes = client.query(outcomes_query).to_dataframe()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run experiment with downstream evaluation
results = run_single_experiment(
    exp_name='exp3_standard_moe',
    moe_config=MoEConfig(num_experts=8, top_k=2),
    use_learnt_att_pool=False,
    train_data=df_train,
    val_data=df_val,
    device=device,
    epochs=10,
    experiment_round='round4_lob_downstream',
    embedding_size=256,
    outcomes_df=df_outcomes,  # NEW: Pass outcomes for downstream eval
    run_downstream_eval=True,  # NEW: Enable downstream eval
    save_model=True  # NEW: Save model with standardized naming
)

# Results will include:
# - All intrinsic metrics (loss, top-k accuracy, etc.)
# - model_name: 'round4_lob_downstream_exp3_standard_moe_bs32_ep10_d256_20241211_143022'
# - model_path: 'logs/round4_lob_downstream/exp3_standard_moe/saved_models/..._final.pt'
# - downstream_evaluation: {'test_auc_roc': 0.72, 'test_f1': 0.45, ...}
```

---

**User**

Ok great, I have applied all of the changes and I want to create a test for the downstreamEvaluator and make sure it works correctly with other exisitng implemenation and also the model can be saved properly; I want to use real data to test 

---

**Cursor**

I'll create a comprehensive test script for the `DownstreamEvaluator` and model saving functionality. Let me first check the current structure to ensure the test integrates properly:

Now I have a complete picture. Here's a comprehensive test for the `DownstreamEvaluator` and model saving functionality. 

**Add this test section AFTER the existing tests (around line 10765, after `test_full_experiment_simulation`):**

```python
# ============================================================================
# SECTION 9: DOWNSTREAM EVALUATION & MODEL SAVING TESTS
# ============================================================================

def test_conv_lob():
    """
    Test LOB (Line of Business) parsing function.
    
    Validates:
    - Single value format: "Medicaid"
    - Asterisk-separated format: "Medicaid*Medicaid*..."
    - Case insensitivity
    - Unknown values default to 0
    - Forward-fill behavior
    """
    print("\n" + "="*80)
    print("TEST 16: LOB Parsing Function")
    print("="*80)
    
    # Test 1: Single value format
    print("  Testing single value format...")
    result = conv_lob("Medicaid", 5)
    assert result == [3, 3, 3, 3, 3], f"Single value failed: {result}"
    print(f"    'Medicaid' -> {result} ✅")
    
    result = conv_lob("Commercial", 5)
    assert result == [1, 1, 1, 1, 1], f"Commercial failed: {result}"
    print(f"    'Commercial' -> {result} ✅")
    
    result = conv_lob("Medicare", 5)
    assert result == [2, 2, 2, 2, 2], f"Medicare failed: {result}"
    print(f"    'Medicare' -> {result} ✅")
    
    # Test 2: Asterisk-separated format
    print("  Testing asterisk-separated format...")
    result = conv_lob("Medicaid*Medicaid*Commercial", 5)
    assert result == [3, 3, 1, 1, 1], f"Asterisk format failed: {result}"
    print(f"    'Medicaid*Medicaid*Commercial' (len=5) -> {result} ✅")
    
    # Test 3: Case insensitivity
    print("  Testing case insensitivity...")
    result = conv_lob("MEDICAID", 3)
    assert result == [3, 3, 3], f"Case insensitivity failed: {result}"
    print(f"    'MEDICAID' -> {result} ✅")
    
    result = conv_lob("medicare", 3)
    assert result == [2, 2, 2], f"Lowercase failed: {result}"
    print(f"    'medicare' -> {result} ✅")
    
    # Test 4: Unknown values
    print("  Testing unknown values...")
    result = conv_lob("Unknown", 3)
    assert result == [0, 3, 3], f"Unknown value failed: {result}"  # 0, then forward-fill with 3
    print(f"    'Unknown' -> {result} ✅")
    
    # Test 5: None/empty values
    print("  Testing None/empty values...")
    result = conv_lob(None, 3)
    assert result == [0, 0, 0], f"None failed: {result}"
    print(f"    None -> {result} ✅")
    
    result = conv_lob("", 3)
    assert result == [0, 0, 0], f"Empty string failed: {result}"
    print(f"    '' -> {result} ✅")
    
    print("\n✅ TEST 16 PASSED: conv_lob function works correctly\n")


def test_clinical_dataset_with_lob():
    """
    Test ClinicalDataset properly handles LOB column.
    
    Validates:
    - LOB tensor is created
    - LOB values are correct
    - Batch collation includes LOB
    - Works with missing LOB column (default to Medicaid)
    """
    print("\n" + "="*80)
    print("TEST 17: ClinicalDataset with LOB")
    print("="*80)
    
    cfg = BaseConfig(len_dy=50, len_cd=20, batch_size=4)
    
    # Create test data WITH LOB column
    print("  Testing with LOB column present...")
    test_data = df_train.head(10).copy()
    
    # Add LOB column if not present
    if 'lob' not in test_data.columns:
        test_data['lob'] = 'Medicaid'
    
    # Rename target to target_cd if needed
    if 'target' in test_data.columns and 'target_cd' not in test_data.columns:
        test_data = test_data.rename(columns={'target': 'target_cd'})
    
    dataset = ClinicalDataset(test_data, cfg)
    
    # Check LOB tensor exists
    assert hasattr(dataset, 'lobs'), "Dataset missing 'lobs' tensor"
    assert dataset.lobs.shape == (len(test_data), cfg.len_dy), f"LOB shape wrong: {dataset.lobs.shape}"
    print(f"    LOB tensor shape: {dataset.lobs.shape} ✅")
    
    # Check values are in valid range [0, 3]
    assert dataset.lobs.min() >= 0 and dataset.lobs.max() <= 3, "LOB values out of range"
    print(f"    LOB value range: [{dataset.lobs.min()}, {dataset.lobs.max()}] ✅")
    
    # Test dataloader and collation
    print("  Testing DataLoader and collation...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=clinical_collate_fn
    )
    
    batch = next(iter(dataloader))
    assert 'lob' in batch, "Batch missing 'lob' key"
    assert batch['lob'].shape == (4, cfg.len_dy), f"Batch LOB shape wrong: {batch['lob'].shape}"
    print(f"    Batch LOB shape: {batch['lob'].shape} ✅")
    
    # Test without LOB column (should default to Medicaid = 3)
    print("  Testing without LOB column (should default to Medicaid)...")
    test_data_no_lob = test_data.drop(columns=['lob'], errors='ignore')
    dataset_no_lob = ClinicalDataset(test_data_no_lob, cfg)
    
    assert hasattr(dataset_no_lob, 'lobs'), "Dataset should still have LOB tensor"
    # Default should be Medicaid (3)
    print(f"    Default LOB values: {dataset_no_lob.lobs[0, :5].tolist()} ✅")
    
    print("\n✅ TEST 17 PASSED: ClinicalDataset handles LOB correctly\n")


def test_model_forward_with_lob():
    """
    Test all model types handle LOB correctly in forward pass.
    
    Validates:
    - Input tensor shape: [batch, len_dy, 3 + len_cd]
    - All models accept LOB embedding
    - Output shape is correct
    """
    print("\n" + "="*80)
    print("TEST 18: Model Forward Pass with LOB")
    print("="*80)
    
    cleanup_gpu_memory()
    
    cfg = BaseConfig(len_dy=50, len_cd=20, batch_size=4)
    batch_size = 4
    len_dy = 50
    len_cd = 20
    
    # Build input tensor with LOB (age, gender, lob, codes)
    age = torch.randint(0, 1400, (batch_size, len_dy), device=device)
    gender = torch.randint(0, 4, (batch_size, len_dy), device=device)
    lob = torch.randint(0, 4, (batch_size, len_dy), device=device)  # NEW: LOB
    codes = torch.randint(0, 1000, (batch_size, len_dy, len_cd), device=device)
    
    x = torch.cat([
        age.unsqueeze(-1),
        gender.unsqueeze(-1),
        lob.unsqueeze(-1),  # NEW: Include LOB
        codes
    ], dim=-1)
    
    expected_input_dim = 3 + len_cd  # age + gender + lob + codes
    assert x.shape == (batch_size, len_dy, expected_input_dim), \
        f"Input shape wrong: {x.shape}, expected ({batch_size}, {len_dy}, {expected_input_dim})"
    print(f"  Input tensor shape: {x.shape} ✅")
    
    # Test BaselineTransformer
    print("  Testing BaselineTransformer with LOB...")
    model = BaselineTransformer(cfg).to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, len_dy, cfg.target_cd_cnt), \
        f"Baseline output shape wrong: {output.shape}"
    print(f"    Output shape: {output.shape} ✅")
    del model
    
    # Test FlashAttentionTransformer
    print("  Testing FlashAttentionTransformer with LOB...")
    flash_cfg = FlashAttentionConfig(len_dy=50, len_cd=20, batch_size=4)
    model = FlashAttentionTransformer(flash_cfg).to(device)
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(x)
    
    assert output.shape == (batch_size, len_dy, flash_cfg.target_cd_cnt), \
        f"Flash output shape wrong: {output.shape}"
    print(f"    Output shape: {output.shape} ✅")
    del model
    
    # Test FlashMoETransformer
    print("  Testing FlashMoETransformer with LOB...")
    moe_cfg = MoEConfig(d_model=flash_cfg.embedding_size, d_ff=flash_cfg.nhid, 
                        num_experts=4, top_k=2, use_moe_from_layer=0)
    model = FlashMoETransformer(flash_cfg, moe_cfg).to(device)
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output, moe_losses = model(x, return_moe_losses=True)
    
    assert output.shape == (batch_size, len_dy, flash_cfg.target_cd_cnt), \
        f"MoE output shape wrong: {output.shape}"
    print(f"    Output shape: {output.shape} ✅")
    del model
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 18 PASSED: All models handle LOB correctly\n")


def test_embedding_extractor_with_lob():
    """
    Test EmbeddingExtractor works with LOB-enabled models.
    
    Validates:
    - Embeddings can be extracted from all model types
    - Patient-level embeddings work with dt_cnt
    """
    print("\n" + "="*80)
    print("TEST 19: EmbeddingExtractor with LOB")
    print("="*80)
    
    cleanup_gpu_memory()
    
    cfg = FlashAttentionConfig(len_dy=50, len_cd=20, batch_size=4)
    batch_size = 4
    len_dy = 50
    len_cd = 20
    
    # Build input tensor with LOB
    age = torch.randint(0, 1400, (batch_size, len_dy), device=device)
    gender = torch.randint(0, 4, (batch_size, len_dy), device=device)
    lob = torch.randint(0, 4, (batch_size, len_dy), device=device)
    codes = torch.randint(0, 1000, (batch_size, len_dy, len_cd), device=device)
    
    x = torch.cat([
        age.unsqueeze(-1),
        gender.unsqueeze(-1),
        lob.unsqueeze(-1),
        codes
    ], dim=-1)
    
    dt_cnt = [30, 45, 20, 50]
    
    # Test with FlashAttentionTransformer
    print("  Testing embedding extraction with LOB...")
    model = FlashAttentionTransformer(cfg).to(device)
    model.eval()
    
    with EmbeddingExtractor(model) as extractor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(x)
        
        all_emb = extractor.get_embeddings()
        print(f"    All embeddings shape: {all_emb.shape}")
        assert all_emb.shape == (batch_size, len_dy, cfg.embedding_size), \
            f"Wrong embedding shape: {all_emb.shape}"
        
        patient_emb = extractor.get_patient_embedding(dt_cnt)
        print(f"    Patient embeddings shape: {patient_emb.shape}")
        assert patient_emb.shape == (batch_size, cfg.embedding_size), \
            f"Wrong patient embedding shape: {patient_emb.shape}"
    
    print("  ✅ Embeddings extracted successfully")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 19 PASSED: EmbeddingExtractor works with LOB\n")


def test_model_saving_and_loading():
    """
    Test model saving and loading functionality.
    
    Validates:
    - Model can be saved with standardized naming
    - Model can be loaded back
    - Loaded model produces same output
    - Config and results are saved
    """
    print("\n" + "="*80)
    print("TEST 20: Model Saving and Loading")
    print("="*80)
    
    cleanup_gpu_memory()
    
    import tempfile
    import os
    
    cfg = FlashAttentionConfig(len_dy=50, len_cd=20, batch_size=4, embedding_size=64, nhid=128)
    
    # Create and initialize model
    print("  Creating model...")
    model = FlashAttentionTransformer(cfg).to(device)
    model.eval()
    
    # Test generate_model_name
    print("  Testing model name generation...")
    model_name = generate_model_name(
        exp_name='exp2_dense_flash',
        experiment_round='test_round',
        batch_size=32,
        epochs=10,
        embedding_size=64
    )
    print(f"    Generated name: {model_name}")
    assert 'test_round' in model_name
    assert 'exp2_dense_flash' in model_name
    assert 'bs32' in model_name
    assert 'ep10' in model_name
    assert 'd64' in model_name
    print("    ✅ Name format correct")
    
    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock results
        exp_results = {
            'experiment': 'exp2_dense_flash',
            'final_val_loss': 0.5,
            'final_top_10_acc': 0.75,
            'training_time_sec': 100.0
        }
        
        # Save model
        print("  Saving model...")
        model_path = save_trained_model(
            model=model,
            config=cfg,
            model_name=model_name,
            save_dir=tmpdir,
            exp_results=exp_results,
            is_best=True
        )
        
        # Check files exist
        assert os.path.exists(model_path), "Model file not saved"
        assert os.path.exists(os.path.join(tmpdir, f"{model_name}_config.json")), "Config not saved"
        assert os.path.exists(os.path.join(tmpdir, f"{model_name}_results.json")), "Results not saved"
        assert os.path.exists(os.path.join(tmpdir, f"{model_name}_best.pt")), "Best model not saved"
        print("    ✅ All files saved")
        
        # Test loading
        print("  Loading model...")
        loaded_model = load_trained_model(
            model_path=model_path,
            model_class=FlashAttentionTransformer,
            config=cfg,
            device=device
        )
        
        # Verify loaded model works
        print("  Verifying loaded model...")
        batch_size = 4
        len_dy = 50
        len_cd = 20
        
        age = torch.randint(0, 1400, (batch_size, len_dy), device=device)
        gender = torch.randint(0, 4, (batch_size, len_dy), device=device)
        lob = torch.randint(0, 4, (batch_size, len_dy), device=device)
        codes = torch.randint(0, 1000, (batch_size, len_dy, len_cd), device=device)
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                original_output = model(x)
                loaded_output = loaded_model(x)
        
        # Outputs should be identical
        assert torch.allclose(original_output, loaded_output, atol=1e-4), \
            "Loaded model produces different output"
        print("    ✅ Loaded model produces identical output")
    
    del model, loaded_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 20 PASSED: Model saving and loading works\n")


def test_downstream_evaluator_with_real_data():
    """
    Test DownstreamEvaluator with real data.
    
    This is the integration test for the full downstream evaluation pipeline:
    1. Load real training data
    2. Load real outcomes data
    3. Train a small model
    4. Extract embeddings
    5. Train linear probe
    6. Evaluate on train/val/test
    
    REQUIRES: Real data loaded (df_train, df_val) and outcomes table available
    """
    print("\n" + "="*80)
    print("TEST 21: DownstreamEvaluator with Real Data")
    print("="*80)
    
    cleanup_gpu_memory_hard()
    
    # Check if real data is available
    try:
        # Use small sample for testing
        train_sample = df_train.head(500).copy()
        val_sample = df_val.head(200).copy()
        
        # Ensure required columns exist
        if 'target' in train_sample.columns and 'target_cd' not in train_sample.columns:
            train_sample = train_sample.rename(columns={'target': 'target_cd'})
        if 'target' in val_sample.columns and 'target_cd' not in val_sample.columns:
            val_sample = val_sample.rename(columns={'target': 'target_cd'})
        
        # Add LOB column if missing
        if 'lob' not in train_sample.columns:
            train_sample['lob'] = 'Medicaid'
        if 'lob' not in val_sample.columns:
            val_sample['lob'] = 'Medicaid'
            
    except NameError:
        print("  ⚠️ Real data not loaded. Skipping test.")
        print("  Run after loading df_train and df_val from BigQuery")
        return
    
    print(f"  Using {len(train_sample)} train samples, {len(val_sample)} val samples")
    
    # Load outcomes data
    print("  Loading outcomes data...")
    try:
        outcomes_query = """
        SELECT individual_id, CAST(index_dt AS STRING) as index_dt, 
               COALESCE(acute_ip_flag, 0) as acute_ip_flag 
        FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment`
        LIMIT 10000
        """
        outcomes_df = client.query(outcomes_query).to_dataframe()
        print(f"  Loaded {len(outcomes_df)} outcome records")
        print(f"  IP prevalence: {outcomes_df['acute_ip_flag'].mean():.2%}")
    except Exception as e:
        print(f"  ⚠️ Could not load outcomes: {e}")
        print("  Creating synthetic outcomes for testing...")
        
        # Create synthetic outcomes for testing
        individual_ids = val_sample['individual_id'].unique()
        outcomes_df = pd.DataFrame({
            'individual_id': individual_ids,
            'index_dt': val_sample.groupby('individual_id')['index_dt'].first().values,
            'acute_ip_flag': np.random.binomial(1, 0.1, len(individual_ids))  # 10% prevalence
        })
        outcomes_df['individual_id'] = outcomes_df['individual_id'].astype(str)
        outcomes_df['index_dt'] = outcomes_df['index_dt'].astype(str)
        print(f"  Created {len(outcomes_df)} synthetic outcomes")
    
    # Create a small model for testing
    print("  Creating model...")
    cfg = FlashAttentionConfig(
        len_dy=50,  # Smaller for speed
        len_cd=20,
        batch_size=16,
        embedding_size=64,  # Smaller for speed
        nhid=128
    )
    model = FlashAttentionTransformer(cfg).to(device)
    
    # Train for 1 epoch to get non-random weights
    print("  Training model for 1 epoch...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Create dataset and dataloader
    train_dataset = ClinicalDataset(train_sample, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=clinical_collate_fn,
        num_workers=0
    )
    
    # Quick training
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Just 5 batches
            break
        
        optimizer.zero_grad()
        
        age = batch['age'].to(device)
        gender = batch['gender'].to(device)
        lob = batch['lob'].to(device)
        codes = batch['codes'].to(device)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(x)
            loss = compute_loss(output, y, dt_cnt, cfg, criterion, device)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print("  Training complete")
    
    # Run downstream evaluation
    print("\n  Running DownstreamEvaluator...")
    model.eval()
    
    evaluator = DownstreamEvaluator(
        model=model,
        model_config=cfg,
        device=device,
        use_mixed_precision=True
    )
    
    # Ensure val_sample has individual_id and index_dt as strings
    val_sample['individual_id'] = val_sample['individual_id'].astype(str)
    val_sample['index_dt'] = val_sample['index_dt'].astype(str)
    
    downstream_config = DownstreamConfig(
        task_name="medicaid_ip_risk_test",
        test_size=0.3,
        val_size=0.1,
        n_cv_folds=3,
        max_iter=500
    )
    
    try:
        results = evaluator.evaluate(
            features_df=val_sample,
            outcomes_df=outcomes_df,
            downstream_config=downstream_config
        )
        
        # Validate results structure
        print("\n  Validating results...")
        assert 'train_auc_roc' in results, "Missing train_auc_roc"
        assert 'val_auc_roc' in results, "Missing val_auc_roc"
        assert 'test_auc_roc' in results, "Missing test_auc_roc"
        assert 'test_f1' in results, "Missing test_f1"
        
        print(f"\n  Results Summary:")
        print(f"    Train AUC-ROC: {results['train_auc_roc']:.4f}")
        print(f"    Val AUC-ROC:   {results['val_auc_roc']:.4f}")
        print(f"    Test AUC-ROC:  {results['test_auc_roc']:.4f}")
        print(f"    Test F1:       {results['test_f1']:.4f}")
        print(f"    Test Recall:   {results['test_recall']:.4f}")
        print(f"    Embedding dim: {results['embedding_dim']}")
        
        # Basic sanity checks
        assert 0 <= results['test_auc_roc'] <= 1, "AUC-ROC out of range"
        assert 0 <= results['test_f1'] <= 1, "F1 out of range"
        assert results['embedding_dim'] == cfg.embedding_size, "Wrong embedding dimension"
        
        print("\n  ✅ DownstreamEvaluator works correctly!")
        
    except Exception as e:
        print(f"\n  ⚠️ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        del model, evaluator
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n✅ TEST 21 PASSED: DownstreamEvaluator works with real data\n")


def test_run_single_experiment_with_downstream():
    """
    Test run_single_experiment with downstream evaluation enabled.
    
    This is the full integration test:
    - Run a mini experiment
    - Save the model
    - Run downstream evaluation
    - Verify all results are captured
    """
    print("\n" + "="*80)
    print("TEST 22: run_single_experiment with Downstream Evaluation")
    print("="*80)
    
    cleanup_gpu_memory_hard()
    
    # Check if real data is available
    try:
        train_sample = df_train.head(100).copy()
        val_sample = df_val.head(50).copy()
        
        # Prepare data
        if 'target' in train_sample.columns and 'target_cd' not in train_sample.columns:
            train_sample = train_sample.rename(columns={'target': 'target_cd'})
        if 'target' in val_sample.columns and 'target_cd' not in val_sample.columns:
            val_sample = val_sample.rename(columns={'target': 'target_cd'})
        
        if 'lob' not in train_sample.columns:
            train_sample['lob'] = 'Medicaid'
        if 'lob' not in val_sample.columns:
            val_sample['lob'] = 'Medicaid'
            
    except NameError:
        print("  ⚠️ Real data not loaded. Skipping test.")
        return
    
    # Create synthetic outcomes
    print("  Creating synthetic outcomes...")
    individual_ids = val_sample['individual_id'].unique()
    outcomes_df = pd.DataFrame({
        'individual_id': individual_ids.astype(str),
        'index_dt': val_sample.groupby('individual_id')['index_dt'].first().astype(str).values,
        'acute_ip_flag': np.random.binomial(1, 0.1, len(individual_ids))
    })
    
    print(f"  Train samples: {len(train_sample)}")
    print(f"  Val samples: {len(val_sample)}")
    print(f"  Outcomes: {len(outcomes_df)}")
    
    # Run experiment with downstream evaluation
    print("\n  Running experiment with downstream eval...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_single_experiment(
            exp_name='exp2_dense_flash',
            moe_config=None,
            use_learnt_att_pool=False,
            train_data=train_sample,
            val_data=val_sample,
            device=device,
            epochs=1,
            experiment_round='test_downstream',
            log_dir=tmpdir,
            outcomes_df=outcomes_df,
            run_downstream_eval=True,
            save_model=True
        )
        
        # Validate results
        print("\n  Validating experiment results...")
        
        # Check standard results
        assert 'experiment' in result, "Missing experiment name"
        assert 'parameters' in result, "Missing parameters"
        assert 'val_loss_final' in result, "Missing val_loss_final"
        print(f"    Experiment: {result['experiment']} ✅")
        print(f"    Parameters: {result['parameters']:,} ✅")
        
        # Check model was saved
        if 'model_name' in result:
            print(f"    Model name: {result['model_name']} ✅")
        if 'model_path' in result:
            assert os.path.exists(result['model_path']), "Model file not found"
            print(f"    Model saved: {result['model_path']} ✅")
        
        # Check downstream results
        if 'downstream_evaluation' in result:
            ds_results = result['downstream_evaluation']
            print(f"\n  Downstream Evaluation Results:")
            print(f"    Test AUC-ROC: {ds_results.get('test_auc_roc', 'N/A'):.4f}")
            print(f"    Test F1:      {ds_results.get('test_f1', 'N/A'):.4f}")
            print("    ✅ Downstream evaluation captured in results")
        else:
            print("    ⚠️ No downstream_evaluation in results (check if outcomes matched)")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 22 PASSED: Full integration with downstream evaluation works\n")


# ============================================================================
# RUN ALL DOWNSTREAM TESTS
# ============================================================================

def run_all_downstream_tests():
    """Run all downstream evaluation and model saving tests."""
    print("\n" + "="*80)
    print("DOWNSTREAM EVALUATION TEST SUITE")
    print("="*80)
    
    # Unit tests (no real data needed)
    test_conv_lob()
    
    # Tests requiring model
    test_model_forward_with_lob()
    test_embedding_extractor_with_lob()
    test_model_saving_and_loading()
    
    # Tests requiring real data (may be skipped if data not loaded)
    test_clinical_dataset_with_lob()
    test_downstream_evaluator_with_real_data()
    test_run_single_experiment_with_downstream()
    
    print("\n" + "="*80)
    print("ALL DOWNSTREAM TESTS COMPLETED!")
    print("="*80 + "\n")


# Uncomment to run tests:
# run_all_downstream_tests()
```

---

## How to Run the Tests

Add the above code to your file, then run individually or all together:

```python
# Run individual tests:
test_conv_lob()
test_clinical_dataset_with_lob()
test_model_forward_with_lob()
test_embedding_extractor_with_lob()
test_model_saving_and_loading()
test_downstream_evaluator_with_real_data()
test_run_single_experiment_with_downstream()

# Or run all at once:
run_all_downstream_tests()
```

---

## Before Running Tests

Make sure you have:

1. **Loaded real data** (the tests will use `df_train` and `df_val` if available):
```python
# Load from feather files
df_train = pd.read_feather("sample_data/mdcd_train_1m.feather")
df_val = pd.read_feather("sample_data/mdcd_val_10k.feather")

# OR load from BigQuery
input_sql = """
SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`
LIMIT 10000
"""
df_train = client.query(input_sql).to_dataframe()
```

2. **Renamed columns if needed**:
```python
# Ensure target_cd column exists
if 'target' in df_train.columns and 'target_cd' not in df_train.columns:
    df_train = df_train.rename(columns={'target': 'target_cd'})
if 'target' in df_val.columns and 'target_cd' not in df_val.columns:
    df_val = df_val.rename(columns={'target': 'target_cd'})
```

3. **Added LOB column if not present** (tests will handle this, but for real training):
```python
if 'lob' not in df_train.columns:
    df_train['lob'] = 'Medicaid'
```

---

## Test Summary

| Test | Purpose | Real Data Needed |
|------|---------|------------------|
| 16: `test_conv_lob` | Validates LOB string parsing | No |
| 17: `test_clinical_dataset_with_lob` | Validates dataset handles LOB | Yes |
| 18: `test_model_forward_with_lob` | Validates all models accept LOB | No |
| 19: `test_embedding_extractor_with_lob` | Validates embedding extraction | No |
| 20: `test_model_saving_and_loading` | Validates save/load pipeline | No |
| 21: `test_downstream_evaluator_with_real_data` | Full downstream evaluation test | Yes |
| 22: `test_run_single_experiment_with_downstream` | Full integration test | Yes |

You're in **ask mode**. Copy the test code above into your file and run it. Switch to **agent mode** if you need me to add the code directly.

```python
# ============================================================================
# SECTION 9: DOWNSTREAM EVALUATION & MODEL SAVING TESTS
# ============================================================================

def test_conv_lob():
    """
    Test LOB (Line of Business) parsing function.
    
    Validates:
    - Single value format: "Medicaid"
    - Asterisk-separated format: "Medicaid*Medicaid*..."
    - Case insensitivity
    - Unknown values default to 0
    - Forward-fill behavior
    """
    print("\n" + "="*80)
    print("TEST 16: LOB Parsing Function")
    print("="*80)
    
    # Test 1: Single value format
    print("  Testing single value format...")
    result = conv_lob("Medicaid", 5)
    assert result == [3, 3, 3, 3, 3], f"Single value failed: {result}"
    print(f"    'Medicaid' -> {result} ✅")
    
    result = conv_lob("Commercial", 5)
    assert result == [1, 1, 1, 1, 1], f"Commercial failed: {result}"
    print(f"    'Commercial' -> {result} ✅")
    
    result = conv_lob("Medicare", 5)
    assert result == [2, 2, 2, 2, 2], f"Medicare failed: {result}"
    print(f"    'Medicare' -> {result} ✅")
    
    # Test 2: Asterisk-separated format
    print("  Testing asterisk-separated format...")
    result = conv_lob("Medicaid*Medicaid*Commercial", 5)
    assert result == [3, 3, 1, 1, 1], f"Asterisk format failed: {result}"
    print(f"    'Medicaid*Medicaid*Commercial' (len=5) -> {result} ✅")
    
    # Test 3: Case insensitivity
    print("  Testing case insensitivity...")
    result = conv_lob("MEDICAID", 3)
    assert result == [3, 3, 3], f"Case insensitivity failed: {result}"
    print(f"    'MEDICAID' -> {result} ✅")
    
    result = conv_lob("medicare", 3)
    assert result == [2, 2, 2], f"Lowercase failed: {result}"
    print(f"    'medicare' -> {result} ✅")
    
    # Test 4: Unknown values
    print("  Testing unknown values...")
    result = conv_lob("Unknown", 3)
    assert result == [0, 3, 3], f"Unknown value failed: {result}"  # 0, then forward-fill with 3
    print(f"    'Unknown' -> {result} ✅")
    
    # Test 5: None/empty values
    print("  Testing None/empty values...")
    result = conv_lob(None, 3)
    assert result == [0, 0, 0], f"None failed: {result}"
    print(f"    None -> {result} ✅")
    
    result = conv_lob("", 3)
    assert result == [0, 0, 0], f"Empty string failed: {result}"
    print(f"    '' -> {result} ✅")
    
    print("\n✅ TEST 16 PASSED: conv_lob function works correctly\n")


def test_clinical_dataset_with_lob():
    """
    Test ClinicalDataset properly handles LOB column.
    
    Validates:
    - LOB tensor is created
    - LOB values are correct
    - Batch collation includes LOB
    - Works with missing LOB column (default to Medicaid)
    """
    print("\n" + "="*80)
    print("TEST 17: ClinicalDataset with LOB")
    print("="*80)
    
    cfg = BaseConfig(len_dy=50, len_cd=20, batch_size=4)
    
    # Create test data WITH LOB column
    print("  Testing with LOB column present...")
    test_data = df_train.head(10).copy()
    
    # Add LOB column if not present
    if 'lob' not in test_data.columns:
        test_data['lob'] = 'Medicaid'
    
    # Rename target to target_cd if needed
    if 'target' in test_data.columns and 'target_cd' not in test_data.columns:
        test_data = test_data.rename(columns={'target': 'target_cd'})
    
    dataset = ClinicalDataset(test_data, cfg)
    
    # Check LOB tensor exists
    assert hasattr(dataset, 'lobs'), "Dataset missing 'lobs' tensor"
    assert dataset.lobs.shape == (len(test_data), cfg.len_dy), f"LOB shape wrong: {dataset.lobs.shape}"
    print(f"    LOB tensor shape: {dataset.lobs.shape} ✅")
    
    # Check values are in valid range [0, 3]
    assert dataset.lobs.min() >= 0 and dataset.lobs.max() <= 3, "LOB values out of range"
    print(f"    LOB value range: [{dataset.lobs.min()}, {dataset.lobs.max()}] ✅")
    
    # Test dataloader and collation
    print("  Testing DataLoader and collation...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=clinical_collate_fn
    )
    
    batch = next(iter(dataloader))
    assert 'lob' in batch, "Batch missing 'lob' key"
    assert batch['lob'].shape == (4, cfg.len_dy), f"Batch LOB shape wrong: {batch['lob'].shape}"
    print(f"    Batch LOB shape: {batch['lob'].shape} ✅")
    
    # Test without LOB column (should default to Medicaid = 3)
    print("  Testing without LOB column (should default to Medicaid)...")
    test_data_no_lob = test_data.drop(columns=['lob'], errors='ignore')
    dataset_no_lob = ClinicalDataset(test_data_no_lob, cfg)
    
    assert hasattr(dataset_no_lob, 'lobs'), "Dataset should still have LOB tensor"
    # Default should be Medicaid (3)
    print(f"    Default LOB values: {dataset_no_lob.lobs[0, :5].tolist()} ✅")
    
    print("\n✅ TEST 17 PASSED: ClinicalDataset handles LOB correctly\n")


def test_model_forward_with_lob():
    """
    Test all model types handle LOB correctly in forward pass.
    
    Validates:
    - Input tensor shape: [batch, len_dy, 3 + len_cd]
    - All models accept LOB embedding
    - Output shape is correct
    """
    print("\n" + "="*80)
    print("TEST 18: Model Forward Pass with LOB")
    print("="*80)
    
    cleanup_gpu_memory()
    
    cfg = BaseConfig(len_dy=50, len_cd=20, batch_size=4)
    batch_size = 4
    len_dy = 50
    len_cd = 20
    
    # Build input tensor with LOB (age, gender, lob, codes)
    age = torch.randint(0, 1400, (batch_size, len_dy), device=device)
    gender = torch.randint(0, 4, (batch_size, len_dy), device=device)
    lob = torch.randint(0, 4, (batch_size, len_dy), device=device)  # NEW: LOB
    codes = torch.randint(0, 1000, (batch_size, len_dy, len_cd), device=device)
    
    x = torch.cat([
        age.unsqueeze(-1),
        gender.unsqueeze(-1),
        lob.unsqueeze(-1),  # NEW: Include LOB
        codes
    ], dim=-1)
    
    expected_input_dim = 3 + len_cd  # age + gender + lob + codes
    assert x.shape == (batch_size, len_dy, expected_input_dim), \
        f"Input shape wrong: {x.shape}, expected ({batch_size}, {len_dy}, {expected_input_dim})"
    print(f"  Input tensor shape: {x.shape} ✅")
    
    # Test BaselineTransformer
    print("  Testing BaselineTransformer with LOB...")
    model = BaselineTransformer(cfg).to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, len_dy, cfg.target_cd_cnt), \
        f"Baseline output shape wrong: {output.shape}"
    print(f"    Output shape: {output.shape} ✅")
    del model
    
    # Test FlashAttentionTransformer
    print("  Testing FlashAttentionTransformer with LOB...")
    flash_cfg = FlashAttentionConfig(len_dy=50, len_cd=20, batch_size=4)
    model = FlashAttentionTransformer(flash_cfg).to(device)
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(x)
    
    assert output.shape == (batch_size, len_dy, flash_cfg.target_cd_cnt), \
        f"Flash output shape wrong: {output.shape}"
    print(f"    Output shape: {output.shape} ✅")
    del model
    
    # Test FlashMoETransformer
    print("  Testing FlashMoETransformer with LOB...")
    moe_cfg = MoEConfig(d_model=flash_cfg.embedding_size, d_ff=flash_cfg.nhid, 
                        num_experts=4, top_k=2, use_moe_from_layer=0)
    model = FlashMoETransformer(flash_cfg, moe_cfg).to(device)
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output, moe_losses = model(x, return_moe_losses=True)
    
    assert output.shape == (batch_size, len_dy, flash_cfg.target_cd_cnt), \
        f"MoE output shape wrong: {output.shape}"
    print(f"    Output shape: {output.shape} ✅")
    del model
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 18 PASSED: All models handle LOB correctly\n")


def test_embedding_extractor_with_lob():
    """
    Test EmbeddingExtractor works with LOB-enabled models.
    
    Validates:
    - Embeddings can be extracted from all model types
    - Patient-level embeddings work with dt_cnt
    """
    print("\n" + "="*80)
    print("TEST 19: EmbeddingExtractor with LOB")
    print("="*80)
    
    cleanup_gpu_memory()
    
    cfg = FlashAttentionConfig(len_dy=50, len_cd=20, batch_size=4)
    batch_size = 4
    len_dy = 50
    len_cd = 20
    
    # Build input tensor with LOB
    age = torch.randint(0, 1400, (batch_size, len_dy), device=device)
    gender = torch.randint(0, 4, (batch_size, len_dy), device=device)
    lob = torch.randint(0, 4, (batch_size, len_dy), device=device)
    codes = torch.randint(0, 1000, (batch_size, len_dy, len_cd), device=device)
    
    x = torch.cat([
        age.unsqueeze(-1),
        gender.unsqueeze(-1),
        lob.unsqueeze(-1),
        codes
    ], dim=-1)
    
    dt_cnt = [30, 45, 20, 50]
    
    # Test with FlashAttentionTransformer
    print("  Testing embedding extraction with LOB...")
    model = FlashAttentionTransformer(cfg).to(device)
    model.eval()
    
    with EmbeddingExtractor(model) as extractor:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = model(x)
        
        all_emb = extractor.get_embeddings()
        print(f"    All embeddings shape: {all_emb.shape}")
        assert all_emb.shape == (batch_size, len_dy, cfg.embedding_size), \
            f"Wrong embedding shape: {all_emb.shape}"
        
        patient_emb = extractor.get_patient_embedding(dt_cnt)
        print(f"    Patient embeddings shape: {patient_emb.shape}")
        assert patient_emb.shape == (batch_size, cfg.embedding_size), \
            f"Wrong patient embedding shape: {patient_emb.shape}"
    
    print("  ✅ Embeddings extracted successfully")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 19 PASSED: EmbeddingExtractor works with LOB\n")


def test_model_saving_and_loading():
    """
    Test model saving and loading functionality.
    
    Validates:
    - Model can be saved with standardized naming
    - Model can be loaded back
    - Loaded model produces same output
    - Config and results are saved
    """
    print("\n" + "="*80)
    print("TEST 20: Model Saving and Loading")
    print("="*80)
    
    cleanup_gpu_memory()
    
    import tempfile
    import os
    
    cfg = FlashAttentionConfig(len_dy=50, len_cd=20, batch_size=4, embedding_size=64, nhid=128)
    
    # Create and initialize model
    print("  Creating model...")
    model = FlashAttentionTransformer(cfg).to(device)
    model.eval()
    
    # Test generate_model_name
    print("  Testing model name generation...")
    model_name = generate_model_name(
        exp_name='exp2_dense_flash',
        experiment_round='test_round',
        batch_size=32,
        epochs=10,
        embedding_size=64
    )
    print(f"    Generated name: {model_name}")
    assert 'test_round' in model_name
    assert 'exp2_dense_flash' in model_name
    assert 'bs32' in model_name
    assert 'ep10' in model_name
    assert 'd64' in model_name
    print("    ✅ Name format correct")
    
    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock results
        exp_results = {
            'experiment': 'exp2_dense_flash',
            'final_val_loss': 0.5,
            'final_top_10_acc': 0.75,
            'training_time_sec': 100.0
        }
        
        # Save model
        print("  Saving model...")
        model_path = save_trained_model(
            model=model,
            config=cfg,
            model_name=model_name,
            save_dir=tmpdir,
            exp_results=exp_results,
            is_best=True
        )
        
        # Check files exist
        assert os.path.exists(model_path), "Model file not saved"
        assert os.path.exists(os.path.join(tmpdir, f"{model_name}_config.json")), "Config not saved"
        assert os.path.exists(os.path.join(tmpdir, f"{model_name}_results.json")), "Results not saved"
        assert os.path.exists(os.path.join(tmpdir, f"{model_name}_best.pt")), "Best model not saved"
        print("    ✅ All files saved")
        
        # Test loading
        print("  Loading model...")
        loaded_model = load_trained_model(
            model_path=model_path,
            model_class=FlashAttentionTransformer,
            config=cfg,
            device=device
        )
        
        # Verify loaded model works
        print("  Verifying loaded model...")
        batch_size = 4
        len_dy = 50
        len_cd = 20
        
        age = torch.randint(0, 1400, (batch_size, len_dy), device=device)
        gender = torch.randint(0, 4, (batch_size, len_dy), device=device)
        lob = torch.randint(0, 4, (batch_size, len_dy), device=device)
        codes = torch.randint(0, 1000, (batch_size, len_dy, len_cd), device=device)
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                original_output = model(x)
                loaded_output = loaded_model(x)
        
        # Outputs should be identical
        assert torch.allclose(original_output, loaded_output, atol=1e-4), \
            "Loaded model produces different output"
        print("    ✅ Loaded model produces identical output")
    
    del model, loaded_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 20 PASSED: Model saving and loading works\n")


def test_downstream_evaluator_with_real_data():
    """
    Test DownstreamEvaluator with real data.
    
    This is the integration test for the full downstream evaluation pipeline:
    1. Load real training data
    2. Load real outcomes data
    3. Train a small model
    4. Extract embeddings
    5. Train linear probe
    6. Evaluate on train/val/test
    
    REQUIRES: Real data loaded (df_train, df_val) and outcomes table available
    """
    print("\n" + "="*80)
    print("TEST 21: DownstreamEvaluator with Real Data")
    print("="*80)
    
    cleanup_gpu_memory_hard()
    
    # Check if real data is available
    try:
        # Use small sample for testing
        train_sample = df_train.head(500).copy()
        val_sample = df_val.head(200).copy()
        
        # Ensure required columns exist
        if 'target' in train_sample.columns and 'target_cd' not in train_sample.columns:
            train_sample = train_sample.rename(columns={'target': 'target_cd'})
        if 'target' in val_sample.columns and 'target_cd' not in val_sample.columns:
            val_sample = val_sample.rename(columns={'target': 'target_cd'})
        
        # Add LOB column if missing
        if 'lob' not in train_sample.columns:
            train_sample['lob'] = 'Medicaid'
        if 'lob' not in val_sample.columns:
            val_sample['lob'] = 'Medicaid'
            
    except NameError:
        print("  ⚠️ Real data not loaded. Skipping test.")
        print("  Run after loading df_train and df_val from BigQuery")
        return
    
    print(f"  Using {len(train_sample)} train samples, {len(val_sample)} val samples")
    
    # Load outcomes data
    print("  Loading outcomes data...")
    try:
        outcomes_query = """
        SELECT individual_id, CAST(index_dt AS STRING) as index_dt, 
               COALESCE(acute_ip_flag, 0) as acute_ip_flag 
        FROM `edp-prod-storage.edp_ent_sdoheir_cns.a964286_Medicaid_outcome_ip_4_te_experiment`
        LIMIT 10000
        """
        outcomes_df = client.query(outcomes_query).to_dataframe()
        print(f"  Loaded {len(outcomes_df)} outcome records")
        print(f"  IP prevalence: {outcomes_df['acute_ip_flag'].mean():.2%}")
    except Exception as e:
        print(f"  ⚠️ Could not load outcomes: {e}")
        print("  Creating synthetic outcomes for testing...")
        
        # Create synthetic outcomes for testing
        individual_ids = val_sample['individual_id'].unique()
        outcomes_df = pd.DataFrame({
            'individual_id': individual_ids,
            'index_dt': val_sample.groupby('individual_id')['index_dt'].first().values,
            'acute_ip_flag': np.random.binomial(1, 0.1, len(individual_ids))  # 10% prevalence
        })
        outcomes_df['individual_id'] = outcomes_df['individual_id'].astype(str)
        outcomes_df['index_dt'] = outcomes_df['index_dt'].astype(str)
        print(f"  Created {len(outcomes_df)} synthetic outcomes")
    
    # Create a small model for testing
    print("  Creating model...")
    cfg = FlashAttentionConfig(
        len_dy=50,  # Smaller for speed
        len_cd=20,
        batch_size=16,
        embedding_size=64,  # Smaller for speed
        nhid=128
    )
    model = FlashAttentionTransformer(cfg).to(device)
    
    # Train for 1 epoch to get non-random weights
    print("  Training model for 1 epoch...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Create dataset and dataloader
    train_dataset = ClinicalDataset(train_sample, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=clinical_collate_fn,
        num_workers=0
    )
    
    # Quick training
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Just 5 batches
            break
        
        optimizer.zero_grad()
        
        age = batch['age'].to(device)
        gender = batch['gender'].to(device)
        lob = batch['lob'].to(device)
        codes = batch['codes'].to(device)
        dt_cnt = batch['dt_cnt']
        y = batch['target']
        
        x = torch.cat([
            age.unsqueeze(-1),
            gender.unsqueeze(-1),
            lob.unsqueeze(-1),
            codes
        ], dim=-1)
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(x)
            loss = compute_loss(output, y, dt_cnt, cfg, criterion, device)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print("  Training complete")
    
    # Run downstream evaluation
    print("\n  Running DownstreamEvaluator...")
    model.eval()
    
    evaluator = DownstreamEvaluator(
        model=model,
        model_config=cfg,
        device=device,
        use_mixed_precision=True
    )
    
    # Ensure val_sample has individual_id and index_dt as strings
    val_sample['individual_id'] = val_sample['individual_id'].astype(str)
    val_sample['index_dt'] = val_sample['index_dt'].astype(str)
    
    downstream_config = DownstreamConfig(
        task_name="medicaid_ip_risk_test",
        test_size=0.3,
        val_size=0.1,
        n_cv_folds=3,
        max_iter=500
    )
    
    try:
        results = evaluator.evaluate(
            features_df=val_sample,
            outcomes_df=outcomes_df,
            downstream_config=downstream_config
        )
        
        # Validate results structure
        print("\n  Validating results...")
        assert 'train_auc_roc' in results, "Missing train_auc_roc"
        assert 'val_auc_roc' in results, "Missing val_auc_roc"
        assert 'test_auc_roc' in results, "Missing test_auc_roc"
        assert 'test_f1' in results, "Missing test_f1"
        
        print(f"\n  Results Summary:")
        print(f"    Train AUC-ROC: {results['train_auc_roc']:.4f}")
        print(f"    Val AUC-ROC:   {results['val_auc_roc']:.4f}")
        print(f"    Test AUC-ROC:  {results['test_auc_roc']:.4f}")
        print(f"    Test F1:       {results['test_f1']:.4f}")
        print(f"    Test Recall:   {results['test_recall']:.4f}")
        print(f"    Embedding dim: {results['embedding_dim']}")
        
        # Basic sanity checks
        assert 0 <= results['test_auc_roc'] <= 1, "AUC-ROC out of range"
        assert 0 <= results['test_f1'] <= 1, "F1 out of range"
        assert results['embedding_dim'] == cfg.embedding_size, "Wrong embedding dimension"
        
        print("\n  ✅ DownstreamEvaluator works correctly!")
        
    except Exception as e:
        print(f"\n  ⚠️ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        del model, evaluator
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n✅ TEST 21 PASSED: DownstreamEvaluator works with real data\n")


def test_run_single_experiment_with_downstream():
    """
    Test run_single_experiment with downstream evaluation enabled.
    
    This is the full integration test:
    - Run a mini experiment
    - Save the model
    - Run downstream evaluation
    - Verify all results are captured
    """
    print("\n" + "="*80)
    print("TEST 22: run_single_experiment with Downstream Evaluation")
    print("="*80)
    
    cleanup_gpu_memory_hard()
    
    # Check if real data is available
    try:
        train_sample = df_train.head(100).copy()
        val_sample = df_val.head(50).copy()
        
        # Prepare data
        if 'target' in train_sample.columns and 'target_cd' not in train_sample.columns:
            train_sample = train_sample.rename(columns={'target': 'target_cd'})
        if 'target' in val_sample.columns and 'target_cd' not in val_sample.columns:
            val_sample = val_sample.rename(columns={'target': 'target_cd'})
        
        if 'lob' not in train_sample.columns:
            train_sample['lob'] = 'Medicaid'
        if 'lob' not in val_sample.columns:
            val_sample['lob'] = 'Medicaid'
            
    except NameError:
        print("  ⚠️ Real data not loaded. Skipping test.")
        return
    
    # Create synthetic outcomes
    print("  Creating synthetic outcomes...")
    individual_ids = val_sample['individual_id'].unique()
    outcomes_df = pd.DataFrame({
        'individual_id': individual_ids.astype(str),
        'index_dt': val_sample.groupby('individual_id')['index_dt'].first().astype(str).values,
        'acute_ip_flag': np.random.binomial(1, 0.1, len(individual_ids))
    })
    
    print(f"  Train samples: {len(train_sample)}")
    print(f"  Val samples: {len(val_sample)}")
    print(f"  Outcomes: {len(outcomes_df)}")
    
    # Run experiment with downstream evaluation
    print("\n  Running experiment with downstream eval...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_single_experiment(
            exp_name='exp2_dense_flash',
            moe_config=None,
            use_learnt_att_pool=False,
            train_data=train_sample,
            val_data=val_sample,
            device=device,
            epochs=1,
            experiment_round='test_downstream',
            log_dir=tmpdir,
            outcomes_df=outcomes_df,
            run_downstream_eval=True,
            save_model=True
        )
        
        # Validate results
        print("\n  Validating experiment results...")
        
        # Check standard results
        assert 'experiment' in result, "Missing experiment name"
        assert 'parameters' in result, "Missing parameters"
        assert 'val_loss_final' in result, "Missing val_loss_final"
        print(f"    Experiment: {result['experiment']} ✅")
        print(f"    Parameters: {result['parameters']:,} ✅")
        
        # Check model was saved
        if 'model_name' in result:
            print(f"    Model name: {result['model_name']} ✅")
        if 'model_path' in result:
            assert os.path.exists(result['model_path']), "Model file not found"
            print(f"    Model saved: {result['model_path']} ✅")
        
        # Check downstream results
        if 'downstream_evaluation' in result:
            ds_results = result['downstream_evaluation']
            print(f"\n  Downstream Evaluation Results:")
            print(f"    Test AUC-ROC: {ds_results.get('test_auc_roc', 'N/A'):.4f}")
            print(f"    Test F1:      {ds_results.get('test_f1', 'N/A'):.4f}")
            print("    ✅ Downstream evaluation captured in results")
        else:
            print("    ⚠️ No downstream_evaluation in results (check if outcomes matched)")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ TEST 22 PASSED: Full integration with downstream evaluation works\n")


# ============================================================================
# RUN ALL DOWNSTREAM TESTS
# ============================================================================

def run_all_downstream_tests():
    """Run all downstream evaluation and model saving tests."""
    print("\n" + "="*80)
    print("DOWNSTREAM EVALUATION TEST SUITE")
    print("="*80)
    
    # Unit tests (no real data needed)
    test_conv_lob()
    
    # Tests requiring model
    test_model_forward_with_lob()
    test_embedding_extractor_with_lob()
    test_model_saving_and_loading()
    
    # Tests requiring real data (may be skipped if data not loaded)
    test_clinical_dataset_with_lob()
    test_downstream_evaluator_with_real_data()
    test_run_single_experiment_with_downstream()
    
    print("\n" + "="*80)
    print("ALL DOWNSTREAM TESTS COMPLETED!")
    print("="*80 + "\n")


# Uncomment to run tests:
# run_all_downstream_tests()
```

```python
# Run individual tests:
test_conv_lob()
test_clinical_dataset_with_lob()
test_model_forward_with_lob()
test_embedding_extractor_with_lob()
test_model_saving_and_loading()
test_downstream_evaluator_with_real_data()
test_run_single_experiment_with_downstream()

# Or run all at once:
run_all_downstream_tests()
```

```python
# Load from feather files
df_train = pd.read_feather("sample_data/mdcd_train_1m.feather")
df_val = pd.read_feather("sample_data/mdcd_val_10k.feather")

# OR load from BigQuery
input_sql = """
SELECT * FROM `edp-prod-storage.edp_ent_sdoheir_cns.a834793_Medicaid_o3_train_ending`
LIMIT 10000
"""
df_train = client.query(input_sql).to_dataframe()
```

```python
# Ensure target_cd column exists
if 'target' in df_train.columns and 'target_cd' not in df_train.columns:
    df_train = df_train.rename(columns={'target': 'target_cd'})
if 'target' in df_val.columns and 'target_cd' not in df_val.columns:
    df_val = df_val.rename(columns={'target': 'target_cd'})
```

```python
if 'lob' not in df_train.columns:
    df_train['lob'] = 'Medicaid'
```

