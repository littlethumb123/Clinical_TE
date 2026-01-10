# Session Summary: Memory OOM in `prepare_data_once`

---

## 1. Problem Statement

**Symptom**: `prepare_data_once()` function kills the GCP Vertex AI Workbench kernel when processing **3.4M members** data. The kernel dies silently (no explicit OOM error reported).

**Observations**:
- ✅ 1.5M members with 624 GB memory: **Works**
- ❌ 3.4M members with 624 GB memory: **Kernel dies mid-processing**
- ❌ 3.4M members with 416 GB memory: **Kernel dies immediately**

---

## 2. Root Cause Analysis

### 2.1 Memory Calculation for `ClinicalDataset`

The `ClinicalDataset.__init__` pre-allocates these tensors with `dtype=torch.long` (int64 = 8 bytes):

```python
self.ages = torch.zeros(len(df), config.len_dy, dtype=torch.long)     # (N, 200)
self.genders = torch.zeros(len(df), config.len_dy, dtype=torch.long)  # (N, 200)
self.codes = torch.zeros(len(df), config.len_dy, config.len_cd, dtype=torch.long)  # (N, 200, 80) 🚨
self.lobs = torch.zeros(len(df), config.len_dy, dtype=torch.long)     # (N, 200)
self.targets = []  # Python list
```

### 2.2 Memory Breakdown for 3.4M Samples

| Component | Calculation | Memory |
|-----------|-------------|--------|
| `self.codes` | 3.4M × 200 × 80 × 8 bytes | **435.2 GB** 🚨 |
| `self.ages` | 3.4M × 200 × 8 bytes | 5.44 GB |
| `self.genders` | 3.4M × 200 × 8 bytes | 5.44 GB |
| `self.lobs` | 3.4M × 200 × 8 bytes | 5.44 GB |
| `self.targets` (Python list) | ~60 GB overhead | ~60 GB |
| **Subtotal (Tensors)** | | **~511 GB** |
| Original DataFrames (not cleaned) | | **~200 GB** |
| Intermediate string lists | | **~80 GB** |
| **PEAK TOTAL** | | **~821 GB** |

### 2.3 Root Causes Identified

| Root Cause | Problem | Impact |
|------------|---------|--------|
| **#1: Wrong dtype** | `torch.long` (int64) used for all tensors | `codes` alone uses 435 GB instead of 109 GB |
| **#2: No memory cleanup** | DataFrames and string lists never deleted | +280 GB wasted memory |
| **#3: Over-sized dtypes** | int64 for ages (max 1439), genders (max 3), lobs (max 3) | +15 GB wasted |

### 2.4 Why 1.5M Worked, 3.4M Failed

| Dataset | Peak Memory | 624 GB Instance | 416 GB Instance |
|---------|-------------|-----------------|-----------------|
| 1.5M | ~362 GB | ✅ Fits | ✅ Fits (barely) |
| 3.4M | ~821 GB | ❌ OOM | ❌ Immediate OOM |

---

## 3. Solution

### 3.1 Fix Data Types (Saves ~340 GB)

| Tensor | Before | After | Justification |
|--------|--------|-------|---------------|
| `codes` | `int64` (8 bytes) | `int32` (4 bytes) | Max value 75,516 fits in int32 |
| `ages` | `int64` (8 bytes) | `int16` (2 bytes) | Max value 1,439 fits in int16 |
| `genders` | `int64` (8 bytes) | `int8` (1 byte) | Max value 3 fits in int8 |
| `lobs` | `int64` (8 bytes) | `int8` (1 byte) | Max value 3 fits in int8 |

### 3.2 Add Memory Cleanup (Saves ~280 GB)

- Delete intermediate string lists after processing
- Call `gc.collect()` periodically and after completion

### 3.3 Keep `targets` as Python List (No Change)

- The variable-length-per-day structure is intentional
- `clinical_collate_fn` already converts to `target_multihot` tensor during batching
- No need to introduce new constants like `MAX_CODES_PER_DAY`

---

## 4. Final Memory Estimate After Optimization

| Component | After Optimization |
|-----------|-------------------|
| `self.codes` (int32) | **108.8 GB** |
| `self.ages` (int16) | 1.4 GB |
| `self.genders` (int8) | 0.7 GB |
| `self.lobs` (int8) | 0.7 GB |
| `self.targets` (Python list) | ~60 GB |
| DataFrames (deleted) | 0 GB |
| String lists (deleted) | 0 GB |
| **PEAK TOTAL** | **~172 GB** |

**Result**: Fits comfortably in 416 GB with ~240 GB headroom.

---

## 5. Implementation

Switch to Agent mode to apply the corrected `ClinicalDataset.__init__` with:
1. `dtype=torch.int32` for codes
2. `dtype=torch.int16` for ages
3. `dtype=torch.int8` for genders and lobs
4. Memory cleanup with `del` + `gc.collect()`
5. Keep `self.targets` as Python list (unchanged)
6. Add `.long()` conversion in `__getitem__` for embedding compatibility