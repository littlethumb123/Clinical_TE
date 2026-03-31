# Memory issues with large dataset preprocessing

## Metadata

- **Conversation ID:** 5afa2068-ee35-4936-a19c-9bdb8b95d8e9
- **Mode:** agent
- **Created:** 2026-03-18 13:34:46
- **Last Updated:** 2026-03-18 15:30:25
- **Branch:** main
- **Files Changed:** 1
- **Lines Added:** 1170
- **Lines Removed:** 1017
- **Summary:** Edited legacy_full_training.ipynb

---

## Conversation

### 🧑 User

When I run 11M data to preprocess it in the code @dev/legacy/legacy_full_training.ipynb I got the died kernel when it runs Pre-processing 1800000/7907218; I think it is OOM error and too large dataset due to the ClinicalDataset class. inspect the root cause and understand which parts take the most memory and why and proposed solutions for this issue to make the functions to accomodate the 11M data.

---

### 🤖 Assistant

I'll start by reading the notebook and understanding the `ClinicalDataset` class and preprocessing pipeline to diagnose the OOM issue.

---

### 🤖 Assistant

Let me find the `ClinicalDataset` class in the notebook specifically, along with the preprocessing functions.

---

### 🤖 Assistant

Excellent, I can see the prior work on `ClinicalDatasetLazy` for `moe_flashattn_4.py`. Now let me read the actual `ClinicalDataset` class in the legacy notebook, plus the preprocessing functions it uses.

---

### 🤖 Assistant

Now let me read the `conv_lob` function and check where the data is being loaded and preprocessed in the actual execution cells.

---

### 🤖 Assistant

Now let me read the actual data loading and preprocessing execution cell more carefully, and the `CachedClinicalDataset` / save/load functions.

---

### 🤖 Assistant

Now I have a comprehensive understanding of the problem. Let me also check the `TRAINING_TABLE` variable and see if there's a reference to the 11M table.

---

### 🤖 Assistant

Here is the full root cause analysis and proposed solutions.

---

## Root Cause Analysis: OOM at ~1.8M/7.9M during `ClinicalDataset.__init__`

The kernel dies at **"Pre-processing 1800000/7907218"** because `ClinicalDataset` is an **eager dataset** — it parses every row and stores the fully materialized numpy arrays in a Python list (`self.samples`) during `__init__`. This is the exact same architectural problem that was diagnosed and solved in `moe_flashattn_4.py` on March 3, 2026 (documented in `docs/progress/2026-03-03_lazy_dataset_scaling_11M.md`), but that fix was never ported to this legacy notebook.

### Memory math — where it all goes

Each sample in `self.samples` stores:

| Field | Shape | dtype | Bytes/sample |
|---|---|---|---|
| `codes` | `(200, 80)` | int64 | **128,000 B** |
| `age` | `(200,)` | int64 | 1,600 B |
| `gender` | `(200,)` | int64 | 1,600 B |
| `lob` | `(200,)` | int64 | 1,600 B |
| `target` | ~200 lists of lists | Python objects | ~4,000-8,000 B |
| dict overhead + `entity_id` + `dt_cnt` | — | — | ~500 B |
| **Total per sample** | | | **~137 KB** |

The **`codes` array is the dominant cost** at 128 KB per sample (93% of per-sample memory).

Scaling this up:

| Dataset size | `codes` memory alone | Total `self.samples` memory |
|---|---|---|
| 1.8M (where it dies) | **220 GB** | ~240 GB |
| 7.9M (full query) | **965 GB** | ~1,032 GB |
| 11M (target) | **1,340 GB** | ~1,440 GB |

Your machine almost certainly has somewhere between 128-416 GB of RAM. The kernel gets SIGKILL'd by the Linux OOM killer at ~1.8M samples because the cumulative `self.samples` list plus the still-alive DataFrame (`df`) plus Python/PyTorch overhead exceeds available memory. Since SIGKILL cannot be caught, there's no Python traceback — just "Kernel died."

### Secondary memory pressure: concurrent DataFrame

During `ClinicalDataset.__init__`, the original `df` (with 7.9M rows of string columns) is still in memory. For 7.9M rows, the DataFrame itself is roughly **60-100 GB** (string columns for `cd`, `age_in_months`, `lob`, `target_cd` are large). So at the 1.8M mark, you have:

- **~240 GB** in `self.samples` (growing)
- **~60-100 GB** in `df` (static, not freed until after `ClinicalDataset.__init__` returns)
- **~10-20 GB** in Python/PyTorch/CUDA overhead

Total: ~310-360 GB, which triggers OOM on most machines.

### Additional inefficiency: int64 everywhere

The `ClinicalDataset` stores all arrays as **int64** (8 bytes per element). But:
- `codes` values are 0-75,516 → fits in **int32** (4 bytes) → would halve `codes` memory
- `age` values are 0-1,439 → fits in **int16** (2 bytes)
- `gender` values are small → fits in **int16**
- `lob` values are 0-3 → fits in **int8** (1 byte)

The `CachedClinicalDataset` already uses these compact dtypes, but `ClinicalDataset` does not — it converts to int64 numpy arrays at parse time.

---

## Proposed Solutions

### Solution 1: Port `ClinicalDatasetLazy` from `moe_flashattn_4.py` (Recommended)

This is the exact same solution that was designed and validated for the MoE pipeline. The approach:

- **Store raw strings** from the DataFrame in `__init__` (just `.tolist()` per column — takes ~1-2 seconds)
- **Parse on-the-fly** in `__getitem__` using the existing `conv_*` / `_parse_*` functions
- **DataLoader workers** parallelize the parsing, fully hiding the ~5ms/sample CPU cost behind GPU computation

Memory impact for 11M:
- Strings: ~20-30 GB (vs 1,440 GB for materialized arrays)
- Init time: ~2 seconds (vs hours)
- Training throughput: unchanged (CPU parsing overlaps with GPU)

This requires adding `ClinicalDatasetLazy` to the notebook and routing the data loading cell to use it.

### Solution 2: Chunked preprocessing with disk-backed shards (Build on existing cache)

You already have `save_dataset_cache` / `load_dataset_cache` with `CachedClinicalDataset`. The problem is you can't get through `ClinicalDataset.__init__` to *create* the cache. The fix:

- Process in chunks (e.g., 500K rows at a time) from the DataFrame
- Write each chunk's shard to disk immediately, then delete from memory
- After all shards are written, use `load_dataset_cache` to load them back as compact numpy arrays (int32/int16/int8)

Memory math with compact dtypes for 11M loaded back:
- `codes`: 11M × 200 × 80 × 4B (int32) = **~670 GB** — still too large for RAM

This approach works for **saving** without OOM, but `CachedClinicalDataset` still loads all `codes` arrays into a contiguous numpy array at load time. For 11M, that's ~670 GB even with int32. So the cache approach alone doesn't solve the problem unless you also add memory-mapped loading.

### Solution 3: Memory-mapped cache (Extension of Solution 2)

Write the compact arrays to disk, then load them via `np.memmap` instead of `np.load`. The OS pages in only the data being accessed.

- Init: near-instant (just opens file handles)
- RAM: only the pages being actively used (~a few GB)
- I/O overhead: depends on SSD speed, ~1-5ms/batch additional latency

This is more complex to implement but would make even 11M trivially loadable on a 64 GB machine.

### Recommended approach: Solution 1 (Lazy Dataset)

This is the highest-impact, lowest-complexity fix. It was already designed, implemented, and validated at 6.3M scale in `moe_flashattn_4.py`. Porting it to the legacy notebook requires:

1. Add `ClinicalDatasetLazy` class (stores strings, parses in `__getitem__`)
2. Modify the data loading cell to use `ClinicalDatasetLazy(df)` instead of `ClinicalDataset(df)`
3. Add streaming code frequency computation (parse target strings without materializing full targets list)
4. Delete `df` after creating the dataset (strings are already copied)

Would you like me to implement Solution 1 — porting `ClinicalDatasetLazy` into the legacy notebook?

---

### 🧑 User

Ok add ClinicalDatasetLazy to the legacy full training notebook

---

### 🤖 Assistant

Let me first read the existing `ClinicalDatasetLazy` from `moe_flashattn_4.py` to port it faithfully, and also re-read the exact notebook cells around where I need to insert.

---

### 🤖 Assistant

Now let me also read the `conv_age_gender`, `conv_lob`, and `conv_target` functions from `moe_flashattn_4.py` so I understand how they differ from the legacy notebook's `_parse_*` methods.

---

### 🤖 Assistant

Now let me read the exact notebook cells to understand where to insert. I need to see the cell boundaries around the `ClinicalDataset` class, the `conv_lob` function, and the data loading cell.

---

### 🤖 Assistant

Now I need to figure out the notebook cell structure. Let me find the cell that contains `ClinicalDataset` and its boundaries.

---

### 🤖 Assistant

Now let me identify the cell indices. The `ClinicalDataset` is in a cell that starts at line 1117 (the code cell). Let me find which cell index that is, and read the cell structure around it.

---

### 🤖 Assistant

Good. Now I have the full picture. The cell at line 1117 contains `conv_lob`, `ClinicalDataset`, `clinical_collate_fn`, and `load_training_data` — all in one cell. I need to count which cell index this is.

Let me count the cell_type lines up to 1117: lines 4, 42, 85, 93, 175, 184, 582, 728, 905, 992, 1001, 1109, 1117 → that's cell index 12 (0-indexed). The markdown "## Dataset & Data Loading" at line 1109 is index 11, and the code cell at 1117 is index 12.

Now let me also read the data loading execution cell (around line 3177) and the cell just before it to understand the full context I need to modify.

---

### 🤖 Assistant

Now I have a complete understanding. Let me identify the cell indices precisely, then make the edits.

The cell_type markers are at lines: 4(md), 42(code), 85(md), 93(code), 175(md), 184(code), 582(code), 728(code), 905(code), 992(md), 1001(code), 1109(md), 1117(code), 1298(md), 1307(code)...

So the cell containing `ClinicalDataset` (starting at line 1117) is **cell index 12** (0-indexed).

The data loading execution cell (starting at line 3177) — let me count: ...1460(md), 1470(code), 1786(code), 1899(code), 1973(md), 1981(code), 2127(md), 2139(code), 2198(code), 2355(code), 2425(code), 2483(code), 2545(code), 2625(code), 2697(code), 2806(code), 2865(code), 2965(md), 2977(code), 3104(md), 3113(code), 3158(code), 3177(code) — that's cell index **34**.

Now I'll implement the changes. The strategy:

1. **Cell 12**: Add `ClinicalDatasetLazy` class and `compute_code_freq_from_strings` function to the same cell, after the existing code
2. **Cell 34**: Add a new cell with the lazy data loading pipeline

Let me first verify my cell counting by reading the cell ID of cell 12.

---

### 🤖 Assistant

Good, cell id `c95320b9` with execution_count 27. Now let me implement. I'll add `ClinicalDatasetLazy` and `compute_code_freq_from_strings` to the existing Dataset cell (cell 12), and then add a new lazy data loading cell after the existing eager loading cell.

First, let me edit cell 12 to append `ClinicalDatasetLazy` and the streaming frequency function after `load_training_data`.

---

### 🤖 Assistant

Now let me read the exact content of cell 12 to find the right string to match for the edit.

---

### 🤖 Assistant

I'll append `ClinicalDatasetLazy` and the streaming frequency function to the end of cell 12.

---

### 🤖 Assistant

Now I need to add a new cell with the lazy data loading pipeline. Since the cell indices shifted after the edit, let me re-find the data loading cell. First let me verify the current state by locating the existing eager loading cell.

---

### 🤖 Assistant

Now I need to find the cell index of the eager loading cell (id `00e0793b`). Let me count the cell_type lines in the current state. Since I added content to cell 12, the line numbers shifted. Let me recount.

---

### 🤖 Assistant

Let me count more precisely. Cell index 0 is at line 4 (markdown), 1 at line 42 (code), 2 at 85 (md), 3 at 93 (code), 4 at 174 (md), 5 at 183 (code), 6 at 580 (code), 7 at 725 (code), 8 at 901 (code), 9 at 987 (md), 10 at 996 (code), 11 at 1103 (md), 12 at 1111 (code — this is the one I edited), 13 at 1441 (md), 14 at 1450 (code)...

Continuing: 15 at 1602 (md), 16 at 1612 (code), 17 at 1928 (code), 18 at 2040 (code), 19 at 2113 (md), 20 at 2121 (code), 21 at 2266 (md), 22 at 2278 (code), 23 at 2336 (code), 24 at 2492 (code), 25 at 2561 (code), 26 at 2618 (code), 27 at 2679 (code), 28 at 2758 (code), 29 at 2829 (code), 30 at 2937 (code), 31 at 2995 (code), 32 at 3094 (md), 33 at 3106 (code), 34 at 3232 (md), 35 at 3241 (code), 36 at 3285 (code), 37 at 3304 (code — let me check this).

---

### 🤖 Assistant

So cell 37 is the `import time` cell (id `57286622`). Cell 38 is the eager loading cell (id `00e0793b`). Let me verify.

---

### 🤖 Assistant

Cell index 38 is the eager loading cell. I'll insert a new cell at index 39 (right after it) with the lazy loading alternative.

---

### 🤖 Assistant

Let me verify the notebook is valid by reading the edited cells and checking the structure.

---

### 🤖 Assistant

Good. Now let me verify the full content of the new lazy loading cell and the end of the definitions cell to make sure they're correct.

---

### 🤖 Assistant

Now let me verify the lazy loading cell that was inserted.

---

### 🤖 Assistant

Let me read the rest of the new cell.

---

### 🤖 Assistant

Everything looks correct. Now let me also verify the notebook parses as valid JSON.

---

### 🤖 Assistant

Let me also do a quick structural validation that the notebook has the expected number of cells and the new class/function are properly placed.

---

### 🤖 Assistant

All changes are in place. Here's what was added:

---

### Changes made to `dev/legacy/legacy_full_training.ipynb`

**Cell 12** (Dataset & Data Loading definitions) — appended two new components:

1. **`ClinicalDatasetLazy`** class — drop-in replacement for `ClinicalDataset`:
   - `__init__` stores raw strings via `.tolist()` per column (~2 seconds, ~20-30 GB for 11M vs ~1,440 GB eager)
   - `_parse_age_gender`, `_parse_codes`, `_parse_target` methods identical to `ClinicalDataset` but with added NaN/None guards for robustness
   - `__getitem__` parses on-the-fly and returns the exact same dict format — `clinical_collate_fn`, DataLoaders, and training loop require zero changes
   - Applies the same `minimum_mth_training` filter as `ClinicalDataset`

2. **`compute_code_freq_from_strings`** function — streaming code frequency computation:
   - Parses target strings directly without materializing the full `targets` list
   - Matches the eager frequency computation behavior (skips code 0)
   - Progress reporting every 1M samples

**Cell 39** (new) — Lazy data loading pipeline:
   - Same structure as the eager cell (cell 38) but uses `ClinicalDatasetLazy` instead of `ClinicalDataset`
   - Uses `compute_code_freq_from_strings` for streaming frequency computation
   - Produces the same `dataset`, `train_dataset`, `val_dataset`, `code_freq` variables — all downstream cells work unchanged

### How to use

For 11M data: run the **"1-ALT. LOAD DATA WITH LAZY DATASET"** cell (cell 39) **instead of** the eager "1. LOAD DATA" cell (cell 38). Everything downstream — DataLoaders, model creation, training loop — works identically since `ClinicalDatasetLazy.__getitem__` returns the same dict format.

### Expected performance for 11M

| Metric | Eager (`ClinicalDataset`) | Lazy (`ClinicalDatasetLazy`) |
|---|---|---|
| Init time | 8-14 hours | ~2 seconds |
| Peak RAM (init) | ~1,440 GB | ~20-30 GB |
| Code freq compute | seconds (from materialized list) | ~5-10 minutes (streaming) |
| Training throughput | baseline | same (parsing hidden behind GPU) |

---

### 🧑 User

Ok when I run the save_dataset_cache() and it reports the following error; thegiven the change in lazy function, anything we need to change in save and load dataset section SAVE / LOAD PREPROCESSED DATASET. Carefully inspect and think about it; and make correspodning modifications to the functions making sure both saving and load functions works; also for the folllowing DataLoader; making sure it compatible with the changes as well (if any)

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[22], line 2
      1 # Save it for further usage. 
----> 2 save_dataset_cache(dataset, train_dataset, val_dataset, code_freq)

Cell In[21], line 65, in save_dataset_cache(dataset, train_dataset, val_dataset, code_freq, cache_dir)
     62 ids_shard = []
     64 for j, i in enumerate(range(start, end)):
---> 65     s = dataset.samples[i]
     66     age_arr[j] = s['age'].astype(np.int16)
     67     gender_arr[j] = s['gender'].astype(np.int16)

AttributeError: 'ClinicalDatasetLazy' object has no attribute 'samples'

---

### 🤖 Assistant

The error is clear — `save_dataset_cache` accesses `dataset.samples[i]`, which only exists on the eager `ClinicalDataset`. With `ClinicalDatasetLazy`, there's no `.samples` list; the data lives as raw strings. Let me inspect the full save/load/dataloader cells to understand all the incompatibilities.

---

### 🤖 Assistant

Now let me read the full save/load cell and the DataLoader cell.

---

### 🤖 Assistant

Now let me read the DataLoader cell too.

---

### 🤖 Assistant

Good, I now have the full picture. Let me analyze the issues:

**`save_dataset_cache` issues:**
1. Accesses `dataset.samples[i]` — doesn't exist on `ClinicalDatasetLazy`
2. Assumes each sample has pre-computed numpy arrays (`.astype(np.int16)` etc.) — `ClinicalDatasetLazy` has no pre-computed arrays
3. Accesses `s[entity_id]` — lazy dataset stores IDs in `self.ids` list, not per-sample dicts

**`load_dataset_cache` issues:** None — it creates a `CachedClinicalDataset` from disk, which is independent of how data was saved. But the load path concatenates all `codes` arrays into one big numpy array, which for 11M would be ~670 GB in int32. This defeats the purpose of lazy loading. So `load_dataset_cache` needs an option to return a lazy-compatible format for large datasets.

**DataLoader cell:** No issues — it uses `train_dataset` and `val_dataset` (which are `Subset` wrappers), and `clinical_collate_fn`, all of which work with both `ClinicalDatasetLazy` and `CachedClinicalDataset`.

The fix for `save_dataset_cache`: use `dataset[i]` (the `__getitem__` protocol) instead of `dataset.samples[i]`. This works for both eager and lazy datasets. The returned dict has tensors (from lazy) or numpy arrays (from eager), so the conversion needs to handle both.

For `load_dataset_cache`: for 11M, loading all arrays into RAM defeats the purpose. But the save/load pattern is genuinely useful for medium-scale data (1.5M-6.3M) — it avoids re-querying BigQuery. I'll keep `load_dataset_cache` returning `CachedClinicalDataset` for moderate sizes, and add an alternative `load_dataset_cache_lazy` that loads only strings for 11M scale.

Actually, on reflection, the main purpose of save/load cache is to avoid re-running BigQuery + preprocessing. For the lazy path, you don't need to save parsed arrays — you could save the raw strings. But the user is currently trying to call `save_dataset_cache` on a `ClinicalDatasetLazy` dataset, so the most practical fix is to make `save_dataset_cache` work with both dataset types by using `__getitem__`.

Let me now find the cell index for the save/load cell.

---

### 🤖 Assistant

Let me identify which cell_type line corresponds to the save/load definitions cell. The `SAVE / LOAD PREPROCESSED DATASET` cell starts around line 3525 in the current file. Looking at the list, it's the cell at line 3505 (cell_type appears a few lines before the source). Let me verify:

---

### 🤖 Assistant

That's the save/load cell. Let me count from the beginning to get the cell index. I'll use Python for this.

---

### 🤖 Assistant

Now I have the exact cell indices. Let me now make the edits. The key change in `save_dataset_cache` is to use `dataset[i]` (the `__getitem__` protocol) instead of `dataset.samples[i]`, and handle the fact that lazy `__getitem__` returns tensors while eager returns numpy arrays.

---

### 🤖 Assistant

Now let me check that the `load_dataset_cache` function doesn't need changes. Let me re-read it to confirm, along with verifying the DataLoader cell.

---

### 🤖 Assistant

Let me verify the full edited cell looks correct by reading the save function.

---

### 🤖 Assistant

Everything looks correct. Here's the summary of what was changed and why:

---

### Changes to `save_dataset_cache` (cell 40)

**Root cause:** The function accessed `dataset.samples[i]`, which is an internal attribute of the eager `ClinicalDataset`. `ClinicalDatasetLazy` doesn't have `.samples` — it stores raw strings and parses on-the-fly via `__getitem__`.

**Two changes made:**

1. **Added `_to_numpy` helper** — `ClinicalDatasetLazy.__getitem__` returns `torch.Tensor` values (via `torch.tensor(...)`), while the eager `ClinicalDataset` returns `numpy.ndarray`. The save function needs numpy arrays for `np.savez_compressed`. The helper handles both:

```python
def _to_numpy(val, dtype):
    if isinstance(val, torch.Tensor):
        return val.numpy().astype(dtype)
    return np.asarray(val, dtype=dtype)
```

2. **Replaced `dataset.samples[i]` with `dataset[i]`** — uses the standard `__getitem__` protocol, which works universally across all three dataset types (`ClinicalDataset`, `ClinicalDatasetLazy`, `CachedClinicalDataset`). Also changed `s[entity_id]` to `s.get(entity_id)` for safety.

### What did NOT need changes

- **`load_dataset_cache`** — Unchanged. It reads from disk and creates a `CachedClinicalDataset` with compact numpy arrays. This path is independent of how the data was originally saved. For 1.5M-6.3M scale, loading into `CachedClinicalDataset` works fine (memory ~100-400 GB depending on scale).

- **DataLoader cell** — Unchanged. It wraps `train_dataset` / `val_dataset` (which are `Subset` objects) with `clinical_collate_fn`. The collate function uses `torch.stack` on the returned tensors, which works identically regardless of whether the underlying dataset is eager, lazy, or cached.

### Important caveat for 11M

`save_dataset_cache` with `ClinicalDatasetLazy` will work (no error), but it will be **slow** for 11M — it calls `dataset[i]` for every sample, which triggers on-the-fly parsing. For 11M samples × ~5ms/sample = ~15 hours single-threaded. The save is primarily useful for moderate scales (1.5M-6.3M) where you want to avoid re-querying BigQuery. For 11M, the lazy path is meant to skip caching entirely and parse on-the-fly during training.

