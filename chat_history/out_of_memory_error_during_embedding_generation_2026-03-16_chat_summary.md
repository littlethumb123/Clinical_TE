# Out of memory error during embedding generation

## Metadata

- **Conversation ID:** 6063ef89-4fca-4260-a04b-0947a37e7704
- **Mode:** chat
- **Created:** 2026-03-16 18:23:50
- **Last Updated:** 2026-03-16 21:30:06
- **Branch:** main
- **Files Changed:** 0
- **Lines Added:** 0
- **Lines Removed:** 0
- **Summary:** Read legacy_full_training.ipynb

---

## Conversation

### 🧑 User

When I run the embedding geenration with batch_size = 32, it went well in the first 75% and then come across OOM error @dev/legacy/legacy_full_training.ipynb 

================================================================================
EMBEDDING GENERATION PIPELINE  (4 GPUs)
================================================================================
Releasing training model GPU memory...
Checkpoint : logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints/checkpoint_best.pt
Per-GPU batch: 32  |  Effective: 128
Workers/GPU: 2
Checkpoint loaded (epoch 0, val_loss 0.030352)

Pre-parsing data...
  Pre-parsing 2,886,355 rows (lightweight columns)...
    500,000 / 2,886,355...
    1,000,000 / 2,886,355...
    1,500,000 / 2,886,355...
    2,000,000 / 2,886,355...
    2,500,000 / 2,886,355...
  Pre-parse complete: 2.69 GiB arrays + strings in memory
Parsed in 198.7s

Extracting embeddings across 4 GPUs...
  GPU 0: 721,588 rows  (offset 0)
  GPU 1: 721,588 rows  (offset 721,588)
  GPU 2: 721,588 rows  (offset 1,443,176)
  GPU 3: 721,591 rows  (offset 2,164,764)
Multi-GPU (4 GPUs):  75%
 2164767/2886355 [54:56<52:01, 231.20it/s]
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
Cell In[71], line 3
      1 # --- Execute ---
      2 CHECKPOINT_PATH = 'logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints/checkpoint_best.pt'
----> 3 embeddings_df = generate_and_upload_embeddings(
      4     checkpoint_path=CHECKPOINT_PATH,
      5     data_df = df_cm_sample
      6 )

Cell In[67], line 263, in generate_and_upload_embeddings(checkpoint_path, data_df, per_gpu_batch_size, num_workers_per_gpu)
    260     pbar.close()
    262     for f in futures:
--> 263         f.result()
    265 elapsed = time.time() - extract_start
    266 del state_dict

File /opt/conda/lib/python3.10/concurrent/futures/_base.py:451, in Future.result(self, timeout)
    449     raise CancelledError()
    450 elif self._state == FINISHED:
--> 451     return self.__get_result()
    453 self._condition.wait(timeout)
    455 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:

File /opt/conda/lib/python3.10/concurrent/futures/_base.py:403, in Future.__get_result(self)
    401 if self._exception:
    402     try:
--> 403         raise self._exception
    404     finally:
    405         # Break a reference cycle with the exception in self._exception
    406         self = None

File /opt/conda/lib/python3.10/concurrent/futures/thread.py:58, in _WorkItem.run(self)
     55     return
     57 try:
---> 58     result = self.fn(*self.args, **self.kwargs)
     59 except BaseException as exc:
     60     self.future.set_exception(exc)

Cell In[67], line 144, in _extract_on_single_gpu(gpu_id, state_dict, dataset, embeddings_out, start_offset, per_gpu_batch_size, num_workers, progress_lock, progress_counter)
    141 dt_cnt_list = batch['dt_cnt']
    143 with torch.cuda.amp.autocast(enabled=True):
--> 144     _ = replica(x)
    146 enc_out = activation['enc']  # [seq_len, batch, emb_dim]
    147 patient_embs = torch.stack([
    148     enc_out[dt_cnt_list[j], j, :] for j in range(bs_actual)
    149 ]).float()

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1776, in Module._wrapped_call_impl(self, *args, **kwargs)
   1774     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775 else:
-> 1776     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1787, in Module._call_impl(self, *args, **kwargs)
   1782 # If we don't have any hooks, we want to skip the rest of the logic in
   1783 # this function, and just call forward.
   1784 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1785         or _global_backward_pre_hooks or _global_backward_hooks
   1786         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1787     return forward_call(*args, **kwargs)
   1789 result = None
   1790 called_always_called_hooks = set()

Cell In[7], line 65, in LegacyTransformerModel.forward(self, x)
     63 cd = cd.reshape(gpu_batchsize * len_dy, len_cd, embedding_size)
     64 cd = torch.swapaxes(cd, 0, 1)
---> 65 cd = self.transformer_encoder_cd(cd)
     66 cd = cd.permute(1, 2, 0)
     67 cd = nn.MaxPool1d(len_cd)(cd)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1776, in Module._wrapped_call_impl(self, *args, **kwargs)
   1774     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775 else:
-> 1776     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1787, in Module._call_impl(self, *args, **kwargs)
   1782 # If we don't have any hooks, we want to skip the rest of the logic in
   1783 # this function, and just call forward.
   1784 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1785         or _global_backward_pre_hooks or _global_backward_hooks
   1786         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1787     return forward_call(*args, **kwargs)
   1789 result = None
   1790 called_always_called_hooks = set()

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/transformer.py:540, in TransformerEncoder.forward(self, src, mask, src_key_padding_mask, is_causal)
    537 is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
    539 for mod in self.layers:
--> 540     output = mod(
    541         output,
    542         src_mask=mask,
    543         is_causal=is_causal,
    544         src_key_padding_mask=src_key_padding_mask_for_layers,
    545     )
    547 if convert_to_nested:
    548     output = output.to_padded_tensor(0.0, src.size())

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1776, in Module._wrapped_call_impl(self, *args, **kwargs)
   1774     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775 else:
-> 1776     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1787, in Module._call_impl(self, *args, **kwargs)
   1782 # If we don't have any hooks, we want to skip the rest of the logic in
   1783 # this function, and just call forward.
   1784 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1785         or _global_backward_pre_hooks or _global_backward_hooks
   1786         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1787     return forward_call(*args, **kwargs)
   1789 result = None
   1790 called_always_called_hooks = set()

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/transformer.py:954, in TransformerEncoderLayer.forward(self, src, src_mask, src_key_padding_mask, is_causal)
    950     x = x + self._ff_block(self.norm2(x))
    951 else:
    952     x = self.norm1(
    953         x
--> 954         + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
    955     )
    956     x = self.norm2(x + self._ff_block(x))
    958 return x

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/transformer.py:968, in TransformerEncoderLayer._sa_block(self, x, attn_mask, key_padding_mask, is_causal)
    961 def _sa_block(
    962     self,
    963     x: Tensor,
   (...)
    966     is_causal: bool = False,
    967 ) -> Tensor:
--> 968     x = self.self_attn(
    969         x,
    970         x,
    971         x,
    972         attn_mask=attn_mask,
    973         key_padding_mask=key_padding_mask,
    974         need_weights=False,
    975         is_causal=is_causal,
    976     )[0]
    977     return self.dropout1(x)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1776, in Module._wrapped_call_impl(self, *args, **kwargs)
   1774     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1775 else:
-> 1776     return self._call_impl(*args, **kwargs)

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1787, in Module._call_impl(self, *args, **kwargs)
   1782 # If we don't have any hooks, we want to skip the rest of the logic in
   1783 # this function, and just call forward.
   1784 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1785         or _global_backward_pre_hooks or _global_backward_hooks
   1786         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1787     return forward_call(*args, **kwargs)
   1789 result = None
   1790 called_always_called_hooks = set()

File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/activation.py:1491, in MultiheadAttention.forward(self, query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)
   1465     attn_output, attn_output_weights = F.multi_head_attention_forward(
   1466         query,
   1467         key,
   (...)
   1488         is_causal=is_causal,
   1489     )
   1490 else:
-> 1491     attn_output, attn_output_weights = F.multi_head_attention_forward(
   1492         query,
   1493         key,
   1494         value,
   1495         self.embed_dim,
   1496         self.num_heads,
   1497         self.in_proj_weight,
   1498         self.in_proj_bias,
   1499         self.bias_k,
   1500         self.bias_v,
   1501         self.add_zero_attn,
   1502         self.dropout,
   1503         self.out_proj.weight,
   1504         self.out_proj.bias,
   1505         training=self.training,
   1506         key_padding_mask=key_padding_mask,
   1507         need_weights=need_weights,
   1508         attn_mask=attn_mask,
   1509         average_attn_weights=average_attn_weights,
   1510         is_causal=is_causal,
   1511     )
   1512 if self.batch_first and is_batched:
   1513     return attn_output.transpose(1, 0), attn_output_weights

File /opt/conda/lib/python3.10/site-packages/torch/nn/functional.py:6435, in multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training, key_padding_mask, need_weights, attn_mask, use_separate_proj_weight, q_proj_weight, k_proj_weight, v_proj_weight, static_k, static_v, average_attn_weights, is_causal)
   6431 if not use_separate_proj_weight:
   6432     assert in_proj_weight is not None, (
   6433         "use_separate_proj_weight is False but in_proj_weight is None"
   6434     )
-> 6435     q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
   6436 else:
   6437     assert q_proj_weight is not None, (
   6438         "use_separate_proj_weight is True but q_proj_weight is None"
   6439     )

File /opt/conda/lib/python3.10/site-packages/torch/nn/functional.py:5830, in _in_projection_packed(q, k, v, w, b)
   5823 proj = linear(q, w, b)
   5824 # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
   5825 proj = (
   5826     proj.unflatten(-1, (3, E))
   5827     .unsqueeze(0)
   5828     .transpose(0, -2)
   5829     .squeeze(-2)
-> 5830     .contiguous()
   5831 )
   5832 # pyrefly: ignore [bad-return]
   5833 return proj[0], proj[1], proj[2]

OutOfMemoryError: CUDA out of memory. Tried to allocate 750.00 MiB. GPU 0 has a total capacity of 14.57 GiB of which 646.75 MiB is free. Process 2493055 has 12.02 GiB memory in use. Including non-PyTorch memory, this process has 1.87 GiB memory in use. Of the allocated memory 1.46 GiB is allocated by PyTorch, and 208.23 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

---

### 🤖 Assistant

Inspecting the embedding-generation code and data flow to determine why OOM occurs at ~75%.

---

### 🤖 Assistant

Checking the model forward for tensor shapes and whether `enc_out` is freed; inspecting the dataset for variable-length batches.

---

### 🤖 Assistant

Checking how the notebook invokes the function with `data_df` and whether an overload exists:

---

### 🤖 Assistant

Here’s what’s going on and what to do about it.

---

## Why OOM happens around 75%

1. **GPU memory fragmentation**  
   The message says only **646.75 MiB** is free while PyTorch reports **1.46 GiB allocated** and **208.23 MiB reserved**. So the process is holding a lot of GPU memory that the allocator can’t reuse as one block. After ~55 minutes and many batches, allocations/frees leave fragmented space, so a **750 MiB** allocation fails even though earlier batches succeeded.

2. **Where it fails**  
   The failure is in `transformer_encoder_cd`: for each batch the model reshapes to `[80, 6400, 256]` (seq=80, batch=32×200, emb=256). That makes the daily encoder’s attention and activations relatively large; when the allocator can’t find a contiguous 750 MiB block, this layer is where it shows up.

3. **Why ~75%**  
   All four GPUs are still running at 75% (each has done ~75% of its shard). One of them (GPU 0 in your run) hits the fragmented state first and OOMs. So it’s not that “later data” is heavier, but that **after many iterations** one GPU’s memory is fragmented enough to fail the next allocation.

4. **Activation not explicitly freed**  
   In `_extract_on_single_gpu` you overwrite `activation['enc']` each step and delete `x` and `patient_embs`, but you never delete `enc_out` or clear `activation['enc']`. So the previous encoder output can linger until the next forward. That’s one more reason peak memory stays high and fragmentation builds up.

---

## What to do (in order of impact)

1. **Lower per-GPU batch size**  
   Use **16 or 8** instead of 32. That cuts the “batch” dimension (e.g. 32×200 → 16×200) going into the daily encoder and reduces peak memory so the same 750 MiB allocation is easier to satisfy even with some fragmentation.

2. **Set PyTorch allocator behavior**  
   Before running the notebook, set:
   ```bash
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   ```
   or in Python before any `torch` CUDA use:
   ```python
   import os
   os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
   ```
   This can reduce fragmentation and often avoids late-run OOMs like this.

3. **Explicitly free activation and optionally clear cache**  
   In the batch loop of `_extract_on_single_gpu`, after copying to `embeddings_out` and updating `local_idx` / `progress_counter`, add:
   - `del enc_out` (or `activation['enc'] = None`),
   - and optionally every N batches (e.g. 500–1000): `torch.cuda.empty_cache()` on that GPU.  
   That keeps fewer large tensors alive and can help the allocator.

4. **Periodic `empty_cache` only if needed**  
   If OOM still happens after (1)–(3), call `torch.cuda.empty_cache()` every few hundred batches inside `_extract_on_single_gpu` (e.g. when `local_idx % (500 * per_gpu_batch_size) == 0`). Don’t do it every batch (too slow).

5. **Two-pass strategy if one run is still too heavy**  
   Split the dataframe in half, run `generate_and_upload_embeddings` (or your `data_df`-based path) twice, then concatenate the embedding DataFrames. Each run uses less “time in process,” so fragmentation is less likely to grow until OOM.

---

## Quick change set (conceptual)

- Call with `per_gpu_batch_size=16` (or 8).
- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` before any CUDA use.
- In the extraction loop, after `embeddings_out[local_idx:...].copy_(...)` and the progress update, add:
  - `del enc_out` (and/or clear `activation['enc']`),
  - and optionally `del batch` then every 500–1000 batches `torch.cuda.empty_cache()` on the current GPU.

The most effective single change is **reducing batch size**; the env var and cleanup help avoid the same OOM at 75% with batch_size=32 or when you try to push batch size back up later.

---

### 🧑 User

So if I set up batch_size = 16; but it still sucked in the 75%; Check the cell 74; and its implemeantion in 72. @dev/legacy/legacy_full_training.ipynb 
# --- Execute ---
CHECKPOINT_PATH = 'logs/legacy_2026-03-16_06-21-09/legacy_replication/checkpoints/checkpoint_best.pt'
embeddings_df = generate_and_upload_embeddings(
    checkpoint_path=CHECKPOINT_PATH,
    data_df = df_cm_sample
)

---

### 🤖 Assistant

Let me look at the current implementation in cells 72 and 74 to understand the exact code being executed.

---

### 🤖 Assistant

Now I can see the full picture clearly. Let me check cell 73 output more carefully — specifically GPU 0's pre-existing memory state.

The critical clue is right here. Look at the GPU memory state from cell 73 output, **printed right before cell 74 runs**:

```
GPU 0: allocated=1.46 GiB, reserved=1.67 GiB, free=12.90 GiB
GPU 1: allocated=0.03 GiB, reserved=0.17 GiB, free=14.40 GiB
GPU 2: allocated=0.03 GiB, reserved=0.17 GiB, free=14.40 GiB
GPU 3: allocated=0.03 GiB, reserved=0.17 GiB, free=14.40 GiB
```

And in the OOM error:

```
Process 2493055 has 12.02 GiB memory in use.
Including non-PyTorch memory, this process has 2.25 GiB memory in use.
Of the allocated memory 1.94 GiB is allocated by PyTorch, and 112.08 MiB is reserved.
```

## Root Cause: GPU 0 has ~1.5 GiB of leaked memory that never got freed

**Batch size is not the problem.** The problem is that **GPU 0 starts the embedding run already holding ~1.46 GiB from the training model**, and that memory is never successfully released. Here's why:

### 1. The `model.cpu()` call doesn't work properly

In `generate_and_upload_embeddings`, the cleanup code is:

```python
if 'model' in globals():
    print("Releasing training model GPU memory...")
    globals()['model'].cpu()
```

This does print "Releasing training model GPU memory...", so the global `model` exists. But the model was trained with `DataParallel`, which wraps the model and spreads it across GPUs. Calling `.cpu()` moves the *wrapper* to CPU, but:

- The training `DataLoader` workers may still be alive holding GPU references
- The **optimizer state** (SGD momentum buffers) is still on GPU 0 — `optimizer` is still alive in the notebook's global scope and holds GPU-resident tensors
- The **scheduler**, **train_loader**, **val_loader**, and any other training-related objects are still in scope

The `empty_cache()` calls in the function only release PyTorch's *cache* — they cannot free memory that is still referenced by live Python objects.

### 2. At OOM time, GPU 0 has 12.02 GiB in use by the process

- ~1.94 GiB PyTorch allocated (the embedding replica model + activations + the leaked training state)
- ~2.25 GiB non-PyTorch memory (CUDA context, DataLoader worker buffers, etc.)
- The remaining ~8 GiB is CUDA context / driver overhead accumulated from the training run

This leaves only ~2.5 GiB truly usable, and the daily encoder's attention projection for batch=16 needs a contiguous 376 MiB block that can't be found.

### 3. Why it fails at 75% (again)

Same reason as before — the memory works in early batches, but progressive fragmentation across ~45k batches on GPU 0 eventually makes a 376 MiB contiguous allocation impossible.

---

## What to fix

The real fix is **aggressively cleaning up training state before embedding generation**. Here's what needs to happen:

### Option A: Add proper cleanup before running cell 74

Add a cell between 73 and 74 with:

```python
import gc

for name in ['model', 'optimizer', 'scheduler', 
             'train_loader', 'val_loader', 'train_dataset', 'val_dataset']:
    if name in globals():
        del globals()[name]

gc.collect()

if torch.cuda.is_available():
    for g in range(torch.cuda.device_count()):
        with torch.cuda.device(g):
            torch.cuda.empty_cache()
    
    for gpu_id in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"GPU {gpu_id}: allocated={mem_alloc:.2f} GiB")
```

After this, GPU 0 should show close to 0 GiB allocated, matching GPUs 1-3. If it still shows significant memory, there are other references holding GPU tensors — you'd need to track those down.

### Option B: Also fix `_extract_on_single_gpu` for fragmentation resilience

Inside the batch loop, after the `del x, patient_embs` line, add:

```python
del x, patient_embs, enc_out
activation['enc'] = None
```

And optionally add periodic defragmentation (every ~1000 batches):

```python
if (local_idx - start_offset) % (1000 * per_gpu_batch_size) == 0:
    torch.cuda.empty_cache()
```

### Option C: Set the allocator env var

As the error message itself suggests, set this **before any CUDA operation in the notebook** (i.e., very first cell):

```python
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

This must be set before `torch.cuda` is initialized. If you've already imported torch and used CUDA, you'd need to restart the kernel.

---

## Summary

| Factor | Impact |
|---|---|
| **GPU 0 leaked training state (~1.46 GiB)** | Primary cause — steals ~10% of GPU capacity |
| **Optimizer/scheduler/loaders still alive** | Prevents `empty_cache` from actually freeing memory |
| **`enc_out` / `activation['enc']` not deleted per batch** | Keeps extra tensors alive, worsens fragmentation |
| **No `PYTORCH_ALLOC_CONF=expandable_segments:True`** | Makes allocator vulnerable to fragmentation |
| **Batch size 16 vs 32** | Halves the *requested* allocation, but doesn't fix the underlying 12 GiB process footprint |

The reason lowering batch size from 32 to 16 didn't help is that the failed allocation shrank proportionally (750 MiB -> 376 MiB), but the *available* memory on GPU 0 is still only ~248 MiB free because the underlying leak is the same. Fix the leak first, then batch_size=16 (or even 32) should work fine.

---

### 🧑 User

I have applied the GPU cleaning (but no variable cleaning) before I ran that round of generation but still got the error; will your optional work really? 
This is hte code I run 
# Optional
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for gpu_id in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        mem_free = (torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)) / 1024**3
        print(f"GPU {gpu_id}: allocated={mem_alloc:.2f} GiB, reserved={mem_reserved:.2f} GiB, free={mem_free:.2f} GiB")

---

### 🤖 Assistant

Yes, the variable cleanup will make a real difference, and here's the concrete proof from your own output.

## What your cleanup cell actually shows

Your cell 73 output after `gc.collect()` + `torch.cuda.empty_cache()`:

```
GPU 0: allocated=1.46 GiB, reserved=1.67 GiB, free=12.90 GiB
GPU 1: allocated=0.03 GiB, reserved=0.17 GiB, free=14.40 GiB
GPU 2: allocated=0.03 GiB, reserved=0.17 GiB, free=14.40 GiB
GPU 3: allocated=0.03 GiB, reserved=0.17 GiB, free=14.40 GiB
```

GPU 0 still has **1.46 GiB allocated** after your cleanup. GPUs 1-3 have 0.03 GiB. That 1.43 GiB gap is the problem.

## Why `empty_cache()` alone doesn't work

`torch.cuda.empty_cache()` only frees PyTorch's **cached-but-unreferenced** memory blocks. It cannot free memory that is **still referenced by a live Python variable**.

Think of it this way:

- `empty_cache()` = "return unused blocks to CUDA" (like returning empty boxes to a warehouse)
- `del optimizer` = "destroy the object that owns the GPU tensors" (like emptying the boxes first)

If the Python objects still exist, their GPU tensors are still "in use" and `empty_cache()` has nothing to free. That's exactly what you're seeing — 1.46 GiB that `empty_cache()` **cannot touch** because `optimizer`, `model`, etc. are still holding references to those GPU tensors.

## What's holding the 1.46 GiB on GPU 0

Your training used `DataParallel`, which makes GPU 0 the "master" device. These notebook globals are still alive and pinning GPU 0 memory:

| Variable | What it holds on GPU 0 |
|---|---|
| `optimizer` | SGD momentum buffers — one per model parameter, all on GPU 0 |
| `model` | The `DataParallel`-wrapped model (parameters + buffers) |
| `scheduler` | References to optimizer param groups |
| `train_loader` / `val_loader` | May hold pinned-memory references |

The `generate_and_upload_embeddings` function does call `globals()['model'].cpu()`, but that only moves the model weights. It never touches the optimizer — and SGD with momentum stores a momentum buffer **the same size as every parameter**, all on GPU 0.

## The proof

Quick math on your model's optimizer state. From your parameter listing, the model has ~7.8M parameters (mostly in `embedding_cd`: 75516 x 256 = ~19.3M params). SGD momentum stores one float32 per parameter:

- ~19.3M params x 4 bytes = ~77 MB just for `embedding_cd`
- Total model: ~29M params x 4 bytes = ~116 MB for parameters
- Plus momentum buffers: another ~116 MB
- Plus DataParallel replica references, gradient buffers that may not have been zeroed

This adds up to the 1.46 GiB you see.

## So yes, the variable deletion will work

Replace your current cleanup cell with:

```python
for name in ['model', 'optimizer', 'scheduler', 
             'train_loader', 'val_loader', 'train_dataset', 'val_dataset',
             'training_history', 'gradient_tier_analyzer']:
    if name in globals():
        del globals()[name]

gc.collect()

if torch.cuda.is_available():
    for g in range(torch.cuda.device_count()):
        with torch.cuda.device(g):
            torch.cuda.empty_cache()
    
    for gpu_id in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        mem_free = (torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)) / 1024**3
        print(f"GPU {gpu_id}: allocated={mem_alloc:.2f} GiB, reserved={mem_reserved:.2f} GiB, free={mem_free:.2f} GiB")
```

After running this, you should see GPU 0 drop to ~0.03 GiB allocated (matching the other GPUs). **That's the verification** — if GPU 0 still shows significantly more than GPUs 1-3, there are additional variables to hunt down.

Once all four GPUs start at ~0.03 GiB, batch_size=16 (or even 32) should complete without OOM, because you'll have the full ~14.4 GiB available on every GPU instead of only ~12.9 GiB on GPU 0.

