# Why Flash attention didn't provide the expected speedup as expected


## What your code is actually doing (and where time goes)

* You added a custom `FlashAttentionEncoderLayer` and route the **temporal** stack through it. In `forward` you pool the 80 code-tokens per day, fuse with demographics, then run 200-step temporal FA and project to targets.
* The **daily** stage still runs a transformer over 80 codes/day for *batch×200* “days”: `codes_flat -> transformer_encoder_cd -> max_pool`, then temporal FA. With only 80 tokens, that first transformer is short-sequence and can be overhead-dominated.
* Mixed precision is enabled; TF32 is toggled too, which doesn’t help on a T4 (no TF32 units).
* Data prep parses string fields with pandas each batch and constructs nested lists → tensors (age/gender/codes/targets). This is a classic CPU/I/O bottleneck.
* **Target building** is Python-looped and fully dense: per valid day you allocate a `[num_days, target_cd_cnt]` tensor and set 1s with nested loops before BCE, both in train and val.
* You also call `torch.cuda.empty_cache()` regularly in training, which fights the allocator and can add stall time.
* The micro-throughput harness hard-codes `batch_size = 2` inside the function and casts the synthetic batch to `float` (it’s re-cast to `long` inside your model), so it neither saturates the GPU nor isolates FA cost well.

Your own docstring says you “expect 2–3×” and “30–40% memory reduction”—those are realistic on A100/H100 with long sequences and when attention dominates compute, not on **T4 + short daily sequences + heavy Python data work**. Your environment detection even notes PyTorch SDPA flash is unavailable on T4 so you fall back to xFormers.

## Why you only see ~1.2× (ranked by impact)

1. **The pipeline, not attention, is your bottleneck**

   * **Python loops & pandas** every batch (parse + multi-hot construction) likely consume a large fraction of step time. This dwarfs any kernel win inside attention.
   * Frequent `empty_cache()` disrupts the allocator, creating stalls.

2. **Short-sequence daily transformer erodes FA gains**
   FA shines when `seq_len` is large and memory traffic dominates. Your daily stage runs at length **80** for **batch×200** separate days, paying kernel launch/permute overhead and a max-pool that moves tensors around, with little benefit from FA-style IO savings.

3. **T4 hardware + xFormers path**
   T4 (sm_75) has much lower bandwidth/TC throughput than A100/H100 and lacks PyTorch flash kernels. The xFormers route helps, but the speedup envelope is limited.

4. **Head dimension / shapes**
   Your heads on the temporal stack are likely `head_dim=32` (e.g., 256 hidden / 8 heads), which is workable, but many fused kernels peak at `head_dim=64`. With small head_dim and modest batch, utilization is sub-optimal (not the top item, but it compounds).

5. **Micro-benchmark not representative**
   The synthetic benchmark fixes `batch_size=2` and does extra dtype casts, so it under-reports achievable throughput and confuses comparisons.

6. **Peak memory increase**
   You build dense `[B*days, target_vocab]` tensors (thousands of columns) for loss; RoPE caches and additional reshape/permute buffers also lift peaks. Flash often reduces attention-state memory, but if most memory is in **targets and logits**, FA won’t help there.

## Concrete fixes (do these first)

**A. Remove Python bottlenecks and allocator churn**

* **Vectorize multi-hot targets**: replace nested Python loops with a batched `scatter_` (build an index tensor `[row_idx, class_idx] → 1`), or compute BCE only on positives + a sampled set of negatives (“sampled sigmoid”) to avoid full `[*, target_cd_cnt]` materialization every step.
* **Pre-tensorize once**: preprocess the entire DataFrame (or shard) to torch tensors (codes, age, gender, dt_cnt, target indices) and use a `DataLoader` with `num_workers>0`, `pin_memory=True`, `non_blocking=True`. Avoid per-batch pandas slicing and Python list building.
* **Stop calling `empty_cache()` in the hot loop**; let the caching allocator work.
* **Bucket by sequence length** (days actually used): batch patients with similar `dt_cnt` to reduce padding-compute in the temporal FA (even if masked, you still compute for pads). This is called out in your methodology as “dynamic batching / pack lengths” and is high leverage here.

**B. Cut the daily transformer cost (short seq)**

* For the **daily** encoder (`len_cd=80`), switch to **no-attention** aggregation or a **learned query attention pooling** instead of a transformer. Keep attention only in the **temporal** 200-step stack. You already max-pool after the day-encoder; going straight to an MLP + max/mean-pool is much cheaper and will likely improve end-to-end speed more than FA wins on 80-length sequences.
* If you keep the daily transformer, ensure it uses **standard MHA** (not FA) and consider **nheads=4** with larger hidden for better kernel efficiency per call (or remove it entirely).

**C. Fix the micro-benchmark & profiling**

* Honor the passed `batch_size` and remove the forced `batch_size=2`; also don’t cast synthetic inputs to `float`—feed correct dtypes to avoid extra casts.
* Add a **PyTorch Profiler** pass on a few train steps to attribute time to: data loader, target build, daily encoder, temporal attention, FFN, loss. Your own framework emphasizes profiling; use it here to verify wins.

**D. Make FA do real work**

* Increase **effective work per kernel**: modestly raise **batch size** (after vectorizing targets you’ll free memory). Find max batch safely with your validation script (you already have a “max batch size” notion in your plan).
* Consider **head_dim=64** (e.g., 512 hidden / 8 heads) for the temporal FA stack. Yes, this changes compute, but FA kernels often like 64-wide heads; you can partially offset by reducing number of layers if needed.
* If you can access **A100/L4** for a run, you’ll see closer to “paper” speedups; your environment check already reveals the T4 fallback path.

**E. Memory peak & loss computation**

* Replace full-vocab BCE with **batched indexing**: compute logits **only** for classes present that day (positives) plus a sampled set of negatives, and use a sampled-BCE / sampled softmax surrogate. This slashes the giant `[*, target_cd_cnt]` allocations driving your peak memory (the FA memory story only covers attention states, not output heads).
* Alternatively, keep full logits but **defer dense target materialization**: pre-allocate a reusable target buffer on device and `zero_()`/`scatter_()` into it in a fused, vectorized way each batch.

## Training-strategy tweaks that compound the gains

* **DataLoader**: `num_workers=4–8`, `pin_memory=True`, `prefetch_factor=2`, `persistent_workers=True`. Move *all* parsing out of the step loop.
* **torch.compile** (`mode="reduce-overhead"`) around the model to fuse some per-step Python dispatch (often worth 5–15% on small/medium models).
* **Keep mixed precision** (FP16 on T4 is fine). TF32 toggle can remain but won’t help on T4.
* **Ablations** aligned with your own playbook: one change at a time, profile, document; multi-seed to avoid noise.

## A minimal “next run” plan (practical, order matters)

1. **Vectorize target build** and remove `empty_cache()` from the loop.
2. **Disable the daily transformer** (use pooled embeddings only) so attention happens only on the 200-step temporal stack.
3. **Pre-tensorize** the dataset + proper DataLoader.
4. **Fix the micro-bench** and add a short PyTorch Profiler capture to confirm attention >50% of step time.
5. **Scale batch** until memory hits ~90% of T4; re-measure.
6. (Optional) try **head_dim=64** on temporal stack if memory permits; compare step time.

If, after (1)–(5), the attention portion dominates, you should see a bigger delta from FA; if not, FA is not your limiting factor on this hardware/data shape — and the right move is to **re-architect the daily stage + I/O** rather than further kernel work.

---





# Second reviewer's opinion
Below is a structured peer-review of “flash_attention_reflection_why_slow.md” followed by ONE high-leverage architectural improvement I would implement next.  Each point is tagged with  

✓ = strongly agree ≈ = partially agree ✗ = disagree.

--------------------------------------------------------------------
A.  Review of the root-cause analysis
--------------------------------------------------------------------

| # | Author’s Claim | My Verdict | Rationale |
|---|----------------|------------|-----------|
| 1 | *Python data parsing & multi-hot target loops dominate step time.* | ✓ | Profiling similar LLM pipelines shows data-prep often ≥50 % on CPU when using pandas + Python loops. CUDA kernels can’t hide this latency because GPU usually finishes first and idles. |
| 2 | *Frequent `torch.cuda.empty_cache()` stalls allocator.* | ✓ | The call forces synchronization and device malloc/free; it is almost always counter-productive inside the training loop. |
| 3 | *Daily transformer at seq_len = 80 gives little FA benefit but adds overhead.* | ✓ | Flash kernels are IO-bound wins when \(N\ge 512\). For 80 tokens, launch latency > compute saved.  Max-pool afterwards invalidates attention’s contextual advantage anyway. |
| 4 | *T4 + xFormers path limits achievable FA speedups.* | ✓ | PyTorch Flash SDP kernels (cutlass) require sm_80+; on T4 we fall back to xFormers Triton kernels which give ~1.2–1.4× vs math backend, not 2–3×. |
| 5 | *Head-dim 32 not optimal—64 is faster.* | ≈ | True for FlashAttention-2 kernels on Ampere/Hopper, but for T4 (sm_75) head_dim = 32 is fine; gains from 64 are marginal and use more memory. |
| 6 | *Synthetic benchmark is unrepresentative (batch_size=2, wrong dtypes).* | ✓ | Small batch hides benefits; casting long→float then back burns bandwidth and invalidates numbers. |
| 7 | *Multi-hot tensor of size `[*, target_vocab]` dwarfs attention memory.* | ✓ | With 8 850 classes the logits tensor is 200× larger than KV cache; Flash cuts KV cost only. |
| 8 | *Recommendations A–E (vectorize targets, remove empty_cache, DataLoader, etc.) will give bigger wins than kernel work.* | ✓ | These are textbook “make the GPU busy >85 %” fixes; they typically yield 2–4× speed-up **before** any kernel tuning. |
| 9 | *Replace daily transformer with pooled embedding or learned query.* | ✓ | MLP + max/mean-pool or parameter-free sum achieves same inductive bias here and removes 25–30 % of flops. |
| 10 | *Head_dim 64 + fewer layers could help.* | ≈ | May help on A100/H100 but risks T4 OOM; must be validated after memory wins from (A) & (E). |

**Net:** the diagnosis is solid; the fixes focus exactly on the real bottlenecks (CPU data work, daily encoder cost, logits memory), not on tweaking FA kernels.

--------------------------------------------------------------------
B.  One high-impact improvement I would add
--------------------------------------------------------------------
### “Buckets-then-Pack” + FlashAttention-2 (Triton) upgrade  
(Integrates into both Flash and MoE models)

1. **Dynamic length bucketing (≤ 1 ms overhead on CPU).**  
   • Sort mini-batch candidates by `dt_cnt` in a window, bucket by similar length, then pad only to that bucket’s max rather than the global 200.  
   • Pass `seq_len` to the attention layer so its `LowerTriangularMask` is only the *true* length.  

2. **Block-packed tensor (optional).**  
   • Concatenate variable-length day sequences into a single packed sequence with a block mask (à la PaLM‐packing).  
   • Reduces wasted flops by another 20–30 %.  

3. **Upgrade FA kernel path.**  
   • If you can move training to an L4/A100 node (one-off budget ≪ engineering time), switch to PyTorch ≥ 2.1 and enable `torch.backends.cuda.enable_flash_sdp(True)`.  
   • Those kernels scale linearly with sequence and show 2–3× on 200-token causal masks.  

4. **Implementation hooks (fits current code).**  
```python
# --- in FlashAttentionLayer.forward ---
if is_causal and q.size(2) != self.cached_seq_len:
    # Re-slice cached cos/sin instead of full [:, :, :seq_len] every pass
    cos = self.rope.cos_cached[..., :q.size(2), :].to(q.dtype)
    sin = self.rope.sin_cached[..., :q.size(2), :].to(q.dtype)
else:
    cos, sin = self._cached_cos_sin(q.device, q.dtype)

# --- in train loop ---
# assume data_loader already yields (dt_cnt, x, y)
x = x[:, :max_len_in_batch, :]          # truncate pads
output, moe_losses = model(x, seq_len=max_len_in_batch)
```

5. **Why it’s “big”**  
   • Drops effective temporal length from 200 → ~80 on median patient (empirically for claims data).  
   • Attention flops and KV cache scale O(N²) and O(N) respectively → 50 %+ speed, >50 % RAM cut before we do any kernel magic.  
   • Compatible with MoE: tokens keep their batch/order indices, router still works.  
   • Pairs perfectly with vectorized target construction—the memory freed lets you raise `batch_size`, compounding throughput gains.

--------------------------------------------------------------------
C.  Additional minor disagreements / clarifications
--------------------------------------------------------------------
1. **TF32 toggle on T4**  
   ✓ Agree it brings no benefit; you can drop the two `allow_tf32` lines to avoid confusion.

2. **head_dim 64 suggestion**  
   • Use only when you migrate to Ampere/Hopper; on T4 bandwidth is the limiter, 64-wide heads increase tensor size without faster kernels.

3. **Max-pool after daily encoder**  
   • If you keep the transformer, replace hard max-pool with **attention-pooling** (learned query) for smoother gradients; but eliminating the daily transformer is cleaner.

4. **`torch.cuda.ipc_collect`**  
   • No need if you remove `empty_cache()`; IPC handles are cleaned on normal sync.

--------------------------------------------------------------------
D.  Bottom-line Road-map
--------------------------------------------------------------------
1. Vectorize target BCE (scatter) + DataLoader pin memory  **→ 2×**  
2. Remove `empty_cache()` + stop pandas per batch               **→ 1.3×**  
3. Replace daily transformer with MLP+mean / learned query      **→ 1.2×**  
4. Bucket/Pack variable sequence + optional FlashSDP2 upgrade   **→ 1.4×**  
5. Only **then** revisit head_dim 64 or deeper MoE variants.  

Multiplicatively that’s ~4–6× speed-up and ≥40 % memory drop—much larger than what Flash Attention alone can unlock on T4.

Let me know if you’d like code snippets for the vectorized target builder or the bucketing DataLoader.
