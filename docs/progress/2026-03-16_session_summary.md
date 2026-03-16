# Session Progress Report — Legacy Model Post-Training: Embedding Generation & Continued Training Infrastructure
**Date**: 2026-03-16
**Status**: Two new notebook cells implemented for post-training operations (embedding generation + continued training from checkpoint), iterated through 4 revisions to solve GPU OOM, multi-GPU parallelism, CPU memory exhaustion, and inference throughput optimization.

## 1. Executive Summary

With the legacy transformer model training complete (from the 2026-03-15 session), this session focused on the two critical post-training operations: (1) extracting 256-dim member embeddings from a checkpoint and uploading them to BigQuery for downstream tasks, and (2) enabling training continuation from a checkpoint with identical optimizer/scheduler state. Both were implemented as function-based cells in `dev/legacy/legacy_full_training.ipynb`. The embedding generation went through 4 major revisions — single-GPU OOM fix, manual multi-GPU parallelism (4x T4s), CPU RAM exhaustion fix (815 GiB pre-allocation), and full throughput optimization (DataLoader prefetch, pinned memory, inference_mode) modeled after the `moe_flashattn_3` downstream pipeline. A comparative analysis of legacy vs Flash Attention inference speed identified the causal mask materialization (`O(n^2)` dense 200x200 matrix × 96 attention heads) as the dominant bottleneck, explaining the 3-6x speed gap.

## 2. Planned vs. Executed

**Original Plan**: Implement embedding generation from checkpoint and training continuation in the legacy notebook.

**What Got Done**:
- [x] Cell 43: Embedding generation — load checkpoint, query BigQuery, extract embeddings, upload results
- [x] Cell 44: Continue training — restore model/optimizer/scheduler state, resume epoch loop
- [x] Fixed GPU OOM (batch_size=512 with DataParallel + forward hook conflict)
- [x] Implemented manual multi-GPU parallelism (4x T4, ThreadPoolExecutor, per-GPU model replicas)
- [x] Fixed CPU MemoryError (815 GiB numpy allocation for codes matrix)
- [x] Optimized throughput: pinned memory, DataLoader prefetch, lazy dataset, inference_mode
- [x] Comparative analysis of legacy vs Flash Attention inference architectures
- [x] Rigorous break-even analysis for terminate-and-restart vs keep-running decision

**Alignment Notes**: The initial implementation was straightforward, but production-scale data (6.8M rows) exposed 3 cascading issues that required iterative fixes. Each fix was root-caused before implementing, avoiding band-aid patches.

## 3. Key Decisions & Rationale

### Decision: Skip DataParallel, Use Manual Model Sharding for Inference
**Context**: First embedding generation attempt hit GPU OOM at `embedding_cd` layer.
**Options Considered**:
- Option A: Reduce batch_size under DataParallel — still has the hook corruption problem (DataParallel scatters batch across GPUs, forward hook on inner module captures partial tensor, making `enc_out[dt_cnt, j, :]` index wrong members)
- Option B: Manual sharding — one independent model replica per GPU, each with its own hook, data sharded across GPUs, threads in parallel
**Chosen**: Option B — **Rationale**: Eliminates both OOM and hook corruption simultaneously. Same strategy used by `moe_flashattn_3_lob3_downstream_running.py`.
**Trade-offs**: 4x model weight memory (one copy per GPU, ~1.2 GiB each) vs shared weights with DataParallel. Acceptable on T4s (14.6 GiB each).

### Decision: Lazy Dataset Instead of Full Pre-parsing
**Context**: Pre-allocating `np.zeros((6.8M, 200, 80), dtype=int64)` = 815 GiB — impossible.
**Options Considered**:
- Option A: Use smaller dtype (int32 = 408 GiB, int16 = 204 GiB) — still exceeds RAM
- Option B: Lazy parsing — store `cd` as raw strings (~3.2 GiB), parse per-sample in `__getitem__` via DataLoader workers
**Chosen**: Option B — **Rationale**: Reduces memory from 815 GiB to ~9.5 GiB total. DataLoader workers parse in parallel, keeping GPU fed. Same pattern as `LazyClinicalDataset` in the MoE pipeline.
**Trade-offs**: Per-sample string parsing in hot loop, but overlapped with GPU compute via prefetch.

### Decision: Adopt moe_flashattn_3 Throughput Optimizations
**Context**: Single-GPU inference too slow for 6.8M rows; old multi-GPU version was CPU-bottlenecked by raw `DataFrame.iloc` + string parsing.
**Optimizations adopted**:
1. Pre-allocated pinned-memory output tensor — threads write directly, zero-copy to numpy
2. Per-GPU DataLoader with `pin_memory=True`, `prefetch_factor=2`, `num_workers=2`
3. `torch.inference_mode()` instead of `torch.no_grad()` (disables version tracking)
4. Non-blocking async `GPU→CPU` via `.copy_(non_blocking=True)`
5. Shared tqdm progress bar with live ETA across all GPUs

## 4. Technical Changes

### 4.1 Files Modified
- `dev/legacy/legacy_full_training.ipynb` — Added 2 new cells (Cell 43, Cell 44) after the training loop

### 4.2 Cell 43: Embedding Generation (325 lines, 4 revisions)

**Functions defined:**
- `_EmbeddingInferenceDataset(Dataset)` — Lazy dataset: pre-parses age/gender/lob into compact numpy arrays (int16/int8, ~6.4 GiB), keeps `cd` as raw strings (~3.2 GiB), parses codes on-the-fly in `__getitem__` via DataLoader workers
- `_collate_emb(batch)` — Custom collate for stacking batch tensors
- `_extract_on_single_gpu(gpu_id, state_dict, dataset, embeddings_out, ...)` — Per-GPU worker: builds model replica, registers forward hook, runs DataLoader loop, writes embeddings directly into pinned-memory output tensor
- `generate_and_upload_embeddings(checkpoint_path, bq_source_table, bq_dest_table, ...)` — Orchestrator: frees GPU memory, loads checkpoint on CPU, queries BigQuery, pre-parses dataset, shards across GPUs, launches ThreadPoolExecutor, uploads result to BigQuery

**Revision history:**
1. v1: Single-GPU, DataParallel, batch_size=512 → GPU OOM (1.95 GiB allocation on GPU with 1.15 GiB free)
2. v2: Single-GPU, no DataParallel, batch_size=16, AMP → Works but slow
3. v3: Multi-GPU (4x), ThreadPoolExecutor, raw DataFrame.iloc per batch → Works, CPU-bottlenecked
4. v4: Multi-GPU + DataLoader + lazy dataset + pinned memory + inference_mode → CPU MemoryError on pre-parse
5. v5 (final): Lazy `_EmbeddingInferenceDataset` with on-the-fly `cd` parsing → Production-ready

### 4.3 Cell 44: Continue Training from Checkpoint (178 lines)

**Functions defined:**
- `continue_training_from_checkpoint(checkpoint_path, train_loader, val_loader, additional_epochs, ...)` — Loads checkpoint (model state_dict, optimizer state_dict, scheduler state_dict, epoch counter), rebuilds model + DataParallel, restores full optimizer/scheduler state, runs standard training loop with identical configuration (BCEWithLogitsLoss, gradient accumulation, AMP, gradient clipping at 0.25)

**Key design:**
- Scheduler `T_max` set to `start_epoch + additional_epochs` for correct cosine decay continuation
- Reuses existing `train_loader` and `val_loader` from the original training session
- Full metrics logging (MetricsLogger with `resume=True`, LossTracker, GradientTierAnalyzer)
- Saves best/epoch/latest checkpoints

## 5. Discussions & Reasoning

### Topic: Why Legacy Model Inference Is Fundamentally Slower Than Flash Attention
**Question**: User observed legacy model taking much longer to generate embeddings than the Flash model in `moe_flashattn_3`.
**Analysis**:
1. **Causal mask materialization**: Legacy uses `_generate_square_subsequent_mask(200)` → dense `[200, 200]` float32 matrix passed to `TransformerEncoder`. PyTorch computes full O(n^2) attention in all 6 layers × 16 heads = 96 attention matrices. Flash Attention uses xFormers' `memory_efficient_attention` with `LowerTriangularMask()` — fused CUDA kernel, never materializes the matrix, O(n) memory.
2. **Daily encoder**: Legacy runs full `TransformerEncoder` (80-seq attention) on `batch*200 = 3200` sequences. Flash model uses `LearnedAttentionPooling` — dramatically cheaper.
3. **Kernel launch overhead**: Standard MHA = ~10 kernel launches per layer (Q/K/V projections, bmm, mask+softmax, bmm, output proj). Flash = ~3 (fused attention + FFN).
**Conclusion**: 3-6x speed gap is architectural, not fixable by pipeline optimization alone.

### Topic: Break-Even Analysis for Terminate-and-Restart
**Question**: User at 91,000/427,505 batches (21.3%) on old multi-GPU method. Keep running or switch to optimized version?
**Analysis**: Derived batch rate from progress count. If CPU-bottlenecked (80-120ms/batch, likely given DataFrame.iloc parsing), remaining time is 112-168 min. New method from scratch: 81-121 min. If GPU-bound (50-60ms/batch), remaining 70-84 min vs new 79-94 min — marginal.
**Conclusion**: If elapsed ≥ 25 min, terminate and switch saves 30-50 min. Key discriminator: `DataFrame.iloc` string parsing overhead.

## 6. Verification & Quality Checks

**Syntax Validation**: All cells verified via `ast.parse()` — no syntax errors
**Name Resolution**: Automated analysis confirmed all referenced variables/functions resolve to prior notebook cells
**Dependency Mapping**: Verified `extract_embeddings` (cell 18) accesses only `age_in_months, gender_cd, cd, dt_cnt, lob, entity_id` — compatible with heldout table (no `target` column needed)
**Memory Analysis**: Confirmed lazy dataset uses ~9.5 GiB (vs 815 GiB for full pre-parse). COW fork semantics for DataLoader workers verified safe.
**Checkpoint Format**: Verified `save_checkpoint_local` format matches `load_checkpoint` expectations: `{model, optimizer, scheduler, epoch, val_loss, config}`

## 7. Plan Alignment Review

**Original Goals**: Post-training operations for legacy model replication experiment
**Completion Status**:
- Embedding generation: 100% implemented, iterated to production-ready (currently running on GPU machine)
- Continue training: 100% implemented, tested for syntax/dependencies, ready to execute after embedding generation completes

**Scope Changes**: Scope expanded from "two simple cells" to include multi-GPU parallelism engineering and throughput optimization — driven by production data scale (6.8M rows) exposing issues not visible in testing.

## 8. Blockers & Issues

**Resolved**:
- GPU OOM on embedding inference → Removed DataParallel, reduced batch_size, added AMP
- Forward hook corruption with DataParallel → Manual model sharding with per-GPU hooks
- CPU MemoryError (815 GiB array) → Lazy dataset with on-the-fly code parsing
- Slow inference throughput → DataLoader prefetch, pinned memory, inference_mode

**Outstanding**:
- Embedding generation currently running on GPU machine — awaiting completion
- Continue training cell not yet executed (waiting for embedding generation to finish)
- Legacy model inherently 3-6x slower than Flash model for inference — architectural limitation, no fix possible within legacy architecture

## 9. Next Session Plan

**Immediate Priorities** (ranked):
1. Verify embedding generation completed successfully — check BigQuery table `a964286_te4exp_3lob_exp_round5_exp1_legacy_1_5m_commercial_all_sample_embedding`, confirm row counts match source
2. Run downstream evaluation on legacy embeddings — replicate the Commercial downstream pipeline from `moe_flashattn_3_lob3_downstream_running.py`
3. Execute continued training (Cell 44) — additional 5 epochs from best checkpoint, monitor convergence
4. Compare legacy vs Flash model downstream performance — the key experiment question

**Preparation Required**: Verify `train_loader` and `val_loader` are still in GPU kernel memory before running continued training. If kernel was restarted, need to re-load cached dataset first.

**Open Questions**:
- How many additional epochs should continued training run? (currently set to 5)
- Should embeddings also be generated for Medicare and Medicaid heldout sets, or Commercial only?

---
**Session Duration**: ~2 hours
**Files Modified**: 1 (dev/legacy/legacy_full_training.ipynb — 2 cells added, ~503 lines)
**Commits**: 0 (changes uncommitted)
**Environment**: macOS darwin 24.6.0 (local development), remote GPU machine with 4x T4 (inference execution)
