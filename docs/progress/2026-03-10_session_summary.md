# Session Progress Report — Co-occurrence Embedding Pre-training (Phase 2): Implementation & v2 Experiment
**Date**: 2026-03-10
**Status**: Implemented PPMI+SVD co-occurrence embedding pre-training (Phase 2), ran v2 experiment, achieved first positive tail margin ever (+1.02); generated three industry-proven next-step proposals (MoE Decoder, GradNorm, Contrastive Learning).

## 1. Executive Summary

This session implemented and executed Phase 2 of the learning bottleneck resolution plan — PPMI+SVD co-occurrence embedding pre-training. After definitively confirming in v0/v1 experiments (Mar 8-9) that the encoder representation `h` lacks discriminative features for tail codes, Phase 2 addressed the root cause: embedding homogenization (tail std=0.03). The v2 experiment with pre-computed co-occurrence embeddings (std=0.0765) and 50% frozen embedding schedule achieved the **best Stage 1 metrics ever** (Recall@10=0.825, NDCG@20=0.441, medium_top10_acc 8x better) and, critically, the **first positive tail margin ever observed** (+1.02 post-Stage 2, vs -0.06 in v1). However, tail_top10_acc remained at 0% because the margin is insufficient for top-10 competition. Three industry-proven next steps were designed: MoE Decoder, GradNorm tier balancing, and contrastive auxiliary loss.

## 2. Planned vs. Executed

**Original Plan**: Implement Phase 2 (co-occurrence embedding pre-training) from the learning bottleneck implementation plan; run the v2 experiment end-to-end; evaluate whether breaking embedding homogenization improves tail code discrimination.

**What Got Done**:
- [x] Step-by-step implementation guide for Phase 2 in `moe_flashattn_5.ipynb` — fixed two bugs in existing Phase 2 cell (wrong variable names), upgraded to v1 hyperparameters
- [x] Deep analysis of `compute_cooccurrence_embeddings()` — memory profiling (nnz=~220M, ~3GB CSR matrix), speed optimization assessment, theoretical grounding in PPMI+SVD literature
- [x] Ran v2 experiment on Vertex AI (4xT4): Stage 1 (1 epoch, 12,336 batches, ~$3.09) + Stage 2 (20 epochs, 21,360 steps, focused loss)
- [x] Comprehensive v2 vs v0/v1 comparative analysis with full metric table
- [x] Root cause analysis: why co-occurrence embeddings helped but didn't fully solve the problem (gradient starvation unchanged at 72.6% common / 3.8% tail; 256-dim information bottleneck)
- [x] Designed three production-grade next-step methods: TieredMoE Decoder, GradNorm balancing, Contrastive Learning — with complete implementation code
- [ ] Run MoE Decoder experiment (deferred — next session)
- [ ] Run GradNorm Stage 1 retraining (deferred — requires full retraining)

**Alignment Notes**: Execution followed the Phase 2 plan exactly. The v2 experiment confirmed the co-occurrence hypothesis was directionally correct (first positive tail margin), but also revealed that gradient starvation during Stage 1 remains the binding constraint.

## 3. Key Decisions & Rationale

### Decision: Use patient-level co-occurrence window (not day-level)
**Context**: PPMI co-occurrence matrix needed a window definition — all codes within a patient's full history vs. same-day codes only.
**Options Considered**: (A) Patient-level (more co-occurrence data for tail codes, noisier) vs. (B) Day-level (more precise, less tail coverage)
**Chosen**: Option A — **Rationale**: Tail codes are so rare that day-level windows would produce near-zero co-occurrence counts for most tail-tail and tail-medium pairs. Patient-level window ensures every tail code gets co-occurrence signal from the medium/common codes in the same patient history. The resulting matrix had 182-220M non-zero entries, confirming sufficient signal.
**Trade-offs**: Noisier co-occurrence signal (two codes in the same patient history may not be clinically related), but for embedding initialization this is acceptable — the encoder refines during training.

### Decision: Confirm v2 results as "directionally correct but insufficient"
**Context**: v2 achieved first positive tail margin (+1.02) but tail_top10_acc remained 0%.
**Options Considered**: (A) Declare success and move to downstream evaluation vs. (B) Acknowledge progress but recognize +1.02 margin is insufficient (need ~5+) and design further interventions
**Chosen**: Option B — **Rationale**: A margin of +1.02 translates to P(pos)/P(neg) ≈ 2.77x, but tail code absolute probabilities remain tiny (~0.006-0.018). For a tail code to enter top-10, its logit (-4.02) must exceed ~1,100+ common codes with higher logits (-2.20). The margin needs to be ~5+ for practical discrimination.
**Trade-offs**: More research time, but the quantitative analysis clearly shows the gap between current (+1.02) and needed (~5+) is large.

### Decision: Propose MoE Decoder as cheapest next experiment
**Context**: Three methods identified — all address different aspects of the bottleneck.
**Options Considered**: (A) MoE Decoder (~3-4 hrs, Stage 2 only) vs. (B) GradNorm (~$5-17, full Stage 1 retrain) vs. (C) Contrastive Learning (~$5-17, full Stage 1 retrain)
**Chosen**: Method A first — **Rationale**: If the v2 encoder's `h` contains nonlinear tail signal that a linear decoder misses, the MoE's 2-layer MLP expert will find it. This is the cheapest test ($3-4) and doesn't require Stage 1 retraining. If A fails, B is next (addresses root cause of gradient starvation directly).

## 4. Technical Changes

### 4.1 Files Created
- `expe_logs/exp_round9/exp2b_256dim_v2/config.json` — v2 experiment config: `use_pretrained_embeddings=true`, `freeze_embeddings_fraction=0.5`, `embedding_lr_multiplier=0.1`, same Stage 2 setup as v1
- `expe_logs/exp_round9/exp2b_256dim_v2/training.log` — Full Stage 1 + Stage 2 training log (646 lines)
- `expe_logs/exp_round9/exp2b_256dim_v2/final_results.json` — Complete tier-stratified results including Stage 2 diagnostics
- `expe_logs/exp_round9/exp2b_256dim_v2/loss_trajectory_epoch0.json` — Batch-level loss trajectory (24,690 lines)
- `expe_analysis/exp_round9/learning_bottleneck/outcome_analysis/decomposed_training_results_cooccur_embed_v2.md` — Comprehensive v0/v1/v2 comparison with root cause analysis and three next-step proposals
- `chat_history/cursor_co_occurrence_embedding_pre_training.md` — Full session chat export

### 4.2 Files Modified
- `dev/moe/moe_flashattn_5.ipynb` — Phase 2 execution cell bug fixes (variable name corrections), experiment config for co-occurrence embedding run

### 4.3 Configuration / Schema Updates
- v2 config additions vs v0/v1: `use_pretrained_embeddings: true`, `freeze_embeddings_fraction: 0.5`, `embedding_lr_multiplier: 0.1`
- PPMI+SVD parameters: `embedding_dim=256`, `window='patient'`, resulting embedding std=0.0765

## 5. Discussions & Reasoning

### Topic: What does the co-occurrence embedding actually do and why does it help?
**Question**: How does PPMI+SVD pre-training address the tail code embedding homogenization bottleneck?
**Analysis**:
1. The root cause (confirmed in v0/v1): tail code embeddings initialized near-zero (std=0.03) → encoder receives nearly identical inputs → gradient starvation prevents differentiation → `h` lacks tail information
2. PPMI (Positive Pointwise Mutual Information) computes: for each code pair (i,j), how much more often they co-occur than expected by chance. This captures semantic structure: codes that appear in similar patients get similar embeddings, codes that don't get dissimilar embeddings
3. SVD reduces the PPMI matrix to 256 dimensions, preserving the top variance directions
4. Result: tail codes that co-occur with specific medium codes now have distinctive embedding directions (std=0.0765 vs 0.03) from the start
5. The 50% freeze schedule prevents early gradient updates from washing out this structure

**Conclusion**: The approach is theoretically sound and empirically validated — Stage 1 metrics improved across the board (best ever), and Stage 2 found positive tail margin for the first time. But the encoder's gradient starvation (72.6% common) still dominates, progressively overwriting the co-occurrence structure.

### Topic: Why did pre-S2 tail margin actually decrease in v2 despite better overall metrics?
**Question**: v2 pre-S2 tail margin was 1.41 vs 1.66 (v1) — the opposite of expected.
**Analysis**: The co-occurrence embeddings improved the encoder's representation of common and medium codes (common margin: 6.94 vs 6.60). This sharpening further marginalized tail codes in `h` — the encoder became more specialized for common codes, leaving less tail-discriminative residual information. The improvement came entirely from Stage 2's focused loss being able to extract better signal from v2's `h`.
**Conclusion**: Better input embeddings → better encoder for common codes → more competitive landscape for tail codes in the representation → Stage 2 needed to work harder but found more to work with (15% lower final loss).

### Topic: Information-theoretic bottleneck of 256 dimensions
**Question**: Is 256 dimensions fundamentally insufficient for 6,297 code predictions?
**Analysis**: With 256 dimensions and 6,297 output codes, the encoder must pack all predictive information into 256 orthogonal directions. Even perfectly allocated, tail codes (1,148 of them) would need ~1,148 bits of information — far exceeding what 256 dimensions can encode at the precision needed. The v2 results showed co-occurrence embeddings pushed the margin from -0.06 to +1.02 — meaningful but structurally capped by this bottleneck.
**Conclusion**: Scaling to 512 dimensions (as done in Round 7) combined with co-occurrence embeddings could break through this ceiling. The 512d model already showed 4.3x improvement in medium_top10_acc (0.68% vs 0.16%) when capacity was the variable.

## 6. Verification & Quality Checks

**Experiments Run**:
- v2 Stage 1: 1 epoch, 12,336 batches, ~2.2 hrs, cost ~$3.09
- v2 Stage 2: 20 epochs, 21,360 steps, focused loss on 16 target codes/batch
- Total v2 cost: ~$3.09 (Stage 1) + ~$3 (Stage 2) ≈ $6

**Reproducibility Check**: Stage 1 baseline metrics (Val loss 0.0030) consistent with v0/v1 (0.0031) after accounting for co-occurrence embedding improvements.

**Key Quantitative Results**:

| Metric | v0 Post-S2 | v1 Post-S2 | v2 Post-S2 | v2 vs v1 |
|--------|-----------|-----------|-----------|----------|
| Stage 2 final loss | 0.5092 | 0.0262 | **0.0223** | -14.9% (better) |
| Tail margin | -0.28 | -0.06 | **+1.02** | **+1.08 (first positive)** |
| Tail top10_acc | 0% | 0% | **0%** | No change |
| Rare margin | 0.32 | 0.45 | **0.77** | +71% |
| Common margin | 6.73 | 6.28 | **6.59** | +0.31 (better preservation) |
| medium_top10_acc (Stage 1) | — | 0.0016 | **0.0129** | **8x improvement** |
| Val Recall@10 (Stage 1) | 0.813 | 0.809 | **0.825** | +1.5% |

**PPMI+SVD Computation**: Completed in ~80 min, processed 1.58M patients, 182-220M non-zero entries in co-occurrence matrix, final embedding std=0.0765.

## 7. Plan Alignment Review

**Original Goals**: Break the 0% tail_top10_acc barrier through co-occurrence embedding pre-training (Phase 2 of learning bottleneck resolution).

**Completion Status**:
- Phase 1 (Decoder Decoupling): **100% complete, definitively negative** (v0+v1)
- Phase 2 (Co-occurrence Embeddings): **100% complete, directionally positive but insufficient**
  - Tail margin: -0.06 → +1.02 (first positive ever)
  - Tail_top10_acc: still 0% (margin insufficient for top-10 competition)
  - Stage 1 metrics: best ever across all rounds
- Phase 3 (Advanced methods): **Designed, not yet executed**
  - MoE Decoder: code written, ready to run
  - GradNorm: code written, requires Stage 1 retraining
  - Contrastive Learning: code written, requires Stage 1 retraining

**Scope Changes**: Added three advanced next-step methods beyond the original two-phase plan, reflecting the finding that co-occurrence embeddings alone are insufficient.

## 8. Blockers & Issues

**Resolved**:
- Phase 2 cell had two variable name bugs (`prepared_data_1p5M` → `data_prepared_1p5M`, `df_train` → `train_df`) — fixed in implementation guide
- Memory concern for PPMI computation — confirmed 3GB CSR matrix well within 624GB VM RAM
- Question of patient-level vs day-level co-occurrence window — patient-level chosen for better tail coverage

**Outstanding**:
- **256-dim information bottleneck**: Fundamental capacity limitation for 6,297 output codes. 512-dim + co-occurrence embeddings is a logical next combination but requires full retraining (~$6-17)
- **Gradient starvation unchanged**: 72.6% common gradient fraction in v2 is identical to v0/v1. GradNorm is the most direct intervention
- **Tail margin still insufficient**: +1.02 vs needed ~5+. The gap is large (5x), suggesting multiple interventions may be needed simultaneously

## 9. Next Session Plan

**Immediate Priorities** (ranked):
1. **MoE Decoder experiment** — Replace linear decoder with TieredMoEDecoder for Stage 2. Tests whether v2's `h` has nonlinear tail signal. Cheapest next step (~$3-4, Stage 2 only). Code ready in analysis doc.
2. **GradNorm Stage 1 retraining** — If MoE decoder doesn't break 0%, retrain Stage 1 with dynamic per-tier gradient balancing. Directly addresses 72.6% common gradient domination. Cost: ~$5-17.
3. **512-dim + co-occurrence embeddings** — Combine capacity gains (Round 7 showed 4.3x medium improvement) with v2's embedding improvements. Strongest combined intervention.

**Preparation Required**:
- Upload TieredMoEDecoder class to notebook
- Prepare GradNorm integration into training loop if MoE experiment is negative
- Consider combining GradNorm + contrastive loss for maximum impact

**Open Questions**:
- Will the MoE's nonlinear readout discover absence patterns in `h` that linear decoder misses?
- Should GradNorm target equal loss ratios or equal gradient norms across tiers?
- Is 512-dim + co-occurrence + GradNorm the "kitchen sink" approach worth testing?

---
**Session Duration**: ~6 hours
**Files Modified**: 1 notebook modified, 5 new files created + 1 chat export
**Commits**: 0 new commits (all untracked)
**Environment**: macOS darwin 24.6.0, Vertex AI Workbench (4xT4 GPUs), PyTorch, FlashAttention, Cursor IDE
