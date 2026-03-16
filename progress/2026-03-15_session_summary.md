# Session Progress Report — Multi-Stream Day: Legacy Replication, SHAP Feature Importance, Architecture Brainstorming Corrections, & Training Diagnostics
**Date**: 2026-03-15
**Status**: Four parallel workstreams executed — legacy model replication planned & notebook created, SHAP + multi-model embedding generation implemented in downstream pipeline, temporal architecture brainstorming doc deeply corrected, and legacy training diagnostics diagnosed as optimizer dynamics (not a bug).

## 1. Executive Summary

This was a high-throughput day spanning four major workstreams across the Clinical TE project. (1) A comprehensive analysis of legacy vs. refactored training pipelines identified 9 critical differences and produced a detailed implementation plan for a standalone legacy replication notebook, which was then executed to create `dev/legacy/legacy_full_training.ipynb` with full metrics tracking, gradient analysis, and compatibility with the MoE experiment logging infrastructure. (2) A model-agnostic SHAP feature importance module and multi-model embedding generation for 3 new checkpoints (rounds 7, 9, 10) across Medicare and Medicaid were implemented in the downstream evaluation pipeline (+463 lines). (3) The temporal architecture brainstorming document received 9 critical corrections — most importantly discovering that the TE already predicts next-day codes via LEAD-shifted SQL targets, invalidating the report's central proposal. (4) A live training diagnostic confirmed that persistently low R@10 metrics in the legacy model are expected behavior given SGD optimization dynamics on sparse multi-label classification, not a metrics bug.

## 2. Planned vs. Executed

**Original Plan**: Execute multiple workstreams for the Clinical TE project including legacy replication, downstream enhancement, brainstorming corrections, and training diagnostics.

**What Got Done**:
- [x] Deep comparison of `transformer_training_scoring.py` vs `transformer_training_pipeline.py` — identified log_softmax + BCEWithLogitsLoss bug
- [x] Deep comparison of `exp1_dense_baseline` vs legacy scripts — identified 9 critical differences (nhid, LOB, optimizer, scheduler, batch size, etc.)
- [x] Created implementation plan for legacy replication (`docs/plans/2026-03-15-legacy-model-replication-full-training.md`, 800 lines)
- [x] Created standalone legacy training notebook (`dev/legacy/legacy_full_training.ipynb`, 3775 lines)
- [x] Created implementation plan for SHAP + embedding generation (`docs/plans/2026-03-15-feature-importance-and-multi-model-embedding-generation.md`, 774 lines)
- [x] Implemented shared SHAP feature importance module in downstream pipeline
- [x] Implemented Commercial SHAP evaluation cells
- [x] Implemented Medicaid SHAP evaluation cells
- [x] Implemented Medicare embedding generation for 3 new models (rounds 7, 9, 10)
- [x] Implemented Medicaid embedding generation for 3 new models (rounds 7, 9, 10)
- [x] Applied 9 corrections to temporal architecture brainstorming document (+196 lines)
- [x] Deep-inspected SQL pipelines to discover LEAD-shifted targets (next-day prediction already in place)
- [x] Diagnosed low R@10 in legacy training — confirmed not a bug, explained SGD vs AdamW dynamics
- [x] Diagnosed DataLoader worker OOM crash when restarting training

**Alignment Notes**: The legacy replication workstream uncovered significantly more differences than expected, leading to the correct decision to create a standalone notebook rather than modify the existing MoE framework. The brainstorming corrections session surfaced a critical factual error that required rewriting 3 major sections.

## 3. Key Decisions & Rationale

### Decision: Create Standalone Legacy Notebook Instead of Modifying moe_flashattn_4
**Context**: After identifying 9 critical differences between the legacy training and the exp1_dense_baseline configuration, needed to decide whether to modify the existing framework or create a new independent notebook.
**Options Considered**:
- Option A: Modify `moe_flashattn_4.ipynb` to add a new experiment config replicating legacy (risk: regressions in other experiments, hidden nhid/LOB/scheduler differences)
- Option B: Create standalone `dev/legacy/legacy_full_training.ipynb` with transparent, auditable code
**Chosen**: Option B — **Rationale**: The existing framework bakes in non-legacy values (nhid=1024, LOB embedding, per-step scheduling, DataParallelWrapper) that are deeply embedded in shared abstractions. Overriding them for one experiment would require modifying `_create_model`, `_calculate_model_dimensions`, and the training loop — all shared by other experiments. A standalone notebook provides clear auditability for the "reference backup" use case.
**Trade-offs**: Code duplication between legacy notebook and MoE framework; requires independent maintenance.

### Decision: Preserve Three Bug Fixes from Refactored Code in Legacy Replication
**Context**: The legacy `min_transformer_train.py` has three genuine bugs: log_softmax + BCEWithLogitsLoss mismatch, gradient clipping after optimizer step, and double weight update.
**Options Considered**:
- Option A: Replicate legacy exactly, bugs and all (pure reproduction)
- Option B: Apply the three bug fixes while keeping everything else legacy (corrected replication)
**Chosen**: Option B — **Rationale**: The bugs are definitively incorrect (mathematically wrong loss function, wasted gradient clipping, unintended 2-3x effective learning rate). Replicating them would not produce a meaningful baseline. The corrected LR (1e-3 vs 1e-2) already accounts for the double-update removal.

### Decision: Reframe Brainstorming Doc from "Target Shifting" to "Loss Function Formulation"
**Context**: Deep SQL inspection revealed the TE already predicts next-day codes via LEAD, invalidating the document's central proposal.
**Options Considered**:
- Option A: Simply note the error and remove Proposal A
- Option B: Reframe the entire analysis to focus on loss function formulation (BCE vs CrossEntropy) and provide two new actionable proposals (A1: full event-level, A2: hybrid CrossEntropy)
**Chosen**: Option B — **Rationale**: The underlying insight (bag-of-codes convergence) is still valid; only the cause attribution was wrong. The reframing provides more rigorous and actionable next steps.

### Decision: Diagnose Legacy Training Low R@10 as Expected Behavior
**Context**: User observed R@10 of 0.001-0.007 in legacy model training while loss decreased normally from 0.80 to 0.76.
**Analysis**: Line-by-line comparison of `compute_batch_metrics_legacy` vs `compute_batch_metrics_lightweight` confirmed functional identity. Mathematical analysis showed P(hit@10) ≈ 0.005 for random predictions on 6297-code vocabulary with ~3 positives/day, exactly matching observed values. SGD at lr=0.001 on sparse multi-label BCE is dominated by negative class gradients (6294/6297 codes), causing loss decrease via negative suppression before positive code elevation.
**Chosen**: Not a bug — **Rationale**: Metric implementations are identical; observed values match random baseline probability; loss decrease pattern consistent with SGD learning to suppress negatives first. Recommended AdamW switch to isolate architecture vs optimizer as bottleneck.

## 4. Technical Changes

### 4.1 Files Created
- `dev/legacy/legacy_full_training.ipynb` (3,775 lines) — Standalone legacy transformer training notebook with full metrics tracking (R@10, R@20, uR@10, P@10, NDCG@20, PosBrier), gradient tier analysis, loss tracking, GCS checkpoint saving, and compatibility with MoE experiment log folder structure. Includes test cells for all functional components.
- `docs/plans/2026-03-15-legacy-model-replication-full-training.md` (800 lines) — Comprehensive implementation plan documenting all 9 critical differences, configuration decisions with rationale, and copy-ready code for each notebook cell.
- `docs/plans/2026-03-15-feature-importance-and-multi-model-embedding-generation.md` (774 lines) — 7-task implementation plan for SHAP module and multi-model embedding generation.
- `docs/progress/2026-03-15_session_summary.md` — This progress report (replaces earlier version from brainstorming session).

### 4.2 Files Modified
- `dev/downstream/moe_flashattn_3_lob3_downstream_running.py` (+463 lines, 4247 → 4711 lines)
  - **Added**: Shared SHAP feature importance module (`compute_shap_feature_importance` + `run_shap_for_all_feature_sets`) — model-agnostic, supports LogReg/CatBoost/XGBoost/LightGBM via automatic SHAP backend selection
  - **Added**: Commercial SHAP evaluation cells (train CatBoost hybrid, run SHAP, export to Excel)
  - **Added**: Medicaid SHAP evaluation cells (same approach, Medicaid-specific data objects)
  - **Added**: Medicare embedding generation for 3 new model checkpoints (rounds 10, 7, 9) with GCP table upload
  - **Added**: Medicaid embedding generation for 3 new model checkpoints with Medicaid-specific naming conventions
- `expe_analysis/exp_round10/temporal_architecture_research_and_brainstorming.md` (+196 lines, 996 → 1192 lines)
  - **Section 0**: Added table of 9 input code types with examples and vocabulary contributions
  - **Section 1.2**: Added correction note explaining conditional distributional advantage mechanism
  - **Section 3.1**: Critical rewrite — corrected factual error (TE already predicts next-day codes via LEAD)
  - **Section 3.1**: Changed "redundant" to "entirely subsumed" with numerical analysis
  - **Section 3.3**: Major rewrite — replaced shallow comparison with rigorous assessment of why BCE permits bag-of-codes + event-level autoregressive comparison
  - **Section 3.7**: Updated summary table
  - **Section 4.1**: Complete rewrite — replaced invalid "shift by 1 day" with two new proposals (A1: full event-level, A2: hybrid CrossEntropy)
  - **Multiple sections**: Consistency updates throughout
  - **Added**: Corrections Log at document end documenting all 9 changes
- `dev/moe/moe_flashattn_4.ipynb` — Modified (3685 insertions/1520 deletions per git diff stat, likely ongoing experiment iterations)

### 4.3 Configuration / Schema Updates
- Legacy notebook configured with exact legacy parameters: nhid=512, nlayers=6, nhead=16, embedding_size=256, SGD(lr=1e-3, momentum=0.9), CosineAnnealingLR(T_max=num_epochs), gradient_clip=0.25, batch_size=512, no LOB embedding
- SHAP module configured for top-10/20/50 cutoff analysis with 2000-sample cap for computation speed
- New embedding tables follow existing naming conventions:
  - Medicare: `a964286_te4exp_3lob_{safe_exp_name}_medicare_all_sample_embedding`
  - Medicaid: `a964286_te4exp_{safe_exp_name}_medicaid_heldout_embedding`

## 5. Discussions & Reasoning

### Topic: Legacy vs Refactored Pipeline — Critical Bug Discovery
**Question**: Are `transformer_training_scoring.py` and `transformer_training_pipeline.py` functionally equivalent? Can either be used as the reference base?
**Analysis**:
1. Model architecture is identical in both scripts
2. Two meaningful training differences: optimizer (SGD vs AdamW) and scheduler T_max (300 vs num_epochs)
3. **Critical bug in both**: Model outputs `F.log_softmax()` but uses `BCEWithLogitsLoss`, creating `sigmoid(log(softmax(logits)))` — a double nonlinearity that severely compresses gradients
4. The `exp1_dense_baseline` in `moe_flashattn_4.py` correctly fixes this by returning raw logits
**Conclusion**: Use scoring script as cleaner reference base, but apply the log_softmax correction from the MoE framework. The double nonlinearity is likely the single most impactful source of suboptimal downstream performance in any training that used these scripts.
**Citations**: `data_ingestion/Legacy/Train/python/min_transformer_train.py`, `dev/transformer_training_pipeline.py`, `dev/legacy/transformer_training_scoring.py`

### Topic: Why Multi-Label BCE Permits Bag-of-Codes Despite Next-Day Targets
**Question**: User challenged: "h_t IS conditioned on history via causal attention — why would it not learn temporal patterns?"
**Analysis**:
1. The hidden state h_t DOES have full access to temporal history — not in dispute
2. BCE decomposes into independent per-code binary losses with no interaction terms
3. ~99.5% of BCE terms are negative class (6k codes, typically 5-30 positive per day)
4. Chronic codes (diabetes, hypertension) appear on nearly every visit → model learns stable patient-level P(code_j | patient) approximately constant across days
5. This creates a broad "patient profile attractor" basin — model falls in at step ~15k/85k and has insufficient gradient pressure to escape
6. CrossEntropy/softmax forces competition between codes, creating discriminative gradient pressure
**Conclusion**: Problem is optimization landscape, not information availability. The model CAN learn temporal patterns under BCE but is not sufficiently FORCED to. Hypothesis supported by 10 rounds of empirical evidence but not proven — controlled BCE vs CrossEntropy ablation would provide definitive evidence.

### Topic: SGD vs AdamW on Sparse Multi-Label Classification
**Question**: Why does legacy training show near-random R@10 while loss decreases?
**Analysis**:
1. With target_cd_cnt=6297 and ~3 positive codes/day, P(hit@10) = 1 - (6294/6297)^10 ≈ 0.005 — exactly matching observed 0.001-0.007
2. SGD applies uniform learning rate to all parameters; with 6294 negative codes dominating gradients, model learns to suppress all logits (reducing loss) before elevating specific positives (improving ranking)
3. AdamW's per-parameter adaptation via second moment gives rarely-positive codes meaningful updates, enabling faster ranking improvement
4. Legacy lr=1e-3 is 10x lower than original 1e-2 (compensating for double-update bug removal), further slowing convergence
**Conclusion**: Not a metrics bug. Expected behavior for SGD on sparse multi-label with 6297 targets at 4% through epoch 1.

## 6. Verification & Quality Checks

**Downstream Pipeline (`moe_flashattn_3_lob3_downstream_running.py`)**:
- Syntax verification: `python -c "import ast; ast.parse(open('...').read())"` — PASS
- Linter: Zero errors
- All 5 insertion blocks verified at correct boundary lines
- Code review: Confirmed model-agnostic SHAP backend selection, correct data structure duck-typing, correct table naming conventions

**Legacy Notebook (`dev/legacy/legacy_full_training.ipynb`)**:
- Full notebook created with model architecture, dataset, collate function, loss computation, metrics tracking, training loop, gradient analysis, and GCS checkpoint saving
- Test cells included for all functional components
- Configuration documented with rationale for each parameter choice

**Brainstorming Document**:
- Grep verification: All stale "predict current day" references replaced; old Proposal A text removed; consistency updates across sections 5.2, 5.4, 6.1, 6.4, 7.1, Appendix A, Appendix B

**No production code was executed** — all changes are local file modifications and notebook creation.

## 7. Plan Alignment Review

**Workstream 1 — Legacy Replication**:
- Plan: `docs/plans/2026-03-15-legacy-model-replication-full-training.md` — Fully executed
- Notebook created with all specified components
- Bug fixes applied as documented (log_softmax, gradient clip order, double update)

**Workstream 2 — SHAP + Embedding Generation**:
- Plan: `docs/plans/2026-03-15-feature-importance-and-multi-model-embedding-generation.md` — Fully executed (7 tasks, commercial embedding generation correctly excluded per user feedback)
- SHAP module shared between Commercial and Medicaid (no code duplication)
- 3 model checkpoints configured for Medicare and Medicaid embedding generation

**Workstream 3 — Brainstorming Corrections**:
- 9 corrections applied with SQL evidence
- Reframed from invalid "target shifting" to valid "loss function formulation"

**Workstream 4 — Training Diagnostics**:
- Confirmed metrics implementations are functionally identical
- Explained SGD dynamics; recommended AdamW switch
- Diagnosed DataLoader OOM as stale `persistent_workers` — recommended kernel restart or `num_workers` reduction

## 8. Blockers & Issues

### Resolved
- **Factual error in brainstorming doc** (TE predicts current-day → actually next-day) — Corrected with SQL evidence from 3 LOB pipelines
- **Low R@10 mystery in legacy training** — Diagnosed as expected SGD behavior, not a metrics bug
- **DataLoader worker OOM on re-run** — Identified as `persistent_workers=True` with dead worker PIDs; advised kernel restart

### Outstanding
- **All new files are uncommitted** — `dev/legacy/legacy_full_training.ipynb`, both plan docs, brainstorming corrections, downstream pipeline changes, and this progress report
- **Legacy notebook not yet run on GPU cluster** — Created locally, needs deployment to remote Jupyter for actual training
- **SHAP + embedding code not yet run on remote** — User plans to manually add to remote Jupyter; precise insertion instructions provided
- **No controlled BCE vs CrossEntropy experiment exists** — The optimization-landscape hypothesis is supported but unproven; Phase 1 (Option A2) would provide definitive evidence
- **Temporal shuffle test never run** — Listed as Phase 0 in updated brainstorming roadmap; single most informative free experiment

## 9. Next Session Plan

**Immediate Priorities** (ranked):
1. **Deploy legacy notebook to GPU cluster and start training** — Verify all components work end-to-end on full `a834793_Combined_All_LOB_o3_train_ending` dataset. Consider switching from SGD to AdamW if early metrics remain stagnant.
2. **Execute SHAP + embedding generation on remote Jupyter** — Add the 5 code blocks to the remote downstream notebook using the provided insertion instructions; run Medicare and Medicaid embedding generation for rounds 7, 9, 10.
3. **Run temporal shuffle test on existing R10 checkpoint** (~$5, Phase 0) — Shuffle day order in validation data, compare R@10/loss vs unshuffled. If negligible change → confirms bag-of-codes hypothesis.
4. **Commit all local changes** — Stage and commit the 6+ untracked/modified files as a logical unit.

**Preparation Required**:
- Remote Jupyter environment with GPU access for legacy training
- BigQuery permissions for new embedding table uploads
- Design temporal shuffle test protocol (which checkpoint, which data subset, metric comparison)

**Open Questions**:
- Should legacy training use SGD (pure replication) or AdamW (likely better results)? Initial run showed SGD is very slow to improve ranking metrics.
- For Option A2 (CrossEntropy experiment), how to select the single target code per day? (Highest-severity? Most novel? Random?)
- Is `moe_flashattn_4.py` the right codebase for the CrossEntropy experiment, or should a new `moe_flashattn_5.py` be created?

---
**Session Duration**: ~8 hours (multiple concurrent chat sessions)
**Files Modified**: 3 (`dev/downstream/moe_flashattn_3_lob3_downstream_running.py`, `expe_analysis/exp_round10/temporal_architecture_research_and_brainstorming.md`, `dev/moe/moe_flashattn_4.ipynb`)
**Files Created**: 4 (`dev/legacy/legacy_full_training.ipynb`, `docs/plans/2026-03-15-legacy-model-replication-full-training.md`, `docs/plans/2026-03-15-feature-importance-and-multi-model-embedding-generation.md`, `docs/progress/2026-03-15_session_summary.md`)
**Commits**: 0 (all changes uncommitted)
**Environment**: macOS darwin 24.6.0, Cursor IDE, Python 3.10 (remote GPU cluster for training)

**Chat Sessions Referenced**:
- [Legacy Replication Planning](cb274a24-a090-4090-84cb-66f31f8e7fa4) — Deep pipeline comparison, implementation plan, decision to create standalone notebook
- [SHAP & Embedding Plan+Execute](df26bee0-fe38-4b54-b05d-b19d545c98ad) — Plan creation, execution of 6 tasks, precise insertion instructions for remote Jupyter
- [Legacy Training Notebook Creation](d8ee0b51-88d7-431a-bb38-35b1a2b1cd1a) — Full notebook implementation with metrics, gradient analysis, test cells
- [Training Diagnostics (R@10 + OOM)](1dcd15d3-a65b-4ac2-9857-d63e1234135c) — Line-by-line metrics comparison, SGD dynamics diagnosis, DataLoader OOM troubleshooting
