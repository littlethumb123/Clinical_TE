# Session Progress Report — Temporal Architecture Brainstorming Deep Review & Critical Corrections
**Date**: 2026-03-13
**Status**: Major corrections applied to `temporal_architecture_research_and_brainstorming.md` — discovered the TE already does next-day prediction via LEAD-shifted targets; the bag-of-codes convergence occurs despite next-day targets, not because of missing target shift. Reframed the entire autoregressive proposal from "shift targets by 1 day" to "change from multi-label BCE to categorical CrossEntropy / event-level prediction."

## 1. Executive Summary

This session was a critical review and correction cycle for the Round 10 temporal architecture brainstorming report (`expe_analysis/exp_round10/temporal_architecture_research_and_brainstorming.md`). Through deep inspection of the data ingestion SQL pipelines (Medicaid, Commercial, Medicare), the legacy training code (`min_transformer_train.py`, `min_transformer_finetune.py`), and the current MoE architecture (`moe_flashattn_4.py`), a **critical factual error** was identified: the report's central claim that the TE predicts "current day's codes" was wrong — all three LOB SQL pipelines apply `LEAD` to shift targets by 1 day, meaning the TE already predicts next-day codes. This invalidated the report's highest-priority proposal (Proposal A: "shift prediction target by 1 day"). The session produced 9 corrections that reframe the problem from "target shifting" to "loss function formulation" (multi-label BCE vs. categorical CrossEntropy) and "sequence granularity" (day-level set prediction vs. event-level autoregressive prediction), with rigorous analysis of why multi-label BCE permits bag-of-codes convergence despite having access to temporal information.

## 2. Planned vs. Executed

**Original Plan**: User asked 5 specific questions about the brainstorming document; then requested corrections be applied.

**What Got Done**:
- [x] Q0: Clarified that raw codes include 9 code types (~84k vocab), not only diagnoses
- [x] Q1: Re-examined the test vs. OOT-strict distributional exposure argument — refined to explain conditional mechanism (temporal distribution shift, not general exposure)
- [x] Q2: Deep-inspected SQL pipelines and training code to determine what the TE actually predicts — discovered LEAD shift already in place
- [x] Q3: Reframed "R10 hybrid = tabular baseline = 0.831" from "redundant" to "entirely subsumed / non-contributive"
- [x] Q4: Compared LearnedAttentionPooling vs. legacy within-day transformer
- [x] Applied all 9 corrections to the brainstorming document
- [x] Addressed follow-up challenge: "Why does multi-label BCE treat codes independently if h_t is conditioned on history?" — produced rigorous optimization-landscape analysis

**Alignment Notes**: The session surfaced a fundamental error in the report that required rewriting 3 major sections (3.1, 3.3, 4.1) rather than minor edits. The user's challenge on the multi-label BCE argument pushed the analysis from a shallow claim to a rigorous optimization-dynamics explanation.

## 3. Key Decisions & Rationale

### Decision: Frame the TE's Problem as Optimization Landscape, Not Information Availability
**Context**: The user correctly challenged: "h_t IS conditioned on history via causal attention — why would it not learn temporal patterns?" The original report implied the model couldn't access temporal info. That was wrong.
**Options Considered**:
- Option A: Claim multi-label BCE is information-theoretically unable to learn temporal patterns (wrong — it CAN)
- Option B: Frame as optimization landscape problem — BCE *permits* a bag-of-codes shortcut that the model falls into early in training
**Chosen**: Option B — **Rationale**: The hidden state h_t has full access to temporal info. The issue is that independent sigmoid decomposition, negative class dominance (~99.5% of BCE terms are y=0), and chronic code stability in claims data create a broad "patient profile attractor" basin in the loss landscape. The model CAN learn temporal patterns but is not sufficiently FORCED to because the loss provides an easier shortcut. This is supported by the empirical evidence (loss floor at step ~15k of ~85k) but is honestly framed as a hypothesis, not a proven fact.
**Trade-offs**: This framing is more nuanced and less actionable than "just shift targets" — it requires controlled experimentation (BCE vs. CrossEntropy on same architecture) to validate.

### Decision: Split Proposal A into Two Options (A1: Full Event-Level, A2: Hybrid)
**Context**: The original Proposal A was "shift targets by 1 day" — already done. The real change needed is fundamentally different.
**Options Considered**:
- A1: Full CLMBR-style event-level autoregressive (flatten hierarchy, decoder-only, CrossEntropy). HIGH complexity.
- A2: Keep existing architecture, replace BCE with CrossEntropy using a single primary-code target per day. MODERATE complexity.
**Chosen**: Recommend A2 first as a controlled experiment — **Rationale**: A2 tests the optimization hypothesis directly. If CrossEntropy on the same architecture escapes bag-of-codes, the loss was the root cause. If not, the daily pooling bottleneck is the binding constraint and A1 is needed. This is more scientifically rigorous than jumping straight to A1.

### Decision: Acknowledge Uncertainty Explicitly
**Context**: The argument for autoregressive superiority comes from different systems on different data (CLMBR on Stanford EHR, LMM on GenHealth claims). No controlled experiment exists comparing BCE vs. CrossEntropy on the same TE data.
**Chosen**: Explicitly label the optimization-landscape argument as "supported by empirical evidence but not proven by first principles" — **Rationale**: Intellectual honesty. The 10 rounds of bag-of-codes convergence are consistent with the hypothesis but do not prove that CrossEntropy would solve it.

## 4. Technical Changes

### 4.1 Files Created
- `progress/2026-03-15_session_summary.md` — This progress report

### 4.2 Files Modified
- `expe_analysis/exp_round10/temporal_architecture_research_and_brainstorming.md` — 9 corrections applied (996 → 1192 lines, +196 lines)
  - **Section 0 (Executive Summary)**: Added table of 9 input code types with examples and vocabulary contributions
  - **Section 1.2 (Hypothesis B)**: Added correction note explaining conditional distributional advantage mechanism + the more damning evidence (transductive access to test members still fails to outperform)
  - **Section 3.1**: **Critical rewrite** — corrected factual error; TE already predicts next-day codes via LEAD-shifted targets in all 3 LOB SQL pipelines. Added SQL evidence with specific line numbers.
  - **Section 3.1 (R10 hybrid)**: Changed "redundant" to "entirely subsumed" with precise numerical analysis (TE alone 0.8096, tabular 0.831, hybrid 0.831 = zero marginal)
  - **Section 3.3**: **Major rewrite** — replaced shallow comparison with 2 new subsections (3.3.1: rigorous assessment of why BCE permits bag-of-codes; 3.3.2: event-level autoregressive comparison). Includes 4-point analysis of BCE optimization dynamics, comparison table, honest caveats.
  - **Section 3.7**: Updated summary table — TE objective corrected, implementation complexity for autoregressive increased to HIGH
  - **Section 4.1 (Proposal A)**: **Complete rewrite** — replaced invalid "shift by 1 day" with two options (A1: full event-level, A2: hybrid CrossEntropy). Recommends A2 first as controlled experiment.
  - **Section 2.3.1 (CLMBR)**: Updated "Critical distinction from TE" to reflect corrected understanding
  - **Sections 4.2, 5.2, 5.4, 6.1, 6.4, 7.1, 7.2, Appendix A, Appendix B**: Updated for consistency
  - **Added**: Corrections Log at document end documenting all 9 changes

### 4.3 Evidence Inspected (Not Modified)
The following files were read during analysis to build evidence for corrections:

| File | What Was Found |
|------|---------------|
| `data_ingestion/TE_pretraining_data_ingestion/medicaid_for_training.sql` (line 1391) | `LEAD(target, 1) OVER (... ORDER BY dt ASC) AS target_next_day` — confirms next-day prediction |
| `data_ingestion/TE_pretraining_data_ingestion/commercial_for_training.sql` (line 1458) | `x1a.seqno = x1b.seqno + 1` — confirms LEAD-style join for next-day target |
| `data_ingestion/TE_pretraining_data_ingestion/medicare_for_training.sql` (line 1732) | Same LEAD logic for Medicare pipeline |
| `data_ingestion/TE_pretraining_data_ingestion/create_w2ind_target_from_w2ind.sql` (lines 105-165) | 9 code types with grouping rules — confirms multi-domain vocabulary |
| `data_ingestion/Legacy/Train/python/min_transformer_finetune.py` (line 332) | `target = 'target'` — reads LEAD-shifted column; line 329: `BCEWithLogitsLoss` |
| `data_ingestion/Legacy/Train/python/min_transformer_train.py` (line 313) | `criterion = nn.NLLLoss()` — legacy IP fine-tuning uses NLLLoss (different from pretraining) |
| `dev/moe/moe_flashattn_4.py` (lines 2428-2453) | Causal attention mask → temporal encoder → `decoder_cd(cd)` returns `[batch, len_dy, target_cd_cnt]` logits |
| `dev/moe/moe_flashattn_4.py` (lines 1095-1158) | `DataParallelWrapper.forward()` — BCE loss on flattened output vs. multi-hot targets |
| `dev/moe/moe_flashattn_4.py` (lines 1732-1809) | `LearnedAttentionPooling` — single learned query, K/V projections, no code-code interaction |

## 5. Discussions & Reasoning

### Topic: Why Multi-Label BCE Permits Bag-of-Codes Despite Next-Day Targets
**Question**: The user challenged: "h_t is conditioned on previous codes via causal attention. Why do you think multi-label BCE fails to learn temporal patterns? The codes ARE conditioned on history."
**Analysis**:
1. The hidden state h_t DOES have full access to temporal history — this is not in dispute
2. The issue is whether the LOSS FUNCTION creates sufficient gradient pressure to USE that temporal info
3. BCE decomposes into independent per-code binary losses: loss_j = -[y_j·log(σ(h_t·w_j)) + (1-y_j)·log(1-σ(h_t·w_j))]
4. No interaction term between codes — predicting code A doesn't affect loss for code B
5. ~99.5% of BCE terms are negative class (6k codes, typically 5-30 positive per day)
6. Chronic codes (diabetes, hypertension) appear on nearly every visit → model can learn stable patient-level P(code_j | patient) that is approximately constant across days
7. This creates a broad "patient profile attractor" basin — model falls in at step ~15k/85k and has insufficient gradient pressure to escape
8. CrossEntropy/softmax forces competition between codes (raising P(code_A) requires lowering P(code_B)), creating discriminative gradient pressure
**Conclusion**: The problem is an optimization landscape issue, not information availability. The model CAN learn temporal patterns under BCE but is not sufficiently FORCED to. This is a hypothesis supported by 10 rounds of empirical evidence but not proven by first principles — a controlled BCE vs. CrossEntropy ablation on the same architecture would provide definitive evidence.

### Topic: Test vs. OOT Distributional Exposure Mechanism
**Question**: User challenged: "If TE has distributional advantage on OOT, it also saw test members during pretraining — why doesn't it dominate on test too?"
**Analysis**:
1. The advantage is NOT about seeing individual members — it's about being calibrated for the temporal PERIOD's code distribution
2. Test split = same time period as training → PCA/AE/SelectKBest are well-calibrated (same distribution) → TE's additional exposure provides negligible benefit
3. OOT-strict = post-cutoff period → PCA/AE/SelectKBest degrade (trained on pre-cutoff only) → TE remains calibrated (saw post-cutoff during pretraining)
4. The more damning evidence: TE had TRANSDUCTIVE access to test members' actual clinical sequences during pretraining, yet barely matches SelectKBest on test AUC (0.8096 vs. 0.8093) and LOSES on test Lift@1% (14.59 vs. 21.75)
**Conclusion**: TE's OOT advantage = (TE calibrated for OOT) + (other methods degrading on OOT). On test, factor 2 doesn't exist, so no advantage. The transductive failure on test is the strongest evidence that TE captures only aggregate code statistics.

### Topic: LearnedAttentionPooling vs. Legacy Within-Day Transformer
**Question**: Does removing the within-day transformer structure impact downstream performance?
**Analysis**:
1. Legacy: 1-layer, 4-head TransformerEncoder (no dropout) → MaxPool. Codes attend to each other.
2. LearnedAttentionPooling: Single learned query attends to all codes independently. No code-code interaction.
3. Theoretical concern: Within-day code combinations (e.g., diabetes + renal failure + dialysis same day) carry meaning beyond individual codes. Pooling loses this.
4. Practical counter: (a) Legacy transformer is very shallow (1 layer, no dropout), (b) MaxPool discards most interaction nuance, (c) residual connection `cd_res = cd.sum(-2)` provides bag-of-codes regardless, (d) if temporal encoder converges to aggregate statistics anyway, within-day interactions are washed out.
**Conclusion**: Second-order concern relative to the fundamental loss function problem. If moving to event-level autoregressive (Option A1), the daily pooling disappears entirely.

## 6. Verification & Quality Checks

**Document Consistency Checks**:
- Grep for "predict.*current.day|one-index shift" → All occurrences are in correction context (explaining what was wrong) ✓
- Grep for "Implementation complexity: LOW. The main change" (old Proposal A) → No matches ✓
- Grep for stale "multi-label BCE" references → All updated or in correction notes ✓
- Sections 5.2, 5.4, 6.1, 6.4, 7.1 updated for consistency with new framing ✓
- Appendix A and B tables updated with corrected TE descriptions ✓
- Document grew from 996 → 1192 lines (net +196 lines of corrections and analysis)

**No code was executed** — this session was purely analytical review and document correction.

## 7. Plan Alignment Review

**Original Goals** (from the brainstorming document's stated purpose):
- Evidence-based review of temporal architectures: **Maintained** — all architecture reviews (Sections 2.1-2.8) remain unchanged and accurate
- Design brainstorm for next-gen TE: **Corrected** — Proposal A fundamentally rewritten from invalid to valid
- Test vs. OOT discrepancy analysis: **Refined** — mechanism clarified, additional evidence added

**Scope Changes**: The corrections revealed the gap between "what the document claimed the TE does" and "what the TE actually does" is larger than expected — requiring 3 major section rewrites rather than minor edits.

## 8. Blockers & Issues

### Resolved
- **Factual error in brainstorming doc** (TE predicts current-day codes → actually predicts next-day) — Corrected with SQL evidence
- **Imprecise distributional exposure argument** — Refined with conditional mechanism
- **Shallow "multi-label BCE treats codes independently" claim** — Replaced with rigorous optimization-landscape analysis addressing user's valid challenge

### Outstanding
- **Brainstorming doc is uncommitted** — The file appears in `git status` as untracked
- **No controlled BCE vs. CrossEntropy experiment exists** — The optimization-landscape hypothesis is empirically supported but unproven. Phase 1 (Option A2) in the roadmap would provide definitive evidence.
- **Temporal shuffle test has never been run** — This is the single most informative free experiment. Listed as Phase 0 in the updated roadmap.

## 9. Next Session Plan

**Immediate Priorities** (ranked):
1. **Run temporal shuffle test on existing R10 checkpoint** (~$5, Phase 0) — Shuffle day order in validation data, compare R@10/loss vs. unshuffled. If negligible change → confirms bag-of-codes, validates loss-function hypothesis.
2. **Implement Option A2 controlled experiment** (~$17, Phase 1) — Keep architecture, replace BCEWithLogitsLoss with CrossEntropy, select single primary-code target per day. Train and compare temporal shuffle test results.
3. **If A2 succeeds** → proceed to Phase 2 (temporal attention bias + CCS auxiliary loss)
4. **If A2 fails** → implement Option A1 (full event-level CLMBR-style architecture, Phase 1b)

**Preparation Required**:
- Design temporal shuffle test protocol (which data to shuffle, which metrics to compare)
- Define "primary code" selection logic for Option A2 (highest-severity? most novel? random?)
- Identify which checkpoint to use for temporal shuffle test

**Open Questions**:
- For Option A2, how to select the single target code per day? (Highest-severity new code? Random sample from day's codes? Most informative code by TF-IDF?)
- Should the temporal shuffle test be run on pretraining metrics (R@10, loss) or downstream metrics (AUC, Lift) or both?
- Is the `moe_flashattn_4.py` the right codebase to modify, or should a new `moe_flashattn_5.py` be created for the experimental branch?

---
**Session**: 1 session (this chat)
**Prior Context**: [Temporal Architecture Brainstorming](expe_analysis/exp_round10/temporal_architecture_research_and_brainstorming.md) produced 2026-03-14
**Files Modified**: 1 (`expe_analysis/exp_round10/temporal_architecture_research_and_brainstorming.md`, +196 lines of corrections)
**Files Created**: 1 (`progress/2026-03-15_session_summary.md`)
**Commits**: 0 (document is untracked, uncommitted)
**Environment**: macOS darwin 24.6.0, Cursor IDE
