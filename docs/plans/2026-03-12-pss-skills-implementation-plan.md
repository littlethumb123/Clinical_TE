# PSS Skills Portfolio Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create 12 artifacts (8 skills, 2 rules, 2 commands) that extend a junior AI engineer's capability to think, practice, design, debug, and problem-solve like a staff-level AI engineer when co-piloting with LLM and Cursor.

**Architecture:** Layered skill architecture — 3 foundation skills (thinking frameworks), 5 specialized skills (domain-informed handbooks), 2 always-on rules, 2 user-invokable commands. Each skill follows the writing-skills TDD methodology (RED-GREEN-REFACTOR). Skills live in `~/.cursor/skills/` as SKILL.md files.

**Tech Stack:** Cursor skills framework (SKILL.md YAML frontmatter), Cursor rules (.mdc files in `.cursor/rules/`), Cursor commands (.md files in `.cursor/commands/`).

**Design Reference:** `docs/plans/2026-03-12-pss-skills-design.md`

**PSS Assessment Reference:** `docs/pss/self-reflections/PSS_patterns_1.md`

**Key Sources:**
- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- Karpathy Recipe (karpathy.github.io/2019/04/25/recipe/)
- Kang et al. ICLR 2020 (Decoupling Representation and Classifier)
- Menon et al. NeurIPS 2020 (Logit Adjustment)

---

## Shared Principles (Apply to ALL Skills)

Every skill MUST embed these principles (do not repeat them in each task below, but ensure they are present in every skill):

1. **Maturity Filter**: When recommending methods/algorithms/tools, classify as Production-Proven / Established Research / Emerging / Experimental. Default to Production-Proven. Never recommend Experimental.

2. **Evidence Cross-Validation**: Every recommendation must cite credible sources. No single source is sufficient. Cross-reference across Google Tuning Playbook, published papers, and industry post-mortems.

3. **General Frameworks**: Skills must be applicable to LLM/transformer training broadly. Project-specific scenarios (clinical multi-label, class imbalance) appear as ONE example, not the dominant framing.

4. **Knowledge Retrieval Guidance**: Skills should guide the agent to SEARCH for relevant best practices and production norms, not just follow templates. The agent should actively look up what staff engineers know.

---

## Task 1: Infrastructure Setup

**Files:**
- Create: `~/.cursor/skills/hypothesis-driven-diagnosis/` (directory)
- Create: `~/.cursor/skills/rigorous-experiment-protocol/` (directory)
- Create: `~/.cursor/skills/multi-expert-consultation/` (directory)
- Create: `~/.cursor/skills/loss-and-objective-handbook/` (directory)
- Create: `~/.cursor/skills/training-health-and-debugging-handbook/` (directory)
- Create: `~/.cursor/skills/data-understanding-protocol/` (directory)
- Create: `~/.cursor/skills/architecture-and-scaling-advisor/` (directory)
- Create: `~/.cursor/skills/representation-quality-assessor/` (directory)

**Step 1: Create all skill directories**

```bash
mkdir -p ~/.cursor/skills/hypothesis-driven-diagnosis
mkdir -p ~/.cursor/skills/rigorous-experiment-protocol
mkdir -p ~/.cursor/skills/multi-expert-consultation
mkdir -p ~/.cursor/skills/loss-and-objective-handbook
mkdir -p ~/.cursor/skills/training-health-and-debugging-handbook
mkdir -p ~/.cursor/skills/data-understanding-protocol
mkdir -p ~/.cursor/skills/architecture-and-scaling-advisor
mkdir -p ~/.cursor/skills/representation-quality-assessor
```

**Step 2: Verify directories exist**

```bash
ls -la ~/.cursor/skills/ | grep -E "(hypothesis|rigorous|multi-expert|loss-and|training-health|data-understanding|architecture-and|representation)"
```

Expected: 8 directories listed

**Step 3: Commit**

```bash
# No git commit needed for ~/.cursor/skills/ (outside repo)
# Skills are personal workspace configuration
```

---

## Task 2: Foundation Skill — `hypothesis-driven-diagnosis`

**Files:**
- Create: `~/.cursor/skills/hypothesis-driven-diagnosis/SKILL.md`

**Step 1: Write the SKILL.md**

The skill must contain these sections (full content for each):

```markdown
---
name: hypothesis-driven-diagnosis
description: Use when model results are unexpected, performance underperforms expectations, training produces anomalies, or any situation requiring root cause investigation before consulting experts or changing the model
---

# Hypothesis-Driven Diagnosis

## Overview

Structured diagnostic framework that guides you through priority-ordered investigation of ML training issues. Instead of presenting raw symptoms and asking experts to diagnose, this skill forces you to form hypotheses at each bottleneck level, search for production best practices, and design minimum-cost experiments before seeking external input.

**Core principle:** Observe → Hypothesize (guided by priority hierarchy) → Design cheapest test → Execute → Update. Never skip hypothesis formation, even when uncertain.

## When to Use

- Model metrics underperform expectations (loss plateau, metric stagnation, poor downstream performance)
- Training produces unexpected behavior (divergence, instability, anomalous curves)
- Results contradict prior understanding or expert predictions
- Before opening any diagnostic consultation session
- After any experiment produces surprising results

**When NOT to use:**
- Routine hyperparameter tuning with expected results
- Initial project setup (use data-understanding-protocol instead)

## The Diagnostic Framework

### Phase 1: Observe and Document

Before ANY hypothesis generation:

1. **Collect all relevant evidence** — loss curves, gradient norms, metric trajectories, GPU utilization, training logs
2. **State the specific discrepancy** — "I expected X because Y, but observed Z"
3. **Quantify** — use specific numbers, not narrative ("R@10 dropped from 0.83 to 0.79" not "performance got worse")

### Phase 2: Priority-Guided Hypothesis Generation

Navigate this hierarchy TOP-DOWN. At each level, you MUST:
- Search for relevant production best practices and known failure modes
- Form a hypothesis (even if uncertain)
- Specify what experiment would confirm or refute it
- Only proceed to the next level after addressing the current one

```
Level 1: DATA
  Questions:
  - Is the data distribution well-understood?
  - Has the data quality been verified?
  - Is there train/val/test distribution alignment?
  - Has data leakage been ruled out?
  
  If uncertain → invoke data-understanding-protocol
  Hypothesis template: "The data [is/isn't] the bottleneck because [evidence]"

Level 2: LOSS / OBJECTIVE ALIGNMENT  
  Questions:
  - Does the loss function optimize what the evaluation metric measures?
  - Is there loss-metric divergence? (loss improving but metric stalling)
  - Is the gradient distribution appropriate for all classes/groups?
  
  If uncertain → invoke loss-and-objective-handbook
  Hypothesis template: "The loss [is/isn't] aligned because [evidence]"

Level 3: TRAINING DYNAMICS
  Questions:
  - Are gradients healthy? (no vanishing, exploding, or concentration)
  - Is the learning rate schedule appropriate?
  - Are there stability issues?
  - Is the optimizer configured correctly?
  
  If uncertain → invoke training-health-and-debugging-handbook
  Hypothesis template: "Training dynamics [are/aren't] the bottleneck because [evidence]"

Level 4: ARCHITECTURE / SCALING
  Questions (ONLY after eliminating L1-L3):
  - Does the model have sufficient capacity?
  - What does the scaling law predict?
  - Can this be solved with a training strategy change instead?
  
  If uncertain → invoke architecture-and-scaling-advisor
  Hypothesis template: "Architecture [is/isn't] the bottleneck because [evidence]"
```

### Phase 3: Cheapest Experiment Design

For the highest-priority hypothesis:

1. **Estimate cost** — compute hours, wall clock time
2. **Apply the 30-minute rule**: if the diagnostic costs <4 GPU-hours, run it BEFORE consulting anyone
3. **Pre-register**: write down what result confirms and what refutes your hypothesis
4. **Common cheap diagnostics** (run these before expensive experiments):

| Diagnostic | Cost | What It Reveals |
|-----------|------|-----------------|
| Overfit single batch | ~5 min | Model capacity sufficient? |
| Loss at initialization | ~1 min | Expected random baseline? |
| Gradient norms per layer | ~30 min added | Gradient health |
| Logit/output histograms | ~10 min on checkpoint | Output collapse? |
| 2-layer vs N-layer comparison | ~1 hour | Depth contribution |
| Loss decomposition by group | ~10 min | Which groups drive loss |

### Phase 4: Evidence Cross-Validation

Before acting on any hypothesis:

1. **Search production best practices** — What do Google Tuning Playbook, published papers, industry post-mortems say about this failure mode?
2. **Apply maturity filter** — Are the proposed solutions production-proven?
3. **Cross-validate** — Do multiple authoritative sources agree?

### Handling Insufficient Knowledge

If you lack the domain knowledge to form a mechanistic hypothesis:

**Do NOT skip to expert consultation.** Instead, write:

```
I observe: [specific data]
I expected: [what and why]
I lack understanding of: [precise knowledge gap]
The most naive explanation would be: [simplest possible cause]
To resolve my uncertainty, I need to learn about: [specific topic]
The cheapest thing I can try: [minimum-cost action]
```

This precise uncertainty articulation makes expert consultation dramatically more efficient.

## Common Mistakes

| Mistake | Correction |
|---------|------------|
| Presenting raw data and asking "diagnose this" | Form your own hypothesis first, even if uncertain |
| Starting with architecture changes | Check data, loss, training dynamics FIRST |
| Consulting 4+ experts before running any experiment | Run the cheapest diagnostic BEFORE consulting |
| Changing multiple variables at once | One hypothesis, one experiment, one variable |
| Accepting expert advice without evidence cross-validation | Verify recommendations against production best practices |

## Related Skills

- **REQUIRED FOLLOW-UP:** After diagnosis, use rigorous-experiment-protocol to test hypotheses
- **OPTIONAL:** multi-expert-consultation if hypothesis confidence is low AFTER Phase 3
- **REFERENCE:** data-understanding-protocol (Level 1), loss-and-objective-handbook (Level 2), training-health-and-debugging-handbook (Level 3), architecture-and-scaling-advisor (Level 4)
```

**Step 2: Verify the skill file exists and has correct frontmatter**

```bash
head -5 ~/.cursor/skills/hypothesis-driven-diagnosis/SKILL.md
```

Expected: YAML frontmatter with name and description

**Step 3: Word count check**

```bash
wc -w ~/.cursor/skills/hypothesis-driven-diagnosis/SKILL.md
```

Expected: <800 words (skills should be concise; if over, trim reference sections)

**Step 4: Commit**

```bash
cd ~/.cursor/skills && git add hypothesis-driven-diagnosis/SKILL.md 2>/dev/null || echo "Not a git repo - skill saved to filesystem"
```

---

## Task 3: Foundation Skill — `rigorous-experiment-protocol`

**Files:**
- Create: `~/.cursor/skills/rigorous-experiment-protocol/SKILL.md`

**Step 1: Write the SKILL.md**

Key sections to include:

**Frontmatter:**
```yaml
---
name: rigorous-experiment-protocol
description: Use when designing any ML experiment, ablation study, or hyperparameter search — before running training jobs, before comparing model variants, or when experiment results are inconclusive
---
```

**Content must cover:**

1. **Pre-Experiment Phase**
   - Pre-registration template (hypothesis, expected result, refutation criterion, single variable)
   - Baseline verification gate: "Does a simple baseline exist? If not, create one FIRST"
   - Hyperparameter classification (Google Tuning Playbook: scientific vs nuisance vs fixed)
   - Cost estimation and the 30-minute rule
   - Method maturity check: are methods being tested production-proven?

2. **Experiment Design Principles**
   - One-variable-at-a-time as default
   - Factorial design when multiple variables MUST be tested (with justification)
   - Search space design (Google Tuning Playbook: quasi-random for exploration, Bayesian for exploitation)
   - Baseline requirements: trivial predictor + simple model + existing production model

3. **Execution Phase**
   - Mandatory instrumentation checklist (loss curves, gradient norms, LR trajectory, per-group metrics)
   - Artifact saving standards (config, logs, trajectories, metrics in structured format)
   - Checkpoint strategy

4. **Post-Experiment Phase**
   - Post-mortem template (expected vs actual, hypothesis status, surprise, lesson, next experiment)
   - Learning extraction: "What do I now know that I didn't before?"
   - Knowledge update: feed back into hypothesis-driven-diagnosis

5. **Common Mistakes**
   - Confounding variables (changing architecture + optimizer + LR simultaneously)
   - No baseline (comparing complex models without simple reference point)
   - No pre-registration (post-hoc rationalization of results)
   - Insufficient search space sampling (Google Playbook guidance)

**Step 2: Verify and word count check**

```bash
head -5 ~/.cursor/skills/rigorous-experiment-protocol/SKILL.md && echo "---" && wc -w ~/.cursor/skills/rigorous-experiment-protocol/SKILL.md
```

**Step 3: Commit**

---

## Task 4: Foundation Skill — `multi-expert-consultation`

**Files:**
- Create: `~/.cursor/skills/multi-expert-consultation/SKILL.md`

**Step 1: Write the SKILL.md**

**Frontmatter:**
```yaml
---
name: multi-expert-consultation
description: Use when expert input is needed for ML training diagnosis, architecture decisions, or method selection — after forming an initial hypothesis via hypothesis-driven-diagnosis, when confidence is low, or when the problem requires cross-disciplinary perspectives
---
```

**Content must cover:**

1. **Hard Dependency Gate**
   - hypothesis-driven-diagnosis MUST be completed first
   - If not completed, redirect to that skill
   - Consultation brief template: observation, hypothesis, uncertainty statement, specific question

2. **Subagent Orchestration Protocol**
   - User specifies N experts (default: 3)
   - Each expert subagent receives FULL context: data, code, results, configuration
   - Each receives the consultation brief with user's hypothesis
   - Each receives instruction: "Challenge the hypothesis. Provide evidence for/against. Recommend cheapest diagnostic. Prioritize production-proven methods. Classify any recommendation by maturity tier."
   - Subagents analyze INDEPENDENTLY and in PARALLEL (use Task tool with subagent_type)

3. **Synthesis Protocol (Primary Agent)**
   - Collect all expert responses
   - Structured analysis:
     ```
     AGREEMENTS: What do all/most experts converge on? → High confidence
     DISAGREEMENTS: Where do experts diverge?
       → For each disagreement:
         - Expert A says: ___ because ___
         - Expert B says: ___ because ___
         - The specific factual claim in dispute: ___
         - Experiment to resolve: ___
     EVIDENCE QUALITY:
       → Which claims cite specific evidence vs speculate?
       → Which recommendations are production-proven vs emerging?
     MATURITY ASSESSMENT:
       → Are recommended methods widely validated?
       → Flag any emerging/experimental recommendations
     ```
   - Output: Updated hypothesis + proposed experiment plan

4. **Post-Consultation Gate**
   - After synthesis: run the proposed experiment BEFORE consulting again
   - Maximum 1 consultation round per experiment cycle
   - This breaks the consultation cascade

5. **Flowchart**
   ```
   hypothesis-driven-diagnosis (completed)
     → Prepare consultation brief
     → User specifies N experts
     → Launch N subagents in parallel
     → Collect responses
     → Synthesize (agreements/disagreements/evidence/maturity)
     → Design experiment to resolve disagreements
     → RUN EXPERIMENT (not another consultation)
   ```

**Step 2: Verify and word count check**

**Step 3: Commit**

---

## Task 5: Specialized Skill — `loss-and-objective-handbook`

**Files:**
- Create: `~/.cursor/skills/loss-and-objective-handbook/SKILL.md`

**Step 1: Write the SKILL.md**

**Frontmatter:**
```yaml
---
name: loss-and-objective-handbook
description: Use when selecting or evaluating loss functions, when training loss and evaluation metrics diverge, when suspecting objective misalignment, or when encountering class imbalance or gradient distribution issues during any ML training task
---
```

**Content must cover (GENERAL, not project-specific):**

1. **Objective-Metric Alignment Analysis** — The core universal question: "What does my loss optimize vs what does my metric reward?" Common misalignment patterns: calibration vs ranking, per-sample vs aggregate, pointwise vs listwise, independent labels vs structured output.

2. **Gradient Distribution Analysis** — How to compute expected gradient contribution per class/group ANALYTICALLY before training. Gradient concentration detection. Template for any loss function.

3. **Loss Selection Decision Tree** — Organized by TASK TYPE, not by loss function name:
   - Classification (single-label): cross-entropy (production standard)
   - Classification (multi-label): BCE (default), focal loss (for imbalance), ASL (established for extreme imbalance)
   - Ranking: listwise losses, contrastive losses
   - Regression: MSE, Huber
   - Generation: cross-entropy with label smoothing
   - All tagged by maturity tier

4. **Class Imbalance and Long-Tail** (one section among many):
   - Detection: power-law analysis, Gini coefficient
   - Production-proven solutions: focal loss (ICCV 2017), two-stage training (Kang et al. ICLR 2020), logit adjustment (Menon et al. NeurIPS 2020)
   - When staged training applies

5. **Multi-Task and Auxiliary Loss Balancing** — Uncertainty weighting, GradNorm, manual balancing guidance

6. **Loss-Metric Divergence Detection** — How to detect (plot loss vs metric), what it means, resolution patterns by divergence type

7. **Common Mistakes** — Using default loss without analysis, ignoring gradient distribution, changing loss without re-tuning LR, applying class weights without gradient budget analysis

**Step 2: Verify and word count check**

**Step 3: Commit**

---

## Task 6: Specialized Skill — `training-health-and-debugging-handbook`

**Files:**
- Create: `~/.cursor/skills/training-health-and-debugging-handbook/SKILL.md`

This is the MOST COMPREHENSIVE skill. Consider splitting into SKILL.md (core) + reference files if content exceeds ~800 words.

**Step 1: Write the SKILL.md**

**Frontmatter:**
```yaml
---
name: training-health-and-debugging-handbook
description: Use when training produces unexpected results, loss diverges or plateaus, training is unstable, GPU memory issues occur, distributed training fails, or any training run needs health assessment or debugging
---
```

**Content must cover (comprehensive, general-purpose):**

1. **Loss Curve Classification** (from Google Tuning Playbook + frontier practices)
   - Divergence (→ inf/NaN): diagnostic tree → fixes (LR, gradient clipping, data pipeline)
   - Plateau: diagnostic tree → fixes (LR schedule, loss alignment, gradient starvation)
   - Instability (oscillating): diagnostic tree → fixes (batch size, warmup, optimizer)
   - Slow convergence: diagnostic tree → fixes (initialization, capacity, data quality)
   - Overfitting: diagnostic tree → fixes (dropout, weight decay, label smoothing, data augmentation)

2. **Gradient Health Monitoring**
   - Per-layer gradient norm tracking
   - Gradient clipping methodology (90th percentile threshold starting point)
   - Vanishing/exploding gradient diagnosis
   - Per-group gradient contribution analysis

3. **Training Stability** (production-proven mechanisms)
   - LR warmup: when needed, how to find warmup_steps (Google Playbook methodology)
   - Gradient clipping: threshold selection, when >50% clipped means LR too high
   - QK-norm, z-loss, logit softcapping (frontier model practices)
   - Pre-norm residual: x + f(Norm(x)), not Norm(x + f(x))
   - ReZero initialization for deep models

4. **Optimizer Tuning** (PRODUCTION-PROVEN ONLY)
   - AdamW: default. Tuning budget guide (Google Playbook):
     - <10 trials: tune LR only
     - 10-25 trials: tune LR + β₁
     - 25+ trials: tune LR + β₁ + ε
     - Substantially more: additionally tune β₂
   - SGD + Momentum/Nesterov: proven for CV, simpler
   - NAdam: proven variant
   - LR schedule: linear decay or cosine decay as defaults
   - WSD (warmup-stable-decay): proven in frontier training

5. **Common Failure Modes** (from Google Tuning Playbook)
   - Search space boundary issues (best point at edge → expand)
   - Insufficient sampling (run more trials)
   - Infeasible trial handling (adjust search space, check for bugs)
   - Training curve examination checklist
   - Batch size interactions with other hyperparameters

6. **Memory and Distributed Training** (from HF Ultra-Scale Playbook)
   - OOM diagnosis and resolution hierarchy
   - Parallelism strategy: Data → ZeRO → Tensor → Pipeline
   - Activation checkpointing: memory-compute tradeoff
   - Mixed precision: bf16 preferred, fp16 with loss scaling
   - Batch size: maximize throughput, not directly tune for validation
   - Multi-GPU: gradient sync, batch norm across devices, RNG seed management

7. **Checkpoint and State Management**
   - Optimizer state restoration on resume (failure → loss spikes)
   - Retrospective optimal checkpoint selection
   - Checkpoint frequency and storage

8. **Minimum Monitoring Requirements** (for ANY training run)
   - Training/validation loss curves
   - LR schedule visualization
   - Gradient norms per layer
   - GPU memory utilization
   - Throughput (examples/second)

**Step 2: If over ~800 words, split reference tables into supporting file**

```bash
wc -w ~/.cursor/skills/training-health-and-debugging-handbook/SKILL.md
# If >800: create failure-modes-reference.md for detailed tables
```

**Step 3: Verify and commit**

---

## Task 7: Specialized Skill — `data-understanding-protocol`

**Files:**
- Create: `~/.cursor/skills/data-understanding-protocol/SKILL.md`

**Step 1: Write the SKILL.md**

**Frontmatter:**
```yaml
---
name: data-understanding-protocol
description: Use before modeling on any new dataset, when revisiting a dataset after unexpected results, when suspecting data quality issues, data leakage, or distribution problems, or when Karpathy Step 1 become-one-with-the-data has not been completed
---
```

**Content must cover (GENERAL, not just class imbalance):**

1. **Distribution Analysis** — Feature distributions, target/label distributions, class balance (if classification), temporal/spatial patterns, outlier detection

2. **Data Quality Assessment** — Missing data rates and patterns, label noise estimation, duplicate detection, consistency checks, annotation quality

3. **Data-Model Alignment** — Is data sufficient for model complexity? (scaling law rough check), is data representative of deployment distribution?, train/val/test alignment, data leakage detection (temporal, feature)

4. **Feature and Signal Analysis** — Feature informativeness (MI, correlation), redundancy, co-occurrence patterns, signal-to-noise ratio

5. **Class Imbalance Profiling** (one section, not dominant) — Frequency histogram, power-law exponent, Gini, tier boundaries, gradient budget prediction

6. **Scale and Compute** — Dataset size vs model parameters ratio, training time/cost estimate, data augmentation opportunities, tokenization analysis

7. **Implications Document Template** — Data-informed recommendations for loss function, expected challenges, baselines, scaling

**Step 2: Verify and commit**

---

## Task 8: Specialized Skill — `architecture-and-scaling-advisor`

**Files:**
- Create: `~/.cursor/skills/architecture-and-scaling-advisor/SKILL.md`

**Step 1: Write the SKILL.md**

**Frontmatter:**
```yaml
---
name: architecture-and-scaling-advisor
description: Use before making architecture changes, scaling decisions, optimizer selection, parallelism strategy choices, or infrastructure decisions for ML training — requires evidence that data, loss, and training dynamics have been investigated first
---
```

**Content must cover (COMPREHENSIVE):**

1. **Pre-Architecture-Change Gate** — Evidence required that L1-L3 bottlenecks have been investigated. Scaling law prediction. Can training strategy change solve it instead?

2. **Architecture Decisions Table** (production-proven only):

   | Component | Production Standard | Alternative | Avoid |
   |-----------|-------------------|-------------|-------|
   | Attention | MHA | GQA (groups 2/4/8), MLA | — |
   | Normalization | RMSNorm (pre-norm) | LayerNorm (pre-norm) | Post-norm |
   | Activation | SwiGLU | GELU | ReLU for transformers |
   | Position | RoPE | ALiBi | Learned absolute |
   | Residual | Pre-norm: x + f(Norm(x)) | — | Post-norm: Norm(x+f(x)) |
   | Init | Xavier/Kaiming | Scaled init, ReZero | Random uniform |

3. **Scaling Laws** — Chinchilla (C ≈ 6ND), beyond-Chinchilla (inference-aware), encoder vs decoder differences

4. **MoE Assessment** — When justified (>1B params, diverse data), load balancing requirements, expert parallelism

5. **Optimizer Selection** (production-proven ONLY) — AdamW default, SGD+Nesterov for CV, NAdam variant. Decision tree: start simple, complexify with evidence

6. **Precision and Memory** — bf16 preferred, fp16 + loss scaling fallback, mixed precision strategy, KV cache optimization

7. **Parallelism Strategy** — DP → ZeRO stages → TP → PP → sequence parallelism. Decision tree by model size and hardware

8. **Vocabulary and Tokenizer** — Size vs compression tradeoff, domain-specific considerations, KV cache implications

9. **Context Length** — Document masking, position extension (YaRN), memory scaling with sequence length

**Step 2: Verify and commit**

---

## Task 9: Specialized Skill — `representation-quality-assessor`

**Files:**
- Create: `~/.cursor/skills/representation-quality-assessor/SKILL.md`

**Step 1: Write the SKILL.md**

**Frontmatter:**
```yaml
---
name: representation-quality-assessor
description: Use when evaluating model quality beyond training loss, when pretraining metrics do not translate to downstream performance, when assessing learned embeddings or representations, or when comparing model checkpoints
---
```

**Content must cover:**

1. **Probing Analysis** — Linear probes on frozen representations, per-layer probing, per-group accuracy, information containment vs extraction gap

2. **Geometric Analysis** — Embedding visualization (t-SNE/UMAP), per-group variance, inter-class vs intra-class distance, homogenization detection

3. **Layer-wise Analysis** — Per-layer probing accuracy (intermediate often outperform final by ~16%), attention entropy, representation similarity (CKA)

4. **Downstream Alignment** — Pretraining-downstream performance gap, when improved pretraining loss doesn't translate

5. **Comparative Analysis** — Before/after intervention comparison, per-group improvement assessment

**Step 2: Verify and commit**

---

## Task 10: Rule — `llm-training-first-principles`

**Files:**
- Create: `.cursor/rules/llm-training-first-principles.mdc`

**Step 1: Write the rule**

```markdown
# LLM Training First Principles

Always-on guidance for LLM/transformer training decisions.

## Priority Hierarchy

Before suggesting architecture changes, verify this order:
1. **Data** — Is the data understood? Quality verified? Distribution analyzed?
2. **Loss/Objective** — Does loss align with evaluation metric? Gradient distribution appropriate?
3. **Training Dynamics** — Gradients healthy? LR schedule appropriate? Stability OK?
4. **Architecture** — Only after eliminating 1-3 as bottlenecks

## Quick Decision Points

- User reports loss-metric divergence → suggest loss-and-objective-handbook BEFORE architecture changes
- User reports performance plateau → suggest training-health-and-debugging-handbook diagnostics FIRST
- User wants to try new architecture → ask "Have data, loss, and training dynamics been ruled out?"
- User reports tail/minority class failure → check gradient distribution and loss alignment FIRST

## Method Maturity Filter

When recommending methods, algorithms, tools, or techniques:
- Prioritize production-proven (used at scale by 3+ organizations)
- Flag emerging methods explicitly as "emerging, not yet production-validated"
- Never recommend experimental/preprint-only methods as defaults

## Anti-Patterns to Watch For

- Changing architecture without evidence that architecture is the bottleneck
- Trying emerging optimizers before exhausting AdamW tuning
- Adding model complexity before establishing simple baselines
- Running expensive experiments before cheap diagnostics
```

**Step 2: Verify the rule is in `.cursor/rules/` and uses .mdc extension**

**Step 3: Commit to repo**

```bash
git add .cursor/rules/llm-training-first-principles.mdc
git commit -m "feat: add llm-training-first-principles always-on rule"
```

---

## Task 11: Rule — `experiment-hygiene`

**Files:**
- Create: `.cursor/rules/experiment-hygiene.mdc`

**Step 1: Write the rule**

```markdown
# Experiment Hygiene

Always-on guidance ensuring experimental rigor in ML training.

## Requirements for Every Experiment

Every experiment MUST have:
- **Hypothesis**: What do you expect and why? (even if uncertain)
- **Single variable**: Only ONE thing changes (if multiple, split into sequential experiments)
- **Expected outcome**: What result would confirm the hypothesis?
- **Refutation criterion**: What result would disprove the hypothesis?

## The 30-Minute Rule

If a diagnostic experiment costs <4 GPU-hours, run it BEFORE consulting any expert.

Common cheap diagnostics: overfit single batch (~5 min), loss at initialization (~1 min), gradient norms per layer (~30 min), logit histograms (~10 min), loss decomposition by group (~10 min).

## Reminders

- Before running: "Has a pre-registration doc been written?" (invoke /pre-experiment)
- After results: "Has a post-mortem been written?" (invoke /post-experiment)
- Before next experiment: "What did I learn from the last one?"

## Baseline Gate

Before complex model experiments: does a simple baseline exist?
- Trivial predictor (frequency-based, random)
- Simple model (logistic regression, MLP)
- Production reference (existing system)

If no baseline exists, establishing one IS the next experiment.
```

**Step 2: Commit**

```bash
git add .cursor/rules/experiment-hygiene.mdc
git commit -m "feat: add experiment-hygiene always-on rule"
```

---

## Task 12: Command — `/pre-experiment`

**Files:**
- Create: `.cursor/commands/pre-experiment.md`

**Step 1: Write the command**

```markdown
# Pre-Experiment Registration

Generate a structured pre-registration document before any ML experiment.

## Instructions

Ask the user for (or infer from context):
1. Experiment name
2. What they're testing (the hypothesis)
3. What they expect to see

Then generate and save this document:

## Template

```markdown
## Pre-Registration: [Experiment Name]
**Date**: [today]
**Hypothesis**: [What the user thinks will happen and why]
**Expected Result**: [Specific metric values or behaviors expected]
**Refutation Criterion**: If I see [specific result], my hypothesis is wrong
**Scientific Variable**: [The ONE thing being changed]
**Nuisance Variables**: [Things being tuned/optimized over]
**Fixed Variables**: [Things held constant]
**Controls**: [What stays the same vs baseline]
**Baseline Reference**: [What existing result this compares against]
**Diagnostics to Instrument**: [Gradient tracking, logit histograms, etc.]
**Estimated Cost**: [GPU-hours, wall time, $]
**Method Maturity**: [Are all methods production-proven?]
```

Save to: `docs/experiments/YYYY-MM-DD-[name]-preregistration.md`
```

**Step 2: Commit**

```bash
git add .cursor/commands/pre-experiment.md
git commit -m "feat: add /pre-experiment command for experiment pre-registration"
```

---

## Task 13: Command — `/post-experiment`

**Files:**
- Create: `.cursor/commands/post-experiment.md`

**Step 1: Write the command**

```markdown
# Post-Experiment Review

Generate a structured post-mortem document after any ML experiment.

## Instructions

Ask the user for (or infer from context):
1. Which experiment (link to pre-registration if exists)
2. What happened
3. Whether it matched expectations

Then generate and save this document:

## Template

```markdown
## Post-Mortem: [Experiment Name]
**Date**: [today]
**Pre-Registration Reference**: [link to pre-reg doc, if exists]

### Expected vs Actual
| Metric | Expected | Actual | Delta |
|--------|----------|--------|-------|
| [metric1] | [value] | [value] | [+/-] |

### Hypothesis Status
[Confirmed / Rejected / Refined to: ___]

### Surprise
What I didn't expect: ___

### Root Cause Analysis
Why the result occurred: ___

### Lesson Learned
What I now know that I didn't before: ___

### Knowledge Update
- Updated understanding: ___
- Revised beliefs about: ___

### Next Experiment
Based on this result, the next experiment should: ___
Hypothesis for next: ___
```

Save to: `docs/experiments/YYYY-MM-DD-[name]-postmortem.md`
```

**Step 2: Commit**

```bash
git add .cursor/commands/post-experiment.md
git commit -m "feat: add /post-experiment command for experiment post-mortem"
```

---

## Task 14: Integration Verification

**Step 1: Verify all skills are discoverable**

```bash
echo "=== Skills ===" && ls ~/.cursor/skills/*/SKILL.md | wc -l
echo "=== Rules ===" && ls .cursor/rules/llm-training-first-principles.mdc .cursor/rules/experiment-hygiene.mdc 2>/dev/null | wc -l
echo "=== Commands ===" && ls .cursor/commands/pre-experiment.md .cursor/commands/post-experiment.md 2>/dev/null | wc -l
```

Expected: Skills: 8, Rules: 2, Commands: 2

**Step 2: Verify frontmatter format for all skills**

```bash
for f in ~/.cursor/skills/*/SKILL.md; do
  echo "--- $(basename $(dirname $f)) ---"
  head -4 "$f"
  echo ""
done
```

Expected: Each skill has valid YAML frontmatter with name and description

**Step 3: Verify word counts (skills should be <800 words for core, with supporting files for heavy reference)**

```bash
for f in ~/.cursor/skills/*/SKILL.md; do
  echo "$(basename $(dirname $f)): $(wc -w < "$f") words"
done
```

**Step 4: Final commit for rules and commands**

```bash
git add .cursor/rules/llm-training-first-principles.mdc .cursor/rules/experiment-hygiene.mdc .cursor/commands/pre-experiment.md .cursor/commands/post-experiment.md
git commit -m "feat: add PSS skills portfolio — 8 skills, 2 rules, 2 commands for staff-level AI engineering capability"
```

---

## Execution Order and Dependencies

```
Task 1:  Infrastructure (directories)          — no dependencies
Task 2:  hypothesis-driven-diagnosis            — no dependencies (HIGHEST PRIORITY)
Task 3:  rigorous-experiment-protocol           — no dependencies
Task 4:  multi-expert-consultation              — references Task 2
Task 5:  loss-and-objective-handbook            — referenced by Task 2 (Level 2)
Task 6:  training-health-and-debugging-handbook — referenced by Task 2 (Level 3)
Task 7:  data-understanding-protocol            — referenced by Task 2 (Level 1)
Task 8:  architecture-and-scaling-advisor       — referenced by Task 2 (Level 4)
Task 9:  representation-quality-assessor        — standalone
Task 10: llm-training-first-principles rule     — no dependencies
Task 11: experiment-hygiene rule                — no dependencies
Task 12: /pre-experiment command                — no dependencies
Task 13: /post-experiment command               — no dependencies
Task 14: Integration verification               — depends on ALL above
```

**Recommended parallel execution:**
- Batch 1: Tasks 1 (infra), 10, 11, 12, 13 (rules + commands — lightweight)
- Batch 2: Tasks 2, 3, 4 (foundation skills — highest priority)
- Batch 3: Tasks 5, 6, 7 (specialized skills — handbooks)
- Batch 4: Tasks 8, 9 (specialized skills — advisors)
- Batch 5: Task 14 (verification)
