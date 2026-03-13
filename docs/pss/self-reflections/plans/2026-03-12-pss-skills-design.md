# PSS Skills Portfolio: Staff-Level AI Engineering Capability Extension

**Date**: 2026-03-12
**Status**: Design — awaiting approval before implementation

## 1. Design Philosophy

### Core Principles

1. **Problem-solving frameworks, not knowledge repositories** — Each skill encodes a THINKING MODE and PROCESS, not a database of techniques. The skills guide the agent through structured reasoning informed by domain knowledge.

2. **Production-proven methods prioritized** — When the skill recommends methods, algorithms, tools, or techniques, it MUST prioritize widely-adopted, production-validated approaches over preliminary, emerging, research-only, or lab-oriented alternatives. The maturity filter: "Is this method used in production at scale by multiple organizations?" must be applied before any recommendation.

3. **Evidence search and cross-validation required** — Every hypothesis, recommendation, and method choice must be backed by evidence from credible sources. Cross-validation across multiple authoritative sources (Google Tuning Playbook, published papers, industry post-mortems) is mandatory. No single source is sufficient.

4. **General frameworks, not project-specific handbooks** — Skills must be applicable to LLM/transformer training broadly. Project-specific scenarios (e.g., clinical multi-label classification) appear as ONE example, not the dominant framing. This prevents constraining the agent to a narrow problem space and missing the real issues.

5. **Junior-to-staff capability bridge** — Skills compensate for the specific gap: the junior engineer has good scientific instincts but lacks the KNOWLEDGE and EXPERIENCE to form mechanistic hypotheses, choose the right methods, and diagnose efficiently. Skills guide knowledge retrieval and application, not just process.

### Maturity Filter for Methods/Algorithms/Tools

When any skill recommends a method, it must classify it:

| Tier | Criteria | Guidance |
|------|----------|----------|
| **Production-Proven** | Used at scale by 3+ well-known big tech companies or unicorn companies or famous research labs/orgs, validated in published results | DEFAULT choice. Prioritize always. |
| **Established Research** | Published at top venue (NeurIPS, ICML, ICLR), reproduced by others | Acceptable when no production-proven alternative exists |
| **Emerging** | Recent publication, limited reproduction, no production track record | Mention as "emerging alternative" only; never default |
| **Experimental** | Preprint only, single-lab results | Do NOT recommend |

---

## 2. Artifact Inventory

| # | Type | Name | Trigger |
|---|------|------|---------|
| 1 | Skill | `hypothesis-driven-diagnosis` | When results are unexpected, model underperforms, or diagnosis needed |
| 2 | Skill | `rigorous-experiment-protocol` | Before designing any experiment |
| 3 | Skill | `multi-expert-consultation` | When expert input needed (after initial hypothesis formation) |
| 4 | Skill | `loss-and-objective-handbook` | When evaluating/selecting loss functions or detecting loss-metric misalignment |
| 5 | Skill | `training-health-and-debugging-handbook` | When training produces unexpected results, instability, or resource issues |
| 6 | Skill | `data-understanding-protocol` | Before modeling on any dataset, or after unexpected results |
| 7 | Skill | `architecture-and-scaling-advisor` | Before architecture changes, scaling decisions, or infrastructure choices |
| 8 | Skill | `representation-quality-assessor` | When evaluating model quality beyond training loss |
| 9 | Rule | `llm-training-first-principles` | Always-on background guidance |
| 10 | Rule | `experiment-hygiene` | Always-on background guidance |
| 11 | Command | `/pre-experiment` | Before any experiment |
| 12 | Command | `/post-experiment` | After any experiment |

---

## 3. Foundation Skills (3)

### Skill 1: `hypothesis-driven-diagnosis`

**Merges former Skills 1 & 2. Addresses:** W1 (outsourcing diagnosis), W4 (architecture-first thinking), W5 (loss analysis), Pattern 1 (consultation cascade), Pattern 2 (symptom reporting).

**Trigger:** When results are unexpected, model underperforms, or diagnosis is needed.

**Framework:**

```
PHASE 1: OBSERVE AND DOCUMENT
  → What specific metrics/behaviors are unexpected?
  → What was the expected outcome and why?
  → Collect all relevant evidence (loss curves, gradient norms, metric trajectories)

PHASE 2: PRIORITY-GUIDED HYPOTHESIS GENERATION
  Navigate the bottleneck hierarchy; at EACH level, form a hypothesis before proceeding:
  
  Level 1: DATA
    → Is the data distribution understood? (frequency analysis, quality assessment)
    → Hypothesis: "The issue is/isn't data-related because ___"
    → If uncertain: run data-understanding-protocol
    
  Level 2: LOSS / OBJECTIVE
    → Does the loss function optimize what the evaluation metric measures?
    → Is there loss-metric divergence? (loss improving but metric stalling)
    → Hypothesis: "The loss is/isn't aligned because ___"
    → If uncertain: run loss-and-objective-handbook
    
  Level 3: TRAINING DYNAMICS
    → Are gradients healthy? Is the LR schedule appropriate? Stability issues?
    → Hypothesis: "Training dynamics are/aren't the bottleneck because ___"
    → If uncertain: run training-health-and-debugging-handbook
    
  Level 4: ARCHITECTURE / SCALING
    → Only after eliminating L1-L3 as bottlenecks
    → Hypothesis: "Architecture is/isn't the bottleneck because ___"
    → If uncertain: run architecture-and-scaling-advisor

PHASE 3: CHEAPEST EXPERIMENT DESIGN
  → For the highest-priority hypothesis, design the minimum-cost test
  → Apply the 30-minute rule: if diagnostic costs <4 GPU-hours, run BEFORE consulting
  → Specify: what result confirms, what result refutes

PHASE 4: EVIDENCE CROSS-VALIDATION
  → Does the hypothesis align with known best practices? (cite specific sources)
  → Has this failure mode been documented in production systems?
  → What do authoritative sources (Google Tuning Playbook, published papers) say?
```

**Key requirements:**
- At each hierarchy level, the agent MUST search for relevant production best practices
- Methods and diagnostic approaches must be production-proven (maturity filter applies)
- Hypotheses must articulate both "what I think" AND "what I don't know"
- If knowledge is insufficient to form a mechanistic hypothesis, articulate the ignorance precisely: "I observe X. I lack understanding of Y. The most naive explanation would be Z."

---

### Skill 2: `rigorous-experiment-protocol`

**Addresses:** W3 (variable confounding), W7 (insufficient baselines), Pattern 3 (breadth-first exploration).

**Trigger:** Before designing any experiment.

**Framework:**

```
PRE-EXPERIMENT PHASE:
  1. Write pre-registration: hypothesis, expected result, refutation criterion
  2. Verify baseline exists:
     - If no baseline: establish simplest possible baseline FIRST
     - Baselines should include trivial predictors AND simple models
  3. Verify ONE variable is changing:
     - If multiple variables: split into sequential experiments
     - Classify hyperparameters as scientific/nuisance/fixed (Google Tuning Playbook)
  4. Estimate cost/time:
     - If diagnostic costs <4 GPU-hours: run before consulting any expert
  5. Evidence cross-validation:
     - Is this experiment design aligned with published methodology?
     - Are the methods being tested production-proven? (maturity filter)
     - Has a similar experiment been conducted elsewhere? What were the results?

EXECUTION PHASE:
  6. Instrument with comprehensive monitoring:
     - Loss decomposition, gradient norms, learning rate trajectory
     - Tier/class/group-level metrics (not just aggregate)
  7. Save ALL artifacts in standard format: config, logs, trajectories, metrics

POST-EXPERIMENT PHASE:
  8. Compare actual vs expected from pre-registration
  9. Write post-mortem:
     - What happened vs what was expected
     - Why (root cause analysis)
     - What I learned (update knowledge base)
     - Hypothesis status: confirmed / rejected / refined to ___
  10. Design next experiment based on updated understanding
```

**Key requirements:**
- Pre-registration is MANDATORY for every experiment
- Baselines must exist before complex model experiments
- One-variable-at-a-time is the default; multi-variable only with factorial design justification
- Methods tested must be production-validated (maturity filter)
- Post-mortem is MANDATORY — learning extraction is as important as results

---

### Skill 3: `multi-expert-consultation`

**Addresses:** Pattern 1 (consultation cascade), W1 (outsourcing diagnosis). Enhanced with multi-agent orchestration.

**Trigger:** When expert input is needed, AFTER initial hypothesis formation (hypothesis-driven-diagnosis must be completed first).

**Framework:**

```
PRE-CONSULTATION (required):
  1. Complete hypothesis-driven-diagnosis output (hard dependency)
  2. Prepare consultation brief:
     - Observation (specific data, not narrative)
     - My hypothesis and reasoning
     - What I DON'T understand (precise uncertainty statement)
     - Specific question for experts to address

CONSULTATION ORCHESTRATION:
  3. User specifies N experts (default: 3 subagents)
  4. Each expert subagent receives:
     - Full context: data, code, results, configuration
     - The consultation brief (including user's hypothesis)
     - Instruction: "Challenge the hypothesis. Provide evidence for/against. 
       Recommend the cheapest diagnostic experiment."
     - Instruction: "Prioritize production-proven methods and widely-validated 
       best practices. Classify any method recommendation by maturity tier."
  5. Subagents analyze INDEPENDENTLY and in PARALLEL

SYNTHESIS (primary agent):
  6. Collect all expert responses
  7. Analyze:
     - AGREEMENTS: What do all/most experts converge on? (high confidence)
     - DISAGREEMENTS: Where do experts diverge? What are the reasons for each?
     - EVIDENCE QUALITY: Which claims are well-evidenced vs speculative?
     - MATURITY: Are recommended methods production-proven or emerging?
  8. For each disagreement:
     - Identify the specific factual claim in dispute
     - Design a minimum-cost experiment to resolve it
  9. Synthesize into: updated hypothesis + proposed experiment plan

POST-CONSULTATION:
  10. Run the proposed experiment BEFORE consulting again
      (breaks the consultation cascade: max 1 consultation round per experiment cycle)
```

**Key requirements:**
- Hypothesis-driven-diagnosis MUST be completed before consultation
- Frame as "challenge my hypothesis" not "diagnose for me"
- Maximum 1 consultation round per experiment cycle (then run experiment)
- User controls number of expert subagents
- Synthesis must explicitly catalog agreements, disagreements, and evidence quality
- All expert recommendations run through the maturity filter

---

## 4. Specialized Skills (5)

### Skill 4: `loss-and-objective-handbook`

**Broadened from original Skill 5.** General framework for any loss-objective misalignment, not just class imbalance.

**Trigger:** When evaluating/selecting loss functions, or when loss-metric divergence is detected.

**Coverage (general, not project-specific):**

1. **Objective-Metric Alignment Analysis**
   - What does the loss mathematically optimize? (Write the gradient)
   - What does the evaluation metric reward?
   - Are these the same? Where do they diverge?
   - Common misalignments: calibration vs ranking, per-sample vs aggregate, pointwise vs listwise

2. **Gradient Distribution Analysis**
   - For any loss: compute expected gradient contribution per class/group ANALYTICALLY (before training)
   - Detect gradient concentration (is >80% of gradient from a small subset?)
   - Gradient starvation diagnosis patterns

3. **Class Imbalance and Long-Tail Issues** (one section, not dominant)
   - Power-law detection and quantification
   - Production-proven solutions: focal loss, class-balanced sampling, two-stage training (Kang et al. ICLR 2020)
   - When to use staged training: representation learning → classifier re-balancing

4. **Multi-Task and Auxiliary Loss Balancing**
   - Loss weighting strategies (production-proven: uncertainty weighting, GradNorm)
   - When auxiliary losses help vs hurt

5. **Loss Selection Decision Tree**
   - Classification: cross-entropy (production standard), focal loss (for imbalance)
   - Ranking: listwise losses, contrastive losses
   - Regression: MSE, Huber (robust to outliers)
   - Multi-label: BCE (default), ASL (for extreme imbalance, established)
   - All recommendations tagged with maturity tier

6. **Loss-Metric Divergence Detection**
   - How to detect: loss improving but metric stalling
   - Diagnostic: plot loss vs metric over training steps
   - Resolution strategies by divergence pattern

---

### Skill 5: `training-health-and-debugging-handbook`

**Expanded from original Skill 6.** Comprehensive, general-purpose training diagnosis incorporating Google Tuning Playbook and HuggingFace Ultra-Scale Playbook.

**Trigger:** When training produces unexpected results, instability, or resource issues.

**Coverage:**

1. **Loss Curve Classification and Resolution**
   (From Google Tuning Playbook + frontier model training practices)
   - Divergence (→ inf/NaN): LR too high, gradient explosion, data pipeline issues
   - Plateau (stalled): LR schedule, loss-objective misalignment, gradient starvation
   - Instability (oscillating): Batch size, LR warmup needs, optimizer selection
   - Slow convergence: Initialization, capacity, data quality
   - Overfitting: Regularization needed (dropout, weight decay, label smoothing)
   - Each pattern: diagnostic checklist → root cause tree → recommended fixes

2. **Gradient Health Monitoring**
   - Per-layer gradient norm tracking (vanishing/exploding detection)
   - Per-group/tier gradient contribution analysis
   - Gradient clipping: when and how (90th percentile threshold as starting point)
   - Outlier gradient detection and remediation

3. **Training Stability Mechanisms** (production-proven)
   - Learning rate warmup: when needed, how to tune warmup_steps
   - Gradient clipping: threshold selection methodology
   - QK-norm, z-loss, logit softcapping (used in frontier models)
   - RMSNorm placement (inside residual: x + f(Norm(x)))
   - Residual initialization patterns (ReZero)

4. **Optimizer Tuning** (production-proven only)
   - AdamW: default choice; tuning guidance per Google Playbook
     (<10 trials: tune LR only; 10-25: tune LR + β₁; 25+: tune LR + β₁ + ε)
   - SGD + Momentum/Nesterov: proven for CV workloads, simpler
   - NAdam: proven variant with Nesterov momentum
   - LR schedule: linear decay or cosine decay as defaults; WSD for frontier training

5. **Common Failure Modes** (from Google Tuning Playbook)
   - Search space boundary issues: best point at edge → expand search
   - Insufficient search space sampling → run more trials
   - Infeasible trial handling → adjust search space, check for bugs
   - Training curve examination checklist: overfitting, step-to-step variance, compute-bound vs not

6. **Memory Optimization and Distributed Training**
   (From HuggingFace Ultra-Scale Playbook)
   - OOM diagnosis and resolution
   - Parallelism strategy selection: Data (default) → Tensor → Pipeline → ZeRO stages
   - Activation checkpointing: when to use, memory-compute tradeoff
   - Mixed precision training: bf16 preferred, fp16 with loss scaling
   - Batch size selection: maximize throughput, not validation performance
   - Multi-GPU debugging: gradient synchronization, batch norm across devices

7. **Checkpoint and State Management**
   - Optimizer state restoration on resume (common failure: loss spikes)
   - Retrospective optimal checkpoint selection (Google Playbook)
   - Checkpoint frequency and storage strategy

8. **Diagnostic Instrumentation Requirements**
   - Minimum monitoring for ANY training run:
     - Training/validation loss curves
     - Learning rate schedule visualization
     - Gradient norms per layer
     - GPU memory utilization
     - Throughput (examples/second)

---

### Skill 6: `data-understanding-protocol`

**Broadened from original Skill 7.** General data understanding framework, not just class imbalance.

**Trigger:** Before modeling on any dataset, or after unexpected results.

**Coverage:**

1. **Distribution Analysis** (general)
   - Feature distributions (continuous: histograms, discrete: frequency counts)
   - Target/label distributions
   - Class balance assessment (if classification)
   - Temporal/spatial patterns
   - Outlier detection and characterization

2. **Data Quality Assessment**
   - Missing data patterns and rates
   - Label noise estimation
   - Duplicate detection
   - Data consistency checks (schema violations, type mismatches)
   - Annotation quality sampling

3. **Data-Model Alignment**
   - Is the data SUFFICIENT for the model complexity? (rough scaling law check)
   - Is the data REPRESENTATIVE of the deployment distribution?
   - Train/validation/test distribution alignment verification
   - Data leakage detection (temporal leakage, feature leakage)

4. **Feature and Signal Analysis**
   - Feature informativeness (mutual information, correlation with target)
   - Feature redundancy detection
   - Co-occurrence / co-variation patterns
   - Signal-to-noise ratio estimation

5. **Class Imbalance Profiling** (one section, not dominant)
   - Frequency histogram (log scale)
   - Power-law exponent, Gini coefficient
   - Head/medium/tail tier boundaries
   - Gradient budget prediction under proposed loss

6. **Scale and Compute Analysis**
   - Dataset size vs model parameters ratio
   - Estimated training time and cost
   - Data augmentation opportunities
   - Tokenization and representation analysis (for text/sequence data)

7. **Implications Document**
   - "This data implies the loss function should be: ___ because ___"
   - "Expected challenges: ___"
   - "Recommended baselines: ___"
   - "Scaling considerations: ___"

---

### Skill 7: `architecture-and-scaling-advisor`

**Expanded from original Skill 8.** Comprehensive, production-proven architecture and scaling decisions.

**Trigger:** Before architecture changes, scaling decisions, or infrastructure choices.

**Coverage:**

1. **Pre-Architecture-Change Gate**
   - REQUIRED: Evidence that data (L1), loss (L2), and training dynamics (L3) have been investigated as bottlenecks
   - What does the scaling law predict? If 2x params gives <1% improvement → capacity is NOT the issue
   - Can this be solved with a training strategy change instead? (cheaper, faster)

2. **Architecture Decisions** (production-proven choices)

   | Component | Production Standard | Alternatives | Avoid |
   |-----------|-------------------|--------------|-------|
   | Attention | Multi-Head Attention | GQA (small groups 2/4/8), MLA | — |
   | Normalization | RMSNorm (pre-norm) | LayerNorm (pre-norm) | Post-norm |
   | Activation | SwiGLU | GELU | ReLU for transformers |
   | Position Encoding | RoPE | ALiBi | Learned absolute |
   | Residual Pattern | Pre-norm: x + f(Norm(x)) | — | Post-norm: Norm(x + f(x)) |
   | Initialization | Xavier/Kaiming | scaled init, ReZero for deep | Random uniform |

3. **Scaling Law Application**
   - Chinchilla: C ≈ 6ND, ~20 tokens per parameter for compute-optimal
   - Beyond Chinchilla: account for inference cost; often train smaller models longer
   - Encoder-only models: scaling relationships differ from decoder-only
   - When to add parameters vs add data vs add compute

4. **MoE Applicability Assessment**
   - Benefits emerge at >1B total parameters with diverse data
   - Load balancing is non-negotiable (auxiliary/bias balancing)
   - Expert parallelism infrastructure requirements
   - Decision: Is MoE justified for this scale and data?

5. **Optimizer Selection** (production-proven only)
   - AdamW: default for transformer training (Google, Meta, OpenAI)
   - SGD + Nesterov: proven for some CV workloads
   - NAdam: proven variant
   - Decision tree: start simple, complexify only with evidence

6. **Precision and Memory**
   - bf16: preferred default (better dynamic range than fp16)
   - fp16 + loss scaling: when bf16 not available
   - Mixed precision: which ops in full vs half precision
   - KV cache optimization for inference

7. **Parallelism Strategy**
   - Data Parallelism: default starting point
   - ZeRO stages: Stage 1 → 2 → 3 progression
   - Tensor Parallelism: within-node, limited by bandwidth
   - Pipeline Parallelism: for 70B+ models
   - Sequence/Context Parallelism: for long sequences
   - Decision tree: single GPU → multi-GPU → multi-node

8. **Vocabulary and Tokenizer**
   - Vocabulary size vs token compression tradeoff
   - Domain-specific tokenizer considerations
   - KV cache cost implications

9. **Context Length and Sequence Handling**
   - Document masking for mixed-length training
   - Position extension strategies (YaRN, NTK-aware)
   - Memory scaling with sequence length (quadratic attention)

---

### Skill 8: `representation-quality-assessor`

**Unchanged from original.** General enough for any representation learning task.

**Trigger:** When evaluating model quality beyond training loss, or when pretraining metrics don't translate to downstream performance.

**Coverage:**

1. **Probing Analysis**
   - Linear probes on frozen representations per layer
   - Per-group/tier probing accuracy
   - Does the representation contain information the decoder can't extract?

2. **Geometric Analysis**
   - Embedding space visualization (t-SNE/UMAP by groups)
   - Per-group embedding variance (homogenization detection)
   - Inter-class vs intra-class distance

3. **Layer-wise Analysis**
   - Per-layer probing accuracy (intermediate layers often outperform final by up to 16%)
   - Attention entropy evolution across layers
   - Representation similarity across layers (CKA)

4. **Downstream Alignment**
   - Gap between pretraining metrics and downstream task performance
   - Does improved pretraining loss translate to improved downstream?
   - If not → representation-metric misalignment

5. **Comparative Analysis**
   - Before vs after intervention: which representations changed?
   - Per-group: which improved, which didn't?

---

## 5. Rules (2) — Approved As-Is

### Rule 9: `llm-training-first-principles`

Always-on workspace rule encoding the priority hierarchy: Data → Loss → Training Strategy → Architecture. Reminds the agent to verify lower levels before suggesting higher-level changes. Contains quick-reference decision points and common anti-patterns.

### Rule 10: `experiment-hygiene`

Always-on workspace rule ensuring experiment rigor. Every experiment must have: hypothesis, single variable, expected outcome, refutation criterion. Enforces the 30-minute rule and pre/post-registration reminders.

---

## 6. Commands (2) — Approved As-Is

### Command 11: `/pre-experiment`

Generates structured pre-registration document: hypothesis, expected result, refutation criterion, variables, controls, diagnostics, estimated cost. Saves to `docs/experiments/`.

### Command 12: `/post-experiment`

Generates structured post-mortem: expected vs actual, hypothesis update, lessons learned, next experiment. Saves to `docs/experiments/`.

---

## 7. Dependencies and Invocation Flow

```
hypothesis-driven-diagnosis
  ├── may invoke → data-understanding-protocol (at Level 1)
  ├── may invoke → loss-and-objective-handbook (at Level 2)
  ├── may invoke → training-health-and-debugging-handbook (at Level 3)
  └── may invoke → architecture-and-scaling-advisor (at Level 4)
      
multi-expert-consultation
  └── REQUIRES → hypothesis-driven-diagnosis (completed first)

rigorous-experiment-protocol
  ├── invokes → /pre-experiment (at start)
  └── invokes → /post-experiment (at end)

representation-quality-assessor
  └── standalone, no dependencies
```

---

## 8. Total Artifact Count

- **3 Foundation Skills** (thinking frameworks)
- **5 Specialized Skills** (domain-informed handbooks)
- **2 Rules** (always-on guidance)
- **2 Commands** (user-invokable actions)
- **Total: 12 artifacts**
