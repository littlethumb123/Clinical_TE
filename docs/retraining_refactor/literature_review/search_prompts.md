# LITERATURE REVIEW REQUEST: Mixture of Experts (MoE) Architectures for Patient Historical Claims Transformers

## OBJECTIVE
Conduct a comprehensive literature review to identify research papers that compare different Mixture of Experts (MoE) architecture variants applied to patient historical claims data or similar healthcare sequential data using transformer models.

## RESEARCH CONTEXT & BACKGROUND

### My Current Architecture
I have developed a **hierarchical transformer model** for processing patient healthcare claims data with the following architecture:

**Data Structure:**
- **Input**: Patient medical claims sequences (up to 200 days)
- **Daily Level**: Up to 80 medical codes per day (ICD-10 diagnoses, CPT procedures, medications)
- **Vocabulary**: ~84,000-98,000 unique medical codes
- **Demographics**: Age (in months) and gender embeddings
- **Prediction Task**: Specialist physician visits, readmissions, or disease progression

**Two-Level Hierarchical Architecture:**

**Level 1: Daily Code Encoder**
- Purpose: Encode co-occurring medical codes within a single day
- Architecture: 1-layer transformer encoder
- Configuration: 4 attention heads, 256 hidden dimensions, no dropout
- Aggregation: Max-pooling across codes to create daily representation
- Output: Single 256-dimensional vector per day

**Level 2: Temporal Sequence Encoder** 
- Purpose: Model disease progression and temporal patterns across days
- Architecture: 6-layer transformer encoder
- Configuration: 16 attention heads, 512 FFN dimensions, 0.1 dropout
- Causal masking: Prevents looking at future days
- Output: Sequence of temporally-aware daily representations

**Model Parameters:**
- Embedding size: 256 dimensions
- Total parameters: ~3-5M (depending on vocabulary size)
- Sequence length: 200 days
- Training: Masked language modeling (MLM) pre-training on millions of patient sequences

### My Proposed MoE Enhancement
I am designing a Mixture of Experts extension to this architecture with the following specifications:

**Proposed MoE Design:**
- **Expert Placement**: Replace FFN layers in temporal encoder layers 2-5 with MoE layers
- **Number of Experts**: 8 experts (testing 4, 8, 16 in ablations)
- **Routing Strategy**: Learned top-K routing (K=2, comparing K=1, 2, 4)
- **Expert Architecture**: Each expert is a standard FFN (256→512→256)
- **Load Balancing**: Importance-load loss with auxiliary weight 0.01
- **Motivation**: Capture patient subpopulation heterogeneity (chronic vs. acute, disease-specific patterns, age/complexity groups)

**Hypothesis:**
Different patient subpopulations (e.g., diabetic patients, cardiovascular patients, acute care patients, complex comorbidities) follow distinct temporal patterns. MoE experts should specialize in these subpopulations, improving prediction accuracy while maintaining computational efficiency through sparse activation.

## RESEARCH QUESTIONS

I want to understand whether prior research has:

1. **Applied MoE to healthcare claims/EHR sequential data** using transformer architectures
2. **Compared different MoE variants** (expert counts, top-K values, routing strategies) on healthcare data
3. **Analyzed expert specialization patterns** in medical domains (do experts learn clinically meaningful specializations?)
4. **Conducted ablation studies** on MoE design choices (layer placement, load balancing strategies, capacity factors)
5. **Benchmarked MoE vs. dense transformers** for healthcare prediction tasks
6. **Studied hierarchical or two-level transformers** combined with MoE for healthcare

## SPECIFIC SEARCH CRITERIA

### PRIMARY SEARCH: Direct Comparisons (Highest Priority)
Look for papers that:
- Use MoE architectures with transformers on patient claims, EHR, or medical code sequences
- Compare multiple MoE variants (e.g., Switch Transformer vs. Mixtral vs. Soft MoE)
- Perform ablation studies on MoE hyperparameters (expert count, top-K, layer placement)
- Focus on temporal/sequential medical data (not just static patient snapshots)
- Report expert specialization analysis (what different experts learn)

### SECONDARY SEARCH: Related Healthcare MoE Applications
Look for papers that:
- Use MoE in healthcare but on different data types (imaging, text, multi-modal)
- Apply MoE to patient stratification or subpopulation modeling
- Use RNNs/LSTMs with MoE for clinical sequences (not transformers)
- Employ ensemble or multi-expert approaches for patient modeling
- Study patient heterogeneity and subpopulation-specific models

### TERTIARY SEARCH: Foundational MoE Papers with Healthcare Applications
Look for:
- Switch Transformer, Mixtral, ST-MoE, GLaM applied to healthcare
- MoE architecture surveys mentioning healthcare use cases
- Clinical AI papers comparing sparse vs. dense models
- Healthcare transformer papers (BEHRT, Med-BERT, ClinicalBERT) with MoE extensions

### QUATERNARY SEARCH: Related Architectural Patterns
Look for:
- Hierarchical transformers for healthcare (two-level encoding)
- Conditional computation in medical AI
- Patient trajectory modeling with expert networks
- Domain adaptation in healthcare using multiple specialized models
- Routing mechanisms or attention variants for patient subpopulations

## KNOWN LITERATURE (Already Found - Do Not Re-Report)

I am already aware of these papers:

1. **"Learning to Adapt Clinical Sequences with Residual Mixture of Experts"** (Lee & Hauskrecht, 2022) - Uses MoE with RNNs, not transformers
2. **"NeuroMoE"** (Raza et al., 2025) - MoE for neurological disorder classification with MRI/imaging
3. **"Enhancing Healthcare Recommendation Systems with Multimodal LLMs-based MOE"** (Xu & Wang, 2024) - Recommendation systems, not sequential prediction
4. **"Improving Transformer Performance for French Clinical Notes Classification"** (2024) - Clinical text classification
5. **Med-BERT, BEHRT** - Healthcare transformers without MoE
6. **Switch Transformer, Mixtral** - Foundational MoE papers (NLP, not healthcare)

## LITERATURE REVIEW INSTRUCTIONS

### Phase 1: Systematic Database Search
Search the following databases and sources:
- **Academic**: PubMed, IEEE Xplore, ACM Digital Library, Google Scholar
- **Preprints**: arXiv (cs.LG, cs.CL, stat.ML sections), bioRxiv, medRxiv
- **Conference Proceedings**: NeurIPS, ICML, ICLR, AAAI, AMIA, ML4H, CHIL
- **Journals**: JAMIA, Nature Digital Medicine, NPJ Digital Medicine, JBI, Artificial Intelligence in Medicine

### Phase 2: Search Query Construction
Use combinations of these keyword groups:

**Group A (Architecture):**
- "mixture of experts", "MoE", "sparse experts", "conditional computation", "expert networks", "multi-expert", "Switch Transformer", "Mixtral", "top-k routing", "sparse activation"

**Group B (Healthcare Data):**
- "healthcare", "medical", "clinical", "patient", "EHR", "electronic health records", "claims data", "medical codes", "ICD codes", "diagnosis codes", "procedure codes", "patient history", "medical sequences"

**Group C (Models):**
- "transformer", "BERT", "attention", "self-attention", "temporal model", "sequential model", "RNN", "LSTM", "deep learning"

**Group D (Tasks):**
- "patient stratification", "disease prediction", "subpopulation", "patient trajectories", "disease progression", "readmission", "risk prediction", "clinical outcomes"

**Example Queries:**
- "mixture of experts transformer healthcare claims"
- "MoE architecture electronic health records sequential"
- "sparse experts patient stratification temporal"
- "top-k routing clinical sequences prediction"
- "Switch Transformer medical diagnosis"
- "expert specialization patient subpopulation"

### Phase 3: Paper Screening Criteria

**INCLUDE papers that:**
- ✓ Use MoE or multiple expert architectures
- ✓ Apply to healthcare/medical sequential or temporal data
- ✓ Use transformers, RNNs, or deep learning architectures
- ✓ Discuss patient heterogeneity or subpopulation modeling
- ✓ Compare architectural variants or conduct ablation studies
- ✓ Published 2018-2025 (transformers era)

**EXCLUDE papers that:**
- ✗ Only static patient features (no temporal/sequential modeling)
- ✗ Image-only analysis without claims/EHR codes
- ✗ Traditional machine learning (no deep learning)
- ✗ Only theoretical MoE (no healthcare application)
- ✗ Simple ensembles (not true MoE with routing)

### Phase 4: Information Extraction

For each relevant paper found, extract:

1. **Bibliographic Info**: Authors, title, venue, year, DOI/arXiv ID
2. **Architecture Details**:
   - MoE variant (Switch, Mixtral, custom)
   - Number of experts
   - Routing strategy (top-k, soft, hierarchical, learned, hash-based)
   - Top-K value
   - Load balancing method
   - Base model (transformer, RNN, hybrid)
   - Expert placement (which layers)
3. **Healthcare Application**:
   - Data type (claims, EHR, codes, sequences)
   - Patient cohort size
   - Prediction task
   - Temporal span (days, visits, years)
   - Vocabulary size (if applicable)
4. **Experiments & Results**:
   - Did they compare MoE variants? (ablation study)
   - Performance metrics and baselines
   - Efficiency gains (compute, memory, inference time)
   - Expert specialization analysis (if any)
   - Interpretability findings
5. **Relevance Score**: Rate 1-5 stars for relevance to my research
   - ⭐⭐⭐⭐⭐: Direct comparison of MoE variants on claims/EHR transformers
   - ⭐⭐⭐⭐: MoE transformers on healthcare sequences (no variant comparison)
   - ⭐⭐⭐: MoE on healthcare (different model or data type)
   - ⭐⭐: Healthcare transformers (no MoE) or MoE (no healthcare)
   - ⭐: Tangentially related

## DELIVERABLE FORMAT

Provide your findings in this structure:

### Executive Summary
- Total papers found and reviewed
- Number of highly relevant papers (4-5 stars)
- Key finding: Does prior work exist that compares MoE variants on healthcare claims?
- Identified research gaps

### Section 1: Direct Applications (5-star papers)
For each paper: