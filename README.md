# GARD: Genomic Data based Drug Repurposing in Head and Neck Cancer with Large Language Model Validation

## Introduction to the GARD Pipeline

The **Genomic Alteration-based Repurposing for Drugs (GARD)** pipeline is a comprehensive computational framework designed to identify drug repurposing opportunities by analyzing the genomic landscape of cancer cohorts. Specifically applied to head and neck squamous cell carcinoma (HNSC) using The Cancer Genome Atlas (TCGA) data, GARD systematically discovers both direct and indirect therapeutic candidates that target genomically altered pathways.

### What GARD Does

GARD performs multi-dimensional genomic analysis to identify statistically significant genetic alterations and their druggable targets:

1. **Stratifies patients by clinically relevant biomarkers** (HPV status) to enable personalized therapeutic strategies
2. **Identifies significantly altered genes** through rigorous statistical testing of both copy number variations (CNV) and somatic mutations (SOM)
3. **Maps druggable targets** using two complementary approaches:
   - **Direct targeting**: Drugs that directly act on mutated genes
   - **Indirect targeting**: Drugs that act on proteins functionally connected to mutated genes through protein-protein interaction (PPI) networks
4. **Validates findings through literature mining** using GPU-accelerated NLP to confirm drug-gene relationships in published research
5. **Prioritizes candidates** based on multiple statistical metrics including binomial testing, hypergeometric enrichment, empirical permutation testing, and false discovery rate (FDR) correction

### How GARD Works

The pipeline employs a multi-stage analytical workflow:

#### Stage 1: Data Integration and Cohort Stratification
- Integrates TCGA genomic data (CNV and somatic mutations) with clinical annotations
- Stratifies the HNSC cohort into HPV-positive (n=72) and HPV-negative (n=448) subgroups based on validated HPV status from Nulton et al.
- Consolidates drug-gene interaction data from DrugBank (~21,714 interactions)
- Constructs high-confidence protein-protein interaction networks from STRING database (confidence ≥700)

#### Stage 2: Parallel Genomic Analysis
Two parallel analytical branches process CNV and SOM data independently:

**CNV Analysis Branch:**
- Calculates GISTIC scores to quantify amplification and deletion events
- Performs binomial significance testing for each gene
- Validates significance through 1,000 empirical permutations
- Applies Benjamini-Hochberg FDR correction

**SOM Analysis Branch:**
- Quantifies mutation frequency across the cohort
- Performs binomial significance testing accounting for gene length and mutation rate
- Validates significance through 10,000 empirical permutations
- Applies frequency cutoffs and FDR correction

#### Stage 3: Drug Target Discovery
- **Direct candidates**: Matches significantly altered genes directly to DrugBank interactions
- **Indirect candidates**: Expands target space through PPI networks to identify drugs acting on functionally connected proteins
- Performs hypergeometric enrichment testing to assess statistical over-representation of drug targets
- Calculates empirical FDR through permutation testing (10,000 iterations)

#### Stage 4: Literature Validation (Optional)
- Extracts PubMed IDs (PMIDs) for disease-gene relationships from the biomedical literature
- Uses GPU-accelerated Gemma 2B language model to analyze full-text articles and identify genes implicated in head and neck cancer
- Provides independent validation that genomically identified genes are also recognized in published research
- **Applied after statistical drug-gene filtering:** Drugs passing statistical significance are retained only if they target literature-validated genes (direct) or connect to literature-validated risk genes via PPI (indirect)
- **Literature filtering occurs during File 05 (Final result creation)** where drugs are filtered based on literature-validated gene targets

#### Stage 5: Results Integration and Visualization
- Consolidates findings from CNV and SOM analyses
- Creates comprehensive tables with mutation types, statistical metrics, and DrugBank annotations
- Generates interactive Sankey diagrams visualizing drug-gene-target relationships
- Produces network visualizations of validated drug repurposing candidates

### Overall Goal

The GARD pipeline aims to accelerate the drug repurposing process for cancer treatment by:

- **Leveraging existing genomic data** from large-scale projects like TCGA to identify patient subgroups with distinct genomic profiles
- **Discovering precision medicine opportunities** through biomarker-stratified analysis (HPV+ vs HPV-)
- **Expanding the druggable target space** beyond direct gene products to include functionally connected pathways
- **Providing evidence-based prioritization** through multi-level statistical validation and literature confirmation
- **Generating actionable candidate lists** for experimental validation and potential clinical translation

By integrating genomic alterations, protein networks, drug databases, and published literature, GARD provides a systematic approach to identify repurposing opportunities that may have been overlooked by traditional single-gene or single-drug analyses. The stratification by HPV status is particularly relevant for HNSC, as HPV-positive and HPV-negative tumors have distinct molecular profiles, treatment responses, and clinical outcomes, making personalized therapeutic strategies essential.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Directory Structure](#project-directory-structure)
3. [Required Input Data Sources](#required-input-data-sources)
4. [Output File Descriptions](#output-file-descriptions)
5. [Getting Started](#getting-started)
   - [System Requirements](#system-requirements)
   - [Installation](#installation)
   - [Quick Start](#quick-start)
   - [Expected Runtime](#expected-runtime)
   - [Troubleshooting](#troubleshooting)
6. [File-by-File Breakdown](#file-by-file-breakdown)
   - [00 Data viewing.ipynb](#00-data-viewingipynb)
   - [01 determine HPV status.ipynb](#01-determine-hpv-statusipynb)
   - [02 CNV identify mutation gene.ipynb](#02-cnv-identify-mutation-geneipynb)
   - [02.2 CNV key mutation identification.ipynb](#022-cnv-key-mutation-identificationipynb)
   - [02.50 CNV drug repurposing candidates.ipynb](#0250-cnv-drug-repurposing-candidates-copyipynb)
   - [03 SOM identify key mutation gene.ipynb](#03-som-identify-key-mutation-geneipynb)
   - [03.5 SOM drug repurpose.ipynb](#035-som-drug-repurposeipynb)
   - [03.75 result viewing.ipynb](#0375-result-viewingipynb)
   - [04 SOM and CNV results comparison.ipynb](#04-som-and-cnv-results-comparisonipynb)
   - [05 Final result creation.ipynb](#05-final-result-creationipynb)
   - [06 Graph direct results.ipynb](#06-graph-direct-resultsipynb)
   - [07_sankey_diagram_builder.ipynb](#07_sankey_diagram_builderipynb)
7. [Key Databases and Resources](#key-databases-and-resources)
8. [Output Summary](#output-summary)
9. [Validation Pipeline (Literature Mining)](#validation-pipeline-literature-mining)
   - [00 extract_pmids.bash](#00-extract_pmidsbash)
   - [01 extract based on pmid.ipynb](#01-extract-based-on-pmidipynb)
   - [02 GPU Extraction Scripts](#02-gpu-extraction-scripts)
   - [03 data viewing.ipynb](#03-data-viewingipynb)
   - [Validation Pipeline Workflow Summary](#validation-pipeline-workflow-summary)
   - [Output Integration](#validation-pipeline-output-integration)
10. [Usage Notes](#usage-notes)
11. [Data Availability and Reproducibility](#data-availability-and-reproducibility)
12. [Repository Information](#repository-information)
13. [Contact and Support](#contact-and-support)
14. [Key References](#key-references)

---


## Recent Update (April 2026)

**File 02 CNV identify mutation gene.ipynb was updated:** 

The default value for cnv.append in both deletion and amplification event handling was updated to append a value of 2 (representing the normal diploid state) when encountering missing or undefined CNV values. This improves pipeline robustness and ensures stable, consistent execution across all samples.

File counting in HPV-negative samples was refined to use only unique files, eliminating redundant entries and improving the precision of GISTIC score and frequency distributions. Significance cutoffs were re-evaluated and adjusted to optimally fit the revised distributions while maintaining continuity with the original analytical framework.

These refinements produced marginal shifts in intermediate p-values and gene counts near significance boundaries, with minor adjustments to background drug candidates. Core findings, top-ranked candidates, and overall biological conclusions remained fully consistent.

**File 03 SOM identify key mutation gene.ipynb was updated:**

Normalization by gene length now applies a fallback value of 2021 bp (the approximate mean protein-coding gene length) for genes absent from the reference annotation, improving numerical stability and consistency across samples.

The empirical p-value calculation was updated to use the standard pseudocount correction: (count of simulations ≥ observed + 1) / (number of simulations + 1). This eliminates zero-valued p-values and aligns with established best practices for permutation-based inference.

The parse_gtf function was refined to compute CDS gene lengths without double-counting overlapping exons, ensuring each genomic interval is counted once per gene. This improves biological accuracy and consistency with GENCODE annotation standards. HPV-negative case handling was similarly updated to use only unique cases, improving mutation frequency distributions. Cutoffs were re-evaluated and adjusted to best reflect the refined distributions while remaining in similar regions as the original analysis.

These improvements produced minor shifts in borderline significant genes and background drug candidates, with no impact on core findings or top candidates.

**File 2.5 and 3.5**

Added sort in identifying repurposing candidates for reproducibility consistency of the empirical p value calculations. Also filtered null gene values from all druggable genes. These were cosmetic and led to no changes in final values or drugs.  
   - Removed filtering to only keep first drug-gene connection in indirect aggregation function, this was cosmetic to allow more understanding of what genes were connected to drugs. Increased some drugs support in later stages but did not affect final drugs. This was a cosmetic change as well.

**File 05**

Ensure gene.strip() is applied when merging literature validation results to prevent mismatches due to whitespace. This was a cosmetic change that did not affect any values or drug candidates, but improved consistency and accuracy of literature validation integration.

**File 07**
Added a sort to the final drug list to ensure consistent ordering of genes in the Sankey diagrams. This was a cosmetic change that did not affect any values or drug candidates, but improved reproducibility and consistency of visual outputs. Only affected fostamatinib as it had many gene targets, now the top 15 genes shown are consistent across runs. This was a cosmetic change that did not affect any values or drug candidates, but improved reproducibility and consistency of visual outputs.


**Collective Impact:**

These updates collectively improved pipeline efficiency and numerical stability. Gene-level shifts were marginal, primarily affecting borderline cases near significance cutoffs including minor changes to background genes such as PDE3A and COL1A2, which were edge cases with no connection to final drug candidates. Total gene and drug counts saw small adjustments limited to background candidates. Core findings, top-ranked drug candidates, and all biological conclusions remained fully consistent with the original analysis, demonstrating the robustness of the pipeline to these technical refinements.

Updated final results are available in Results/Final Results/. For reference to the original pre-update code and results that aligns with the paper identically, see the paper-legacy branch. The paper-legacy branch preserves the complete original analysis for full reproducibility and direct comparison with the published results. These updates represent standard pipeline refinements aimed at improving analytical robustness and reproducibility, and do not constitute changes to the scientific conclusions, core findings, or interpretations reported in the published manuscript. The main branch reflects all April 2026 improvements.

---

## Overview

This pipeline performs a comprehensive genomic analysis of the TCGA Head and Neck Squamous Cell Carcinoma (HNSC) cohort to identify drug repurposing candidates stratified by HPV status. The workflow integrates copy number variation (CNV) data, somatic mutation (SOM) data, protein-protein interaction (PPI) networks, and DrugBank annotations to discover both **direct** (drug directly targets mutated genes) and **indirect** (drug targets genes connected via PPI to mutated genes) therapeutic candidates. 

The pipeline consists of 8 major analysis notebooks (00-07) plus a validation pipeline that sequentially process genomic data, identify significantly mutated genes in HPV-positive and HPV-negative cohorts, discover drug candidates through rigorous statistical enrichment testing and literature validation, and visualize the results. Key statistical methods include Binomial significance testing, hypergeometric testing for drug-gene enrichment, empirical permutation testing for significance validation, and FDR correction using Benjamini-Hochberg methodology. Results are validated against PubMed literature using GPU-accelerated natural language processing (NLP) with the Gemma 2B model to confirm drug-gene interactions from published research.

The final outputs include curated tables of drug repurposing candidates with DrugBank-verified mechanisms of action, literature validation from PubMed mining, mutation types (amplification, deletion, somatic), statistical significance metrics (Binomial, hypergeometric and empirical FDR), and interactive Sankey diagrams and network visualizations showing drug-gene-target relationships for both HPV+ and HPV- patient cohorts.

---

## Pipeline Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA ACQUISITION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ • TCGA HNSC: CNV (gene-level) + SOM (MAF format)                            │
│ • Nulton et al.: HPV status validation                                      │
│ • DrugBank: Drug-gene interactions (XML)                                    │
│ • STRING: Protein-protein interactions (confidence ≥700)                    │
│ • Gencode: Gene annotations and chromosome lengths                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    00 DATA VIEWING & EXPLORATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Parse DrugBank XML (~21,714 drug-gene interactions)                       │
│ • Load and explore PPI network structure                                    │
│ • Visualize CNV and SOM data distributions                                  │
│ • Clinical data exploration (age, gender, diagnosis)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    01 HPV STATUS STRATIFICATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Load Nulton HPV+ cases (n=72)                                             │
│ • Identify HPV- cases (n=448)                                               │
│ • Filter somatic mutations to Nulton cohort                                 │
│ Output: HPV+ patients.csv, HPV- patients.csv                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────┴───────────────────────────┐
        ↓                                                       ↓
┌──────────────────────────┐                         ┌──────────────────────────┐
│   CNV ANALYSIS BRANCH    │                         │   SOM ANALYSIS BRANCH    │
├──────────────────────────┤                         ├──────────────────────────┤
│ 02 CNV Mutation Genes    │                         │ 03 SOM Mutation Genes    │
│ • GISTIC score calc      │                         │ • Mutation frequency     │
│ • Binomial testing       │                         │ • Binomial testing       │
│ • 1,000 permutations     │                         │ • 10,000 permutations    │
│ • FDR correction (BH)    │                         │ • FDR correction (BH)    │
│ Output: Significant CNV  │                         │ • Frequency cutoffs      │
│   genes (HPV+/HPV-)      │                         │ • Score distributions    │
├──────────────────────────┤                         │ Output: Significant SOM  │
│ 02.2 CNV Gene Filter     │                         │   genes (HPV+/HPV-)      │
│ • Distribution analysis  │                         ├──────────────────────────┤
│ • Apply GISTIC cutoffs   │                         │ 03.5 SOM Drug Repurpose  │
│ • Apply frequency cutoff │                         │ • Hypergeometric test    │
│ Output: Top CNV genes    │                         │ • 100K permutations      │
├──────────────────────────┤                         │ • Direct candidates      │
│ 02.50 CNV Drug Repurpose │                         │ • Indirect (PPI) cand.   │
│ • Hypergeometric test    │                         │ Output: SOM drug cand.   │
│ • 100K permutations      │                         │   (HPV+/HPV-)            │
│ • Direct candidates      │                         └──────────────────────────┘
│ • Indirect (PPI) cand.   │                                     │
│ Output: CNV drug cand.   │                                     │
│   (HPV+/HPV-)            │                                     │
└──────────────────────────┘                                     │
               │                                                 │
               └───────────────────┬─────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    04 CNV AND SOM COMPARISON                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Calculate gene overlap (CNV ∩ SOM)                                        │
│ • Calculate drug overlap across mutation types                              │
│ • Identify converging evidence (amp + del + somatic)                        │
│ • Preliminary aggregation                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              05 FINAL RESULT CREATION & AGGREGATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Merge CNV + SOM gene results                                              │
│ • Aggregate direct drug candidates (CNV + SOM)                              │
│ • Aggregate indirect drug candidates (CNV + SOM)                            │
│ • Integrate literature validation (PMIDs, article counts)                   │
│ • Take minimum FDR across sources                                           │
│ Output: Final gene results (HPV+/HPV-)                                      │
│ Output: Final direct results (HPV+/HPV-)                                    │
│ Output: Final indirect results (HPV+/HPV-)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────┴───────────────────────────┐
        ↓                                                       ↓
┌──────────────────────────┐                         ┌──────────────────────────┐
│   06 DIRECT DRUG VIZ     │                         │   07 INDIRECT DRUG VIZ   │
├──────────────────────────┤                         ├──────────────────────────┤
│ • Top direct drugs       │                         │ • Top indirect drugs     │
│ • DrugBank action verify │                         │ • PPI pathway validation │
│ • Per-gene ACTION lookup │                         │ • Literature validation  │
│ • Mutation type mapping  │                         │ • Gene family grouping   │
│ • Bipartite network viz  │                         │ • Sankey diagrams:       │
│ Output: Top   Direct     │                         │   - Combined (all drugs) │
│   DrugBank Verified CSV  │                         │   - Individual (per drug)│
│   (HPV+/HPV-)            │                         │ Output: Top   PPI        │
│                          │                         │   Validated CSV          │
│                          │                         │   (HPV+/HPV-)            │
└──────────────────────────┘                         └──────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                         FINAL DELIVERABLES                                 │
├────────────────────────────────────────────────────────────────────────────┤
│ ✓ Significant Genes: HPV+/HPV- gene results (CNV + SOM)                    │
│ ✓ Direct Drugs: Top drugs targeting mutated genes (DrugBank verified)      │
│ ✓ Indirect Drugs: Top drugs targeting PPI-connected genes (validated)      │
│ ✓ Visualizations: Network graphs, Sankey diagrams (interactive HTML)       │
│ ✓ Literature Support: PMIDs, article counts, validation status             │
└────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│              VALIDATION PIPELINE (LITERATURE)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 00: Extract PMIDs from PubMed ("head and neck cancer")                      │
│ 01: Download abstracts and full-text (100K+ articles)                       │
│ 02: GPU-accelerated NLP extraction (Gemma 2B model, 24-72 hours)            │
│ 03: Clean and validate Genes    (remove unknowns, cross-ref DBs)            │
│ → Integration: Merge into File 05 for literature-validated columns          │
│`       (only carry forward literature-validated genes )                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

<!-- ---

## Statistical Methodology and Justification

### Overview of Statistical Framework

This pipeline employs a rigorous dual-validation statistical framework combining parametric tests with non-parametric empirical validation. All analyses control for multiple testing using Benjamini-Hochberg FDR correction.

### 1. Gene-Level Significance Testing

#### CNV Analysis (Copy Number Variation)

**Primary Test: Binomial Test (Right-tailed)**
- **Rationale**: Models high-level CNV events as Bernoulli trials (present/absent in each sample)
- **Null Hypothesis**: Gene experiences CNV events at the genome-wide background rate
- **Test Statistic**: X ~ Binomial(n, p_null), where:
  - n = number of samples in cohort
  - p_null = cohort-specific background CNV rate
  - X = number of samples with high-level CNV in focal gene
- **Threshold**: CNV > 4 (amplification) or CNV < 1 (deletion)

**Empirical Validation: Bootstrap Permutation Test (1,000 iterations)**
- **Rationale**: Validates binomial assumptions using cohort's actual CNV distribution
- **Method**: Bootstrap sample n values from pooled CNV data, count threshold exceedances
- **Advantage**: Accounts for non-normal CNV distributions and outliers
- **Note**: Does NOT shuffle HPV labels; tests against genomic instability baseline

**Key Assumptions**:
- CNV events are independent across samples (reasonable for somatic alterations)
- Background rate adequately captures genome-wide instability
- High-level thresholds (>4, <1) define biologically relevant alterations

#### SOM Analysis (Somatic Mutations)

**Primary Test: Length-Normalized Binomial Test**
- **Rationale**: Accounts for varying gene lengths (longer genes have more mutation opportunities)
- **Null Hypothesis**: Mutations distributed proportionally to CDS length
- **Probability**: p_g = L_g / L_total (gene length / total coding genome length)
- **Test Statistic**: K_g ~ Binomial(N, p_g), where:
  - N = total mutations in cohort
  - K_g = observed mutations in gene g

**Empirical Validation: Multinomial Monte Carlo (10,000 iterations)**
- **Rationale**: Accounts for dependency between genes (mutations must sum to N)
- **Method**: Draw mutation counts for all genes simultaneously from Multinomial(N, **p**)
- **Advantage**: Properly models constraint that total mutations is fixed
- **Superior to**: Independent permutations which violate this constraint

**Key Assumptions**:
- Mutations occur randomly conditional on gene length under null
- CDS length is appropriate measure (excludes non-coding regions)
- Non-synonymous mutations are functionally relevant

### 2. Drug Enrichment Testing

#### Hypergeometric Test (Over-representation)

**Primary Test: Right-tailed Hypergeometric**
- **Rationale**: Classic urn model for testing over-representation
- **Null Hypothesis**: Drug targets randomly distributed among druggable genes
- **Parameters**:
  - M = total druggable genes in universe
  - K = significant genes (from CNV/SOM analysis)
  - n = drug's target gene count
  - x = overlap (drug targets ∩ significant genes)
- **Test**: P(X ≥ x) where X ~ Hypergeometric(M, K, n)

**Empirical Validation: Permutation Test (100,000 iterations)**
- **Rationale**: Validates that enrichment isn't due to network topology or annotation biases
- **Method**: Randomly select n genes from druggable universe, recalculate enrichment
- **Null Distribution**: Empirical distribution of overlaps under random gene selection
- **Advantage**: Accounts for non-uniform gene properties (degree, pleiotropy, etc.)

**Key Assumptions**:
- DrugBank represents unbiased druggable universe (reasonable approximation)
- Drug-gene interactions are independent (conservative assumption)
- Significant genes are representative of altered biological processes

### 3. Multiple Testing Correction

**Method: Benjamini-Hochberg FDR Procedure**
- **Rationale**: Controls false discovery rate while maintaining power
- **Threshold**: FDR < 0.05 (5% expected false discoveries)
- **Applied to**:
  - CNV genes: ~20,000 tests × 4 (amp/del × HPV+/-)
  - SOM genes: ~20,000 tests × 2 (HPV+/-)
  - Drugs: Hundreds of drugs per analysis
- **Total**: >80,000 tests across full pipeline

**Why FDR instead of FWER?**
- FWER (e.g., Bonferroni) is too conservative for exploratory genomics
- FDR provides better balance between discovery and false positives
- Standard in genomics literature (widely accepted)

### 4. Statistical Power Considerations

**Sample Sizes**:
- HPV+ cohort: n=72 (limited power for rare events)
- HPV- cohort: n=448 (adequate power for moderate effect sizes)

**Implications**:
- HPV+ analyses may miss genes altered in <10% of patients
- Conservative thresholds compensate for multiple testing
- Dual validation (parametric + empirical) increases confidence

**Sensitivity Analysis**:
- Frequency thresholds vary by cohort size (higher for HPV+)
- GISTIC score thresholds calibrated to distributions
- Empirical p-values provide robustness check

### 5. Reproducibility

**Random Seeds**: Set to 42 for all stochastic procedures
**Software Versions**: Documented in requirements.txt
**Thresholds**: All cutoffs explicitly documented
**Data Availability**: TCGA public data with case IDs provided -->

---
<!-- 
## Key Statistical Methodology

### 1. Gene-Level Significance Testing

**Copy Number Variation (CNV):**
- **Test:** Binomial test (right-tailed)
- **Null Hypothesis:** Gene alteration frequency = background rate
- **Alternative:** Gene alteration frequency > background rate
- **Background:** Calculated per cohort from all genes
- **Threshold:** CNV > 4 (amplification) or CNV < 1 (deletion)
- **Validation:** 1,000 permutations (shuffle HPV labels)
- **Correction:** Benjamini-Hochberg FDR (q < 0.05)
- **GISTIC Score:** Σ(log₂(CNV)) × frequency (normalized)

**Somatic Mutations (SOM):**
- **Test:** Binomial test (right-tailed)
- **Null Hypothesis:** Mutation frequency = background mutation rate
- **Alternative:** Mutation frequency > background rate
- **Background:** Total mutations / (genes × patients)
- **Validation:** 10,000 permutations
- **Correction:** Benjamini-Hochberg FDR (q < 0.05)

### 2. Drug Enrichment Testing

**Hypergeometric Test:**
- **Question:** Does drug target more significant genes than expected by chance?
- **Parameters:**
  - M = Total genes in DrugBank
  - K = Significant genes from CNV/SOM analysis
  - n = Genes targeted by this drug
  - x = Significant genes targeted by this drug
- **Test:** P(X ≥ x) where X ~ Hypergeom(M, K, n)
- **Direction:** Right-tailed (over-representation)

**Empirical Permutation Validation:**
- **Iterations:** 100,000 permutations
- **Procedure:** 
  1. Randomly shuffle gene labels (preserve network structure)
  2. Recalculate hypergeometric enrichment for each drug
  3. Count permutations with enrichment ≥ observed
  4. Empirical p-value = count / 100,000
- **Rationale:** Validates that enrichment is not due to network topology or database biases

**Multiple Testing Correction:**
- **Method:** Benjamini-Hochberg procedure
- **FDR Threshold:** 0.05
- **Applied to:** Both hypergeometric p-values and empirical p-values
- **Final Filter:** hypergeom_fdr < 0.05 **AND** empirical_fdr < 0.05

### 3. Protein-Protein Interaction (PPI) Analysis

**Network Construction:**
- **Database:** STRING v12.0 (Homo sapiens, organism 9606)
- **Confidence Threshold:** ≥700 (high confidence)
- **Edge Definition:** Experimental and database evidence combined
- **Directionality:** Bidirectional (symmetric interactions)

**Indirect Drug Candidate Identification:**
- **Definition:** Drug targets gene A, gene A interacts with risk gene B via PPI
- **Path Length:** 1 hop (direct PPI connection)
- **Validation:** Both drug→target and target→risk connections must be validated
- **Scoring:** Percentage of risk genes reachable via PPI

### 4. Literature Validation (Optional)

**NLP Extraction:**
- **Model:** Gemma 2B (Google)
- **Input:** PubMed abstracts and full-text articles
- **Task:** Named entity recognition + relation extraction
- **Validation:** Requires ≥2 articles mentioning same drug-gene pair
- **Quality Control:** Cross-reference with DrugBank and HGNC

**Integration:**
- **Primary Evidence:** Genomic data + statistical testing (identifies significant genes and enriched drugs)
- **Secondary Filtering:** Literature validation applied on statistical significant drugs found from drug-gene significance testing
- **Drug Selection:** Statistically significant drugs are retained only if they target literature-validated genes
- **Note:** Absence of literature support filters out the drug candidate (conservative approach prioritizing translational relevance)

### 5. Statistical Significance Criteria

**Gene-Level (CNV or SOM):**
```
Significant if ALL conditions met:
• q_value < 0.05 (parametric FDR)
• empirical_q_value < 0.05 (permutation-based FDR)
• Frequency or GISTIC threshold (category-specific)
```

**Drug-Level (Direct or Indirect):**
```
Significant if ALL conditions met:
• drug_hypergeom_fdr < 0.05
• drug_empirical_fdr < 0.05
• Targets ≥1 significant gene (direct) or connected via PPI (indirect)
```

**Conservative Approach:**
- Dual testing (parametric + empirical) reduces false positives
- High permutation counts (10K-100K) ensure robust p-values
- FDR correction accounts for multiple hypothesis testing

### 6. Effect Size Metrics

**GISTIC Score (CNV):**
- Combines magnitude (log₂ copy number) and frequency
- Higher score = stronger/more frequent alteration
- Used for prioritization (not just significance)

**Frequency Percentage:**
- Proportion of cohort with alteration
- Clinical relevance: Higher frequency = more patients benefit

**Percentage of Targets Hit:**
- Proportion of significant genes targeted by drug
- Drug specificity: Higher percentage = more targeted therapy -->

---

## Project Directory Structure

```
Genetic based drug repurposing Nulton cohort/
│
├── 00 Data viewing.ipynb
├── 01 determine HPV status.ipynb
├── 02 CNV identify mutation gene.ipynb
├── 02.2 CNV key mutation identification.ipynb
├── 02.50 CNV drug repurposing candidates.ipynb
├── 03 SOM identify key mutation gene.ipynb
├── 03.5 SOM drug repurpose.ipynb
├── 04 SOM and CNV results comparison.ipynb
├── 05 Final result creation.ipynb
├── 06 Graph direct results.ipynb
├── 07_sankey_diagram_builder.ipynb
├── README.md
│
├── Data/                                    # INPUT DATA (user must provide)
│   ├── gdc-client                          # GDC data transfer tool (optional)
│   │
│   ├── DGIDB/                              # DrugBank database
│   │   ├── drug_bank.xml                   # REQUIRED: DrugBank XML file (download from DrugBank)
│   │   └── Drug_bank_drug_gene_interactions.csv  # Optional processed version
│   │
│   ├── Protein-protein interaction data/   # STRING PPI database
│   │   ├── 9606.protein.links.v12.0.txt              # REQUIRED: PPI network links
│   │   ├── 9606.protein.links.full.v12.0.txt         # Optional: Full PPI data
│   │   ├── 9606.protein.links.detailed.v12.0.txt     # Optional: Detailed PPI data
│   │   ├── 9606.protein.info.v12.0.txt               # REQUIRED: Protein information
│   │   ├── 9606.protein.aliases.v12.0.txt            # REQUIRED: Protein name aliases
│   │   └── BIOGRID-ORGANISM-Homo_sapiens-4.4.246.psi25.xml  # Optional: Alternative PPI source
│   │
│   ├── Supplementary data Nulton/          # Nulton et al. supplementary data
│   │   ├── 1. HPV Genomes for alignment.csv
│   │   ├── 2. HNC HPV Types.csv            # REQUIRED: HPV+ TCGA case IDs
│   │   ├── 3. HNC HPV State Categories.csv
│   │   ├── 4. HPV-human jxn sites.csv
│   │   ├── 5. HNSC NR4A2,RAD51L1,TRPC4AP.csv
│   │   ├── TCGA CASE ID.csv                # REQUIRED: All cohort case IDs,  
|   |   |                                   extracted from supplementary table 5 listing all TCGA case IDs
│   │   └── [10-15. Additional supplementary files] # OPTIONAL
│   │
│   └── TCGA/                                # TCGA genomic data
│       ├── Clinical data/                   # Clinical information (optional). If available can extract clinical data from here
│       │
│       ├── Gene level CNV/                  # REQUIRED: Copy number variation data
│       │   ├── gdc_sample_sheet.2025-11-03.tsv      # Sample manifest
│       │   └── [Patient folders]/           # Individual patient CNV files, limited to Nulton et al TCGA case IDs
│       │       └── *.tsv                    # gene-level CNV scores for calculating GISTIC scores
│       │
│       └── SOM/                             # REQUIRED: Somatic mutation data
│           ├── cohortMAF.2025-07-29.maf     # Original MAF file from TCGA
│           ├── nulton_somatic_mutation_cohort.csv   # Filtered to Nulton cohort (generated)
│           └── gencode.v48.annotation.gtf   # Gene annotations, extracted from GENCODE to determine gene CDS lengths
│
├── Results/                                 # OUTPUT DATA (pipeline generates)
│   │
│   ├── HPV results/                         # HPV stratification results
│   │   ├── HPV positive patients.csv        # Generated by 01: List of HPV+ case IDs
│   │   └── HPV negative patients.csv        # Generated by 01: List of HPV- case IDs
│   │
│   ├── CNV results/                         # Copy number variation results (intermediate)
│   │   ├── HPV positive CNV top genes.csv   # Generated by 02.2: Significant HPV+ CNV genes
│   │   ├── HPV negative CNV top genes.csv   # Generated by 02.2: Significant HPV- CNV genes
│   │   ├── HPV Positive Top Direct Drug Candidates Aggregated.csv    # Generated by 02.50
│   │   ├── HPV Negative Top Direct Drug Candidates Aggregated.csv    # Generated by 02.50
│   │   ├── HPV Positive Top Indirect Drug Candidates Aggregated.csv  # Generated by 02.50
│   │   ├── HPV Negative Top Indirect Drug Candidates Aggregated.csv  # Generated by 02.50
│   │   ├── hpv_pos_amp_top_drugBank_drug_candidates.csv   # Individual CNV amp results
│   │   ├── hpv_pos_del_top_drugBank_drug_candidates.csv   # Individual CNV del results
│   │   ├── hpv_neg_amp_top_drugBank_drug_candidates.csv   # Individual CNV amp results
│   │   └── hpv_neg_del_top_drugBank_drug_candidates.csv   # Individual CNV del results
│   │
│   ├── SOM results/                         # Somatic mutation results (intermediate)
│   │   ├── HPV positive top genes.csv       # Generated by 03: Significant HPV+ SOM genes
│   │   ├── HPV negative top genes.csv       # Generated by 03: Significant HPV- SOM genes
│   │   ├── hpv_positive_som_top_direct_drug_candidates_agg.csv    # Generated by 03.5
│   │   ├── hpv_negative_som_top_direct_drug_candidates_agg.csv    # Generated by 03.5
│   │   ├── hpv_positive_som_top_indirect_drug_candidates_agg.csv  # Generated by 03.5
│   │   └── hpv_negative_som_top_indirect_drug_candidates_agg.csv  # Generated by 03.5
│   │
│   ├── HPV positive gene results.csv        # Generated by 05: Intermediate unified HPV+ genes (CNV+SOM)
│   ├── HPV negative gene results.csv        # Generated by 05: Intermediate unified HPV- genes (CNV+SOM)
│   │
│   ├── HPV Positive direct results.csv      # Generated by 05: Intermediate HPV+ direct drug candidates
│   ├── HPV Negative direct results.csv      # Generated by 05: Intermediate HPV- direct drug candidates
│   ├── HPV Positive indirect results.csv    # Generated by 05: Intermediate HPV+ indirect drug candidates
│   ├── HPV Negative indirect results.csv    # Generated by 05: Intermediate HPV- indirect drug candidates
│   │
│   └── Final Results/                       # FINAL OUTPUTS
│       ├── HPV Positive validated genes.csv              # Generated by 05: Final validated HPV+ genes
│       ├── HPV Negative validated genes.csv              # Generated by 05: Final validated HPV- genes
│       ├── HPV Positive Validated direct candidates.csv  # Generated by 06: Top HPV+ direct drugs (DrugBank verified)
│       ├── HPV Negative Validated direct candidates.csv  # Generated by 06: Top HPV- direct drugs (DrugBank verified)
│       ├── HPV Positive Validated indirect candidates.csv # Generated by 07: Top HPV+ indirect drugs (PPI validated)
│       └── HPV Negative Validated indirect candidates.csv # Generated by 07: Top HPV- indirect drugs (PPI validated)
│
└── Validation pipeline/                     # Literature validation (optional but recommended)
    ├── 00 extract_pmids.bash                # PubMed ID extraction using NCBI E-utilities
    ├── 01 extract based on pmid.ipynb       # Download abstracts/full text from PubMed
    ├── 02 GPU_full_extract.sh               # GPU NLP extraction script
    ├── 02 GPU_full_extract.py               # Python GPU NLP extraction code
    ├── 03 data viewing.ipynb                # Visualize and clean extracted drug-gene pairs
    ├── pmids.txt                            # List of PubMed IDs for head and neck cancer
    ├── Data/                                # Literature mining input data
    │   ├── DRUGBANK/                        # DrugBank reference for drug name matching
    │   └── Protein-protein interaction data/  # PPI data for validation
    └── Results/                             # Literature mining results
        ├── cleaned_extracted_targets_all_pub_after_2000_GPU_2b_gemma.csv  # Drug-gene from PubMed (Gemma 2B model)
        └── cleaned_extracted_combined_targets_all_pub_after_2000_GPU_2b_gemma.csv  # Combined drug-gene targets
```

### Required Input Data Sources

**1. DrugBank XML Database** (`Data/DGIDB/drug_bank.xml`)
- **Source**: https://go.drugbank.com/releases/latest
- **Access**: Requires free academic account
- **Format**: XML file (~1.5GB)
- **Content**: ~22,820 drug-gene interactions with actions and functions

**2. STRING PPI Database** (`Data/Protein-protein interaction data/`)
- **Source**: https://string-db.org/cgi/download
- **Version**: v12.0 (Human, Homo sapiens, organism 9606)
- **Required Files**:
  - `9606.protein.links.v12.0.txt` (~800MB)
  - `9606.protein.info.v12.0.txt` (~20MB)
  - `9606.protein.aliases.v12.0.txt` (~100MB)

**3. TCGA HNSC Data** (`Data/TCGA/`)
- **Source**: https://portal.gdc.cancer.gov/
- **Project**: TCGA-HNSC (Head and Neck Squamous Cell Carcinoma)
- **Required Data Types**:
  - **CNV**: Gene-level copy number variation (GISTIC2 scores)
    - Download via GDC Data Portal: Data Category = "Copy Number Variation"
    - File Type = "Gene Level Copy Number" (individual patient TSV files)
  - **SOM**: Somatic mutation data (MAF format)
    - Download via GDC Data Portal: Data Category = "Simple Nucleotide Variation"
    - File Type = "Masked Somatic Mutation" (cohort-level MAF file)

**4. Chromosome lengths (Data/TCGA/SOM/gencode.v48.annotation.gtf):**
- **Source**: https://www.gencodegenes.org
- **Version**: Release 48 (GRCh38.p13)
- **Required data files**:
   - Comprehensive gene annotation (GTF format)
   - Comprehensive gene annotation on reference chromosomes
   - Provides CDS lengths of genes for gene-length-normalized mutation analysis 

**5. Nulton et al. Supplementary Data** (`Data/Supplementary data Nulton/`)
- **Source**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5392278/
- **Citation**: Nulton TJ, et al. (2017). Oncotarget. 8(11):17684-17699
- **Required Files**:
  - `2. HNC HPV Types.csv`: HPV+ patient identifiers
  - `TCGA CASE ID.csv`: Complete cohort case list

**6. Literature Validation Data** (`Validation pipeline/`)
- **Source**: PubMed database via NCBI E-utilities API
- **Query**: "head and neck cancer"
- **Processing**: Custom GPU-accelerated NLP extraction pipeline using Gemma 2B language model
- **Computational Requirements**: NVIDIA GPU with ≥8GB VRAM (A100/V100 recommended), ≥32GB system RAM
- **Output**: Drug-gene interaction pairs extracted from scientific literature with PMIDs
- **Purpose**: Secondary validation of DrugBank drug candidates against published clinical and preclinical research
- **Note**: This is an optional but highly recommended validation step that requires significant computational resources


### Output File Descriptions

**Key Final Outputs** (Output files in `Results/Final Results/`):

| File | Generated By | Description |
|------|--------------|-------------|
| `HPV Positive validated genes.csv` | File 05 | **FINAL** validated HPV+ genes with literature support (CNV + SOM) |
| `HPV Negative validated genes.csv` | File 05 | **FINAL** validated HPV- genes with literature support (CNV + SOM) |
| `HPV Positive Validated direct candidates.csv` | File 06 | **FINAL** HPV+ direct drugs with DrugBank action verification |
| `HPV Negative Validated direct candidates.csv` | File 06 | **FINAL** HPV- direct drugs with DrugBank action verification |
| `HPV Positive Validated indirect candidates.csv` | File 07 | **FINAL** HPV+ indirect drugs with PPI network validation |
| `HPV Negative Validated indirect candidates.csv` | File 07 | **FINAL** HPV- indirect drugs with PPI network validation |

**Intermediate Outputs** (used within pipeline, stored in `Results/`):

| File | Generated By | Description |
|------|--------------|-------------|
| `HPV positive gene results.csv` | File 05 | Intermediate unified HPV+ genes (CNV + SOM) |
| `HPV negative gene results.csv` | File 05 | Intermediate unified HPV- genes (CNV + SOM) |
| `HPV Positive direct results.csv` | File 05 | Intermediate HPV+ direct drug candidates (all) |
| `HPV Negative direct results.csv` | File 05 | Intermediate HPV- direct drug candidates (all) |
| `HPV Positive indirect results.csv` | File 05 | Intermediate HPV+ indirect drug candidates (all) |
| `HPV Negative indirect results.csv` | File 05 | Intermediate HPV- indirect drug candidates (all) |

**Pipeline Intermediate Outputs** (supporting files):

- `Results/HPV results/`: HPV stratification (Generated in File 01)
- `Results/CNV results/`: CNV-based gene and drug findings (Generated in Files 02, 02.2, 02.50)
- `Results/SOM results/`: SOM-based gene and drug findings (Generated in Files 03, 03.5)
- Individual mutation type files: Pre-aggregation drug candidates (amp, del, somatic separate)

---


## File-by-File Breakdown

### **00 Data viewing.ipynb**
**Purpose:** Initial data exploration and DrugBank database parsing.

**Key Functions:**
- `structure_drug_bank_data()`: Parses the DrugBank XML file to extract drug-gene interactions including drug names, gene targets, polypeptide information, action mechanisms (inhibitor, antagonist, agonist, etc.), and specific molecular functions.

**Inputs:**
- `Data/DGIDB/drug_bank.xml`: DrugBank XML database containing ~22,820 drug-gene interactions
- `Data/Protein-protein interaction data/9606.protein.links.v12.0.txt`: STRING PPI network links
- `Data/Protein-protein interaction data/9606.protein.info.v12.0.txt`: Protein information
- `Data/Protein-protein interaction data/9606.protein.aliases.v12.0.txt`: Protein name aliases
- `Data/TCGA/Gene level CNV/`: CNV data files for initial exploration, clinical data is taken from here. Limited to Nulton et. al cohort case IDs
- `Data/TCGA/SOM/cohortMAF.2025-07-29.maf`: Somatic mutation data for initial viewing

**Outputs:**
- DrugBank Structured DataFrame with columns: `drug`, `polypeptide`, `gene`, `gene_description`, `action`, `specific_function`

**Key Operations:**
- Loads DrugBank XML using ElementTree parser with namespace handling
    - Extracts drug targets and polypeptide information for viewing
    - Handles cases where polypeptide data is absent (direct gene targeting)
- Views PPI data structure and connectivity
- Explores CNV data format (GISTIC scores)
- Views somatic mutation data structure (MAF format)
- explores clinical data for distribution of:
    - age
    - gender
    - diagnosis

**Libraries:** pandas, numpy, xml.etree.ElementTree, matplotlib, plotly, scipy, statsmodels, networkx

---

### **01 determine HPV status.ipynb**
**Purpose:** Stratify TCGA HNSC patient cohort by HPV status using validated data from Nulton et al. (2017).

**Key Data Sources:**
- Nulton et al. paper (PMC5392278): Provides validated HPV+ cases (72 patients)
- `Data/Supplementary data Nulton/2. HNC HPV Types.csv`: HPV+ case identifiers
- `Data/Supplementary data Nulton/TCGA CASE ID.csv`: All cohort case identifiers
- `Data/TCGA/SOM/cohortMAF.2025-07-29.maf`: Somatic mutation data for the cohort

**Key Operations:**
- Loads paper-identified HPV+ case IDs from Nulton supplementary data
- Loads all TCGA case IDs from the cohort
- Converts TCGA barcodes to case ID format (first 3 segments: TCGA-XX-XXXX)
- Calculates HPV negative cases by set difference: `all_cases - HPV+_cases`
- Filters somatic mutation cohort (originally all HNC patients in TCGA) to Nulton cohort patients
- Exports stratified patient lists for downstream analysis

**Outputs:**
- `Results/HPV results/HPV positive patients.csv`: 72 HPV+ case IDs based on Nulton et al analysis
- `Results/HPV results/HPV negative patients.csv`: Remaining 448 HPV- case IDs based on Nulton et al analysis
- `Data/TCGA/SOM/nulton_somatic_mutation_cohort.csv`: Filtered somatic mutations for the Nulton cohort

**Statistics:**
- HPV+ cases: 72 patients (highly validated)
- Total cohort: 520 patients
- HPV- cases:  448 patients

**Libraries:** pandas, matplotlib, plotly

---

### **02 CNV identify mutation gene.ipynb**
**Purpose:** Identify significantly amplified and deleted genes in HPV+ and HPV- cohorts from copy number variation (CNV) data.

**Key Statistical Methods:**

1. **Binomial Test:** Tests if the CNV of gene amplification/deletion in a cohort exceeds background rates.
   - Null hypothesis: Gene alteration = background rate observed across all genes
   - Alternative hypothesis: Gene alteration > background rate (right-tailed test)
   - **Background rate calculation**: Cohort-specific rate computed as the proportion of all CNV observations (across all genes and samples) that exceed the threshold
     - For amplifications: p_null = (total CNV > 4 events) / (total CNV observations)
     - For deletions: p_null = (total CNV < 1 events) / (total CNV observations)
   - This accounts for genome-wide instability patterns specific to each HPV status group
   
2. **Multiple Testing Correction:** Uses Benjamini-Hochberg FDR correction to control false discovery rate across thousands of genes tested.

3. **Empirical Permutation Testing:** Validates statistical significance through bootstrap sampling from the pooled CNV distribution (1,000 iterations).
   - Null distribution: Randomly sample n CNV values (with replacement) from the entire cohort's CNV distribution across all genes
   - Null hypothesis: Observed gene alteration frequency could occur by chance given the cohort's overall genomic instability pattern
   - Alternative hypothesis: Observed gene alteration frequency exceeds what random sampling would produce
   - Method: For each gene, sample n values from permuted CNV pool, count how many exceed threshold (CNV>4 for amplifications, CNV<1 for deletions)
   - P-value: Proportion of permutations where simulated count ≥ observed count

4. **Calculates GISTIC type scores**: calculates GISTIC score as the amplification or deletion value summed across the cohort multiplied by the frequency of amplification/deletion across the cohort

**Key Operations:**
- Loads TCGA gene-level CNV data from individual patient files
- Stratifies samples by HPV status (HPV+ vs HPV-) based on Nulton et al analysis
- Calculates gene-level amplification/deletion frequencies per cohort
- Performs binomial testing for each gene  significant CNV (CNV>4, or CNV<1) to background
- Applies FDR correction (Benjamini-Hochberg) to control for multiple testing
- Runs empirical permutation tests (1,000 iterations) to validate significance
- Calculate GISTIC scores based on CNV values and alteration frequencies for downstream analysis
- Identifies top significantly altered genes (q-value < 0.05, empirical FDR < 0.05)
- Merges final dataframe with gene chromosome coverage for consideration

**Inputs:**
- `Data/TCGA/Gene level CNV/`: Directory containing individual patient CNV files (TSV format with GISTIC scores)
- `Results/HPV results/HPV positive patients.csv`: HPV+ case IDs
- `Results/HPV results/HPV negative patients.csv`: HPV- case IDs

**Outputs:**
- `Results/CNV results/HPV positive amplification genes.csv`: Significantly amplified genes in HPV+ cohort
- `Results/CNV results/HPV positive deletion genes.csv`: Significantly deleted genes in HPV+ cohort
- `Results/CNV results/HPV negative amplification genes.csv`: Significantly amplified genes in HPV- cohort
- `Results/CNV results/HPV negative deletion genes.csv`: Significantly deleted genes in HPV- cohort

**Key Columns in Output:**
- `gene_name`: Gene symbol
- `MUT_TYPE`: AMPLIFICATION (in amplification files) or DELETION (in deletion files)
- `Cohort_Frequency`: Proportion of patients with the specific alteration (amplification or deletion)
- `Normalized_Count`: Count of patients with the specific alteration
- `q_value`: FDR-corrected p-value (Benjamini-Hochberg)
- `empirical_q_value`: FDR-corrected empirical p-value from permutation testing
- `GISTIC`: GISTIC score for the gene
- `normalized_gistic_score`: Min-max normalized GISTIC score

**Note:** Each output file contains only one mutation type (either amplification or deletion), not both.
**Libraries:** pandas, scipy.stats (binomtest), statsmodels.stats.multitest (multipletests), numpy, matplotlib, plotly

---

### **02.2 CNV key mutation identification.ipynb**
**Purpose:** Exploratory analysis and threshold refinement for CNV gene results to select top candidate genes based on multiple criteria.

**Key Functions:**

**`plot_score_distribution(df)`**
- Visualizes distribution of key metrics across all significant CNV genes
- Plots: frequency_percentage, gistic_score, p-value, q-value, empirical_q_value, coverage_percentage
- Adds mean and 95th percentile reference lines
- Uses log-scale y-axis for better visibility of distributions

**`plot_score_distribution_with_cutoff(df, cutoff)`**
- Same as `plot_score_distribution()` but with custom cutoff lines overlaid
- Helps visualize how cutoff thresholds filter the gene set
- Cutoff dictionary contains threshold values for each metric

**`plot_gistic_distribution(df, title)`**
- Box plot showing GISTIC score distribution per gene using Plotly
- Interactive visualization for exploring individual gene scores
- Adds horizontal cutoff line for threshold reference

Note on thresholds: Cutoffs were determined from the distribution characteristics of GISTIC scores and gene frequency percentages. These thresholds were selected to balance sensitivity and specificity, and small changes may alter which genes are considered significant and therefore change downstream drug repurposing candidates. Thresholds were evaluated across a range of values, and the top candidates remained relatively robust to reasonable variation. Users reproducing or extending these results should document any threshold changes carefully, as even small adjustments can affect the final drug candidate list.

**Key Operations:**

1. **Load CNV Results:**
   - HPV+ amplification and deletion genes from File 02
   - HPV- amplification and deletion genes from File 02
   - DrugBank and PPI databases for druggability assessment

2. **Exploratory Visualization:**
   - Score distribution plots for all metrics
   - Scatter plots: GISTIC vs frequency_percentage, GISTIC vs q-value, GISTIC vs empirical_q_value

3. **Define Cutoff Thresholds:**
   Based on distribution characteristics of GISTIC scores and frequency percentage:
   
   **HPV+ Amplification:**
   - frequency_percentage ≥ 30%
   - gistic_score ≥ 0.296
   - q_value ≤ 0.05
   - empirical_q_value ≤ 0.05
   
   **HPV+ Deletion:**
   - frequency_percentage ≥ 12%
   - gistic_score ≥ 0.091
   - q_value ≤ 0.05
   - empirical_q_value ≤ 0.05
   
   **HPV- Amplification:**
   - frequency_percentage ≥ 35%
   - gistic_score ≥ 0.31
   - q_value ≤ 0.05
   - empirical_q_value ≤ 0.05
   
   **HPV- Deletion:**
   - frequency_percentage ≥ 22.5%
   - gistic_score ≥ 0.12
   - q_value ≤ 0.05
   - empirical_q_value ≤ 0.05

4. **Filter Top Genes:**
   - Apply all cutoff criteria simultaneously (AND logic)
   - Genes must pass all thresholds to be selected
   - Sort by GISTIC score and frequency_percentage for prioritization

5. **Visualization of Top Genes:**
   - Horizontal bar charts showing top genes by GISTIC score
   - Horizontal bar charts showing top genes by frequency_percentage
   - Cutoff lines overlaid for reference

6. **Export Top Gene Sets:**
   - Filtered gene lists for each category (HPV+/- × amplification/deletion)
   - Used as refined input for drug repurposing analysis in File 02.50

**Rationale for Cutoffs:**

- **Frequency_percentage**: Ensures mutations occur in sufficient proportion of patients (clinical relevance)
- **GISTIC_score**: Reflects magnitude of copy number alteration (biological impact)
- **Statistical significance**: Both parametric (q_value) and empirical validation required
- **HPV- vs HPV+**: segregated cohorts for analysis
- **Amplification vs Deletion**: Different thresholds due to different baseline rates

**Inputs:**
- `Results/CNV results/HPV positive CNV genes.csv`: All significant HPV+ CNV genes from File 02
- `Results/CNV results/HPV negative CNV genes.csv`: All significant HPV- CNV genes from File 02
- `Data/DGIDB/drug_bank.xml`: DrugBank database
- `Data/Protein-protein interaction data/9606.protein.links.v12.0.txt`: STRING PPI database

**Outputs:**
- `Results/CNV results/HPV positive amplification top genes.csv`: Top HPV+ amplification genes
- `Results/CNV results/HPV positive deletion top genes.csv`: Top HPV+ deletion genes
- `Results/CNV results/HPV negative amplification top genes.csv`: Top HPV- amplification genes
- `Results/CNV results/HPV negative deletion top genes.csv`: Top HPV- deletion genes

**Key Metrics Explained:**

- **frequency_percentage**: Percentage of patients in cohort with the CNV alteration
- **gistic_score**: GISTIC2.0 normalized copy number score (mean across patients with alteration)
- **coverage_percentage**: Percentage of genome coverage (used for normalization)
- **Amplification_sum**: Total log2(CNV) summed across all samples with amplification

**Use Case:**
This is a quality control and refinement notebook that:
- Ensures only high-confidence genes proceed to drug repurposing analysis
- Balances statistical significance with clinical relevance (frequency)
- Provides visualizations for understanding gene selection criteria
- Reduces false positives by requiring genes to pass multiple filters
- Creates cohort-specific filtered gene lists optimized for druggability analysis

**Relationship to Pipeline:**
- **Receives**: Statistically significant genes from File 02
- **Refines**: Applies additional frequency and GISTIC magnitude thresholds
- **Provides**: High-confidence gene lists to File 02.50 for drug candidate identification

**Libraries:** pandas, matplotlib, plotly, numpy, sklearn.preprocessing (MinMaxScaler), xml.etree.ElementTree

---

### **02.50 CNV drug repurposing candidates.ipynb**
**Purpose:** Identify direct and indirect drug repurposing candidates based on significantly altered CNV genes.

**Key Statistical Methods:**

1. **Hypergeometric Test (Drug Enrichment):**
   - Tests if a drug targets more significant genes than expected by chance
   - Parameters:
     - `M`: Total number of genes in DrugBank
     - `K`: Number of significant genes identified from CNV analysis
     - `n`: Number of genes targeted by the drug
     - `x`: Number of significant genes targeted by the drug
   - Right-tailed test: P(X ≥ x)

2. **Empirical Permutation Testing (100,000 iterations):**
   - Randomly shuffles gene labels to generate null distribution
   - Calculates empirical p-value: proportion of permutations with enrichment ≥ observed
   - Validates statistical significance beyond hypergeometric assumptions

3. **Protein-Protein Interaction (PPI) Network Analysis:**
   - Uses STRING database (confidence ≥ 700) for indirect drug candidates
   - Connects drug targets to significant genes via PPI edges

**Key Functions:**
- `hypergeom_p_right_tail()`: Calculates hypergeometric p-value for drug enrichment
- `identify_significant_drug_candidates()`: Main function that:
  - Matches drug targets to significant genes (direct candidates)
  - Finds PPI connections for indirect candidates
  - Performs hypergeometric testing for each drug
  - Runs 100,000 permutation tests for empirical validation
  - Applies FDR correction (Benjamini-Hochberg)
  - Filters to significant drugs (hypergeom FDR < 0.05, empirical FDR < 0.05)

**Workflow:**

**Direct Drug Candidates:**
1. Match DrugBank drug targets to significant CNV genes
2. Calculate hypergeometric enrichment for each drug
3. Run 100,000 permutations to get empirical p-values
4. Apply FDR correction
5. Filter to FDR < 0.05 for both tests

**Indirect Drug Candidates:**
1. Find PPI connections (STRING confidence ≥ 700) between drug targets and significant genes
2. Calculate percentage of significant genes hit via PPI
3. Perform same statistical testing as direct candidates
4. Filter to FDR < 0.05

**Inputs:**
- `Results/CNV results/HPV positive amplification top genes.csv`: Significant HPV+ amplified genes (from File 02.2)
- `Results/CNV results/HPV positive deletion top genes.csv`: Significant HPV+ deleted genes (from File 02.2)
- `Results/CNV results/HPV negative amplification top genes.csv`: Significant HPV- amplified genes (from File 02.2)
- `Results/CNV results/HPV negative deletion top genes.csv`: Significant HPV- deleted genes (from File 02.2)
- DrugBank structured data (from `structure_drug_bank_data()`)
- `Data/Protein-protein interaction data/9606.protein.links.v12.0.txt`: STRING PPI network links
- `Data/Protein-protein interaction data/9606.protein.info.v12.0.txt`: Protein information
- `Data/Protein-protein interaction data/9606.protein.aliases.v12.0.txt`: Protein aliases

**Outputs:**

**Direct Drug Candidates:**
- `Results/CNV results/hpv_pos_amp_top_drugBank_drug_candidates.csv`: HPV+ amplification direct drugs
- `Results/CNV results/hpv_pos_del_top_drugBank_drug_candidates.csv`: HPV+ deletion direct drugs
- `Results/CNV results/hpv_neg_amp_top_drugBank_drug_candidates.csv`: HPV- amplification direct drugs
- `Results/CNV results/hpv_neg_del_top_drugBank_drug_candidates.csv`: HPV- deletion direct drugs

**Indirect Drug Candidates (via PPI):**
- `Results/CNV results/hpv_pos_amp_top_drugBank_drug_candidates_indirect.csv`: HPV+ amplification indirect drugs
- `Results/CNV results/hpv_pos_del_top_drugBank_drug_candidates_indirect.csv`: HPV+ deletion indirect drugs
- `Results/CNV results/hpv_neg_amp_top_drugBank_drug_candidates_indirect.csv`: HPV- amplification indirect drugs
- `Results/CNV results/hpv_neg_del_top_drugBank_drug_candidates_indirect.csv`: HPV- deletion indirect drugs

**Key Output Columns:**
- `DRUG`: Drug name
- `GENE_TARGET`: Gene(s) directly targeted by drug
- `CONNECTED_TO (risk gene)`: Significant genes connected via PPI (indirect only)
- `NUM_DIRECT_TARGETS_HIT`: Number of significant genes targeted
- `PERCENTAGE_OF_TARGETS_HIT`: Percentage of significant genes targeted
- `ACTION`: DrugBank mechanism of action (inhibitor, antagonist, etc.)
- `SPECIFIC_FUNCTION`: Molecular function from DrugBank
- `drug_hypergeom_p_value`: Hypergeometric test p-value
- `drug_hypergeom_fdr`: FDR-corrected hypergeometric p-value
- `drug_empirical_p_value`: Empirical permutation p-value
- `drug_empirical_fdr`: FDR-corrected empirical p-value

**Filters Applied:**
- Excludes known head/neck cancer chemotherapy medications
- Requires both hypergeom FDR < 0.05 AND empirical FDR < 0.05
- Distribution visualizations (plots)
- PPI confidence threshold ≥ 700 for indirect candidates

---
### **03 SOM identify key mutation gene.ipynb**
**Purpose:** Identify significantly mutated genes from somatic mutation (SOM) data stratified by HPV status using gene-length-normalized dual statistical validation.

**Key Statistical Methods:**

1. **Binomial Test (Parametric Approach):** Tests if observed non-synonymous mutations in a gene exceed random expectation accounting for gene length and genome-wide mutation rate.
   - Null hypothesis: Non-synonymous mutations are randomly distributed across the genome proportional to gene CDS (coding sequence) length
   - **Critical Design Choice**: Uses **ALL mutations (synonymous + non-synonymous)** as denominator to establish the true genome-wide background mutation rate, accounting for regional mutation rate variation and providing stable background estimation
   - Probability that a mutation occurs in gene g: p_g = L_g / L_total (gene length / total coding genome length)
   - Test statistic: k_g ~ Binomial(N_total, p_g) where N_total = total mutations in cohort (including synonymous), k_g = observed non-synonymous mutations in gene g
   - Right-tailed test: P(K ≥ k_g) using binomial distribution

2. **Empirical/Multinomial Test (Non-parametric Approach):** Validates binomial results using Monte Carlo simulation with multinomial distribution.
   - **Critical Design Choice**: Uses **only non-synonymous mutations** as denominator to focus on the functionally relevant mutational landscape
   - Generates null distribution by sampling from Multinomial(N_nonsyn, **p**) where **p** is vector of gene-length-based probabilities
   - 10,000 simulations: In each iteration, all gene mutation counts are drawn simultaneously from multinomial
   - Empirical p-value: (1 + count of simulations where simulated ≥ observed) / (M + 1), with pseudocount adjustment to prevent zero p-values
   - Accounts for dependencies between genes (mutations must sum to total N_nonsyn)

3. **Dual-Denominator Rationale:** The intentional use of different denominators provides complementary validation with distinct null hypotheses:
   - **Binomial test**: Tests whether a gene has excess functional mutations given its overall mutational exposure (including synonymous mutations that reflect regional mutation processes)
   - **Multinomial test**: Tests whether a gene is enriched for functional mutations compared to the empirical distribution of non-synonymous mutations across all genes
   - This dual approach dramatically reduces false positives by requiring genes to pass two independent statistical frameworks, catching true driver genes while filtering out hypermutable regions, random fluctuations, and technical artifacts

4. **Multiple Testing Correction:** Benjamini-Hochberg FDR correction applied to both binomial and empirical p-values.

**Key Operations:**
- Loads somatic mutation data (MAF format) filtered to Nulton cohort
- Filters to non-synonymous mutations only (Frame_Shift, Missense, Nonsense, Splice_Site, etc.)
- Extracts CDS lengths from GENCODE v48 GTF annotation for gene length normalization
- Stratifies mutations by HPV status
- Calculates per-gene mutation counts and cohort frequencies
- Computes gene-length-normalized probabilities: p_g = gene_CDS_length / total_coding_genome_length
- Performs length-normalized binomial testing for each gene
- Runs 10,000 multinomial simulations for empirical validation
- Applies FDR correction (Benjamini-Hochberg) to both test p-values
- Calculates mutation scores (normalized count × frequency percentage, inspired by GISTIC)
- Applies frequency cutoffs and score thresholds for filtering
- Identifies top significantly mutated genes (adjusted p-value < 0.05 for both tests)

**Inputs:**
- `Data/TCGA/SOM/nulton_somatic_mutation_cohort.csv`: Somatic mutations filtered to Nulton cohort patients
- `Data/TCGA/SOM/gencode.v48.annotation.gtf`: GENCODE gene annotations with CDS lengths
- `Results/HPV results/HPV positive patients.csv`: HPV+ case IDs
- `Results/HPV results/HPV negative patients.csv`: HPV- case IDs

**Outputs:**
- `Results/SOM Results/HPV positive top genes.csv`: Significantly mutated genes in HPV+ cohort
- `Results/SOM Results/HPV negative top genes.csv`: Significantly mutated genes in HPV- cohort

**Key Columns in Output:**
- `Gene`: Gene symbol (Hugo symbol)
- `MUT_TYPE`: SOMATIC
- `Mutation_Count`: Raw number of mutations in the gene
- `Cohort_Frequency`: Number of patients with at least one mutation in the gene
- `Frequency_Percentage`: Percentage of patients with mutations in the gene
- `Normalized_Count`: Mutation count divided by gene CDS length (mutations per base pair)
- `Mutation_Score`: Normalized_Count × Frequency_Percentage (composite score)
- `Binomial_P_Value`: Raw binomial test p-value (length-normalized)
- `Empirical_P_Value`: Raw empirical/multinomial test p-value
- `Adjusted_P_Value` (q_value): FDR-corrected binomial p-value
- `Adjusted_Empirical_P_Value` (empirical_q_value): FDR-corrected empirical p-value
- `Normalized_Cohort_Frequency`: Min-max normalized frequency (for visualization)

**Filtering Thresholds:**

**HPV Positive cohort:**
- Frequency_Percentage ≥ 5.0%
- Mutation_Score ≥ 0.002
- Adjusted_P_Value < 0.05 (binomial)
- Adjusted_Empirical_P_Value < 0.05 (multinomial)
- Normalized_Cohort_Frequency ≥ 0.001
- Normalized_Count ≥ 0.001

**HPV Negative cohort:**
- Frequency_Percentage ≥ 0.01%
- Adjusted_P_Value < 0.05 (binomial)
- Adjusted_Empirical_P_Value < 0.05 (multinomial)
- Normalized_Cohort_Frequency ≥ 0.0035
- Normalized_Count ≥ 0.0035

**Common Mutated Genes:** TP53, PIK3CA, CDKN2A (typical in HNSC)

**Libraries:** pandas, scipy.stats (binomtest), statsmodels.stats.multitest, numpy, matplotlib, plotly

**Note on Methodology Update (2026):**
Code updated to align with published methodology. The paper's methods section already described pseudocount adjustment in empirical p-value calculations (formula: (1 + successes) / (M + 1)), but the code originally implemented the simpler formula (successes / M). This update corrects the code to match the published methods. Finalized genes remain unchanged, and empirical q-values shifted only marginally with no impact on final biological conclusions or drug candidates. All results remain consistent with published findings.

---

### **03.5 SOM drug repurpose.ipynb**
**Purpose:** Identify direct and indirect drug repurposing candidates based on significantly mutated somatic genes.

**Key Statistical Methods:** Same as 02.50 CNV drug repurposing (hypergeometric testing + 100,000 empirical permutations).

**Key Differences from 02.50:**
- Input: Somatic mutation (SOM) data instead of CNV data
- Mutation type: SOMATIC instead of AMPLIFICATION/DELETION
- Similar statistical pipeline and filtering criteria

**Key Functions:**
- `hypergeom_p_right_tail()`: Calculates hypergeometric p-value
- `identify_significant_drug_candidates()`: Same workflow as CNV version but adapted for somatic mutations

**Workflow:**
2. Match drug targets to mutated genes (direct candidates)
3. Find PPI connections for indirect candidates
4. Perform hypergeometric enrichment testing
5. Run 100,000 permutation tests for empirical validation
6. Apply FDR correction (Benjamini-Hochberg)
7. Filter to drugs with FDR < 0.05 for both tests

**Inputs:**
- `Results/SOM Results/HPV positive top genes.csv`: Significant HPV+ somatic genes
- `Results/SOM Results/HPV negative top genes.csv`: Significant HPV- somatic genes
- DrugBank structured data
- `Data/Protein-protein interaction data/9606.protein.links.v12.0.txt`: STRING PPI database
- `Data/Protein-protein interaction data/9606.protein.info.v12.0.txt`: Protein information
- `Data/Protein-protein interaction data/9606.protein.aliases.v12.0.txt`: Protein aliases

**Outputs:**
- `Results/hpv_pos_som_top_drugBank_drug_candidates.csv`: HPV+ somatic direct candidates
- `Results/hpv_neg_som_top_drugBank_drug_candidates.csv`: HPV- somatic direct candidates
- Indirect candidates for each category

**Key Output Columns:** Same as 02.50 CNV outputs, with `MUT_TYPE` = SOMATIC

**Libraries:** Same as 02.50

---

### **03.75 result viewing.ipynb**
**Purpose:** Quick visualization and exploratory viewing notebook for SOM drug repurposing results. This is an exploratory notebook for viewing intermediate results.

**Key Operations:**
- Loads SOM drug candidate results (direct and indirect) for HPV+ and HPV-
- Displays top drug candidates in tabular format
- Quick comparison between HPV+ and HPV- results
- Exploratory data viewing without formal analysis or output generation

**Inputs:**
- `Results/SOM results/hpv_positive_som_top_direct_drug_candidates_agg.csv`: HPV+ SOM direct drugs
- `Results/SOM results/hpv_negative_som_top_direct_drug_candidates_agg.csv`: HPV- SOM direct drugs
- `Results/SOM results/hpv_positive_som_top_indirect_drug_candidates_agg.csv`: HPV+ SOM indirect drugs
- `Results/SOM results/hpv_negative_som_top_indirect_drug_candidates_agg.csv`: HPV- SOM indirect drugs

**Outputs:**
- **None** - This is a viewing/exploration notebook only
- Results displayed in notebook cells for review
- No files saved

**Use Case:**
- Quick review of SOM drug repurposing results before final integration
- Exploratory analysis to understand SOM drug candidate characteristics
- Intermediate checkpoint between SOM analysis (File 03.5) and final comparison (File 04)

**Libraries:** pandas

---

### **04 SOM and CNV results comparison.ipynb**
**Purpose:** Compare and visualize overlap between drug candidates identified from CNV (amplifications/deletions) and somatic mutation analyses. This is an exploratory/comparison notebook only - it does not produce output files.

**Key Analyses:**

1. **Gene-Level Overlap:**
   - Calculates overlap between CNV and SOM significant genes
   - Identifies genes with multiple mutation types (e.g., TP53 amplified AND mutated)
   - Stratified by HPV status

2. **Drug-Level Overlap:**
   - Identifies drugs that target genes from multiple mutation sources
   - Calculates overlap between amplification, deletion, and somatic drug candidates
   - Visualizes drugs targeting genes with converging evidence

3. **Comparison Metrics:**
   - Calculates Venn diagram statistics for gene and drug overlaps
   - Identifies drugs appearing in all three categories (amp, del, somatic)
   - Counts number of unique targeted genes and connected genes per drug

**Key Operations:**
- Loads individual CNV drug candidates (amp, del) for HPV+ and HPV-
- Loads SOM drug candidates for HPV+ and HPV-
- Calculates overlap statistics between mutation types
- Visualizes Venn diagrams for gene and drug intersections
- Displays drugs with convergent evidence (appearing across multiple mutation types)
- Performs aggregation analysis within notebook for viewing purposes only

**Inputs:**
- `Results/CNV results/hpv_pos_amp_top_drugBank_drug_candidates.csv`
- `Results/CNV results/hpv_pos_del_top_drugBank_drug_candidates.csv`
- `Results/CNV results/hpv_pos_som_top_drugBank_drug_candidates.csv`
- `Results/CNV results/hpv_neg_amp_top_drugBank_drug_candidates.csv`
- `Results/CNV results/hpv_neg_del_top_drugBank_drug_candidates.csv`
- `Results/CNV results/hpv_neg_som_top_drugBank_drug_candidates.csv`

**Outputs:**
- **None** - This is a viewing/comparison notebook only
- Overlap statistics displayed in notebook cells
- Comparison tables displayed for interactive exploration
- Used to understand convergent evidence before final result creation (File 05)

**Note:** While this file performs aggregation within the notebook for viewing, it does not save these aggregations as output files. The actual aggregation used by downstream files is performed in File 02.50 (for CNV) and File 03.5 (for SOM), which create the "Aggregated" CSV files.

**Key Insights:**
- Drugs targeting genes mutated through multiple mechanisms have stronger biological rationale
- Overlap drugs are prioritized as having convergent evidence
- Helps researchers identify high-confidence candidates before final result creation

**Libraries:** pandas, matplotlib, plotly

---

### **05 Final result creation.ipynb**
**Purpose:** Consolidate and aggregate all genetic results and drug candidates into final unified tables, adding literature validation from PubMed mining.

**Key Operations:**

1. **Gene Results Integration:**
   - Combines CNV and SOM significant genes for each cohort
   - Concatenates mutation types (e.g., "AMPLIFICATION, SOMATIC")
   - Takes minimum q-value and empirical q-value across sources
   - Validates genes against literature-extracted targets

2. **Direct Drug Candidates Aggregation:**
   - Merges CNV and SOM direct drug candidates
   - Groups by drug and aggregates gene targets
   - Combines mutation types and article counts
   - Concatenates PMIDs for literature traceability

3. **Indirect Drug Candidates Aggregation:**
   - Merges CNV and SOM indirect drug candidates
   - Aggregates gene targets and connected risk genes
   - Combines PPI-validated connections
   - Integrates literature validation metrics

4. **Literature Validation and Drug Filtering:**
   - Loads literature-extracted gene-disease relationships from GPU-accelerated NLP pipeline
   - Identifies which genomically significant genes have literature support in HNC research
   - **Drug filtering strategy:** After statistical enrichment testing, drugs are retained only if they:
     - **Direct candidates:** Target genes that appear in the literature validation results
     - **Indirect candidates:** Connect via PPI to risk genes that appear in the literature validation results
   - Adds literature-validated gene lists, article counts, and PMIDs for traceability
   - **This secondary filtering occurs after statistical drug-gene identification** - only drugs targeting literature-validated genes are carried forward to Files 06-07

**Key Data Sources:**
- CNV results: Aggregated HPV+/- drug candidates (amplification + deletion combined)
- SOM results: Aggregated HPV+/- somatic mutation drug candidates
- Literature validation: Drug-gene pairs extracted from PubMed using GPU-accelerated NLP

**Aggregation Strategy:**
- Group by `DRUG` 
- Concatenate gene lists (comma-separated, deduplicated)
- Concatenate mutation types (deduplicated)
- Take minimum FDR values (most significant)
- Sum article counts
- Concatenate PMIDs (deduplicated, cleaned)

**Inputs:**

**Gene Results:**
- `Results/CNV results/HPV positive CNV top genes.csv`: HPV+ CNV significant genes (from File 02.2)
- `Results/CNV results/HPV negative CNV top genes.csv`: HPV- CNV significant genes (from File 02.2)
- `Results/SOM results/HPV positive top genes.csv`: HPV+ SOM significant genes (from File 03)
- `Results/SOM results/HPV negative top genes.csv`: HPV- SOM significant genes (from File 03)

**Drug Candidates (Aggregated files from Files 02.50 and 03.5):**
- `Results/CNV results/HPV Positive Top Direct Drug Candidates Aggregated.csv`: HPV+ CNV direct drugs (amp+del combined, created by File 02.50)
- `Results/CNV results/HPV Positive Top Indirect Drug Candidates Aggregated.csv`: HPV+ CNV indirect drugs (amp+del combined, created by File 02.50)
- `Results/CNV results/HPV Negative Top Direct Drug Candidates Aggregated.csv`: HPV- CNV direct drugs (amp+del combined, created by File 02.50)
- `Results/CNV results/HPV Negative Top Indirect Drug Candidates Aggregated.csv`: HPV- CNV indirect drugs (amp+del combined, created by File 02.50)
- `Results/SOM results/hpv_positive_som_top_direct_drug_candidates_agg.csv`: HPV+ SOM direct drugs (created by File 03.5)
- `Results/SOM results/hpv_positive_som_top_indirect_drug_candidates_agg.csv`: HPV+ SOM indirect drugs (created by File 03.5)
- `Results/SOM results/hpv_negative_som_top_direct_drug_candidates_agg.csv`: HPV- SOM direct drugs (created by File 03.5)
- `Results/SOM results/hpv_negative_som_top_indirect_drug_candidates_agg.csv`: HPV- SOM indirect drugs (created by File 03.5)

**Literature Validation (Optional):**
- `Validation pipeline/Results/cleaned_extracted_targets_all_pub_after_2000_GPU_2b_gemma.csv`: Drug-gene pairs from PubMed
- `Validation pipeline/Results/cleaned_extracted_combined_targets_all_pub_after_2000_GPU_2b_gemma.csv`: Combined drug-gene targets

**Note:** File 05 reads aggregated drug candidate files directly from Files 02.50 (CNV aggregated: amp+del combined) and 03.5 (SOM aggregated). File 04 is used only for comparison/viewing and does not produce output files.

**Outputs:**

**Intermediate Files (Results/):**
- `Results/HPV positive gene results.csv`: Intermediate unified HPV+ significant genes (CNV + SOM)
- `Results/HPV negative gene results.csv`: Intermediate unified HPV- significant genes (CNV + SOM)
- `Results/HPV Positive direct results.csv`: Intermediate aggregated HPV+ direct drug candidates
- `Results/HPV Negative direct results.csv`: Intermediate aggregated HPV- direct drug candidates
- `Results/HPV Positive indirect results.csv`: Intermediate aggregated HPV+ indirect drug candidates (via PPI)
- `Results/HPV Negative indirect results.csv`: Intermediate aggregated HPV- indirect drug candidates (via PPI)

**Final Validated Files (Results/Final Results/):**
- `Results/Final Results/HPV Positive validated genes.csv`: Final validated HPV+ genes with literature support
- `Results/Final Results/HPV Negative validated genes.csv`: Final validated HPV- genes with literature support

**Key Output Columns (Final Results):**
- `DRUG`: Drug name
- `GENE_TARGET` / `LITERATURE_GENE_TARGETS`: Genes targeted by drug (DrugBank and/or literature)
- `LITERATURE_VALIDATED_GENE_TARGETS`: Genes confirmed in PubMed literature
- `CONNECTED_TO (risk gene)`: Risk genes connected via PPI (indirect only)
- `MUT_TYPE`: Mutation type(s) (AMPLIFICATION, DELETION, SOMATIC)
- `NUMBER_OF_ARTICLES`: Count of PubMed articles supporting drug-gene interaction
- `PMID`: PubMed IDs (comma-separated)
- `drug_empirical_fdr`: FDR-corrected empirical p-value
- `drug_hypergeom_fdr`: FDR-corrected hypergeometric p-value
- `PERCENTAGE_OF_TARGETS_HIT`: Percentage of significant genes targeted

**Libraries:** pandas

---

### **06 Graph direct results.ipynb**
**Purpose:** Visualize direct drug-gene targeting relationships and generate DrugBank-verified tables for direct drug candidates.

**Key Functions:**

**`create_direct_gene_table(results_df, drug_bank_df, hpv_gene_results, top_n=50, cohort_name='')`**
- Creates top N direct drug candidate table with DrugBank action verification
<!-- - Handles column name variations (`LITERATURE_VALIDATED_GENE_TARGETS` vs `LITERATURE_GENE_TARGETS`) -->
- Performs per-gene DrugBank action verification:
  - Searches DrugBank for each literature-validated target
  <!-- - Marks genes without DrugBank annotation as "GENE: UNKNOWN" -->
  - Keeps all drugs (does not filter by action availability)
- Maps mutation types from HPV gene results to validated targets
- Collects specific molecular functions from DrugBank (limit 3)
- Cleans and formats PMIDs
- Sorts by `drug_empirical_fdr` (ascending)

**Visualization: Bipartite Drug-Gene Network**
- Drugs positioned on left, genes on right
- Drugs sorted by degree (number of genes targeted, descending)
<!-- - Node colors represent mutation types (amplification: red, deletion: blue, somatic: green) -->
- Node sizes proportional to number of connections
- Edge thickness represents strength of association
- Includes gene degree counts and mutation type labels

**Key Operations:**
1. Load direct results and gene results for HPV+ and HPV-
2. Parse DrugBank XML for action and function verification
3. Create top tables with per-gene action verification:
   - Verify each gene target against DrugBank
   <!-- - Mark missing actions as "GENE: UNKNOWN" -->
   - Map mutation types from gene results
   - Collect molecular functions
   - Clean PMIDs
4. Export tables to CSV
5. Generate bipartite network graphs for visualization

**Inputs:**
- `Results/HPV Positive direct results.csv`
- `Results/HPV Negative direct results.csv`
- `Results/HPV positive gene results.csv`
- `Results/HPV negative gene results.csv`
- `Data/DGIDB/drug_bank.xml`

**Outputs:**
- `Results/Final Results/HPV Positive Validated direct candidates.csv`: Final validated HPV+ direct drugs with DrugBank actions
- `Results/Final Results/HPV Negative Validated direct candidates.csv`: Final validated HPV- direct drugs with DrugBank actions

**Table Columns:**
- `DRUG`: Drug name
- `MUT_TYPE`: Mutation type(s) of targeted genes
- `Literature-Validated Gene Targets`: Genes confirmed in literature
- `Number of Validated Targets`: Count of literature-validated targets
- `Article Count`: Number of PubMed articles
- `PMIDs`: PubMed identifiers
- `ACTION`: DrugBank actions per gene (format: "GENE1: inhibitor; GENE2: UNKNOWN")
- `SPECIFIC_FUNCTION`: Molecular functions (semicolon-separated, max 3)
- `Drug Empirical FDR`: Statistical significance

**Key Design Decision:**
- Changed from filtering drugs without actions to marking individual genes as "GENE: UNKNOWN"
- Rationale: Literature validation is sufficient even without DrugBank confirmation
- Provides transparency about which targets lack DrugBank annotation

**Libraries:** pandas, numpy, matplotlib, networkx

---

### **07_sankey_diagram_builder.ipynb**
**Purpose:** Visualize indirect drug→target→risk gene pathways through Sankey diagrams and generate comprehensive PPI-validated tables showing literature-validated connections only.

**Critical Data Philosophy:**
- **Only uses literature-validated gene targets** (`LITERATURE_GENE_TARGETS` column)
- **Only uses literature-validated risk genes** (`RISK_GENE_LITERATURE_GENE_TARGETS` column)
- All connections must be supported by both PubMed literature AND STRING PPI database
- Ensures all pathways shown are evidence-based and biologically validated

**Sankey Diagram Visualization Strategy:**
The Sankey diagrams visualize drug pathways by showing connections from drugs to indirect gene targets (middle layer) and then to **any validated risk genes** within the HPV cohort (right layer) that are PPI-connected to those gene targets. Critically, the risk genes displayed are not limited to only those through which the drug was identified via significance testing. Instead, the diagrams show **all possible flows** from the drug's significant indirect gene targets to any literature-validated risk genes in the cohort—regardless of whether those risk genes were identified through amplification, deletion, or somatic mutation events. This comprehensive approach ensures that each drug-gene connection reveals the full network of potential therapeutic impacts, including connections to risk genes that may have been discovered through different genomic alteration methods than those initially used for drug discovery.

---

**Key Functions:**

**1. `group_gene_families(gene_list, min_family_size=3)`**
- Groups related genes into families for cleaner Sankey visualization
    - Identifies gene families (e.g., HLA-A, HLA-B, HLA-C → HLA family)
- Only groups if family has ≥ min_family_size members
- Returns mapping: individual gene → family name
- Reduces visual clutter in diagrams with many related genes

**2. `prepare_sankey_data(results_df, ppi_set, top_n=50, required_drugs=None)`**
**Purpose:** Core data preparation function that builds the network structure for Sankey visualization.

**Detailed Workflow:**
1. **Sort and Rank Drugs:**
   - Sorts by `drug_empirical_fdr` (ascending) and `PERCENTAGE_OF_TARGETS_HIT` (descending)
   - Prioritizes statistically significant drugs with high target coverage

2. **Process Required Drugs First:**
   - If `required_drugs` list provided, processes these drugs first (selected based on viewing top 50 and selecting top candidates)
   - Ensures specific drugs of interest are included regardless of ranking

3. **Literature-Validated Gene Parsing:**
   - Extracts `LITERATURE_GENE_TARGETS`: Genes with PubMed evidence of drug interaction
   - Extracts `RISK_GENE_LITERATURE_GENE_TARGETS`: Risk genes with literature validation
   - **Only these literature-validated genes are used** (not all database genes) ensuring secondary validation

4. **PPI Validation:**
   - For each drug, double checks if validated gene targets connect to validated risk genes via PPI (confidence ≥ 700)
   - Creates two link types:
     - `drug_to_target`: Drug → Literature-validated gene target
     - `target_to_risk`: Literature-validated gene target → Literature-validated risk gene (via PPI)
   - **Only includes connections supported by both literature AND PPI**

5. **Filtering for Visualization:**
   - Removes drugs with no valid PPI connections
   - For drugs with > 15 gene targets, keeps top 15 by connection count (reduces clutter) (eg. fostamatinib having over 40 gene targets)
   - Removes genes appearing in both target and risk layers (prevents circular flows)

6. **Gene Family Grouping:**
   - Applies `group_gene_families()` to collapse related genes
   - Creates legend mapping: family → individual genes
   - Re-aggregates links after grouping

7. **Returns Dictionary:**
   - `link_counts`: DataFrame with source→target connections and link types
   - `drugs`: List of drugs with valid connections
   - `gene_targets`: List of literature-validated gene targets (or families)
   - `risk_genes`: List of literature-validated risk genes
   - `top_df`: Filtered results DataFrame
   - `family_to_genes`: Mapping for legend display

**3. `create_combined_sankey(data_dict, title, height=1400, width=2400)`**
**Purpose:** Creates a unified Sankey diagram showing all drugs in one comprehensive view.

**Visualization Details:**
- **Left Column (Blue nodes)**: Drugs
- **Middle Column (Green nodes)**: Literature-validated gene targets (direct drug targets from PubMed)
- **Right Column (Red nodes)**: Literature-validated risk genes (connected via PPI)
- **Flows**: Represent drug→target→risk gene pathways
- **Gene Family Legend**: Shows which individual genes are grouped into families

**Node Positioning:**
- Fixed positions: drugs (x=0.001), targets (x=0.5), risk genes (x=0.999)
- Ensures proper left-to-right flow visualization

**Color Scheme:**
- Blue: Drugs
- Green: Gene targets (literature-validated)
- Red: Risk genes (literature-validated)
- Link transparency: rgba(0,0,0,0.2)

**Interactive Features:**
- Hover to see node labels
- Click-drag to rearrange nodes
- Zoom and pan enabled

**4. `create_individual_sankeys(data_dict, drug_list=None, title_prefix='', ncols=2, height=2200, width=2000, risk_gene_threshold=0)`**
**Purpose:** Creates a grid of individual Sankey diagrams, one per drug, for detailed pathway exploration.

**Grid Layout:**
- Each subplot shows one drug's complete pathway network
- `ncols`: Number of columns in grid (default: 2)
- Automatically calculates rows based on number of drugs
- Subplot titles show individual drug names

**Per-Drug Filtering:**
- For drugs with > 15 gene targets, applies `risk_gene_threshold`
- Keeps only gene targets with ≥ threshold connections to risk genes
- Removes targets with no downstream connections after filtering
- Ensures each diagram shows only meaningful pathways

**Node Ordering:**
- Layer 1: Drug (single node)
- Layer 2: Literature-validated gene targets for this drug
- Layer 3: Literature-validated risk genes connected via PPI
- Maintains strict left-to-right flow

**Visualization Features:**
<!-- - Uniform link thickness (value=1 for all links, focus on topology not weight) -->
- `arrangement='snap'` for optimized node positioning
- Per-drug gene family legend
- Larger font sizes for readability

**5. `create_filtered_top_table(results_df, ppi_set, drug_bank_df, top_n=50, cohort_name='')`**
**Purpose:** Generate publication-ready table with DrugBank action verification for top indirect drug candidates.

**Table Generation Workflow:**

1. **Load Gene Results:**
   - Imports HPV gene results to map mutation types
   - Creates gene→MUT_TYPE dictionary

2. **Sort and Select Top Drugs:**
   - Sorts by `drug_empirical_fdr` (ascending) and `PERCENTAGE_OF_TARGETS_HIT` (descending)
   - Selects top N drugs

3. **Parse Literature-Validated Genes:**
   - Extracts `LITERATURE_GENE_TARGETS`: Genes with PubMed evidence
   - Extracts `RISK_GENE_LITERATURE_GENE_TARGETS`: Risk genes with literature validation
   - **Critical:** Only uses these validated genes, not all database entries, crucial for evidence-based reporting

4. **PPI Validation:**
   - Checks which gene targets connect to risk genes via STRING PPI (confidence ≥ 700)
   - Creates list of `validated_targets`: Gene targets with PPI-confirmed connections

5. **DrugBank Action Verification (Per-Gene):**
   - For each validated target gene:
     - Searches DrugBank for drug-gene interaction
     - Extracts action (inhibitor, antagonist, modulator, etc.)
     - If action found: adds "GENE: action"
     - If no action: adds "GENE: UNKNOWN"
   - Aggregates: "GENE1: inhibitor; GENE2: UNKNOWN; GENE3: modulator"

6. **Specific Function Collection:**
   - Extracts molecular functions from DrugBank for validated targets
   - Limits to 3 functions to keep table readable
   - Format: "function1; function2; function3"

7. **Mutation Type Mapping:**
   - Maps validated risk genes to their mutation types (AMPLIFICATION, DELETION, SOMATIC)
   - Deduplicates: splits compound types like "AMPLIFICATION, DELETION"
   - Joins unique types with ", "

8. **Article Count and PMID Cleaning:**
   - Converts article counts to integers (not strings)
   - Cleans PMIDs: removes 'nan', strips whitespace
   - Separates gene target PMIDs and risk gene PMIDs

9. **Table Assembly:**
   - Creates row with all validated information
   - Adds ALL drugs with PPI validation (not filtered by DrugBank action)
   - Returns DataFrame sorted by statistical significance

**Table Output:**
- All columns contain only literature-validated and PPI-validated information
- Transparency: Shows "UNKNOWN" for genes lacking DrugBank annotation
- Rationale: Literature + PPI validation is sufficient even without DrugBank confirmation

---

**Complete Pipeline Workflow:**

1. **Load Data:**
   - Indirect results (HPV+ and HPV-)
   - Gene results for mutation type mapping
   - DrugBank XML database
   - STRING PPI database (map protein IDs to gene names)

2. **Create PPI Validation Set:**
   - Load protein-protein interaction links
   - Filter to confidence ≥ 700
   - Create bidirectional set: {(geneA, geneB), (geneB, geneA)}

3. **Generate Tables:**
   - Run `create_filtered_top_table()` for HPV+ and HPV-
   - Export top 50 tables with DrugBank verification

4. **Prepare Sankey Data:**
   - Run `prepare_sankey_data()` for HPV+ and HPV-
   - Process top 10-15 drugs for visualization
   - Apply gene family grouping

5. **Create Visualizations:**
   - **Combined Sankey:** All drugs in one comprehensive view
   - **Individual Sankeys:** Grid of per-drug pathway diagrams
   - Generate for both HPV+ and HPV-

6. **Data Quality Checks:**
   - Verify PMID completeness in exported CSVs
   - Confirm no data loss during export
   - Validate that only literature-validated genes appear in outputs

**Inputs:**
- `Results/HPV Positive indirect results.csv`
- `Results/HPV Negative indirect results.csv`
- `Results/HPV positive gene results.csv`
- `Results/HPV negative gene results.csv`
- `Data/DGIDB/drug_bank.xml`
- `Data/Protein-protein interaction data/9606.protein.links.v12.0.txt`
- `Data/Protein-protein interaction data/9606.protein.info.v12.0.txt`
- `Data/Protein-protein interaction data/9606.protein.aliases.v12.0.txt`

**Outputs:**

**Tables (Final Results/):**
- `Results/Final Results/HPV Positive Validated indirect candidates.csv`: Final validated HPV+ indirect drugs
- `Results/Final Results/HPV Negative Validated indirect candidates.csv`: Final validated HPV- indirect drugs

**Sankey Diagrams (Interactive Plotly HTML):**
- Combined Sankey: All drugs in unified view (HPV+ and HPV-)
- Individual Sankeys: Grid of per-drug pathway diagrams (HPV+ and HPV-)

**Table Columns:**
- `DRUG`: Drug name
- `MUT_TYPE`: Mutation type(s) of connected risk genes (from literature-validated risk genes only)
- `Literature-Validated Gene Targets`: Direct targets of drug (from PubMed)
- `Number of Validated Targets`: Count of literature-validated targets
- `Literature-Validated Risk Genes`: Risk genes with PubMed evidence connected via PPI
- `Number of Validated Risk Genes`: Count of literature-validated risk genes
- `Gene Target Article Count`: Articles supporting gene target validation
- `Risk Gene Article Count`: Articles supporting risk gene validation
- `Gene Target PMIDs`: PubMed IDs for gene target validation
- `Risk Gene PMIDs`: PubMed IDs for risk gene validation
- `ACTION`: DrugBank actions per target (format: "GENE1: inhibitor; GENE2: UNKNOWN")
- `SPECIFIC_FUNCTION`: Molecular functions from DrugBank (semicolon-separated, max 3)
- `Drug Empirical FDR`: Statistical significance from permutation testing
- `Percent Targets Hit`: Percentage of literature-validated risk genes connected via PPI

**PPI Validation:**
- Uses STRING database v12.0
- Confidence threshold: ≥ 700 (high confidence)
- Connects drug targets to risk genes through one PPI edge
- Validates that literature-supported connections are also PPI-supported

**Key Design Decisions:**
- Per-gene action verification (not per-drug filtering)
- UNKNOWN marking for genes without DrugBank actions
- Rationale: PPI validation and literature validation are sufficient
- Provides granular information about which targets lack DrugBank annotation
- **Only literature-validated genes used throughout entire pipeline**

**Data Quality Verification:**
- PMID completeness check cell confirms all PMIDs stored completely in CSV exports
- Display truncation ("...") is pandas preview artifact only
- Actual data preserved (verified by re-reading CSV files)

**Libraries:** pandas, numpy, plotly, xml.etree.ElementTree, networkx (implicit through PPI analysis)

---

## Key Databases and Resources

**DrugBank XML (`Data/DGIDB/drug_bank.xml`):**
- ~23,136 drug-gene interactions
- Includes: drug names, gene targets, actions (inhibitor, antagonist, etc.), molecular functions
- Namespace: `http://www.drugbank.ca`

**STRING PPI Database:**
- Human protein-protein interactions (v12.0)
- Confidence scores: 0-1000 (threshold used: ≥700)
- Files: `9606.protein.links.v12.0.txt`, `9606.protein.info.v12.0.txt`, `9606.protein.aliases.v12.0.txt`

**TCGA HNSC Data:**
- Gene-level CNV: GISTIC scores for ~20,000 genes across ~280 patients
- Somatic mutations: MAF format with ~18,000 genes
- Clinical data: HPV status validated by Nulton et al.

**Literature Validation (PubMed Mining):**
- GPU-accelerated NLP pipeline using Gemma 2B model
- Extracts drug-gene interactions from PubMed articles (post-2000)
- Provides PMIDs for citation traceability
- Files: `cleaned_extracted_targets_all_pub_after_2000_GPU_2b_gemma.csv`

---

## Output Summary

**Final Publication-Ready Outputs (Results/Final Results/):**

1. **Gene Results (Significant Mutations with Literature Validation):**
   - `Results/Final Results/HPV Positive validated genes.csv`: Final HPV+ significant genes (CNV + SOM) with literature support
   - `Results/Final Results/HPV Negative validated genes.csv`: Final HPV- significant genes (CNV + SOM) with literature support

2. **Direct Drug Candidates (Drug → Gene):**
   - `Results/Final Results/HPV Positive Validated direct candidates.csv`: Final HPV+ direct drugs with DrugBank verification
   - `Results/Final Results/HPV Negative Validated direct candidates.csv`: Final HPV- direct drugs with DrugBank verification

3. **Indirect Drug Candidates (Drug → Target → Risk Gene via PPI):**
   - `Results/Final Results/HPV Positive Validated indirect candidates.csv`: Final HPV+ indirect drugs with PPI validation
   - `Results/Final Results/HPV Negative Validated indirect candidates.csv`: Final HPV- indirect drugs with PPI validation

**Intermediate Analysis Files (Results/):**
- `Results/HPV positive gene results.csv`: Intermediate HPV+ genes (used by File 06, 07)
- `Results/HPV negative gene results.csv`: Intermediate HPV- genes (used by File 06, 07)
- `Results/HPV Positive direct results.csv`: Intermediate HPV+ direct candidates (all candidates before filtering)
- `Results/HPV Negative direct results.csv`: Intermediate HPV- direct candidates (all candidates before filtering)
- `Results/HPV Positive indirect results.csv`: Intermediate HPV+ indirect candidates (all candidates before filtering)
- `Results/HPV Negative indirect results.csv`: Intermediate HPV- indirect candidates (all candidates before filtering)

**Statistical Thresholds:**
- Genetic significance: q_value < 0.05 AND empirical_q_value < 0.05
- Drug enrichment: hypergeom_fdr < 0.05 AND empirical_fdr < 0.05
- PPI confidence: ≥ 700 (STRING database)

**Validation Layers:**
1. Statistical significance (hypergeometric + permutation testing)
2. DrugBank annotation (action mechanisms, molecular functions)
3. Literature validation (PubMed mining with NLP)
4. PPI network validation (STRING database)

---

## Usage Notes

**Execution Order:**
Run notebooks sequentially from 00 to 07 to reproduce the complete pipeline. File 03.75 is an optional viewing notebook and can be skipped.

**Required Files:**
Files 00-03, 03.5, 04-07 must be run in sequence.
File 03.75 is optional (viewing only).

---

## Getting Started

### System Requirements

**Main Pipeline:**
- **OS:** macOS, Linux, or Windows with WSL2
- **RAM:** ≥16GB (32GB recommended for large-scale permutation testing)
- **CPU:** Multi-core processor (≥4 cores recommended)
- **Storage:** ~100GB free space for TCGA data and intermediate results
- **Python:** 3.8 or higher

**Validation Pipeline:**
- **GPU:** NVIDIA GPU with ≥8GB VRAM (A100, V100, RTX 3090, or equivalent)
- **CUDA:** Version 11.0 or higher
- **RAM:** ≥32GB system memory
- **Storage:** Additional ~50GB for model weights and literature corpus

### Installation

**1. Clone the Repository:**
```bash
git clone https://github.com/pvtanike/Genomic-Landscape-Based-Drug-Repurposing.git
cd Genomic-Landscape-Based-Drug-Repurposing
```

**2. Create Python Environment:**
```bash
# Using conda (recommended)
conda create -n drug_repurpose python=3.9
conda activate drug_repurpose

# Or using venv
python -m venv drug_repurpose_env
source drug_repurpose_env/bin/activate  # On Windows: drug_repurpose_env\Scripts\activate
```

**3. Install Required Packages:**
```bash
# Core dependencies for main pipeline
pip install pandas numpy scipy statsmodels
pip install matplotlib plotly seaborn
pip install networkx tqdm
pip install jupyter notebook

# Additional packages for validation pipeline (if using)
pip install transformers torch accelerate
pip install biopython requests

# If using GPU for validation
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

**4. Download Required Databases:**

See "Required Input Data Sources" section above for detailed instructions on downloading:
- DrugBank XML (requires free academic registration)
- STRING PPI database (public download)
- TCGA HNSC data (via GDC Data Portal)
- Nulton et al. supplementary data (from PMC5392278)
- Gencode gene annotations

### Quick Start

**Step-by-Step Execution:**

1. **Prepare Data Directory Structure:**
```bash
mkdir -p Data/DGIDB
mkdir -p Data/"Protein-protein interaction data"
mkdir -p Data/"Supplementary data Nulton"
mkdir -p Data/TCGA/"Gene level CNV"
mkdir -p Data/TCGA/SOM
mkdir -p Results
```

2. **Place Required Input Files** (see directory structure above)

3. **Run Analysis Pipeline:**
```bash
# Start Jupyter Notebook
jupyter notebook

# Execute notebooks sequentially:
# 00 Data viewing.ipynb → Initial data exploration
# 01 determine HPV status.ipynb → HPV stratification
# 02 CNV identify mutation gene.ipynb → CNV analysis
# 02.2 CNV key mutation identification.ipynb → CNV gene filtering
# 02.50 CNV drug repurposing candidates.ipynb → CNV drug candidates
# 03 SOM identify key mutation gene.ipynb → Somatic mutation analysis
# 03.5 SOM drug repurpose.ipynb → SOM drug candidates
# 03.75 result viewing.ipynb → (Optional) View SOM results
# 04 SOM and CNV results comparison.ipynb → Compare CNV and SOM results
# 05 Final result creation.ipynb → Aggregate final results
# 06 Graph direct results.ipynb → Visualize direct drug-gene relationships
# 07_sankey_diagram_builder.ipynb → Visualize indirect drug-gene pathways
```

4. **Optional - Literature Validation:**
```bash
cd "Validation pipeline"

# Step 1: Extract PMIDs
bash 00\ extract_pmids.bash

# Step 2-4: Run validation notebooks sequentially
# See "Validation Pipeline" section for detailed instructions
```

5. **Access Results:**
```bash
# Final publication-ready output files located in:
Results/Final\ Results/HPV\ Positive\ validated\ genes.csv
Results/Final\ Results/HPV\ Negative\ validated\ genes.csv
Results/Final\ Results/HPV\ Positive\ Validated\ direct\ candidates.csv
Results/Final\ Results/HPV\ Negative\ Validated\ direct\ candidates.csv
Results/Final\ Results/HPV\ Positive\ Validated\ indirect\ candidates.csv
Results/Final\ Results/HPV\ Negative\ Validated\ indirect\ candidates.csv
```

### Expected Runtime

**Main Pipeline (Files 00-07):**
- File 00-01: ~30 minutes (data loading and HPV stratification)
- File 02: ~1 hour (CNV analysis with permutation testing)
- File 02.2-02.50: ~30 minutes (CNV drug candidates with 100,000 permutations)
- File 03: ~30 minutes (SOM analysis with permutation testing)
- File 03.5: ~30 minutes (SOM drug candidates with 100,000 permutations)
- File 03.75: ~5 minutes (optional - SOM result viewing)
- File 04: ~30 minutes (CNV and SOM comparison)
- File 05: ~15 minutes (final aggregation)
- File 06-07: ~30 minutes (visualization)
- **Total: ~4 hours** (can run overnight)

**Validation Pipeline (Optional):**
- File 00: ~5 minutes (PMID extraction)
- File 01: ~6-12 hours (abstract download)
- File 02: ~24-72 hours (GPU extraction, depends on hardware)
- File 03: ~30 minutes (cleaning and validation)
- **Total: 1.5-4 days** (primarily GPU extraction time)

### Computational Requirements by File

**High Memory Files (≥16GB RAM):**
- 02 CNV identify mutation gene.ipynb
- 02.50 CNV drug repurposing candidates.ipynb
- 03 SOM identify key mutation gene.ipynb
- 03.5 SOM drug repurpose.ipynb
- 07_sankey_diagram_builder.ipynb

**High Compute Files (Multi-core beneficial):**
- 02.50 CNV drug repurposing candidates.ipynb (100,000 permutations)
- 03.5 SOM drug repurpose.ipynb (100,000 permutations)

**GPU-Required Files:**
- Validation pipeline/02 GPU_full_extract.py

### Dependencies

**Core Python Packages:**
```
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
statsmodels >= 0.13.0
matplotlib >= 3.4.0
plotly >= 5.3.0
seaborn >= 0.11.0
networkx >= 2.6.0
tqdm >= 4.62.0
jupyter >= 1.0.0
```

**Packages (Validation Pipeline):**
```
transformers >= 4.30.0
torch >= 2.0.0
accelerate >= 0.20.0
biopython >= 1.79
requests >= 2.26.0
```

**System Tools:**
```
NCBI E-utilities (for validation pipeline)
CUDA Toolkit (for GPU validation pipeline)
```

<!-- ### Troubleshooting

**Common Issues:**

1. **Out of Memory Errors:**
   - Reduce permutation count (e.g., 10,000 instead of 100,000)
   - Process in batches
   - Use machine with more RAM

2. **Slow Permutation Testing:**
   - Enable parallel processing if available
   - Run overnight or on HPC cluster
   - Consider reducing iterations for exploratory analysis

3. **Missing Data Files:**
   - Check "Required Input Data Sources" section
   - Verify file paths match expected structure
   - Ensure all required files downloaded

4. **DrugBank XML Parsing Errors:**
   - Verify DrugBank version compatibility
   - Check XML namespace in code matches downloaded file
   - Re-download if file corrupted

5. **GPU Out of Memory (Validation Pipeline):**
   - Reduce batch size in 02 GPU_full_extract.py
   - Use mixed precision (FP16) - already enabled
   - Use smaller model (distilbert instead of Gemma 2B)

**Getting Help:**
- Open issue on GitHub with error message and system specs
- Check existing issues for similar problems
- Email contacts listed in "Contact and Support" section

**Data Updates:**
- DrugBank: Requires updated XML file for latest drug annotations
- STRING: Update PPI database to latest version for current interactions
- TCGA: Data frozen as of analysis date (reflects cohort at time of download)

**Citation:**
When using this pipeline, please cite:
- Nulton et al. (2017) PMC5392278 for HPV status validation
- DrugBank database
- STRING database v12.0
- TCGA HNSC project -->

---



## Validation Pipeline (Literature Mining)

The Validation Pipeline is a supplementary workflow that provides literature-based validation of **gene-disease relationships** identified in the main pipeline. It uses GPU-accelerated natural language processing to extract disease targets (genes associated with head and neck cancer) from PubMed abstracts, providing independent evidence that genes identified through genomic analysis are also implicated in the disease based on published research.

### Overview

**Purpose:** Extract gene-disease relationships from scientific literature to validate that genomically significant genes are also recognized disease targets in published HNC research.

**Approach:** GPU-accelerated NLP using the Gemma 2B language model to parse PubMed articles and identify key disease targets (genes) mentioned in head and neck cancer literature.

**Integration:** After statistical enrichment identifies significant drug-gene connections, drugs are filtered to retain only those targeting literature-validated genes. This ensures drugs target genes with both genomic significance and published disease relevance.

### Pipeline Components

#### **00 extract_pmids.bash**
**Purpose:** Query PubMed for head and neck cancer articles and extract their PMIDs.

**Key Operations:**
- Uses NCBI E-utilities API (`esearch` and `efetch`)
- Query: "head and neck cancer"
- Extracts unique PubMed IDs for all matching articles
- Requires NCBI API key (free registration at NCBI)

**Setup:**
```bash
# Install NCBI E-utilities
# macOS: brew install ncbi-entrez-direct
# Linux: Follow NCBI EDirect installation guide

# Set your NCBI API key
export NCBI_API_KEY=your_api_key_here
```

**Execution:**
```bash
cd "Validation pipeline"
bash 00\ extract_pmids.bash
```

**Output:**
- `pmids.txt`: List of PubMed IDs (one per line) for head and neck cancer literature

**Typical Results:**
- ~400,000+ PMIDs for head and neck cancer publications

---

#### **01 extract based on pmid.ipynb**
**Purpose:** Download abstracts and full-text content from PubMed using the extracted PMIDs.

**Key Functions:**
- `fetch_pubmed_abstract(pmid)`: Downloads abstract for a given PMID using Entrez API
- `fetch_pmc_full_text(pmcid)`: Downloads full-text XML from PubMed Central if available
- Batch processing with rate limiting to comply with NCBI API policies

**Key Operations:**
1. Load PMIDs from `pmids.txt`
2. Query PubMed API for each PMID to retrieve:
   - Article title
   - Abstract text
   - Publication year
   - Authors
   - Journal
   - PMC ID (if full text available)
3. Filter to articles published after year 2000 (more recent literature)
4. Download full-text XML from PMC when available
5. Combine abstracts and full-text into unified corpus
6. Export to CSV for NLP processing

**API Configuration:**
- Uses Biopython `Entrez` module
- Requires email registration with NCBI

**Inputs:**
- `pmids.txt`: List of PubMed IDs

**Outputs:**
- `Data/pubmed_abstracts.csv`: DataFrame with columns `[PMID, Title, Abstract, Year, Journal]`
- `Data/pubmed_full_text/`: Directory with full-text XML files (when available)
- `Data/combined_corpus.csv`: Unified text corpus for NLP processing

**Estimated Runtime:** 6-12 hours for PMIDs (depends on API rate and availability)

**Libraries:** pandas, Biopython (Bio.Entrez), time, requests

---

#### **02 GPU_full_extract.sh** and **02 GPU_full_extract.py**
**Purpose:** GPU-accelerated NLP extraction of disease targets (genes) from PubMed literature corpus using large language models.

**Architecture:**
- **Language Model:** Google Gemma 2B (lightweight, efficient, runs on single GPU)
- **Hardware:** NVIDIA GPU with ≥8GB VRAM (A100, V100, RTX 3090, or equivalent)
- **Framework:** HuggingFace Transformers with PyTorch
- **Task:** Identify key disease targets (genes implicated in head and neck cancer) from abstracts

**NLP Extraction Pipeline:**

1. **Data Loading:**
   - Loads `Data/head and neck cancer query abstracts.csv` from File 01
   - Filters to publications after year 2000 for relevance

2. **Prompt Engineering:**
   - Task description instructs model to identify disease targets from abstracts
   - Prompt template:
     ```
     You are required to read the given abstract and determine any key disease targets 
     for the cancer mentioned. Respond with the disease target, if there are multiple 
     separate by comma. Explain your answer with the sentence where the information is found.
     
     Key disease targets: [Gene symbols separated by comma]
     Sentence where information is found: [Evidence sentence from abstract]
     ```

3. **Model Inference:**
   - Loads Gemma 2B model to GPU
   - Processes each abstract individually through the model
   - Max output tokens: 5000 per abstract
   - Extracts gene targets and supporting sentences from model output

4. **Post-Processing:**
   - `extractTarget()`: Parses model output to extract gene symbols
   - `extractSentences()`: Extracts evidence sentences where targets were mentioned
   - Adds TARGETS and SENTENCE_WHERE_FOUND columns to working data

**Shell Script (02 GPU_full_extract.sh):**
- Batch job submission script for GPU clusters (SLURM/PBS)
- Sets up environment: loads CUDA, Python, model weights
- Configures GPU settings: device, memory limits
- Launches Python extraction script with error handling
- Monitors GPU utilization and memory

**Python Script Components:**
```python
MODEL_NAME = "google/gemma-2b-it"  # Instruction-tuned variant
YEAR_CUTOFF = 2000  # Only analyze recent literature
MAX_LENGTH = 5000   # Maximum output tokens
DEVICE = "cuda"     # GPU device
```

**Inputs:**
- `Data/head and neck cancer query abstracts.csv`: Abstracts from File 01

**Outputs:**
- DataFrame with extracted targets: `[PMID, TITLE, YEAR, ABSTRACT, TARGETS, SENTENCE_WHERE_FOUND]`
- Saved as CSV for downstream validation in File 03

**Output Format:**
```
PMID, TITLE, YEAR, ABSTRACT, TARGETS, SENTENCE_WHERE_FOUND
12345678, "TP53 mutations in HNSCC", 2010, "...", "tp53, cdkn2a", "TP53 mutations were found in 60% of tumors..."
23456789, "EGFR signaling in HNC", 2015, "...", "egfr, pik3ca", "EGFR overexpression correlated with poor prognosis..."
```

**Computational Requirements:**
- **GPU**: NVIDIA A100 (40GB) or V100 (32GB) recommended
  - Can run on RTX 3090 (24GB) with reduced batch size
  - Minimum: 8GB VRAM
- **RAM**: ≥32GB system memory
- **Storage**: ~50GB for model weights and intermediate files
- **Runtime**: 24-72 hours for 100,000 articles (depends on GPU, batch size, article length)

**Performance Optimization:**
- Mixed precision (FP16): 2x speedup
- Dynamic batching: Maximizes GPU utilization
- Checkpoint saving: Resume from interruption
- Multi-GPU: Can parallelize across multiple GPUs (modify script)

**Error Handling:**
- Automatic retries on CUDA out-of-memory errors (reduces batch size)
- Checkpoint every 1,000 articles
- Logs failed extractions for manual review

**Libraries:** transformers (HuggingFace), torch (PyTorch), pandas, numpy, tqdm, accelerate

---

#### **03 data viewing.ipynb**
**Purpose:** Visualize, clean, and validate literature-extracted disease targets (genes) to create a reference set of genes implicated in head and neck cancer based on published research.

**Key Operations:**

1. **Load Extraction Results:**
   - Imports gene targets extracted by Gemma 2B model (File 02)
   - Loads PPI database for gene name standardization
   - Creates protein ID to gene name mapping dictionaries

2. **Gene Symbol Standardization:**
   - Maps protein IDs to preferred gene names using STRING database
   - Handles gene name aliases and synonyms
   - Converts extracted targets to standardized HUGO gene symbols

3. **Quality Filtering:**
   - Removes non-specific extractions ("no information available", "unknown", "not found")
   - Filters to valid gene symbols present in PPI/DrugBank database
   - Validates gene symbols against protein-coding gene lists
   - Removes generic terms that are not actual gene symbols

4. **Aggregation:**
   - Groups by gene symbol
   - Counts number of articles mentioning each gene
   - Collects PMIDs supporting each gene-disease association
   - Ranks genes by literature support (article count)

**Key Visualizations:**

1. **Extraction Statistics:**
   - Total disease targets (genes) extracted
   - Distribution of article counts per gene
   - Most frequently mentioned genes in HNC literature

2. **Gene Symbol Validation:**
   - Bar charts showing gene extraction frequency

**Data Cleaning Operations:**

1. **Gene Symbol Validation:**
   - Filters to valid HUGO gene symbols
   - Cross-references with STRING PPI database preferred names
   - Handles lowercase/uppercase variations

2. **PMID Aggregation:**
   - Collects all PMIDs mentioning each gene
   - Removes duplicate PMIDs per gene
   - Formats as comma or semicolon-separated strings

**Validation of Main Pipeline Genes:**
- Can be used to validate significant genes from Files 02 (CNV) and 03 (SOM)
- Identifies genes with converging evidence:
  - **Both genomic AND literature**: Highest confidence (mutated AND recognized in literature)

**Inputs:**
- Extracted targets from File 02 (GPU extraction output)
- `Data/Protein-protein interaction data/9606.protein.info.v12.0.txt`: Gene name mapping
- `Data/Protein-protein interaction data/9606.protein.aliases.v12.0.txt`: Gene aliases
- `Results/CNV results/HPV positive CNV top genes.csv`: Genomically significant genes (HPV+)
- `Results/CNV results/HPV negative CNV top genes.csv`: Genomically significant genes (HPV-)
- `Results/SOM Results/HPV positive top genes.csv`: Genomically significant genes (HPV+)
- `Results/SOM Results/HPV negative top genes.csv`: Genomically significant genes (HPV-)

**Outputs:**
- `Results/cleaned_extracted_targets_all_pub_after_2000_GPU_2b_gemma.csv`: Cleaned gene-disease pairs
- `Results/cleaned_extracted_combined_targets_all_pub_after_2000_GPU_2b_gemma.csv`: Aggregated gene targets with article counts

**Use as Validation:**
- Provides independent evidence that genomically identified genes are biologically relevant
- Helps distinguish true disease drivers from passenger mutations
- Prioritizes genes with both genomic and literature support for drug development

**Libraries:** pandas, matplotlib, plotly, numpy, tqdm

---

### Validation Pipeline Workflow Summary

**Complete Pipeline Execution:**

```bash
# Step 1: Extract PMIDs (5 minutes)
cd "Validation pipeline"
bash 00\ extract_pmids.bash

# Step 2: Download abstracts (6-12 hours)
jupyter notebook "01 extract based on pmid.ipynb"
# Run all cells

# Step 3: GPU extraction (24-72 hours, requires GPU)
# Option A: Submit to GPU cluster
sbatch 02\ GPU_full_extract.sh

# Option B: Run locally with GPU
python 02\ GPU_full_extract.py

# Step 4: Clean and export results (30 minutes)
jupyter notebook "03 data viewing.ipynb"
```

**Purpose of Validation:**
This pipeline extracts genes mentioned in HNC literature to create a reference set of literature-validated disease targets. Drug candidates identified through statistical enrichment (Files 02.50, 03.5) are then filtered to retain only those targeting these literature-validated genes.

**What Gets Validated:**
- Genes from File 02 (CNV analysis): Amplified/deleted genes mentioned in HNC literature
- Genes from File 03 (SOM analysis): Mutated genes mentioned in HNC literature
- **Result:** A list of literature-validated genes used for drug filtering in File 05

**Drug Filtering Logic:**
- **Statistical identification first:** Genomic analysis identifies significant genes → Drug enrichment testing identifies significant drug-gene connections
- **Literature filtering second:** Only drugs targeting literature-validated genes are retained in final outputs
- **Direct drugs:** Must target genes that appear in literature
- **Indirect drugs:** Must connect via PPI to risk genes that appear in literature

**When to Run Validation Pipeline:**
- **Before File 05:** To create literature-validated gene list for drug filtering
- **Integration point:** File 05 loads literature results and filters drugs based on gene validation
- **Effect:** Final outputs (Files 06-07) contain only drugs targeting literature-supported genes

**Note:** This validation pipeline extracts gene-disease relationships (not drug-gene interactions). Drug-gene relationships come from DrugBank. Literature validation determines which genes are "validated," and drugs are kept if they target those validated genes.

**Computational Cost:**
- **Without GPU:** Not feasible (NLP extraction too slow with CPU)
- **With GPU (A100):** ~36 hours total compute time
- **With GPU (RTX 3090):** ~60 hours total compute time
- **Cost estimate (cloud GPU):** ~$50-100 on AWS/GCP/Azure

**Alternative Approach (if no GPU available):**
- Use cloud-based GPU instances (AWS EC2 p3.2xlarge, GCP Compute Engine with T4/V100)
- Use university/institutional HPC clusters with GPU nodes
- Skip validation pipeline and rely solely on DrugBank annotations (less comprehensive)

---

### Validation Pipeline Output Integration

**How Literature Validation Filters Drug Candidates:**

**Process Flow:**
1. **Statistical Analysis (Files 02-03):** Identifies genomically significant genes
2. **Drug Enrichment (Files 02.50, 03.5):** Identifies drugs significantly enriched for targeting those genes
3. **Literature Validation:** Extracts genes mentioned in HNC literature from PubMed
4. **Drug Filtering (File 05):** Retains only drugs that target literature-validated genes

**Filtering Criteria:**
- **Direct drugs:** Must target ≥1 gene that appears in literature validation results
- **Indirect drugs:** Must connect via PPI to ≥1 risk gene that appears in literature validation results
- **Effect:** Drugs targeting genomically significant but literature-absent genes are filtered out

**Benefits:**
1. **Increased Confidence:** Final candidates have both statistical significance AND published disease relevance
2. **Evidence Traceability:** PMIDs provide direct links to supporting publications for targeted genes
3. **Translational Focus:** Prioritizes drugs targeting well-established disease genes over novel/uncertain targets

**Example Enhanced Result:**
```
Drug: Afatinib
DrugBank Targets: EGFR, ERBB2, ERBB4
Literature-Validated Targets: EGFR, ERBB2 (87 articles)
Risk Genes Connected via PPI: PIK3CA, AKT1, MTOR (23 articles)
PMIDs: 25316818;26177326;27542767;... (PMIDs for EGFR in HNC literature)
Status: RETAINED (targets literature-validated genes)
```
---

## Data Availability and Reproducibility

### Reproducibility and Result Variability


**Important Note on Pipeline Variability and Purpose:**

Any run through the code or use of the pipeline may lead to small variations in the final resulting genes and drugs. Most strong, top candidate genes and drugs should remain robust to these small variations, and tend to persist across runs. Users should be aware that the exact results can vary slightly due to the stochastic and data-driven nature of the pipeline. This variability can arise from differences in software library versions, hardware, GPU operations, operating systems, and parallel processing, as well as updates to external databases and literature sources.

Despite these sources of minor variability, the main point of the paper and the pipeline stands: to identify drug repurposing candidates through a genomics-based, validated computational workflow. The pipeline is designed to robustly highlight strong candidates, and the biological conclusions regarding repurposing opportunities remain consistent even if some specific gene or drug calls change between runs.


**Random Seeds:**
All stochastic procedures in the pipeline use fixed random seeds (typically set to 42) to ensure reproducibility. This includes:
- Permutation testing in CNV analysis (File 02)
- Permutation testing in SOM analysis (File 03)  
- Empirical permutation testing for drug enrichment (Files 02.50 and 03.5)

**Important Disclaimer on Result Variability:**
While random seeds are set throughout the pipeline, **minor variations in results may still occur between runs** due to:
- **Software library versions**: Different versions of NumPy, SciPy, or statistical libraries may implement algorithms differently
- **Hardware differences**: Floating-point arithmetic can vary slightly across different CPUs/architectures
- **GPU variability**: The validation pipeline's NLP extraction (Gemma 2B model) may produce slightly different results even with seeds due to non-deterministic GPU operations
- **Operating system differences**: Thread scheduling and memory allocation patterns can differ between systems
- **Parallel processing**: When operations are parallelized, the order of execution may vary slightly

**Expected Variability:**
- Statistical significance calls (FDR < 0.05) should remain consistent for genes/drugs well above or below the threshold
- Genes/drugs with FDR values very close to 0.05 may occasionally flip between significant/non-significant across runs
- Exact p-values and FDR values may vary slightly in the 3rd-4th decimal place
- Literature extraction results may vary more substantially due to GPU model non-determinism

**Best Practices for Reproducibility:**
- Document software versions (use `requirements.txt`)
- Record hardware specifications and operating system
- Save intermediate results to enable exact reproduction of downstream analyses
- For critical findings near significance thresholds, consider running multiple iterations to assess stability
- When comparing results across runs, focus on biological conclusions rather than exact numerical values

### Data Availability

**Public Datasets:**
- **TCGA HNSC Data:** Available via NCI Genomic Data Commons (GDC) Data Portal
  - URL: https://portal.gdc.cancer.gov/projects/TCGA-HNSC
  - Access: Open access (no authentication required for most data types)
  - Data Type: Copy number variation (CNV) and somatic mutation (SOM) data
  
- **DrugBank:** Available via DrugBank Online
  - URL: https://go.drugbank.com/releases/latest
  - Access: Free academic account required
  - Format: XML download
  
- **STRING PPI Database:** Available via STRING website
  - URL: https://string-db.org/cgi/download
  - Access: Open access
  - Version: v12.0 (update to latest as needed)
  
- **Nulton et al. Supplementary Data:** Available via PubMed Central
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5392278/
  - Access: Open access
  - Format: CSV files in supplementary materials

- **Gencode Gene Annotations:** Available via Gencode
  - URL: https://www.gencodegenes.org/human/
  - Access: Open access
  - Version: Release 48 or later (GRCh38)

**Processed Results:**
- Final output files are included in this repository under `Results/` directory
- Intermediate files can be regenerated by running the pipeline
- Literature validation results (if generated) are available upon request

---

## Repository Information

**Repository:** [Genomic-Landscape-Based-Drug-Repurposing](https://github.com/pvtanike/Genomic-Landscape-Based-Drug-Repurposing)

**Owner:** pvtanike

**Current Branch:** main

**License:** [Specify license - e.g., MIT, GPL-3.0, or proprietary]

**Citation:** If you use this pipeline in your research, please cite:
```
[Authors]. Genomic Landscape-Based Drug Repurposing for Head and Neck Cancer.
GitHub repository: https://github.com/pvtanike/Genomic-Landscape-Based-Drug-Repurposing (2025)
```

**Contributing:**
- Issues and pull requests are welcome
- For major changes, please open an issue first to discuss proposed changes
- Follow existing code style and documentation standards

---

## Contact and Support

For questions about this pipeline, please contact:

**Wu Lab - University of North Carolina at Chapel Hill:**
- Primary Contact: did@live.unc.edu 
- Secondary Contact: pvtanike@live.unc.edu

**Bug Reports and Feature Requests:**
- Open an issue on the [GitHub repository](https://github.com/pvtanike/Genomic-Landscape-Based-Drug-Repurposing/issues)

**Collaboration Inquiries:**
- Email the contacts above with subject line: "Drug Repurposing Pipeline Collaboration"

**Key References:**
- Nulton TJ, et al. (2017). Analysis of The Cancer Genome Atlas sequencing data reveals novel properties of the human papillomavirus 16 genome in head and neck squamous cell carcinoma. Oncotarget. 8(11):17684-17699. PMCID: PMC5392278

---

## Acknowledgments

**Data Sources:**
- The Cancer Genome Atlas (TCGA) Research Network
- National Cancer Institute (NCI) Genomic Data Commons (GDC)
- DrugBank (University of Alberta and The Metabolomics Innovation Centre)
- STRING Database (CPR and EMBL)
- NCBI PubMed and PubMed Central
- Gencode (EMBL-EBI and University of California, Santa Cruz)

**Software and Tools:**
- Python Scientific Computing Stack (NumPy, SciPy, pandas)
- Plotly and Matplotlib visualization libraries
- NetworkX graph analysis library
- HuggingFace Transformers and PyTorch
- Jupyter Notebook interactive computing environment

**Research Support:**
- Wu Lab, University of North Carolina at Chapel Hill
- [Add funding sources if applicable]

**Special Thanks:**
- Nulton TJ et al. for HPV status validation
- TCGA consortium for data generation and access
- Open-source software community

---

## Version History

**Version 1.1 (March 2026):**
- Updated File 03 (SOM identify key mutation gene.ipynb) to align code implementation with published methodology
- The paper's methods section already described pseudocount adjustment formula: `(1 + successes) / (M + 1)`
- Code originally implemented simpler formula: `successes / M`
- This update corrects the code to match the published methods, ensuring full consistency between paper and implementation
- No change to finalized gene selections or biological conclusions
- Empirical q-values shifted marginally without impact on final results
- All results remain robust and consistent with published findings

**Version 1.0 (December 2025):**
- Initial public release
- Complete main pipeline (Files 00-07)
- Optional validation pipeline
- Comprehensive documentation

**Planned Updates:**
- Integration with additional databases (ClinicalTrials.gov, COSMIC)
- Enhanced visualization tools (interactive web dashboard)
- Machine learning-based prioritization
- Multi-cancer application framework

---

<!-- ## License

[Specify license here - common options:]
- MIT License (permissive, allows commercial use)
- GPL-3.0 (copyleft, requires derivative works to be open source)
- CC BY 4.0 (Creative Commons Attribution for documentation/data)
- Proprietary/Custom (if institutional requirements)

[Example - uncomment and modify as appropriate:]
```
MIT License

Copyright (c) 2025 Wu Lab, University of North Carolina at Chapel Hill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``` -->

---

<!-- ## Frequently Asked Questions (FAQ)

**Q1: How long does it take to run the complete pipeline?**
A: Main pipeline: 14-24 hours. Validation pipeline (optional): 1.5-4 days (primarily GPU extraction).

**Q2: Can I run this pipeline without access to a GPU?**
A: Yes! The main pipeline (Files 00-07) does not require a GPU. Only the optional validation pipeline (literature mining) requires GPU acceleration.

**Q3: What if DrugBank requires a paid license at my institution?**
A: Academic licenses are typically free. Contact DrugBank support for academic access. Alternatively, use DGIdb as a substitute (though less comprehensive).

**Q4: Can I apply this pipeline to other cancer types?**
A: Yes! Replace TCGA HNSC data with data from other cancer types (e.g., BRCA, LUAD). Modify HPV stratification logic as appropriate for the cancer type.

**Q5: How do I choose between direct and indirect drug candidates?**
A: Direct candidates have stronger evidence (drug directly targets mutated gene). Indirect candidates may have broader pathway effects. Consider both for comprehensive therapeutic options.

**Q6: What if my results differ slightly from the published outputs?**
A: Minor differences expected due to: (1) stochastic permutation testing, (2) DrugBank database updates, (3) PubMed content changes. Set random seeds for exact reproducibility.

**Q7: Can I modify the statistical thresholds?**
A: Yes! All thresholds are defined in the notebooks and can be adjusted. However, lowering significance thresholds (e.g., FDR < 0.1) increases false positives.

**Q8: How do I cite this pipeline?**
A: See "Citation" section under "Repository Information" above. Include GitHub repository URL and access date.

**Q9: Is patient-level clinical data included?**
A: Clinical data exploration is included in File 00 (age, gender, diagnosis). TCGA provides additional clinical variables via GDC portal.

**Q10: Can I contribute to this project?**
A: Yes! See "Contributing" section under "Repository Information". Open issues for bugs/features, submit pull requests for enhancements. -->

---

**Last Updated:** March 25, 2026

**README Version:** 1.1

**Pipeline Version:** 1.1

---

*For the latest updates and documentation, visit the [GitHub repository](https://github.com/pvtanike/Genomic-Landscape-Based-Drug-Repurposing).*

---
