# OGC Evals – The CitizenQuery Benchmark

## Overview

**OGC Evals** is a robust evaluation framework designed to benchmark Large Language Models (LLMs) on "citizen queries"—open-ended, prompt-response interactions typical of government, civic, and public sector information seeking.

It implements a **FActScore-style** evaluation pipeline, incorporating advanced fact-checking methodologies adapted from **SAFE** (Search-Augmented Factuality Evaluators) and **VeriScore**. By decomposing generated responses into self-contained atomic claims and verifying them against pre-compiled ground-truth reference datasets, the system derives fine-grained metrics for factuality, verbosity, and refusal/abstention behaviors.

This repository implements the evaluation pipeline behind the academic paper:
> **The CitizenQuery Benchmark: A Novel Dataset and Evaluation Pipeline for Measuring LLM Performance in Citizen Query Tasks** (February 2026)

---

## The Standard Pipeline Workflow (`ogc_eval.main`)

To minimize compute costs and token usage, the standard architecture splits evaluation into two distinct phases: **Reference Creation** (`prepare`) and **Model Benchmarking** (`evaluate`).

### Step 1: Precompute Ground Truth Reference (`prepare`)
* **Frequency:** Once per dataset.  
* **Purpose:** The "Ground Truth" answers in your dataset must be decomposed into Atomic Facts before they can be used for evaluation. Since this requires LLM calls (using the AFG module), we precompute and save these facts so they don't need to be regenerated for every model you test.

```bash
# Example: Prepare the sample dataset (Mock mode)
python -m ogc_eval.main prepare --input public_set_1.csv --output public_set_1_reference.csv --mock
```

* **Output:** A **Reference Dataset** containing a `response_facts` column. This file is now the fixed "source of truth" for all future evaluations.

### Step 2: Evaluate Model Outputs (`evaluate`)
* **Frequency:** Many times (Once per model or experiment).  
* **Purpose:** This stage takes your model's generated responses, extracts facts from them, and compares them against the **Reference Dataset** created in Step 1.

```bash
# Example: Evaluate a model using the Reference Dataset (Mock mode)
python -m ogc_eval.main evaluate --input public_set_1_reference.csv --mock
```

---

## The High-Performance Parallel Pipeline (`fast_main.py` & `fast_benchmark.py`)

For large-scale evaluations (>22,000 queries) or high-speed testing situations, we provide a heavily optimized, asynchronous parallel execution pipeline. 

### Why Use the Parallel Pipeline?

#### 1. High Velocity (3x to 7x Speedup)
The standard sequential pipeline evaluates sentences individually, which incurs immense network and token overhead. The parallel pipeline processes data in batches using `ThreadPoolExecutor` and optimized model loaders, accelerating processing times by up to **7.0x**.

#### 2. Resolving the "Over-Decomposition" Precision Penalty
In sequential evaluations, parsing text sentence-by-sentence can sometimes cause the LLM to over-generate highly granular, overlapping, and synonymic claims (often splitting a brief response into **22 to 28 facts**!), depending on sentence structure. This balloons the generated claim count ($\hat{K}$) and artificially penalizes the precision formula:
$$Pr = \frac{Supported}{\hat{K}}$$

By contrast, the **Batch Atomic Fact Generator** parses the generated response *globally* in a single structured call. Guided by retrieved BM25 few-shot demonstrations, it keeps facts concise and dense (averaging **12 to 14 facts**), which generally matches Ground Truth density and yields highly calibrated, representative scores (**0.66 - 0.84** F1@K) that correspond directly with human-annotated baselines.

### Fast Pipeline Commands

#### Step 1: Generate Model Outputs (`fast_benchmark.py`)
Queries provider APIs concurrently (such as Groq or Gemini via LiteLLM) to generate zero-shot and few-shot responses on civic prompts containing demographic metadata.
```bash
python fast_benchmark.py
```
*Saves generated outputs under the `fast_results/` directory.*

#### Step 2: Stitch Ground-Truth Facts (`results_stitch.py`)
Merges the precomputed reference facts from your reference CSV onto the newly generated model outputs.
```bash
python results_stitch.py
```

#### Step 3: Run Abstention Tagging (`fast_main.py abstain`)
Invokes the local sequence-classification PLM on your GPU or CPU to tag whether responses represent disclaimers, refusals, or inability to answer.
```bash
python fast_main.py abstain --input fast_results/kimi_k2_fewshot.csv --device cuda
```

#### Step 4: Batch Verification (`fast_main.py verify`)
Runs parallel API-based fact checks. It bypasses abstained responses, runs batch logical entailment queries in JSON, and implements **resilient sequential verify fallbacks** on format anomalies to guarantee perfect execution without silent score drops.
```bash
python fast_main.py verify --input fast_results/prepped_abstentions/kimi_k2_fewshot_abstentions.csv --reference subset_500_public_set_1_reference.csv --model groq/llama-3.1-8b-instant --api_key YOUR_API_KEY
```

---

## Project Architecture

```
OGC_evals/
├── ogc_eval/
│   ├── main.py         # CLI Entry point (sequential prepare & evaluate)
│   ├── abstention.py   # PLM-based refusal classifier (LibrAI/longformer-action-ro)
│   ├── afg.py          # Atomic Fact Generator
│   ├── afv.py          # Fact Verifier (logical entailment LLM judge)
│   ├── data_loader.py  # Handles schema verification and CSV loading
│   ├── result_writer.py# Generates CSVs and formatted text summary reports
│   ├── logger.py       # Centralized, granular module logging
│   ├── model.py        # Wrapper for OpenAI/HF/Mock models and LiteLLM
│   ├── demons/         # Curated few-shot examples ("demons") for AFG
│   └── prompts/        # System and User prompting text files
│
├── fast_main.py        # Optimized, parallel evaluation pipeline (Batch AFG & AFV)
├── fast_benchmark.py   # Concurrent, multi-threaded LLM response generator
├── results_stitch.py   # Utility to merge reference facts onto model responses
├── evals_visualisation.py # Matplotlib/Seaborn diagnostic dashboard generator
└── requirements.txt    # Python library dependencies
```

---

## Detailed Methodology

### 1. Abstention Detection
Uses a sequence-classification PLM (`LibrAI/longformer-action-ro`) to filter out disclaimers or refusals, preventing "I am unable to answer" statements from being mistakenly penalized or processed as factual claims. Disclaimers (Class 3) and Direct Answers (Class 5) bypass filtration completely. High confidence cutoffs ($AB_{th} = 0.925$) are enforced.

### 2. Atomic Fact Generation (AFG)
Decomposes long-form responses into self-contained atomic propositions.
- **Reference Ground Truth:** Precomputed during the `prepare` stage.
- **Model Output:** Extracted dynamically during evaluation.
- **Mechanism:** Integrates NLTK sentence splitting, Spacy parsing, and an LLM prompted with top-3 similar few-shot "demons" retrieved using BM25 query mapping from `demons/newdemons.json`.

### 3. Automatic Fact Verification (AFV)
An LLM-as-a-judge compares each generated claim against the reference facts to check logical entailment (supported vs. unsupported). It outputs Precision, Recall, and the harmonic mean ($F1@K$), successfully penalizing verbosity ("word salads") and reward-cheating redundancy.

---

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Spacy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Hardware Requirements

- **Abstention Classifier (LibrAI PLM):** Requires ~600MB of storage. Can run on CPU, but uses ~1GB vRAM if `device="cuda"` is specified.
- **Local LLMs:** Running Llama-3-8B locally requires ~16GB-24GB vRAM.
- **API Mode:** If running in API Mode (LiteLLM) or Mock Mode, local GPU/vRAM requirements are **zero**.
