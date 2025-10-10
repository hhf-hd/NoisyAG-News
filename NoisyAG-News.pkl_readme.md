Of course. Here is the professional, pure English translation of your dataset README file.

-----

# Guide to `NoisyAG-News.pkl`

This document provides a quick guide to understanding the origin, column meanings, noise construction rules, and basic usage of `NoisyAG-News.pkl`. It is intended to facilitate research and evaluation of instance-dependent noise in text classification.

## 1\. Dataset Overview

  - **Base Task**: 4-class text classification (consistent with AG-News).
  - **Data Source**: A base of 50,000 samples randomly drawn from the balanced AG-News dataset.
  - **Human Annotation**: 60 graduate students were organized into 3 groups (20 annotators each). Each group annotated the entire set of 50,000 samples, resulting in 3 independent human-provided labels for each sample (`human_label_1/2/3`).
  - **Human Annotator Accuracy**: The accuracies of the three annotation groups are approximately 76.8%, 75.9%, and 77.1% (relative to the `ground_truth`).
  - **Objective**: To construct a benchmark with varying levels of instance-dependent noise for evaluating and improving noise-robust text classification methods.

## 2\. File Structure and Column Groups

The following columns are present in `NoisyAG-News.pkl` and are organized into 6 functional groups.

### Group 1: Core Sample Information

  - `sample_index`: The index of the sample.
  - `text`: The original news article text.
  - `ground_truth`: The true label (4-class; consistent with AG-News).

### Group 2: Human Annotations and Aggregated Labels

  - `human_label_1`, `human_label_2`, `human_label_3`: The three independent labels provided by human annotators.
  - `human_best_label`, `human_middle_label`, `human_worst_label`: The aggregated labels based on the three human annotations, following the Best/Mid/Worst rules (see Section 3).

### Group 3: SOTA LLM Annotations and Aggregated Labels

  - `gemini_2.5_pro_label`, `gpt4_label`, `claude4_label`: Single-pass labels from three State-of-the-Art LLMs.
  - `SOTA_llm_best_label`, `SOTA_llm_middle_label`, `SOTA_llm_worst_label`: Aggregated labels based on the three LLM annotations, following the Best/Mid/Worst rules (see Section 3).

### Group 4: Mistral Multi-Model (Family) Experiment

  - `mistral_3b_label`, `mistral_8b_label`, `mistral_latest_label`: Labels from multiple Mistral variants.
  - `mistral_multi_best_label`, `mistral_multi_middle_label`, `mistral_multi_worst_label`: Aggregated labels based on the Mistral variant annotations, following the Best/Mid/Worst rules (see Section 3).

### Group 5: Mistral High-Temperature (Stochastic) Sampling

  - `mistral_8b_temp1.5_run1`, `mistral_8b_temp1.5_run2`, `mistral_8b_temp1.5_run3`: Three independent sample labels from the same model at a higher temperature (temp=1.5).
  - `mistral_8b_best_label`, `mistral_8b_middle_label`, `mistral_8b_worst_label`: Aggregated labels based on the three high-temperature samples, following the Best/Mid/Worst rules (see Section 3).

### Group 6: Purely Synthetic Noise Baselines

  - `noise_instance_best`, `noise_instance_med`, `noise_instance_worst`
  - `noise_sameNTM_best`, `noise_sameNTM_med`, `noise_sameNTM_worst`
  - `noise_single_best`, `noise_single_med`, `noise_single_worst`
  - `noise_uniform_best`, `noise_uniform_med`, `noise_uniform_worst`

Note: The four prefixes above correspond to four different synthetic noise injection strategies (not dependent on human or LLM labels), constructed at the three target noise rates of Best / Mid / Worst (see Section 3). The exact injection details are provided in the implementation.

## 3\. Noise Aggregation Rules (Best / Mid / Worst)

For the purpose of comparison, all "annotation-based" and "purely synthetic" noise sets follow the same three levels of noise severity:

  - **Best (approx. 10% Noise)**
      - **Rule**: If the ground-truth label is present among the three candidate labels, the output is the ground-truth label. Otherwise, a label is randomly selected from the label space.
  - **Mid (approx. 20% Noise)**
      - **Rule**: A majority vote is performed on the three candidate labels. If no majority exists, the final label is determined by the specific implementation strategy (e.g., random choice or priority).
  - **Worst (approx. 38% Noise)**
      - **Rule**: The output is the ground-truth label only if all three candidate labels are identical to the ground-truth. Otherwise, one of the incorrect candidate labels is randomly selected.

The **source of the "candidate labels"** is as follows:

  - **Human Aggregation**: Candidates are `human_label_1/2/3`.
  - **SOTA LLM Aggregation**: Candidates are `gemini_2.5_pro_label`, `gpt4_label`, `claude4_label`.
  - **Mistral Multi-Model Aggregation**: Candidates are `mistral_3b_label`, `mistral_8b_label`, `mistral_latest_label`.
  - **Mistral High-Temperature Aggregation**: Candidates are `mistral_8b_temp1.5_run1/2/3`.
  - **Purely Synthetic Noise**: These do not rely on candidate labels. Noise is injected directly into the `ground_truth` labels according to the target noise rates of Best/Mid/Worst. The different prefixes (`instance`/`sameNTM`/`single`/`uniform`) represent different variants of the injection strategy.

Note: The noise rates mentioned are approximate overall targets; statistical fluctuations may exist at the per-sample or per-class level.

## 4\. Basic Usage Example

```python
import pandas as pd

df = pd.read_pickle('NoisyAG-News.pkl')
print(len(df), 'rows')
print(df.columns.tolist())

# Select core information columns
core_cols = ['sample_index', 'text', 'ground_truth']
print(df[core_cols].head(2))

# Get a specific noise view (e.g., human-aggregated Mid labels)
y_mid = df['human_middle_label']

# Get an LLM-aggregated view (e.g., SOTA LLM Best labels)
y_llm_best = df['SOTA_llm_best_label']

# Get a purely synthetic noise baseline (e.g., uniform Worst)
y_uniform_worst = df['noise_uniform_worst']
```