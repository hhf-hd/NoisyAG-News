# NoisyAG-News: A Benchmark for Instance-Dependent Label Noise in Text Classification

This repository contains the code and dataset associated with our paper:

> **NoisyAG-News: A Benchmark for Instance-Dependent Label Noise in Text Classification**   
---

## 📰 Overview

Real-world label noise is often **instance-dependent** and more complex than the commonly used synthetic noise in NLP benchmarks. This project introduces **NoisyAG-News**, the **first controlled benchmark** for instance-dependent label noise (IDN) in text classification.

We also implement and evaluate several **Learning with Noisy Labels (LNL)** techniques adapted for the NLP domain. This includes:

- Baseline (without correction)
- Label Smoothing (LS)
- Negative Label Smoothing (NLS)
- Co-Teching (CT)
- SelfMix
- Train By exponential decay (expDecay)

---

## 📁 Repository Structure

```
NoisyAG-News/
├── dataset/         # Dataset loading and processing
├── arguments/       # Argument parsing
├── config/          # Configuration files
├── utils/           # Utility functions ( logging, etc.)
├── logs/            # Training and evaluation logs
├── WN/              # Baseline (without noise handling)
├── LS/              # Label Smoothing method
├── NLS/             # Negative Label Smoothing
├── CT/              # Co-Teching
├── selfMix/         # SelfMix noise-robust training
├── expDecay/        # Exponential Decay training strategy
├── main.py          # Main script to run all methods
├── commandSet.sh    # Shell script with example commands
└── run.sh           # Shell script for quick experiment runs
```



---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- PyTorch >= 1.10
- Transformers (by HuggingFace)
- Datasets
- scikit-learn
- tqdm

## 📦 Dataset

We release the **NoisyAG-News** dataset, constructed by collecting **redundant annotations from 60 crowdworkers** on the standard AG-News dataset.
> **Note:** Due to file size limitations, you need to unzip `NoisyAG-News.zip` to get the `NoisyAG-News.pkl` file.

- The dataset includes:
  - Original AG-News labels
  - Multiple annotator labels per instance
  - Derived IDN labels (majority/minority, label distribution)



## 🧪 How to Run

We provide two convenient ways to run experiments:

### 🔹 Option 1: Run Default Method (`WN`) via `run.sh`

```bash
bash run.sh
```

This will execute the following command by default:

```bash
nohup python -u main.py config/WN.yaml > logs/WN/log.log 2>&1 &
```

It runs the baseline method **WN (without noise handling)** using the YAML configuration at `config/WN.yaml`, and logs output to `logs/WN/log.log`.

---

### 🔹 Option 2: Run Specific Methods via `commandSet.sh`

To run a specific method such as Label Smoothing (`LS`) or SelfMix, you can use the corresponding command in `commandSet.sh`.

Example: run Label Smoothing (`LS`) by executing:

```bash
nohup python -u main.py config/LS.yaml > logs/LS/log.log 2>&1 &
```

You may copy the command directly or run the full script to execute all methods one by one (not recommended unless for full benchmarking):

```bash
bash commandSet.sh
```

---

### 🔧 Customize Hyperparameters

All training settings, including model, optimizer, and method-specific parameters, are configured in the corresponding `.yaml` files located in the `config/` directory:

- `config/WN.yaml` – Baseline without noise handling  
- `config/LS.yaml` – Label Smoothing  
- `config/NLS.yaml` – Negative Label Smoothing 
- `config/CT.yaml` – Co-Teching  
- `config/selfMix.yaml` – SelfMix Method  
- `config/expDecay.yaml` – Exponential Decay Training

Feel free to modify these YAML files to tune parameters like learning rate, batch size, training epochs, or method-specific settings.

---

📌 *All outputs including training logs will be saved under the `logs/` directory by default.*


-----

### Part 2: Reproducing Core Paper Analyses (Sec 5)

The `Analysis-Section5/` directory contains the code to reproduce the core findings from Section 5 of our paper.

#### 🔹 1. Reproducing Learning Dynamics & the "Short-Plank Effect" (Sec 5.1)

This analysis reproduces Figure 10, showing the different impacts of real IDN versus synthetic noise.

1.  **Train and Generate Metrics**:
    Run the `multiEval.sh` script. This will call `train_bert_noise_metrics_multieval.py` to train the model with high-frequency evaluation, saving the 10 core diagnostic metrics required for our analysis.
    ```bash
    cd Analysis-Section5
    bash multiEval.sh
    ```
2.  **Visualize the Results**:
    Open and run the `DynamicLearning.ipynb` Jupyter Notebook. It will load the metrics generated in the previous step and plot the performance curves, visually demonstrating the "Short-Plank Effect."

#### 🔹 2. Reproducing Human vs. LLM Bias Analysis (Sec 5.2)

This analysis reproduces Figures 12 and 13, uncovering the mechanisms behind human and LLM annotation errors via TF-IDF.

  - **Analyze Human "Fallback to World" Bias**:
    Open and run `tf-df-human-to-world.ipynb`. This notebook compares correctly and incorrectly labeled samples to extract the trigger keywords responsible for the "fallback" phenomenon.
  - **Analyze LLM "Collapse to Business" Bias**:
    Open and run `tf-df-llm-to-business.ipynb`. This notebook analyzes samples that LLMs misclassify from 'Sci/Tech' to 'Business' to identify the high-signal business terms causing this "collapse."

-----

## 📊 Key Findings

  - **Real-world IDN is far more destructive than synthetic noise**: While Pre-trained Language Models (PLMs) are robust to synthetic noise, their performance **collapses dramatically** on our realistic IDN benchmark due to the **"Short-Plank Effect."**
  - **Human and LLM errors have different origins**: Human errors stem from **high-level semantic abstraction** (a "Fallback" triggered by macro-narrative frames), whereas LLM errors arise from **over-fitting low-level features** (a "Collapse" triggered by a high density of business keywords).
  - **Sample selection is the most effective mitigation strategy**: Among existing LNL techniques, **Sample Selection** methods like **SelfMix** perform best, as they are most capable of identifying and isolating the plausible-but-incorrect samples characteristic of real-world IDN.

-----

## 🧠 Keywords

`instance-dependent noise`, `noisy labels`, `text classification`, `label noise`, `NLP`, `benchmark dataset`, `AG-News`, `robust training`

---

## 🤝 Acknowledgments

We thank the 60 crowdworkers for their annotations, and the NLP community for ongoing discussions around label noise.

---
