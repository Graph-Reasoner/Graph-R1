<div align="center">

# Graph-R1

<div>
   🧠 A Small <strong>R</strong>easoning Language Model for Complex <strong>Graph</strong> Problems with 10k+ tokens long COT <strong>R</strong>FT and <strong>R</strong>einforcement Learning  🔗
</div>

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-under%20review-red)](https://github.com/Graph-Reasoner/Graph-R1) [![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/HKUST-DSAIL) [![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/HKUST-DSAIL/Graph-R1-SFT-30K) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Paper Abstract

Reasoning Language Models have achieved impressive success across various complex reasoning tasks, but their capabilities in solving complex graph problems remain less explored, especially for small language models. To bridge this gap, we present Graph-R1, a small reasoning language model specifically designed to tackle complex graph problems. Our approach integrates cold-start Rejection Sampling Supervised Fine-Tuning (RFT) and Reinforcement Learning (RL) framework with fine-grained rewards and curriculum learning, enhancing both performance and efficiency.
This repository contains the reproduction code for **Graph-R1**, a small reasoning language model specifically designed to tackle complex graph problems through integrated supervised fine-tuning and reinforcement learning.

## Models and Resources

### 🎯 Models

| Model                                                             | Base                   | Parameters | Description          |
| ----------------------------------------------------------------- | ---------------------- | ---------- | -------------------- |
| [Graph-R1-7B](https://huggingface.co/HKUST-DSAIL/Graph-R1-7B)     | Qwen2.5-7B-Instruct-1M | 7.62B      | Main reasoning model |
| [Graph-R1-1.5B](https://huggingface.co/HKUST-DSAIL/Graph-R1-1.5B) | Qwen2.5-1.5B           | 1.78B      | Lightweight version  |

### 📊 Training Data

| Dataset                                                                          | Size        | Description                     |
| -------------------------------------------------------------------------------- | ----------- | ------------------------------- |
| [Graph-R1-SFT-30K](https://huggingface.co/datasets/HKUST-DSAIL/Graph-R1-SFT-30K) | 30K samples | Ultra-long CoT reasoning traces |

### 🏢 Organization

[**HKUST-DSAIL**](https://huggingface.co/HKUST-DSAIL) - Data Science & AI Lab at HKUST

</div>

## 📈 Performance Overview

### Complex Graph Problems (Small Scale)

| Model             | TSP Acc. | GED Acc. | MCP Acc. | Average  |
| ----------------- | -------- | -------- | -------- | -------- |
| QwQ-32B           | 89.4     | **70.2**     | 96.2     | 85.3 |
| Claude-3.5-Sonnet | 45.4     | 37.2     | 62.2     | 48.3     |
| GPT-4o            | 44.2     | 32.6     | 62.4     | 46.4     |
| **Graph-R1-7B**   | **91.8** | 68.2 | **97.0** | **85.7** |
| **Graph-R1-1.5B** | 44.6     | 28.4     | 53.0     | 42.0     |

### Complex Graph Problems (Large Scale)

| Model             | TSP Acc. | GED Acc. | MCP Acc. | Average  |
| ----------------- | -------- | -------- | -------- | -------- |
| QwQ-32B           | 7.6      | **7.4**  | **64.6** | 26.5 |
| Claude-3.5-Sonnet | 2.8      | 3.4      | 19.0     | 8.4      |
| GPT-4o            | 0.8      | 2.4      | 15.2     | 6.1      |
| **Graph-R1-7B**   | **11.2** | 6.2      | 63.6     | **27.0** |
| **Graph-R1-1.5B** | 1.8      | 3.4      | 21.4     | 8.9      |

### Cross-Domain Transferability

| Model           | AIME25 (pass@64) | AIME24 (pass@64) | Math500 (pass@8) | Avg Improvement |
| --------------- | ---------------- | ---------------- | ---------------- | --------------- |
| Base Model      | 26.7             | 33.3             | 86.2             | -               |
| **RFT Model**   | 33.3 (+24.7%)    | 40.0 (+20.1%)    | 87.0 (+0.9%)     | **+17.9%**      |
| **Graph-R1-7B** | 33.3 (+24.7%)    | 30.0 (-9.9%)     | 88.0 (+2.1%)     | **+7.6%**       |

## Quick Start on Training

### Data Preparation

Before starting training, you need to download and prepare the source data files. These files are large and have been excluded from the repository to keep it lightweight.

#### Download Source Data

```bash
# Create the source directory
mkdir -p verl/verl/utils/reward_score/tasks/source

# Download the required data files
cd verl/verl/utils/reward_score/tasks/source

# Method 1: Using gdown (recommended)
pip install gdown
gdown 1meKois5K3SVfTlEhn1FQNfXzq2S6NFvq

# Method 2: Using Google Drive Link
# access https://drive.google.com/file/d/1meKois5K3SVfTlEhn1FQNfXzq2S6NFvq/view?usp=sharing

# Extract the compressed data
tar -xzf source.tar.gz

# Clean up the compressed file (optional)
rm source.tar.gz
```

### Training Environment Setup

Please Follow Verl Setting Shown in `verl/README.md`.

### Two-Stage Training Pipeline

#### Stage 1: Supervised Fine-Tuning

We recommend VERL's SFT framework for **3x speedup** over the original `360-llama-factory` approach.

```bash
cd verl/recipe/graph-r1/sft/
bash run_sft.sh
```

<div align="center">

#### Stage 2: Reinforcement Learning with Curriculum Learning

</div>

```bash
# Single-level training
bash 7b_norepeat_stage1.sh

# Full curriculum learning (5 levels)
bash curriculum_learning_full.sh
```

## Repository Structure

```
Graph-R1/
├── README.md
├── requirements.txt
├── verl/
│   └── recipe/
│       └── graph-r1/
│           ├── sft/                         # Stage 1: SFT Training
│           │   ├── run_sft.sh              #   Training script
│           │   ├── sft_trainer.yaml        #   Configuration
│           │   └── SFT_README.md           #   Documentation
│           ├── 7b_norepeat_stage1.sh       # Stage 2: Single-level RL
│           ├── curriculum_learning_full.sh # Stage 2: Full curriculum
│           ├── README.md                   # RL training guide
│           └── CURRICULUM_README.md        # Curriculum guide
├── eval/                                   # TODO: Evaluation scripts
├── data/                                   # TODO: Data processing
└── docs/                                   # Documentation
```

## Hardware Requirements

### Recommended Configuration

| Component | Specification |
| --------- | ------------- |
| **GPU**   | 8x A800 80GB  |

## Citation

If you use this code or models in your research, please cite:

```bibtex
@misc{graph-r1-2025,
  title={Graph-R1: A Small Reasoning Language Model for Complex Graph Problems},
  author={HKUST-DSAIL},
  year={2025},
  url={https://github.com/Graph-Reasoner/Graph-R1},
  note={arXiv preprint under review}
}
```
