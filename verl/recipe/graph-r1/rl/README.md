# Graph-R1 Training Script Usage Guide

## Overview

This is a training script for Graph-R1 model that supports Curriculum Learning and GRPO algorithm.

## Environment Variables Configuration

Before using the script, please set the following environment variables:

```bash
# Base model path
export BASE_MODEL_PATH="/path/to/your/base/model"

# Checkpoint save path
export CHECKPOINT_DIR="/path/to/checkpoints"

# Data path
export DATA_BASE_PATH="/path/to/your/data"
```

## Data File Requirements

Prepare the following data files in the `DATA_BASE_PATH` directory:

- Training data: `train_graph_level_{level}_cleaned.parquet`
- Test data: `test_graph_and_math_level_{level}_cleaned.parquet`

Where `{level}` is the curriculum learning level (e.g., 1, 2, 3, etc.).

## Running the Script

```bash
# Basic run
bash 7b_norepeat_stage1.sh

# Run with additional parameters
bash 7b_norepeat_stage1.sh --additional_param value
```

## Main Configuration Description

### Training Parameters

- `PROJECT_NAME`: Project name for wandb logging
- `EXPERIMENT_NAME`: Experiment name
- `MAX_PROMPT_LENGTH`: Maximum prompt length (2048)
- `TOTAL_EPOCHS`: Number of training epochs per level

### Curriculum Learning Configuration

Current configuration for single level training:

- `LEVELS=(1)`: Only train level 1
- `MAX_RESPONSE_LENGTHS=(4096)`: Maximum response length
- `TEMPERATURES=(1.0)`: Sampling temperature
- `SP=(4)`: Sequence parallel size

### Multi-level Training Configuration Example

To enable multi-level curriculum learning, modify the following configuration:

```bash
LEVELS=(1 2 3 4 5)
MAX_RESPONSE_LENGTHS=(4096 5120 6144 7168 8192)
SP=(4 4 4 4 4)
TEMPERATURES=(1.0 1.0 1.0 1.1 1.2)
```

## Hardware Requirements

- GPU: 8 A800 or higher GPUs (adjustable via `CUDA_VISIBLE_DEVICES`)
- Memory: Recommend at least 64GB
- Storage: Ensure sufficient space for checkpoints and logs
