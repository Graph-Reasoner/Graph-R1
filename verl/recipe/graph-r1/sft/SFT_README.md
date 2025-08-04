# Supervised Fine-Tuning (SFT) Configuration Guide

## Overview

This directory contains configuration files for supervised fine-tuning (SFT) of language models using the VERL framework.

## Files

- `run_sft.sh`: Shell script to launch SFT training
- `sft_trainer.yaml`: Configuration file for SFT training parameters

## Environment Variables

Set these environment variables before running:

```bash
# Number of GPUs to use
export NPROC_PER_NODE=8

# Checkpoint directory
export CHECKPOINT_DIR="/path/to/your/checkpoints"

# Configuration paths
export CONFIG_PATH="/path/to/verl/recipe/graph-r1"
export CONFIG_NAME="sft_trainer.yaml"
```

## Data Requirements

Prepare your training data in parquet format:

- **Training data**: Should contain `prompt` and `response` columns
- **Validation data**: Same format as training data for evaluation

Update the paths in `sft_trainer.yaml`:

```yaml
data:
  train_files: /path/to/your/train_data.parquet
  val_files: /path/to/your/val_data.parquet
```

## Model Configuration

Update the base model path in `sft_trainer.yaml`:

```yaml
model:
  partial_pretrain: /path/to/your/base/model
```

## Usage

```bash
# Set environment variables
export NPROC_PER_NODE=8
export CHECKPOINT_DIR="/your/checkpoint/path"
export CONFIG_PATH="/path/to/verl/recipe/graph-r1"

# Run SFT training
bash run_sft.sh
```

## Key Configuration Parameters

### Data Settings

- `train_batch_size`: 8 (global batch size)
- `micro_batch_size_per_gpu`: 1 (batch size per GPU)
- `max_length`: 128000 (maximum sequence length)
- `truncation`: right (truncation strategy)

### Model Settings

- `strategy`: fsdp (training strategy)
- `enable_gradient_checkpointing`: true (memory optimization)
- `ulysses_sequence_parallel_size`: 4 (sequence parallelism)
- `use_remove_padding`: true (efficiency optimization)

### Training Settings

- `total_epochs`: 2 (number of training epochs)
- `lr`: 1.0e-5 (learning rate)
- `weight_decay`: 0.01
- `warmup_steps_ratio`: 0.1
- `lr_scheduler`: cosine

## Hardware Requirements

- **GPU**: 8 A800 or higher GPUs
- **Memory**: At least 64GB system RAM
- **Storage**: Sufficient space for model checkpoints and logs

## Output Structure

```
CHECKPOINT_DIR/
├── sft/
│   ├── checkpoints/
│   ├── logs/
│   └── wandb/
```

## Advanced Configuration

### Multi-turn Support

Enable multi-turn conversation training:

```yaml
multiturn:
  enable: true
  messages_key: messages
  tools_key: tools
```

### LoRA Training

Enable LoRA for memory-efficient training:

```yaml
lora_rank: 16
lora_alpha: 32
target_modules: all-linear
```

### Memory Optimization

For limited GPU memory:

```yaml
fsdp_config:
  cpu_offload: true
  offload_params: true
```

## Troubleshooting

### Common Issues

1. **Out of Memory**

   - Reduce `micro_batch_size_per_gpu`
   - Enable `cpu_offload`
   - Reduce `max_length`

2. **Data Loading Error**

   - Check file paths in `sft_trainer.yaml`
   - Verify data format (parquet with prompt/response columns)
   - Ensure sufficient disk space

3. **Configuration Error**
   - Verify `CONFIG_PATH` points to correct directory
   - Check YAML syntax in `sft_trainer.yaml`

### Performance Tips

- **Increase throughput**: Increase `micro_batch_size_per_gpu` if memory allows
- **Memory efficiency**: Use gradient checkpointing and remove padding
- **Distributed training**: Adjust `ulysses_sequence_parallel_size` for long sequences

## Monitoring

The training progress can be monitored through:

- Console logs
- Weights & Biases (wandb) dashboard
- Checkpoint files for resume capability
