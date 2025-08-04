#!/bin/bash
set -x

# Number of GPUs to use
nproc_per_node=${NPROC_PER_NODE:-8}

# Create checkpoint directory - please modify according to your setup
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"checkpoints/sft"}
mkdir -p ${CHECKPOINT_DIR}

# Configuration path - please modify according to your setup
CONFIG_PATH=${CONFIG_PATH:-"path_to_verl/verl/recipe/graph-r1"}
CONFIG_NAME=${CONFIG_NAME:-"sft_trainer.yaml"}

# Run training
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} 