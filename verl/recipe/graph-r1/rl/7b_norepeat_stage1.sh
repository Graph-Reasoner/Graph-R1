#!/bin/bash
set -x
# export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DISABLE_MEMORY_MONITOR=1
export HYDRA_FULL_ERROR=1
# Set environment variables before starting Ray to prevent overflow
# export RAY_TEMP_DIR=/tmp/ray
# ln -s /tmp/ray /tmp/ray

# Path configuration - please modify according to your actual setup
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/path/to/your/base/model"}

# Training parameters
PROJECT_NAME='verl_grpo_graph_curriculum' # Pure graph training
EXPERIMENT_NAME='7b_norepeat_stage1' 
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/path/to/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
MAX_PROMPT_LENGTH=2048
# TEMPERATURE=0.7
# actor_rollout_ref.rollout.enforce_eager=False \
# actor_rollout_ref.rollout.free_cache_engine=False \
# Complete BASE_CONFIG (preserve all original parameters)
BASE_CONFIG="\
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    "

# Curriculum learning configuration
# LEVELS=(1 2 3 4 5)
# RESUME_MODE="disable"  # Initial mode
# TOTAL_EPOCHS=1        # Train 1 epoch per level
# MAX_RESPONSE_LENGTHS=(4096 5120 6144 7168 8192)
# SP=(4 4 4 4 4)
# TEMPERATURES=(1.0 1.0 1.0 1.1 1.2)
LEVELS=(1)
RESUME_MODE="disable"  # Initial mode
TOTAL_EPOCHS=1        # Train 1 epoch per level
MAX_RESPONSE_LENGTHS=(4096)
SP=(4)
TEMPERATURES=(1.0)

# Data path configuration - please modify according to your actual setup
DATA_BASE_PATH=${DATA_BASE_PATH:-"/path/to/your/data"}

for i in "${!LEVELS[@]}"; do
    level=${LEVELS[$i]}
    max_response_length=${MAX_RESPONSE_LENGTHS[$i]}
    temperature=${TEMPERATURES[$i]}
    sp_size=${SP[$i]}
    
    # Data file paths - please modify according to your actual setup
    TRAIN_DATA="${DATA_BASE_PATH}/train_graph_level_${level}_cleaned.parquet"
    TEST_DATA="${DATA_BASE_PATH}/test_graph_and_math_level_${level}_cleaned.parquet"

    # Parameter validation
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "Error: Training data file not found: $TRAIN_DATA"
        echo "Please set DATA_BASE_PATH environment variable to the correct data directory"
        exit 1
    fi

    COMMAND="python3 -m verl.trainer.main_ppo \
        ${BASE_CONFIG} \
        data.train_files=['$TRAIN_DATA'] \
        data.val_files=['$TEST_DATA'] \
        data.max_response_length=${max_response_length} \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_response_length + MAX_PROMPT_LENGTH)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        trainer.resume_mode=$RESUME_MODE \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        trainer.total_epochs=$TOTAL_EPOCHS $@"

    echo "======= STARTING LEVEL $level ======="
    echo "Training Data: $TRAIN_DATA"
    $COMMAND 2>&1 | tee ${EXPERIMENT_NAME}_L${level}_$(date +%Y%m%d%H%M).log

    # Update resume mode and clear cache
    RESUME_MODE="auto"
    # nvidia-smi --gpu-reset
    TOTAL_EPOCHS=$((TOTAL_EPOCHS + 1))
done