#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# DAPO training on GPUs from the Qwen2.5-7B SFT-1K checkpoint at global_step_62.
#
# This is a single-node style launcher.  It prepares the Python environment,
# resets any existing local Ray runtime, and starts `recipe.dapo.main_dapo` with
# explicit Hydra overrides.
###############################################################################

log() {
  echo -e "[RUN] $(date +'%F %T') $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

###############################################################################
# Python environment
###############################################################################

VENV_PATH="${VENV_PATH:-/mnt/public/dingbowen/uv_envs/retu_verl/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
else
  die "retu_verl virtual environment not found at ${VENV_PATH}"
fi

log "python=$(command -v python)"
python -V

###############################################################################
# Repository and runtime environment
###############################################################################

WORK_DIR="${WORK_DIR:-/mnt/public/dingbowen/RETU/new_verl}"
[[ -d "${WORK_DIR}" ]] || die "WORK_DIR does not exist: ${WORK_DIR}"

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

# Keep CUDA_LAUNCH_BLOCKING disabled for normal multi-GPU runs.  It forces
# synchronous kernels and can make NCCL stalls look worse.  Enable it only for
# narrow single-GPU debugging.
# export CUDA_LAUNCH_BLOCKING=1

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
nccl_timeout_sec="${NCCL_TIMEOUT_SEC:-7200}"

log "WORK_DIR=${WORK_DIR}"

###############################################################################
# Experiment hyperparameters
###############################################################################

prompt_bsz=64
tbz=$((prompt_bsz * 2))
mini_bz=64
rollout_n=8

rollout_temp=0.7
top_p=1.0
val_rollout_temp=0.7
val_top_p=1.0
top_k=-1

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))

LR=1e-6
kl_coef=0.0
adv_estimator=grpo
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

NNODES=1
nnodes="${nnodes:-${NNODES}}"
num_devices="${NUM_DEVICES:-8}"
wandb_project_name="${wandb_project_name:-verl-dapo}"
loss_agg_mode="${loss_agg_mode:-token-mean}"

sft_data="SFT889K"
ckpt=14080
cnt_ckpt=0
state="cnt_from_${cnt_ckpt}step"

###############################################################################
# Performance knobs
###############################################################################

sp_size=1
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=False
gen_tp=2

val_batch_size="${VAL_BATCH_SIZE:-512}"
val_before_train="${VAL_BEFORE_TRAIN:-False}"

###############################################################################
# Data, model, and output paths
###############################################################################


diff_level="win_1_2_3_4_5_6_7"
wandb_experiment_name="dapo_ctrl_tp${gen_tp}_sp_${sp_size}_${diff_level}_grpo_tbz${tbz}_minibz${mini_bz}_promptbsz${prompt_bsz}_rollout${rollout_n}_tmp${val_rollout_temp}_kl_coef${kl_coef}_LR${LR}_CH${clip_ratio_high}_CL${clip_ratio_low}_valTemp${val_rollout_temp}_valTopp${val_top_p}_max_response_length${max_response_length}_sft_ckpt${ckpt}"

DATA_ROOT="${DATA_ROOT:-/mnt/public/dingbowen/RETU/data_zoo}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/public/dingbowen/RETU/model_zoo}"


MODEL_NAME="${SFT_DATA}/global_step_${ckpt}"
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
TRAIN_FILE="${DATA_ROOT}/RL62K.parquet"
TEST_FILE="${DATA_ROOT}/benchmark_data_expanded.parquet"

output_dir="${WORK_DIR}/outputs/Qwen2_5_7B/${wandb_experiment_name}"
tensorboard_save_dir="${WORK_DIR}/tensorboard/${wandb_experiment_name}"
mkdir -p "${output_dir}" "${tensorboard_save_dir}"

export TENSORBOARD_DIR="${tensorboard_save_dir}"
RUNTIME_ENV="${WORK_DIR}/verl/trainer/runtime_env.yaml"

log "MODEL_PATH=${MODEL_PATH}"
log "TRAIN_FILE=${TRAIN_FILE}"
log "TEST_FILE=${TEST_FILE}"
log "output_dir=${output_dir}"
log "RUNTIME_ENV=${RUNTIME_ENV}"
log "nccl_timeout_sec=${nccl_timeout_sec} val_batch_size=${val_batch_size} val_before_train=${val_before_train} gen_tp=${gen_tp}"

###############################################################################
# Training command
###############################################################################

run_dapo_training() {
  local train_args=(
    "algorithm.adv_estimator=${adv_estimator}"
    "algorithm.filter_groups.enable=True"
    "algorithm.filter_groups.max_num_gen_batches=10"
    "algorithm.filter_groups.metric=acc"
    "algorithm.filter_groups.win_num_range=['1','2','3','4','5','6','7']"
    "reward_model.reward_manager=dapo"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    "data.prompt_key=prompt"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.max_prompt_length=${max_prompt_length}"
    "data.max_response_length=${max_response_length}"
    "data.gen_batch_size=${tbz}"
    "data.train_batch_size=${prompt_bsz}"
    "data.val_batch_size=${val_batch_size}"
    "actor_rollout_ref.rollout.n=${rollout_n}"
    "algorithm.use_kl_in_reward=${use_kl_in_reward}"
    "algorithm.kl_ctrl.kl_coef=${kl_coef}"
    "actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}"
    "actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}"
    "actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}"
    "actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}"
    "actor_rollout_ref.actor.clip_ratio_c=10.0"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bz}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.enable_chunked_prefill=True"
    "actor_rollout_ref.actor.entropy_checkpointing=True"
    "actor_rollout_ref.ref.entropy_checkpointing=True"
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
    "actor_rollout_ref.ref.entropy_from_logits_with_chunking=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${offload}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.grad_clip=1.0"
    "actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}"
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model','model','optimizer','extra']"
    "actor_rollout_ref.nccl_timeout=${nccl_timeout_sec}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.70"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.rollout.temperature=${rollout_temp}"
    "actor_rollout_ref.rollout.top_p=${top_p}"
    "actor_rollout_ref.rollout.top_k=${top_k}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${val_rollout_temp}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${offload}"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}"
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
    "custom_reward_function.path=${WORK_DIR}/verl/utils/reward_score/no_format_custom_reward_fn.py"
    "custom_reward_function.name=my_reward_fn"
    "trainer.logger=['console','tensorboard']"
    "trainer.project_name=${wandb_project_name}"
    "trainer.experiment_name=${wandb_experiment_name}"
    "trainer.default_local_dir=${output_dir}/${wandb_experiment_name}"
    "trainer.validation_data_dir=${output_dir}/validation_data"
    "trainer.rollout_data_dir=${output_dir}/rollout_data"
    "trainer.n_gpus_per_node=${num_devices}"
    "trainer.nnodes=${nnodes}"
    "trainer.val_before_train=${val_before_train}"
    "trainer.test_freq=10"
    "trainer.save_freq=10"
    "trainer.total_epochs=4"
  )

  log "Stopping any existing local Ray runtime."
  ray stop || true

  log "Starting DAPO training."
  log "stdout: ${output_dir}/${state}.log"
  log "stderr: ${output_dir}/${state}.progress"
  python3 -m recipe.dapo.main_dapo "${train_args[@]}" >"${output_dir}/${state}.log" 2>"${output_dir}/${state}.progress"
}

run_dapo_training
log "Done. Logs: ${output_dir}/${state}.log ${output_dir}/${state}.progress"
