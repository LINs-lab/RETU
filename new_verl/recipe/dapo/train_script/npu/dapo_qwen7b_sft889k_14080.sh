#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# DAPO training from the Qwen2.5-7B SFT checkpoint at global_step_14080.
#
# This script is intended for a managed multi-node Ascend/NPU environment.  Node
# 0 starts the Ray head and launches training after all workers register their
# NPU resources.  Non-zero nodes only join the Ray cluster.
###############################################################################

log() {
  echo -e "[RUN] $(date +'%F %T') $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

status_check() {
  local ret=$?
  if [[ "${ret}" -ne 0 ]]; then
    die "Command failed with return code ${ret}"
  fi
}

###############################################################################
# Repository and runtime environment
###############################################################################

WORK_DIR="${WORK_DIR:-/mnt/public/dingbowen/RETU/new_verl}"
[[ -d "${WORK_DIR}" ]] || die "WORK_DIR does not exist: ${WORK_DIR}"

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

log "WORK_DIR=${WORK_DIR}"
log "python=$(command -v python)"
python -V

###############################################################################
# Experiment hyperparameters
###############################################################################

loss_agg_mode="token-mean"
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

prompt_bsz=64
tbz=$((prompt_bsz * 2))
mini_bz=64
rollout_n=8

rollout_temp=0.7
top_p=1.0
val_rollout_temp=0.7
val_top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout.

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
offload=True
gen_tp=4

###############################################################################
# Data, model, and output paths
###############################################################################



DATA_ROOT="${DATA_ROOT:-/mnt/public/dingbowen/RETU/data_zoo}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/public/dingbowen/RETU/model_zoo}"

SFT_DATA="SFT889K"
MODEL_NAME="${SFT_DATA}/global_step_${ckpt}"
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
TRAIN_FILE="${DATA_ROOT}/RL62K.parquet"
TEST_FILE="${DATA_ROOT}/benchmark.parquet"

diff_level="win_1_2_3_4_5_6_7"
wandb_project_name="${wandb_project_name:-verl-dapo-npu}"
wandb_experiment_name="dapo_ctrl_tp${gen_tp}_sp_${sp_size}_${diff_level}_grpo_tbz${tbz}_minibz${mini_bz}_promptbsz${prompt_bsz}_rollout${rollout_n}_tmp${val_rollout_temp}_kl_coef${kl_coef}_LR${LR}_CH${clip_ratio_high}_CL${clip_ratio_low}_valTemp${val_rollout_temp}_valTopp${val_top_p}_max_response_length${max_response_length}_sft_ckpt${ckpt}"

output_dir="${WORK_DIR}/ckpts/${wandb_experiment_name}"
tensorboard_save_dir="${WORK_DIR}/tensorboard_log/${wandb_experiment_name}"
mkdir -p "${output_dir}" "${tensorboard_save_dir}"

export TENSORBOARD_DIR="${tensorboard_save_dir}"
RUNTIME_ENV="${WORK_DIR}/verl/trainer/runtime_env.yaml"

log "MODEL_PATH=${MODEL_PATH}"
log "TRAIN_FILE=${TRAIN_FILE}"
log "TEST_FILE=${TEST_FILE}"
log "output_dir=${output_dir}"
log "RUNTIME_ENV=${RUNTIME_ENV}"

###############################################################################
# Ascend/CANN and HCCL environment
###############################################################################

export RAY_DEDUP_LOGS=0
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH}"
export SOC_VERSION="${SOC_VERSION:-ASCEND910B3}"
export COMPILE_CUSTOM_KERNELS="${COMPILE_CUSTOM_KERNELS:-1}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
  log "Ascend toolkit set_env.sh not found; continuing with existing environment."
fi

if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/nnal/atb/set_env.sh
else
  log "ATB set_env.sh not found; continuing with existing environment."
fi

export HCCL_EVENT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_WHITELIST_DISABLE=1
export HCCL_SEND_CQ_DEPTH=16384
export HCCL_DETERMINISTIC=TRUE
export LCCL_DETERMINISTIC=TRUE

###############################################################################
# Cluster metadata
###############################################################################

node_id="${VC_TASK_INDEX:-0}"
master_ip="${MASTER_IP:-${MA_VJ_NAME:-}-${MA_TASK_NAME:-}-0.${MA_VJ_NAME:-}}"
num_npu="${MA_NUM_GPUS:-8}"
nnodes="${MA_NUM_HOSTS:-1}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"
RAY_PORT="${RAY_PORT:-6344}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8260}"

log "node_id=${node_id} master_ip=${master_ip} num_npu=${num_npu} nnodes=${nnodes}"

###############################################################################
# Ray runtime setup
###############################################################################

prepare_ray_tmp() {
  rm -rf /tmp/ray || true
  mkdir -p /cache/ray /cache/record_data /cache/ray_tmp
  ln -s /cache/ray /tmp/ray 2>/dev/null || true
  chmod a+w -R /cache/record_data /cache/ray_tmp || true
  export TMPDIR=/cache/ray_tmp
}

compute_ray_object_store_mem() {
  local mem_bytes shm_bytes target_mem target_shm object_store_mem
  mem_bytes="$(awk '/MemTotal/{printf "%.0f", $2*1024}' /proc/meminfo)"
  shm_bytes="$(df -B1 /dev/shm | awk 'NR==2{print $2}')"
  target_mem=$((mem_bytes * 35 / 100))
  target_shm=$((shm_bytes * 80 / 100))
  object_store_mem=$((target_mem < target_shm ? target_mem : target_shm))

  if [[ "${object_store_mem}" -lt $((512 * 1024 * 1024)) ]]; then
    object_store_mem=$((512 * 1024 * 1024))
  fi

  echo "${object_store_mem}"
}

get_registered_node_count() {
  local status total_npu total_npu_int
  status="$(ray status 2>/dev/null || true)"
  total_npu="$(printf "%s" "${status}" | grep -oE '/[0-9]+(\.[0-9]+)?[[:space:]]*NPU' | head -n 1 | sed -E 's#^/([0-9]+(\.[0-9]+)?).*#\1#' || true)"

  if [[ -z "${total_npu}" ]]; then
    echo 0
    return
  fi

  total_npu_int="${total_npu%%.*}"
  if [[ -z "${total_npu_int}" || "${num_npu}" -eq 0 ]]; then
    echo 0
    return
  fi

  echo $((total_npu_int / num_npu))
}

prepare_ray_tmp
ray_object_store_mem="$(compute_ray_object_store_mem)"
log "ray_object_store_mem=${ray_object_store_mem} bytes"

export RAY_PYTHON_WORKER_COMMAND="$(command -v python) -u"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"
export RAY_USAGE_STATS_ENABLED=0

###############################################################################
# Training command
###############################################################################

run_dapo_training() {
  local train_args=(
    "algorithm.adv_estimator=${adv_estimator}"
    "algorithm.filter_groups.enable=${enable_filter_groups}"
    "algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}"
    "algorithm.filter_groups.metric=${filter_groups_metric}"
    "algorithm.filter_groups.win_num_range=['1','2','3','4','5','6','7']"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    "data.prompt_key=prompt"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.max_prompt_length=${max_prompt_length}"
    "data.max_response_length=${max_response_length}"
    "data.gen_batch_size=${tbz}"
    "data.train_batch_size=${prompt_bsz}"
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
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.80"
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
    "reward_model.reward_manager=dapo"
    "custom_reward_function.path=${WORK_DIR}/verl/utils/reward_score/no_format_custom_reward_fn.py"
    "custom_reward_function.name=my_reward_fn"
    "trainer.logger=['console','tensorboard']"
    "trainer.project_name=${wandb_project_name}"
    "trainer.experiment_name=${wandb_experiment_name}"
    "trainer.default_local_dir=${output_dir}/${wandb_experiment_name}"
    "trainer.validation_data_dir=${output_dir}/validation_data"
    "trainer.rollout_data_dir=${output_dir}/rollout_data"
    "trainer.n_gpus_per_node=${num_npu}"
    "trainer.nnodes=${nnodes}"
    "trainer.val_before_train=True"
    "trainer.test_freq=10"
    "trainer.save_freq=10"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.ref.use_torch_compile=False"
    "trainer.device=npu"
    "trainer.total_epochs=1"
  )

  log "Starting DAPO training. Log file: ${output_dir}/${state}.log"
  python3 -m recipe.dapo.main_dapo "${train_args[@]}" 2>&1 | tee "${output_dir}/${state}.log"
}

###############################################################################
# Ray cluster orchestration
###############################################################################

start_head_and_train() {
  log "Starting Ray head on port ${RAY_PORT}"
  ray start \
    --head \
    --port "${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --object-store-memory="${ray_object_store_mem}" \
    --resources="{\"NPU\": ${num_npu}}"

  local attempt registered_nodes
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    registered_nodes="$(get_registered_node_count)"

    if [[ "${registered_nodes}" -eq "${nnodes}" ]]; then
      log "Ray cluster is ready: ${registered_nodes}/${nnodes} nodes registered."
      run_dapo_training
      return
    fi

    log "[${attempt}/${MAX_ATTEMPTS}] Waiting for Ray nodes: ${registered_nodes}/${nnodes} ready."
    sleep 5
  done

  die "Reached MAX_ATTEMPTS=${MAX_ATTEMPTS} while waiting for Ray cluster readiness."
}

join_head() {
  local attempt
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    log "[${attempt}/${MAX_ATTEMPTS}] Joining Ray head at ${master_ip}:${RAY_PORT}"

    if ray start --address="${master_ip}:${RAY_PORT}" --resources="{\"NPU\": ${num_npu}}" && ray status; then
      log "Successfully connected to the Ray cluster."
      return
    fi

    sleep 5
  done

  die "Reached MAX_ATTEMPTS=${MAX_ATTEMPTS} while joining the Ray cluster."
}

if [[ "${node_id}" == "0" ]]; then
  start_head_and_train
else
  join_head
fi

status_check
