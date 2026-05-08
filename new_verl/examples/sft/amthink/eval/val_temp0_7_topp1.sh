cd /RETU/new_verl/recipe/dapo


eval "$(conda shell.bash hook)"
conda activate retu_verl
echo "python=$(which python)"

ray stop
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

tbz=128
mini_bz=$tbz
prompt_bsz=64
rollout_n=8
rollout_temp=0.7
top_p=1
top_k=-1 
max_prompt_length=1024
max_response_length=8192
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 2))
LR=1e-6
kl_coef=0
cnt_ckpt=0
tp=4
state=cnt_from_${cnt_ckpt}step
ckpt=0


SAVE_DIR=./new_verl/sft_ckpts
WORK_DIR=./new_verl
echo "WORK_DIR=$WORK_DIR"
cd $WORK_DIR
export PYTHONPATH="$WORK_DIR/verl:$PYTHONPATH"


wandb_experiment_name=sft-qwen-2.5-7b-sp2-liger # Taking SFT889K + Qwen2.5-7B as an example
 
MODEL_PATH=${SAVE_DIR}/${wandb_experiment_name}/global_step_${ckpt}/huggingface
TRAIN_FILE=./data_zoo/RL62K.parquet
TEST_FILE=./data_zoo/benchmark.parquet
output_dir=${SAVE_DIR}/$wandb_experiment_name
mkdir -p $output_dir
tensorboard_save_dir=${WORK_DIR}/tensorboard/${wandb_experiment_name}
export TENSORBOARD_DIR="$tensorboard_save_dir"
RUNTIME_ENV="${WORK_DIR}/verl/trainer/runtime_env.yaml"
valid_data_path=$MODEL_ROOT


ckpts=(360 720 1080 1440 1800 3600 5400 7200 10800 14080) # Travel the concered ckpts
for ckpt in "${ckpts[@]}"; do
    echo "Begin eval checkpoint: $ckpt"
    valid_data_path_save_path=${valid_data_path}/validation_curate_data_reorg_reward/${ckpt}
    mkdir -p $valid_data_path_save_path
    MODEL_PATH="${valid_data_path}/global_step_${ckpt}/huggingface"
    # check the model path
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model paths not exist: $MODEL_PATH"
        continue
    fi
    python3 -m recipe.dapo.main_dapo \
        algorithm.adv_estimator=grpo \
        algorithm.filter_groups.enable=True \
        algorithm.filter_groups.metric='seq_final_reward' \
        algorithm.filter_groups.win_num_range=['1','2','3','4','5','6','7']\
        reward_model.reward_manager=dapo \
        data.train_files=$TRAIN_FILE \
        data.val_files=$TEST_FILE \
        data.train_batch_size=${tbz} \
        data.prompt_bsz=${prompt_bsz} \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.prompt_key=prompt \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$mini_bz \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.checkpoint.save_contents=['hf_model','model','optimizer','extra'] \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=${rollout_temp} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${rollout_temp} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=${rollout_n} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        custom_reward_function.path=${WORK_DIR}/verl/utils/reward_score/reward_rule.py \
        custom_reward_function.name=my_reward_fn \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name=$wandb_project_name \
        trainer.experiment_name=$wandb_experiment_name \
        trainer.validation_data_dir=$valid_data_path_save_path \
        trainer.rollout_data_dir=$output_dir/rollout_data \
        trainer.default_local_dir=$output_dir/$wandb_experiment_name \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=100 > $valid_data_path_save_path/val_${ckpt}.log 2> $valid_data_path_save_path/val_${ckpt}.progress 
    echo "checkpoint $ckpt evaluation is finished, log in: ${valid_data_path_save_path}/val_${ckpt}.log" 
done
echo "Finish all evaluations"

