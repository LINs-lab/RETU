#!/bin/bash
#SBATCH --job-name=am-sft-llama3_2-3b-s1K
#SBATCH --partition=linlab
#SBATCH --nodes=2
#SBATCH --nodelist=gnho014,gnho043
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=120
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err



## 

##


cd /RETU/new_verl/examples/sft/amthink

eval "$(conda shell.bash hook)"
conda activate retu_verl
echo "python=$(which python)"

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

WORK_DIR=/RETU/new_verl
SAVE_DIR=/RETU/new_verl/sft_ckpts
cd $WORK_DIR
export PYTHONPATH="$WORK_DIR/verl:$PYTHONPATH"
MODEL_ROOT=/RETU/model_zoo
SAVE_DIR=/RETU/new_verl/sft_ckpts
DATA_ROOT=/RETU/data_zoo/sft_data
MODEL_PATH=${MODEL_ROOT}/Llama-3.2-3B       
TRAIN_FILE_1=$DATA_ROOT/s1k_1_1_sft_format.parquet
TEST_FILE=$DATA_ROOT/am_40m_math_intersect_val_199_sft_format.parquet


wandb_project_name=am-long-cot-sft
wandb_experiment_name=sft-llama-3.2-3b-sp2-liger-s1k_1_1
output_dir=${SAVE_DIR}/$wandb_experiment_name
mkdir -p $output_dir
cnt_from=0 # The start step for the continue training 

echo "=== Begin Training ==="
echo "Time: $(date)"
echo "Nodes List: $SLURM_JOB_NODELIST"


srun torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=gnho009:29600 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE_1 \
    data.val_files=[$TEST_FILE_1,$TEST_FILE_2] \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-5 \
    optim.weight_decay=0.0001 \
    optim.warmup_steps_ratio=0.05 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.truncation='right' \
    data.max_length=16384 \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    model.fsdp_config.offload_params=True \
    model.partial_pretrain=$MODEL_PATH \
    model.use_liger=True \
    trainer.default_local_dir=$output_dir/$wandb_experiment_name \
    trainer.project_name=$wandb_project_name \
    trainer.experiment_name=$wandb_experiment_name \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=62 \
    trainer.save_freq=62 \
    trainer.total_epochs=5 \
    trainer.checkpoint.save_contents=["model","optimizer","extra","hf_model"] \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true >${output_dir}/res_cnt_${cnt_from}.log 2>${output_dir}/res_cnt_${cnt_from}.progress