#!/usr/bin/env bash
# Auto-generated from analysis/flops_estimate_kit/fit_res/uniform102K/ckpt360/best_hparams.json.
# Reproduce the selected scaling curve without hyperparameter search.
set -euo pipefail

KIT_ROOT="${KIT_ROOT:-/mnt/public/dingbowen/RETU/analysis/flops_estimate_kit}"
OUTPUT_ROOT="${FIT_OUTPUT_ROOT:-${KIT_ROOT}/fit_result}"
cd "${KIT_ROOT}"

output_scene="uniform102K"
sft_scene="uniform102K"
rl_start_point=360
max_step="auto"
model_name="qwen_2_5_7b"
rl_method="dapo"
sft_steps=(0 360 720 1080 1440 1800)

sft_dict_file_name="uniform102K_flops2valPerform_sft_dict.pkl"
rl_dict_file_name="${sft_scene}_step2flops2valPerform_dapo_ckpt${rl_start_point}.pkl"

out_dir="${OUTPUT_ROOT}/${output_scene}/ckpt${rl_start_point}"
mkdir -p "${out_dir}"
figure_save_path="${out_dir}/qwen_2_5_7b_sft_uniform102K_dapo_360_max_step_auto_fit70_hparam_best.png"
metrics_json_out="${out_dir}/metrics.json"
log_path="${out_dir}/run.log"

echo "[fit] output_scene=${output_scene} sft_scene=${sft_scene} ckpt=${rl_start_point}"
echo "[fit] figure=${figure_save_path}"

python3 fitting_curves.py \
  --sft_parquets_dir "" \
  --rl_parquets_dir "" \
  --rl_progress_dir "" \
  --sft_dict_file_name "${sft_dict_file_name}" \
  --rl_dict_file_name "${rl_dict_file_name}" \
  --figure_save_path "${figure_save_path}" \
  --metrics_json_out "${metrics_json_out}" \
  --model_name "${model_name}" \
  --sft_scene "${sft_scene}" \
  --rl_method "${rl_method}" \
  --max_step "${max_step}" \
  --sft_steps "${sft_steps[@]}" \
  --rl_start_points "${rl_start_point}" \
  --fit_points_num 70 \
  --lts_alpha 0.75 \
  --outlier_threshold 2.5 \
  --val_most 100 \
  --gap_weight 0.2 \
  2>&1 | tee "${log_path}"
