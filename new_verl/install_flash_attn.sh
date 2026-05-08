export PATH="/usr/local/cuda-12.9/bin:$PATH"   # 按你机器上 nvcc 实际路径改
export CUDA_HOME="/usr/local/cuda-12.9"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=8

/mnt/public/dingbowen/uv_envs/retu_verl/bin/python -m pip uninstall -y flash-attn
/mnt/public/dingbowen/uv_envs/retu_verl/bin/python -m pip cache purge
/mnt/public/dingbowen/uv_envs/retu_verl/bin/python -m pip install flash-attn \
  --no-cache-dir --no-build-isolation