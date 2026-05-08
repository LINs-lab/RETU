# Environment Configuration

## 1. GPU Environment for SFT
These instructions target the **`retu_verl`** environment used for `./new_verl` (verl 0.5.0.dev).
Even though we only did SFT in this environment, you can certainly do RL in this env.

| Component | Version |
|-----------|---------|
| Python | **3.11.15** |
| flash-attn (after build step) | **2.8.3** |

Dependencies are pinned in [`new_verl/requirements.txt`](new_verl/requirements.txt). **flash-attn** is installed separately with [`new_verl/install_flash_attn.sh`](new_verl/install_flash_attn.sh) so it compiles against your local CUDA + PyTorch. We provide two options, **uv** and **Conda**, for setting up the environment.


**Option A: `uv` virtualenv (example layout)**

```bash
pip install uv
```


Example paths from this workspace: venv at `/mnt/public/dingbowen/uv_envs/retu_verl`, project at `RETU/new_verl`. Adjust `RETU_ROOT` / `VENV_DIR` for your machine.

```bash
export RETU_ROOT=/mnt/public/dingbowen/RETU
export VENV_DIR=/mnt/public/dingbowen/uv_envs/retu_verl

uv venv "$VENV_DIR" --python 3.11.15
source "$VENV_DIR/bin/activate"

uv pip install -r "$RETU_ROOT/new_verl/requirements.txt"
```

Then edit [`new_verl/install_flash_attn.sh`](new_verl/install_flash_attn.sh): set `PATH` / `CUDA_HOME` to your CUDA install (must match the `nvcc` you build against), and ensure the script invokes **this** environment’s Python (replace the hard-coded interpreter path if needed). Run:

```bash
bash "$RETU_ROOT/new_verl/install_flash_attn.sh"
```

The script uninstalls any existing wheel, purges pip cache, then installs **flash-attn** with `--no-build-isolation` (you need a working toolchain: `ninja`, compatible compiler, and GPU CUDA matching `CUDA_HOME`).

**Option B: Conda**

```bash
export RETU_ROOT=/mnt/public/dingbowen/RETU

conda create -n retu_verl python=3.11.15 -y
conda activate retu_verl

pip install -r "$RETU_ROOT/new_verl/requirements.txt"
```

Update [`new_verl/install_flash_attn.sh`](new_verl/install_flash_attn.sh) so the `python -m pip` lines use `$CONDA_PREFIX/bin/python` (or your venv path), fix `CUDA_HOME` / `PATH`, then:

```bash
bash "$RETU_ROOT/new_verl/install_flash_attn.sh"
```

**Verify:**

```bash
python --version          # Python 3.11.15
python -m pip show flash-attn   # Version 2.8.3 after install_flash_attn.sh
```


## 2. GPU Environment for Paradigm Comparision
For the **Paradigms Comparison** experiments, we utilized the [Unify-Post-Training](https://github.com/TsinghuaC3I/Unify-Post-Training) codebase, which contains the implementations for UPT, LUFFY, and SRFT and is based on an earlier version of `verl`. To ensure a fair and consistent evaluation, we also implemented baseline RL algorithms (GRPO and DAPO $_d$) within this same framework.


## 3. NPU Environment for RL
For the RL-phase experiments in Section 6.2 of the paper, we use Ascend 910B NPUs (A2-class hardware) for training, with the code in `./new_verl`. The key
point is to keep **CANN**, **torch**, **torch_npu**, **vLLM**, and
**vLLM-Ascend** mutually compatible. 

We recommend starting from the provided Dockerfile, which uses the 910B image
`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.2.rc1-910b-ubuntu22.04-py3.11`,
then installs `torch==2.5.1`, `torch_npu==2.5.1`, `torchvision==0.20.1`,
`vllm==0.9.1`, `vllm-ascend==0.9.1`, MindSpeed, Megatron-LM, and `verl` with
`requirements-npu.txt`.
[`new_verl/docker/Dockerfile.ascend_8.2.rc1_a2`](new_verl/docker/Dockerfile.ascend_8.2.rc1_a2).

```bash
cd RETU/new_verl/docker

# Build a Docker image named verl-ascend.
docker build -f Dockerfile.ascend_8.2.rc1_a2 -t verl-ascend .
```
After entering the container, put the following
in your job script or in `~/.bashrc`. These are the environment variables needed
by torch-npu, vLLM-Ascend, and HCCL:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH
export SOC_VERSION=ASCEND910B3
export COMPILE_CUSTOM_KERNELS=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# The exact subdirectory depends on the CPU architecture.
if [ "$(uname -m)" = "aarch64" ]; then
  export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH
else
  export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/devlib/linux/x86_64:$LD_LIBRARY_PATH
fi

export HCCL_EVENT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_WHITELIST_DISABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
```

Verify that CANN and torch-npu are visible before launching RL:

```bash
npu-smi info
python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu_count:", torch.npu.device_count())
PY
```

**Reference**:
1. https://verl.readthedocs.io/en/v0.5.x/ascend_tutorial/ascend_quick_start.html