#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/hard102K_fit_ckpt0.sh"
bash "${SCRIPT_DIR}/hard102K_fit_ckpt360.sh"
bash "${SCRIPT_DIR}/hard102K_fit_ckpt720.sh"
bash "${SCRIPT_DIR}/hard102K_fit_ckpt1080.sh"
bash "${SCRIPT_DIR}/hard102K_fit_ckpt1440.sh"
bash "${SCRIPT_DIR}/hard102K_fit_ckpt1800.sh"
