#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/s1K_fit_ckpt0.sh"
bash "${SCRIPT_DIR}/s1K_fit_ckpt62.sh"
bash "${SCRIPT_DIR}/s1K_fit_ckpt124.sh"
bash "${SCRIPT_DIR}/s1K_fit_ckpt186.sh"
bash "${SCRIPT_DIR}/s1K_fit_ckpt248.sh"
bash "${SCRIPT_DIR}/s1K_fit_ckpt310.sh"
