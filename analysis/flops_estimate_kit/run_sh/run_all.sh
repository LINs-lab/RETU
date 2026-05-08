#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep all generated artifacts under one configurable root.  Child scripts read
# the same environment variable, so users can redirect a full reproduction run
# with: FIT_OUTPUT_ROOT=/tmp/flops_fit_result bash run_sh/run_all.sh
export FIT_OUTPUT_ROOT="${FIT_OUTPUT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/fit_result}"
mkdir -p "${FIT_OUTPUT_ROOT}"

{
  echo "[run_all] FIT_OUTPUT_ROOT=${FIT_OUTPUT_ROOT}"
  bash "${SCRIPT_DIR}/SFT889K/run_all.sh"
  bash "${SCRIPT_DIR}/easy102K/run_all.sh"
  bash "${SCRIPT_DIR}/hard102K/run_all.sh"
  bash "${SCRIPT_DIR}/s1K/run_all.sh"
  bash "${SCRIPT_DIR}/uniform102K/run_all.sh"
} 2>&1 | tee "${FIT_OUTPUT_ROOT}/run_all.log"
