#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/SFT889K_fit_ckpt0.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt360.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt720.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt1080.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt1440.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt1800.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt3600.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt5400.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt7200.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt9000.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt10800.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt12600.sh"
bash "${SCRIPT_DIR}/SFT889K_fit_ckpt14080.sh"
