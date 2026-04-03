#!/usr/bin/env bash

# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYBIN="${PYBIN:-python}"
SUITE_PY="${ROOT_DIR}/cache_train/run_main_egodex_suite.py"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT_DIR}/scripts/train.sh}"

EGODEX_HF_REPO="${EGODEX_HF_REPO:-haichaozhang/cache}"
DATA_DIR="${DATA_DIR:-hf://datasets/${EGODEX_HF_REPO}/part2}"
CACHE_DIR="${CACHE_DIR:-${DATA_DIR}}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
TEST_MANIFEST="${TEST_MANIFEST:-}"

RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/outputs/eval_main_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEEDS="${SEEDS:-42}"
THINKJEPA_EPOCHS="${THINKJEPA_EPOCHS:-100}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
LR_PRED="${LR_PRED:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
CAMERA_MODE="${CAMERA_MODE:-auto}"
RESUME="${RESUME:-0}"
SKIP_ROLLOUT="${SKIP_ROLLOUT:-1}"

VJEPA2_ROOT="${VJEPA2_ROOT:-${ROOT_DIR}/vjepa2}"
VJEPA2_PARENT="$(dirname -- "${VJEPA2_ROOT}")"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/cache_train:${VJEPA2_ROOT}:${VJEPA2_PARENT}:${PYTHONPATH:-}"
export VJEPA2_ROOT

mkdir -p "${RESULTS_ROOT}"

CMD=(
  "${PYBIN}" "${SUITE_PY}"
  --results_root "${RESULTS_ROOT}"
  --train_script "${TRAIN_SCRIPT}"
  --data_dir "${DATA_DIR}"
  --cache_dir "${CACHE_DIR}"
  --gpu_list "${GPU_LIST}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --seeds ${SEEDS}
  --model_names ThinkJEPA
  --sections main
  --thinkjepa_epochs "${THINKJEPA_EPOCHS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --lr_pred "${LR_PRED}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --test_batch_size "${TEST_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --camera_mode "${CAMERA_MODE}"
)

if [[ -n "${TRAIN_MANIFEST}" ]]; then
  CMD+=(--train_manifest "${TRAIN_MANIFEST}")
fi
if [[ -n "${TEST_MANIFEST}" ]]; then
  CMD+=(--test_manifest "${TEST_MANIFEST}")
fi
if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "${SKIP_ROLLOUT}" == "1" ]]; then
  CMD+=(--skip_rollout)
fi

echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] CACHE_DIR=${CACHE_DIR}"

"${CMD[@]}"
