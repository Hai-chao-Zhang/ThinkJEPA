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
TRAIN_PY="${ROOT_DIR}/cache_train/thinker_train.py"

EGODEX_HF_REPO="${EGODEX_HF_REPO:-haichaozhang/cache}"
DATA_DIR="${DATA_DIR:-hf://datasets/${EGODEX_HF_REPO}/part2}"
CACHE_DIR="${CACHE_DIR:-${DATA_DIR}}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
TEST_MANIFEST="${TEST_MANIFEST:-}"

RUN_NAME="${RUN_NAME:-thinkjepa_train_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}}"
RESULTS_MD="${RESULTS_MD:-${OUT_DIR}/test_results.md}"
OUTPUT_MP4="${OUTPUT_MP4:-${OUT_DIR}/vis/pred}"

GPU_LIST="${GPU_LIST:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SPLIT_SEED="${SPLIT_SEED:-42}"
PAST_T="${PAST_T:-32}"
FUTURE_T="${FUTURE_T:-32}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-1}"
PREDICTOR="${PREDICTOR:-thinkjepa}"
BACKBONE="${BACKBONE:-vjepa}"
CAMERA_MODE="${CAMERA_MODE:-auto}"
USE_NPZ_CACHE="${USE_NPZ_CACHE:-1}"
SKIP_VJEPA="${SKIP_VJEPA:-1}"
THINKJEPA_VLM_SOURCE="${THINKJEPA_VLM_SOURCE:-both}"
THINKJEPA_VLM_LAYER_SELECTOR="${THINKJEPA_VLM_LAYER_SELECTOR:-last}"
THINKJEPA_VLM_COND_MODE="${THINKJEPA_VLM_COND_MODE:-film}"
LR="${LR:-1e-3}"
LR_PRED="${LR_PRED:-1e-4}"
MAX_VIS_BATCHES="${MAX_VIS_BATCHES:-1}"
AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
NO_AMP="${NO_AMP:-0}"
SKIP_NONFINITE_LOSS="${SKIP_NONFINITE_LOSS:-0}"

VJEPA2_ROOT="${VJEPA2_ROOT:-${ROOT_DIR}/vjepa2}"
VJEPA2_PARENT="$(dirname -- "${VJEPA2_ROOT}")"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/cache_train:${VJEPA2_ROOT}:${VJEPA2_PARENT}:${PYTHONPATH:-}"
export VJEPA2_ROOT

mkdir -p "${OUT_DIR}" "${OUT_DIR}/vis"

CMD=(
  "${PYBIN}" "${TRAIN_PY}"
  --data_dir "${DATA_DIR}"
  --cache_dir "${CACHE_DIR}"
  --output_dir "${OUT_DIR}"
  --results_md "${RESULTS_MD}"
  --output_mp4 "${OUTPUT_MP4}"
  --epochs "${EPOCHS}"
  --predictor "${PREDICTOR}"
  --backbone "${BACKBONE}"
  --optimize_together_downstream
  --seed "${SEED}"
  --train_ratio "${TRAIN_RATIO}"
  --split_seed "${SPLIT_SEED}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --test_batch_size "${TEST_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --past_T "${PAST_T}"
  --future_T "${FUTURE_T}"
  --temporal_stride "${TEMPORAL_STRIDE}"
  --camera_mode "${CAMERA_MODE}"
  --thinkjepa_vlm_source "${THINKJEPA_VLM_SOURCE}"
  --thinkjepa_vlm_layer_selector "${THINKJEPA_VLM_LAYER_SELECTOR}"
  --thinkjepa_vlm_cond_mode "${THINKJEPA_VLM_COND_MODE}"
  --lr "${LR}"
  --lr_pred "${LR_PRED}"
  --max_visual_batches "${MAX_VIS_BATCHES}"
)

if [[ "${USE_NPZ_CACHE}" == "1" ]]; then
  CMD+=(--use_npz_cache)
fi
if [[ "${SKIP_VJEPA}" == "1" ]]; then
  CMD+=(--skip_vjepa)
fi
if [[ "${AUTO_RESUME}" == "1" ]]; then
  CMD+=(--auto_resume)
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume_ckpt "${RESUME_CKPT}")
fi
if [[ "${NO_AMP}" == "1" ]]; then
  CMD+=(--no_amp)
fi
if [[ "${SKIP_NONFINITE_LOSS}" == "1" ]]; then
  CMD+=(--skip_nonfinite_loss)
fi
if [[ -n "${TRAIN_MANIFEST}" ]]; then
  CMD+=(--train_manifest "${TRAIN_MANIFEST}")
fi
if [[ -n "${TEST_MANIFEST}" ]]; then
  CMD+=(--test_manifest "${TEST_MANIFEST}")
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] CACHE_DIR=${CACHE_DIR}"
echo "[INFO] VJEPA2_ROOT=${VJEPA2_ROOT}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${CMD[@]:1}"
else
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${CMD[@]}"
fi
