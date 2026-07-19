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

DATA_DIR="${DATA_DIR:-}"
CACHE_DIR="${CACHE_DIR:-}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
TEST_MANIFEST="${TEST_MANIFEST:-}"
VAL_MANIFEST="${VAL_MANIFEST:-${TEST_MANIFEST}}"
SPLIT_META="${SPLIT_META:-}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-best}"

if [[ -z "${DATA_DIR}" || -z "${CACHE_DIR}" ]]; then
  echo "[ERROR] Set DATA_DIR to the local supervision root and CACHE_DIR to the separate causal feature-cache root." >&2
  exit 2
fi
if [[ "${DATA_DIR}" == "${CACHE_DIR}" ]]; then
  echo "[ERROR] DATA_DIR and CACHE_DIR must be separate; refusing a legacy mixed-cache root." >&2
  exit 2
fi

CACHE_PARENT="$(dirname -- "${CACHE_DIR}")"
CACHE_VALIDATION_REPORT="${CACHE_VALIDATION_REPORT:-${CACHE_PARENT}/full_validation.json}"
CACHE_VALIDATION_MARKER="${CACHE_VALIDATION_MARKER:-${CACHE_PARENT}/VALIDATED_SUCCESS}"
if [[ ! -f "${CACHE_VALIDATION_REPORT}" || ! -f "${CACHE_VALIDATION_MARKER}" ]]; then
  echo "[ERROR] Cache validation is incomplete: report=${CACHE_VALIDATION_REPORT} marker=${CACHE_VALIDATION_MARKER}" >&2
  exit 2
fi

HF_HOME="${HF_HOME:-${CACHE_PARENT}/hf_home}"
HF_HUB_CACHE="${HF_HOME}/hub"
HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE

RUN_NAME="${RUN_NAME:-thinkjepa_train_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/${RUN_NAME}}"
RESULTS_MD="${RESULTS_MD:-${OUT_DIR}/validation_results.md}"
OUTPUT_MP4="${OUTPUT_MP4:-${OUT_DIR}/vis/pred}"

GPU_LIST="${GPU_LIST:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
PAST_T="${PAST_T:-32}"
FUTURE_T="${FUTURE_T:-32}"
TEMPORAL_STRIDE="${TEMPORAL_STRIDE:-1}"
BACKBONE="${BACKBONE:-vjepa}"
CAMERA_MODE="${CAMERA_MODE:-egodex}"
LR="${LR:-1e-3}"
LR_PRED="${LR_PRED:-1e-4}"
LAMBDA_TASK="${LAMBDA_TASK:-1.0}"
LAMBDA_PRED="${LAMBDA_PRED:-1.0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
REF_MODE="${REF_MODE:-none}"
VLM_PAD_OLD_TO="${VLM_PAD_OLD_TO:-1280}"
VLM_PAD_NEW_TO="${VLM_PAD_NEW_TO:-16}"
MAX_VIS_BATCHES="${MAX_VIS_BATCHES:-0}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
DEBUG="${DEBUG:-0}"
AUTO_RESUME="${AUTO_RESUME:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
NO_AMP="${NO_AMP:-0}"
DDP="${DDP:-0}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-0}"
PIN_MEMORY="${PIN_MEMORY:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PRELOAD_CACHE_TO_MEMORY="${PRELOAD_CACHE_TO_MEMORY:-0}"
NO_PRELOAD_CACHE_TO_MEMORY="${NO_PRELOAD_CACHE_TO_MEMORY:-0}"
TRAJMODE="${TRAJMODE:-traj}"
JOINT_PRED="${JOINT_PRED:-1}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  DDP=1
fi
if [[ "${PRELOAD_CACHE_TO_MEMORY}" == "1" && "${NO_PRELOAD_CACHE_TO_MEMORY}" == "1" ]]; then
  echo "[ERROR] PRELOAD_CACHE_TO_MEMORY and NO_PRELOAD_CACHE_TO_MEMORY cannot both be enabled." >&2
  exit 2
fi

VJEPA2_ROOT="${VJEPA2_ROOT:-${ROOT_DIR}/vjepa2}"
VJEPA2_PARENT="$(dirname -- "${VJEPA2_ROOT}")"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/cache_train:${VJEPA2_ROOT}:${VJEPA2_PARENT}:${PYTHONPATH:-}"
export VJEPA2_ROOT

mkdir -p "${OUT_DIR}" "${OUT_DIR}/vis"

CMD=(
  "${PYBIN}" "${TRAIN_PY}"
  --data_dir "${DATA_DIR}"
  --cache_dir "${CACHE_DIR}"
  --cache_validation_report "${CACHE_VALIDATION_REPORT}"
  --cache_validation_marker "${CACHE_VALIDATION_MARKER}"
  --output_dir "${OUT_DIR}"
  --results_md "${RESULTS_MD}"
  --output_mp4 "${OUTPUT_MP4}"
  --epochs "${EPOCHS}"
  --checkpoint_selection "${CHECKPOINT_SELECTION}"
  --predictor thinkjepa
  --backbone "${BACKBONE}"
  --optimize_together_downstream
  --seed "${SEED}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --test_batch_size "${TEST_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --past_T "${PAST_T}"
  --future_T "${FUTURE_T}"
  --temporal_stride "${TEMPORAL_STRIDE}"
  --camera_mode "${CAMERA_MODE}"
  --lr "${LR}"
  --lr_pred "${LR_PRED}"
  --lambda_task "${LAMBDA_TASK}"
  --lambda_pred "${LAMBDA_PRED}"
  --grad_accum "${GRAD_ACCUM}"
  --ref_mode "${REF_MODE}"
  --vlm_pad_old_to "${VLM_PAD_OLD_TO}"
  --vlm_pad_new_to "${VLM_PAD_NEW_TO}"
  --max_visual_batches "${MAX_VIS_BATCHES}"
  --max_train_batches "${MAX_TRAIN_BATCHES}"
  --max_eval_batches "${MAX_EVAL_BATCHES}"
)

if [[ "${AUTO_RESUME}" == "1" ]]; then
  CMD+=(--auto_resume)
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume_ckpt "${RESUME_CKPT}")
fi
if [[ "${NO_AMP}" == "1" ]]; then
  CMD+=(--no_amp)
fi
if [[ "${DEBUG}" == "1" ]]; then
  CMD+=(--debug)
fi
if [[ "${DDP}" == "1" ]]; then
  CMD+=(--ddp)
fi
if [[ "${DDP_FIND_UNUSED_PARAMETERS}" == "1" ]]; then
  CMD+=(--ddp_find_unused_parameters)
fi
if [[ "${PIN_MEMORY}" == "1" ]]; then
  CMD+=(--pin_memory)
fi
if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
  CMD+=(--persistent_workers)
fi
if [[ "${PRELOAD_CACHE_TO_MEMORY}" == "1" ]]; then
  CMD+=(--preload_cache_to_memory)
fi
if [[ "${NO_PRELOAD_CACHE_TO_MEMORY}" == "1" ]]; then
  CMD+=(--no_preload_cache_to_memory)
fi
if [[ -n "${TRAJMODE}" ]]; then
  CMD+=(--trajmode "${TRAJMODE}")
fi
if [[ "${JOINT_PRED}" == "1" ]]; then
  CMD+=(--joint_pred)
fi
if [[ -n "${TRAIN_MANIFEST}" ]]; then
  CMD+=(--train_manifest "${TRAIN_MANIFEST}")
fi
if [[ -n "${VAL_MANIFEST}" ]]; then
  CMD+=(--val_manifest "${VAL_MANIFEST}")
fi
if [[ -n "${SPLIT_META}" ]]; then
  CMD+=(--split_meta "${SPLIT_META}")
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] CACHE_DIR=${CACHE_DIR}"
echo "[INFO] VJEPA2_ROOT=${VJEPA2_ROOT}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYBIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${CMD[@]:1}"
else
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${CMD[@]}"
fi
