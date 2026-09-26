#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "${REPO_DIR}/pretrained_models/llava_med_v1.5" ]]; then
    MODEL_PATH="${REPO_DIR}/pretrained_models/llava_med_v1.5"
  elif [[ -d "/home/yoyo/project/pretrained_models/llava_med_v1.5" ]]; then
    MODEL_PATH="/home/yoyo/project/pretrained_models/llava_med_v1.5"
  else
    echo "ERROR: Set MODEL_PATH to the local llava_med_v1.5 directory." >&2
    exit 2
  fi
fi

DATA_ROOT="${DATA_ROOT:-/data3/yoyodata/MedLSC_data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_DIR}/checkpoints/finetune_Hospital_PROG_STANDARD_V10_DEPT_ANCHOR_ADAPTIVE_FUSION_DELTA_MERGE-64-64_llava_med_v1.5}"
ANCHOR_RESULTS_DIR="${ANCHOR_RESULTS_DIR:-${CHECKPOINT_ROOT}/anchors}"
ANCHOR_FILE="${ANCHOR_FILE:-${ANCHOR_RESULTS_DIR}/anchor_lora_router.pt}"
TEST_FEATURE_CACHE_DIR="${TEST_FEATURE_CACHE_DIR:-${ANCHOR_RESULTS_DIR}/test_feature_cache}"

STAGE_ID="${STAGE_ID:-12}"
STAGE_TAGS=(covid-CXP slake-ctxr iu-x-ray slake-mri PCAM pathvqa HAM_skin8 derm Yangxi oct-c8 cervical kvasir hyperkvasir)
if (( STAGE_ID < 0 || STAGE_ID >= ${#STAGE_TAGS[@]} )); then
  echo "ERROR: STAGE_ID must be between 0 and 12" >&2
  exit 2
fi
STAGE_TAG="${STAGE_TAGS[STAGE_ID]}"

ANCHOR_COEFFICIENT="${ANCHOR_COEFFICIENT:-0.5}"
ANCHOR_DEPARTMENT_TOPK="${ANCHOR_DEPARTMENT_TOPK:-1}"
KEY_OUTSIDE_TOPK="${KEY_OUTSIDE_TOPK:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/results/v10_dept_anchor_adaptive_fusion_delta_merge_init${ANCHOR_COEFFICIENT}_atop${ANCHOR_DEPARTMENT_TOPK}_ktop${KEY_OUTSIDE_TOPK}/stage_$(printf '%02d' "${STAGE_ID}")_${STAGE_TAG}}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
DATASETS="${DATASETS:-all-seen}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
MAX_SAMPLES_PER_DATASET="${MAX_SAMPLES_PER_DATASET:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "============================================================"
echo "V10 adaptive-fusion department-aware query-anchor evaluation with exact delta-merge LoRA"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Stage: ${STAGE_ID} (${STAGE_TAG})"
echo "Datasets: ${DATASETS}"
echo "Anchor coefficient: ${ANCHOR_COEFFICIENT}"
echo "Adaptive fusion: learned gate loaded from routing.bin when present"
echo "Anchor department top-k: ${ANCHOR_DEPARTMENT_TOPK}"
echo "Key-based outside top-k: ${KEY_OUTSIDE_TOPK}"
echo "Output: ${OUTPUT_DIR}"
echo "CUDA device: ${CUDA_DEVICE}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${REPO_DIR}/eval_MedLSC.py" \
  --model-path "${MODEL_PATH}" \
  --data-root "${DATA_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --anchor-file "${ANCHOR_FILE}" \
  --test-feature-cache-dir "${TEST_FEATURE_CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --stage-id "${STAGE_ID}" \
  --datasets "${DATASETS}" \
  --anchor-coefficient "${ANCHOR_COEFFICIENT}" \
  --anchor-department-topk "${ANCHOR_DEPARTMENT_TOPK}" \
  --key-outside-topk "${KEY_OUTSIDE_TOPK}" \
  --device cuda:0 \
  --dtype "${MODEL_DTYPE}" \
  --max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
