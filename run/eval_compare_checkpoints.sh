#!/bin/bash
#PBS -q debug-g
#PBS -l select=1:ncpus=72:mpiprocs=1
#PBS -W group_list=gr41
set -euo pipefail

module purge
module load nvidia
module load hdf5

cd $PBS_O_WORKDIR
export CUDA_VISIBLE_DEVICES=0


PI0_CONFIG="pi0_ur3_robotiq"
PI05_CONFIG="pi05_ur3_robotiq"

PI0_EXP="treasured-energy-30"
PI05_EXP="major-sunset-10"
STEP_ARG="20000"
NUM_BATCHES=64
SPLIT="train"
BATCH_SIZE="32"
NUM_WORKERS="4"
SAMPLE_STEPS=10
COMPARE_ACTION_DIM=7
SEED=42
OUT_DIR="eval_results"




mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
PI0_OUT="${OUT_DIR}/pi0_eval_${TS}.json"
PI05_OUT="${OUT_DIR}/pi05_eval_${TS}.json"

set -x
source .venv/bin/activate

EXTRA_ARGS=()
if [[ -n "${STEP_ARG}" ]]; then
  EXTRA_ARGS+=(--step "${STEP_ARG}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  EXTRA_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  EXTRA_ARGS+=(--num-workers "${NUM_WORKERS}")
fi

python scripts/eval_checkpoint_metrics.py "${PI0_CONFIG}" \
  --exp-name "${PI0_EXP}" \
  --num-batches "${NUM_BATCHES}" \
  --split "${SPLIT}" \
  "${EXTRA_ARGS[@]}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --compare-action-dim "${COMPARE_ACTION_DIM}" \
  --seed "${SEED}" \
  --output-json "${PI0_OUT}"

python scripts/eval_checkpoint_metrics.py "${PI05_CONFIG}" \
  --exp-name "${PI05_EXP}" \
  --num-batches "${NUM_BATCHES}" \
  --split "${SPLIT}" \
  "${EXTRA_ARGS[@]}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --compare-action-dim "${COMPARE_ACTION_DIM}" \
  --seed "${SEED}" \
  --output-json "${PI05_OUT}"
set +x

echo
echo "Saved:"
echo "  ${PI0_OUT}"
echo "  ${PI05_OUT}"
