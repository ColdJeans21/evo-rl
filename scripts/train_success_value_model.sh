#!/usr/bin/env bash
# Review the command with --dry-run. Launch only after choosing the training parameters.
set -Eeuo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TRAIN_BIN="/home/xiaozhi/miniconda3/envs/evo-rl/bin/lerobot-value-train"
readonly DATASET_ROOT="/home/xiaozhi/so101_collect_data/success_valueModelTrain"
readonly RUN_NAME="pistar06_pi05_time_takeover_ModelValueTrain_v1"
readonly OUTPUT_DIR="${PROJECT_ROOT}/outputs/value_train/${RUN_NAME}"

if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != "--dry-run" ]]; }; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ! -x "${TRAIN_BIN}" || ! -f "${DATASET_ROOT}/merge_manifest.json" ]]; then
  echo "Missing training executable or verified dataset manifest." >&2
  exit 1
fi

command_args=(
  "${TRAIN_BIN}"
  --dataset.repo_id=success_valueModelTrain
  "--dataset.root=${DATASET_ROOT}"
  --dataset.video_backend=pyav
  --dataset.image_transforms.enable=false
  --value.type=pistar06
  --value.mode=vision_only
  --value.device=cuda
  --value.dtype=bfloat16
  --value.use_gradient_checkpointing=true
  --value.freeze_vision_encoder=false
  --value.push_to_hub=false
  --value.optimizer_lr=5e-5
  --value.optimizer_weight_decay=1e-5
  --value.optimizer_grad_clip_norm=10
  --value.scheduler_warmup_steps=1000
  --value.scheduler_decay_steps=16000
  --value.scheduler_decay_lr=1e-6
  --reward.mode=time_takeover
  --reward.takeover_cost=150
  --reward.intervention_field=complementary_info.is_intervention
  --targets.success_field=episode_success
  --validation.episodes='[1,4,10,21,25,46,50,55,56,67,71,72,89,91,92,93]'
  --validation.freq=8000
  --validation.batch_size=32
  --validation.num_workers=4
  --validation.run_at_start=true
  --batch_size=32
  --steps=16000
  --num_workers=4
  --log_freq=50
  --save_checkpoint=true
  --save_freq=8000
  --wandb.enable=true
  --wandb.project=evo-rl-valueTrain
  --wandb.disable_artifact=true
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${RUN_NAME}"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%q ' "${command_args[@]}"
  printf '\n'
  exit 0
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists: ${OUTPUT_DIR}. Choose a new run or an explicit resume command." >&2
  exit 1
fi

cd -- "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/outputs/value_train"
readonly LOG_PATH="${OUTPUT_DIR}.log"
{
  printf 'Started: %s\n' "$(date -Is)"
  printf 'Git revision: %s\n' "$(git rev-parse HEAD)"
  printf 'Dataset manifest: '
  sha256sum "${DATASET_ROOT}/merge_manifest.json"
  printf 'Command: '
  printf '%q ' "${command_args[@]}"
  printf '\n'
  "${command_args[@]}"
} 2>&1 | tee -a "${LOG_PATH}"
