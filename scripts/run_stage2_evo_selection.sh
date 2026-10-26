#!/usr/bin/env bash
# Run paper-style Evo-RL value selection and weighted policy refinement.
set -Eeuo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ENV_BIN="/home/xiaozhi/miniconda3/envs/evo-rl/bin"
readonly SOURCE_DATASET_ROOT="/home/xiaozhi/so101_collect_data/success_valueModelTrain"
readonly STAGE2_DATASET_ROOT="/home/xiaozhi/so101_collect_data/stage2_evo_pi05_r1_v008k"
readonly STAGE2_DATASET_ID="stage2_evo_pi05_r1_v008k"
readonly VALUE_CHECKPOINT="${PROJECT_ROOT}/outputs/value_train/pistar06_pi05_time_takeover_ModelValueTrain_v1/checkpoints/008000/pretrained_model"
readonly POLICY_CHECKPOINT="${PROJECT_ROOT}/outputs/train/pi05_base1_bc57_v1/checkpoints/030000/pretrained_model"

# Paper-aligned advantage settings. The paper visualizes merged 20-frame
# forward windows; n_step must not exceed PI0.5's 50-step action horizon.
# Top 10% and strictly positive advantages follow Evo-RL.
readonly N_STEP="${N_STEP:-20}"
readonly POSITIVE_RATIO="${POSITIVE_RATIO:-0.10}"
readonly REQUIRE_POSITIVE_ADVANTAGE="${REQUIRE_POSITIVE_ADVANTAGE:-true}"

readonly WINDOW_SIZE="${WINDOW_SIZE:-20}"
readonly DEMO_BASE_WEIGHT="${DEMO_BASE_WEIGHT:-0.50}"
readonly INTERVENTION_WEIGHT="${INTERVENTION_WEIGHT:-0.25}"
readonly DEMO_SELECTED_WEIGHT="${DEMO_SELECTED_WEIGHT:-0.125}"
readonly AUTO_SELECTED_WEIGHT="${AUTO_SELECTED_WEIGHT:-0.125}"
# Demonstrations (0..56) plus the seven verified successful autonomous episodes.
# Intervention episodes are excluded from advantage selection and enter policy
# training only through their contiguous human-controlled segments.
readonly ADVANTAGE_EPISODES='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,65,66,75,79,91,94]'

# Conservative continuation settings for the 16 GiB GPU used by the first run.
readonly BATCH_SIZE="${BATCH_SIZE:-8}"
readonly TRAIN_STEPS="${TRAIN_STEPS:-15000}"
readonly SAVE_FREQ="${SAVE_FREQ:-5000}"
readonly LEARNING_RATE="${LEARNING_RATE:-1e-5}"
readonly NUM_WORKERS="${NUM_WORKERS:-4}"
readonly TAG="evo_r1_v008k_n${N_STEP}_top10"
readonly VALUE_FIELD="complementary_info.value_${TAG}"
readonly ADVANTAGE_FIELD="complementary_info.advantage_${TAG}"
readonly INDICATOR_FIELD="complementary_info.acp_indicator_${TAG}"
readonly INFER_RUN_NAME="pi05_${TAG}"
readonly TRAIN_RUN_NAME="pi05_base2_selective_${TAG}"
readonly INFER_OUTPUT="${PROJECT_ROOT}/outputs/value_infer/${INFER_RUN_NAME}"
readonly TRAIN_OUTPUT="${PROJECT_ROOT}/outputs/train/${TRAIN_RUN_NAME}"
readonly SELECTION_PLAN="${INFER_OUTPUT}/selection_plan.json"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_stage2_evo_selection.sh prepare [--execute]
  scripts/run_stage2_evo_selection.sh infer   [--execute]
  scripts/run_stage2_evo_selection.sh select  [--execute]
  scripts/run_stage2_evo_selection.sh train   [--execute]
  scripts/run_stage2_evo_selection.sh all     [--execute]

Without --execute the script performs preflight checks and prints commands.

Environment overrides requiring experimental justification:
  N_STEP, POSITIVE_RATIO, REQUIRE_POSITIVE_ADVANTAGE, WINDOW_SIZE,
  DEMO_BASE_WEIGHT, INTERVENTION_WEIGHT,
  DEMO_SELECTED_WEIGHT, AUTO_SELECTED_WEIGHT,
  BATCH_SIZE, TRAIN_STEPS, SAVE_FREQ, LEARNING_RATE, NUM_WORKERS
EOF
}

if (( $# < 1 || $# > 2 )); then
  usage >&2
  exit 2
fi
readonly PHASE="$1"
readonly EXECUTE="${2:-}"
if [[ "${PHASE}" != "prepare" && "${PHASE}" != "infer" && "${PHASE}" != "select" && "${PHASE}" != "train" && "${PHASE}" != "all" ]]; then
  usage >&2
  exit 2
fi
if [[ -n "${EXECUTE}" && "${EXECUTE}" != "--execute" ]]; then
  usage >&2
  exit 2
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

check_common() {
  [[ -x "${ENV_BIN}/lerobot-value-infer" ]] || die "Missing lerobot-value-infer in ${ENV_BIN}"
  [[ -x "${ENV_BIN}/lerobot-train" ]] || die "Missing lerobot-train in ${ENV_BIN}"
  [[ -f "${SOURCE_DATASET_ROOT}/merge_manifest.json" ]] || die "Missing verified source manifest"
  [[ -f "${VALUE_CHECKPOINT}/model.safetensors" ]] || die "Missing Value checkpoint"
  [[ -f "${POLICY_CHECKPOINT}/model.safetensors" ]] || die "Missing policy checkpoint"
  [[ "${N_STEP}" =~ ^[0-9]+$ ]] && (( N_STEP > 0 && N_STEP <= 50 )) || die "N_STEP must be in [1, 50]"
  [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] && (( BATCH_SIZE > 0 )) || die "BATCH_SIZE must be positive"
  [[ "${TRAIN_STEPS}" =~ ^[0-9]+$ ]] && (( TRAIN_STEPS > 0 )) || die "TRAIN_STEPS must be positive"
  [[ "${SAVE_FREQ}" =~ ^[0-9]+$ ]] && (( SAVE_FREQ > 0 )) || die "SAVE_FREQ must be positive"
  [[ "${WINDOW_SIZE}" =~ ^[0-9]+$ ]] && (( WINDOW_SIZE > 0 )) || die "WINDOW_SIZE must be positive"
  case "${REQUIRE_POSITIVE_ADVANTAGE}" in true | false) ;; *) die "REQUIRE_POSITIVE_ADVANTAGE must be true or false" ;; esac
}

prepare_dataset() {
  if [[ -e "${STAGE2_DATASET_ROOT}" ]]; then
    if [[ -f "${STAGE2_DATASET_ROOT}/merge_manifest.json" ]]; then
      echo "Stage-2 working copy already exists: ${STAGE2_DATASET_ROOT}"
      return
    fi
    die "Refusing to use incomplete destination: ${STAGE2_DATASET_ROOT}"
  fi
  echo "Create an isolated working copy; the verified source dataset remains unchanged:"
  print_command cp -a --reflink=auto -- "${SOURCE_DATASET_ROOT}" "${STAGE2_DATASET_ROOT}"
  if [[ "${EXECUTE}" == "--execute" ]]; then
    cp -a --reflink=auto -- "${SOURCE_DATASET_ROOT}" "${STAGE2_DATASET_ROOT}"
    [[ -f "${STAGE2_DATASET_ROOT}/merge_manifest.json" ]] || die "Working-copy verification failed"
  fi
}

infer_command=(
  "${ENV_BIN}/lerobot-value-infer"
  "--dataset.repo_id=${STAGE2_DATASET_ID}"
  "--dataset.root=${STAGE2_DATASET_ROOT}"
  "--dataset.episodes=${ADVANTAGE_EPISODES}"
  "--inference.checkpoint_path=${VALUE_CHECKPOINT}"
  --inference.checkpoint_ref=last
  --runtime.device=cuda
  --runtime.batch_size=32
  "--runtime.num_workers=${NUM_WORKERS}"
  --acp.enable=true
  "--acp.n_step=${N_STEP}"
  "--acp.positive_ratio=${POSITIVE_RATIO}"
  "--acp.require_positive_advantage=${REQUIRE_POSITIVE_ADVANTAGE}"
  --acp.force_intervention_positive=false
  --acp.intervention_field=complementary_info.is_intervention
  "--acp.value_field=${VALUE_FIELD}"
  "--acp.advantage_field=${ADVANTAGE_FIELD}"
  "--acp.indicator_field=${INDICATOR_FIELD}"
  --viz.enable=false
  "--output_dir=${INFER_OUTPUT}"
  "--job_name=${INFER_RUN_NAME}"
)

select_command=(
  "${ENV_BIN}/python"
  "${PROJECT_ROOT}/scripts/prepare_evo_stage2_selection.py"
  "--dataset-root=${STAGE2_DATASET_ROOT}"
  "--indicator-field=${INDICATOR_FIELD}"
  "--advantage-field=${ADVANTAGE_FIELD}"
  "--window-size=${WINDOW_SIZE}"
  "--output=${SELECTION_PLAN}"
)

train_command=(
  "${ENV_BIN}/lerobot-train"
  "--dataset.repo_id=${STAGE2_DATASET_ID}"
  "--dataset.root=${STAGE2_DATASET_ROOT}"
  --dataset.video_backend=pyav
  --dataset.image_transforms.enable=false
  "--policy.path=${POLICY_CHECKPOINT}"
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.train_expert_only=true
  --policy.mask_action_padding_loss=true
  "--policy.optimizer_lr=${LEARNING_RATE}"
  --policy.scheduler_warmup_steps=500
  "--policy.scheduler_decay_steps=${TRAIN_STEPS}"
  --policy.scheduler_decay_lr=1e-6
  --policy.push_to_hub=false
  "--batch_size=${BATCH_SIZE}"
  "--steps=${TRAIN_STEPS}"
  "--num_workers=${NUM_WORKERS}"
  --log_freq=50
  --save_checkpoint=true
  "--save_freq=${SAVE_FREQ}"
  --eval_freq=0
  --acp.enable=false
  --evo_selection.enable=true
  "--evo_selection.plan_path=${SELECTION_PLAN}"
  "--evo_selection.demo_base_weight=${DEMO_BASE_WEIGHT}"
  "--evo_selection.intervention_weight=${INTERVENTION_WEIGHT}"
  "--evo_selection.demo_selected_weight=${DEMO_SELECTED_WEIGHT}"
  "--evo_selection.auto_selected_weight=${AUTO_SELECTED_WEIGHT}"
  --wandb.enable=true
  --wandb.project=evo-rl-pi05-stage2
  --wandb.disable_artifact=true
  "--output_dir=${TRAIN_OUTPUT}"
  "--job_name=${TRAIN_RUN_NAME}"
)

run_infer() {
  if [[ "${EXECUTE}" == "--execute" ]]; then
    [[ -f "${STAGE2_DATASET_ROOT}/merge_manifest.json" ]] || die "Run prepare first"
  fi
  [[ ! -e "${INFER_OUTPUT}" ]] || die "Inference output already exists: ${INFER_OUTPUT}"
  echo "Value/advantage annotation command (mutates only the working copy):"
  print_command "${infer_command[@]}"
  if [[ "${EXECUTE}" == "--execute" ]]; then
    mkdir -p "${PROJECT_ROOT}/outputs/value_infer"
    "${infer_command[@]}" 2>&1 | tee "${INFER_OUTPUT}.log"
  fi
}

run_select() {
  if [[ "${EXECUTE}" == "--execute" ]]; then
    [[ -f "${STAGE2_DATASET_ROOT}/merge_manifest.json" ]] || die "Run prepare first"
    [[ -d "${INFER_OUTPUT}" ]] || die "Run infer first"
  fi
  [[ ! -e "${SELECTION_PLAN}" ]] || die "Selection plan already exists: ${SELECTION_PLAN}"
  echo "Paper-style segment-selection command:"
  print_command "${select_command[@]}"
  if [[ "${EXECUTE}" == "--execute" ]]; then
    "${select_command[@]}"
  fi
}

run_train() {
  if [[ "${EXECUTE}" == "--execute" ]]; then
    [[ -f "${STAGE2_DATASET_ROOT}/merge_manifest.json" ]] || die "Run prepare first"
  fi
  [[ ! -e "${TRAIN_OUTPUT}" ]] || die "Training output already exists: ${TRAIN_OUTPUT}"
  if [[ "${EXECUTE}" == "--execute" ]]; then
    [[ -f "${SELECTION_PLAN}" ]] || die "Run select first"
  fi
  echo "Paper-style selective policy continuation command:"
  print_command "${train_command[@]}"
  if [[ "${EXECUTE}" == "--execute" ]]; then
    mkdir -p "${PROJECT_ROOT}/outputs/train"
    "${train_command[@]}" 2>&1 | tee "${TRAIN_OUTPUT}.log"
  fi
}

check_common
echo "Mode: paper-style Evo-RL segment selection and weighted source mixing."
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "Value checkpoint: ${VALUE_CHECKPOINT}"
echo "Policy checkpoint: ${POLICY_CHECKPOINT}"
echo "Working dataset: ${STAGE2_DATASET_ROOT}"
echo "Fields: ${VALUE_FIELD}, ${ADVANTAGE_FIELD}, ${INDICATOR_FIELD}"

case "${PHASE}" in
  prepare) prepare_dataset ;;
  infer) run_infer ;;
  select) run_select ;;
  train) run_train ;;
  all)
    prepare_dataset
    run_infer
    run_select
    run_train
    ;;
esac
