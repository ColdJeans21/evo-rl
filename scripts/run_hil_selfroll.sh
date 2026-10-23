#!/usr/bin/env bash

set -Eeuo pipefail

readonly LEROBOT_HIL_BIN="/home/xiaozhi/miniconda3/envs/evo-rl/bin/lerobot-human-inloop-record"
readonly DATASET_PARENT="/home/xiaozhi/so101_collect_data"
readonly DATASET_PREFIX="selfRoll_episode"
readonly FIRST_EPISODE_INDEX=31
readonly COUNTER_FILE="${DATASET_PARENT}/.selfroll_next_episode"
readonly COUNTER_LOCK_FILE="${DATASET_PARENT}/.selfroll_next_episode.lock"

if [[ ! -x "${LEROBOT_HIL_BIN}" ]]; then
  echo "Error: executable not found: ${LEROBOT_HIL_BIN}" >&2
  exit 1
fi

mkdir -p "${DATASET_PARENT}"

# Reserve one dataset index per script invocation. The lock also makes two
# concurrently started recordings choose different output directories.
exec 9>"${COUNTER_LOCK_FILE}"
flock 9

next_index=""
if [[ -f "${COUNTER_FILE}" ]]; then
  read -r next_index < "${COUNTER_FILE}" || true
fi

if [[ ! "${next_index}" =~ ^[0-9]+$ ]]; then
  max_index=$((FIRST_EPISODE_INDEX - 1))
  shopt -s nullglob
  for dataset_dir in "${DATASET_PARENT}/${DATASET_PREFIX}"*; do
    dataset_name="${dataset_dir##*/}"
    dataset_index="${dataset_name#${DATASET_PREFIX}}"
    if [[ "${dataset_index}" =~ ^[0-9]+$ ]] && (( dataset_index > max_index )); then
      max_index="${dataset_index}"
    fi
  done
  shopt -u nullglob
  next_index=$((max_index + 1))
fi

if (( next_index < FIRST_EPISODE_INDEX )); then
  next_index="${FIRST_EPISODE_INDEX}"
fi

dataset_root="${DATASET_PARENT}/${DATASET_PREFIX}${next_index}"
while [[ -e "${dataset_root}" ]]; do
  next_index=$((next_index + 1))
  dataset_root="${DATASET_PARENT}/${DATASET_PREFIX}${next_index}"
done

counter_tmp="${COUNTER_FILE}.tmp.$$"
printf '%s\n' "$((next_index + 1))" > "${counter_tmp}"
mv "${counter_tmp}" "${COUNTER_FILE}"
flock -u 9

echo "Recording dataset: ${dataset_root}"
echo "The next invocation will start from ${DATASET_PREFIX}$((next_index + 1))."

exec "${LEROBOT_HIL_BIN}" \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46085330-if00 \
  --robot.id=so101 \
  --robot.calibration_dir=/home/xiaozhi/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --robot.max_relative_target=10 \
  --robot.cameras='{front: {type: opencv, index_or_path: "/dev/v4l/by-id/usb-DSJ-250318-J_DSJ-2062-309-video-index0", width: 1280, height: 720, fps: 30, fourcc: "MJPG", backend: 200}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46085343-if00 \
  --teleop.id=so101_leader \
  --teleop.calibration_dir=/home/xiaozhi/.cache/huggingface/lerobot/calibration/teleoperators/so_leader \
  --policy.path=/home/xiaozhi/Evo-RL/outputs/train/pi05_base1_bc57_v1/checkpoints/030000/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=25 \
  --policy.num_inference_steps=10 \
  --acp_inference.enable=false \
  --dataset.repo_id=xz/eval_pi05_base1_ckpt030000_pilot_v1 \
  --dataset.root="${dataset_root}" \
  --dataset.single_task="Pick up the bottle and place it into the box." \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=80 \
  --dataset.reset_time_s=20 \
  --dataset.fps=25 \
  --dataset.video=true \
  --dataset.vcodec=h264 \
  --dataset.push_to_hub=false \
  --dataset.video_encoding_batch_size=1 \
  --require_episode_success_label=true \
  --display_data=false \
  --play_sounds=true \
  --resume=false
