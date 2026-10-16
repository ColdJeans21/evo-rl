"""Success-only Monte Carlo targets with task scales fitted on training episodes."""

import math

import numpy as np

from lerobot.configs.value_reward import ValueRewardConfig
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, normalize_episode_success_label


def build_successful_value_targets(
    dataset,
    reward: ValueRewardConfig,
    success_field: str,
    *,
    task_scales: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return targets in raw frame order; reject partial, failed or unlabelled episodes.

    Only episodes actually present in hf_dataset contribute to the fitted scale.
    Task names, rather than dataset-local task indices, identify checkpoint scales.
    Supplying scales reuses them without refitting or clipping out-of-range returns.
    """
    reward.validate()
    if reward.mode == "legacy":
        raise ValueError("Use the legacy target builder for reward.mode=legacy.")

    frames = dataset.hf_dataset.with_format(None)
    episode_ids = np.asarray(frames["episode_index"], dtype=np.int64).reshape(-1)
    frame_ids = np.asarray(frames["frame_index"], dtype=np.int64).reshape(-1)
    if len(episode_ids) == 0:
        raise ValueError("Dataset has no frames.")
    episodes = dataset.meta.episodes.with_format(None)
    if success_field not in episodes.column_names:
        raise ValueError(f"reward.mode={reward.mode} requires explicit '{success_field}=success' labels.")
    meta = episodes[:]
    active_ids = set(episode_ids.tolist())
    metadata = {}
    for i, episode_id in enumerate(meta["episode_index"]):
        episode_id = int(episode_id)
        if episode_id not in active_ids:
            continue
        if episode_id in metadata:
            raise ValueError(f"Duplicate episode metadata for episode {episode_id}.")
        label = normalize_episode_success_label(meta[success_field][i])
        if label != EPISODE_SUCCESS:
            raise ValueError(
                f"reward.mode={reward.mode} requires only explicitly successful episodes; "
                f"episode {episode_id} has label {label!r}."
            )
        tasks = meta["tasks"][i]
        if not isinstance(tasks, str):
            if len(tasks) != 1:
                raise ValueError(f"Episode {episode_id} must have exactly one task for reward normalization.")
            tasks = tasks[0]
        metadata[episode_id] = (int(meta["length"][i]), str(tasks))
    if active_ids != set(metadata):
        raise ValueError("Missing episode metadata for reward construction.")

    interventions = None
    if reward.mode == "time_takeover":
        if reward.intervention_field not in frames.column_names:
            raise ValueError(f"Missing intervention field '{reward.intervention_field}' for time_takeover.")
        interventions = np.asarray(frames[reward.intervention_field]).reshape(-1)
        if interventions.shape != episode_ids.shape or not np.isin(interventions, [0, 1]).all():
            raise ValueError("Intervention flags must contain one boolean or binary 0/1 value per frame.")
        interventions = interventions.astype(bool)

    returns = np.empty(len(episode_ids), dtype=np.float64)
    fitted_scales: dict[str, float] = {}
    positions_by_task: dict[str, list[np.ndarray]] = {}
    for episode_id in sorted(active_ids):
        positions = np.flatnonzero(episode_ids == episode_id)
        length, task = metadata[episode_id]
        if (
            length <= 0
            or not np.array_equal(frame_ids[positions], np.arange(length))
            or not np.all(np.diff(positions) == 1)
        ):
            raise ValueError(
                f"Episode {episode_id} must contain every frame exactly once, contiguously from 0 to length-1."
            )
        rewards = np.full(length, -1.0, dtype=np.float64)
        rewards[-1] = 0.0
        if interventions is not None and length > 2:
            human = interventions[positions]
            # Penalize the preceding autonomous frame. Entry into the terminal
            # observation and initial human control are not takeover events.
            takeovers = (~human[:-2]) & human[1:-1]
            rewards[:-2] -= reward.takeover_cost * takeovers
        episode_returns = np.cumsum(rewards[::-1])[::-1]
        returns[positions] = episode_returns
        # A one-frame successful episode has zero cost; use scale 1 to avoid 0/0.
        fitted_scales[task] = max(fitted_scales.get(task, 1.0), float(-episode_returns[0]))
        positions_by_task.setdefault(task, []).append(positions)

    scales = dict(fitted_scales if task_scales is None else task_scales)
    targets = np.empty(len(returns), dtype=np.float32)
    for task, chunks in positions_by_task.items():
        scale = scales.get(task)
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Missing or invalid saved reward scale for task {task!r}.")
        positions = np.concatenate(chunks)
        normalized = returns[positions] / scale
        if np.any(normalized < -1.0) or np.any(normalized > 0.0):
            raise ValueError(
                f"Returns for task {task!r} exceed the saved reward scale; refusing to clip or refit."
            )
        targets[positions] = normalized
    return targets, scales
