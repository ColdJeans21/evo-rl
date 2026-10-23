from types import SimpleNamespace

import datasets
import numpy as np
import pytest

from lerobot.configs.value_reward import ValueRewardConfig
from lerobot.values.rewards import build_successful_value_targets


def _dataset(
    episode_lengths: list[int],
    interventions: list[int],
    *,
    labels: list[str] | None = None,
    tasks: list[str] | None = None,
):
    labels = labels or ["success"] * len(episode_lengths)
    tasks = tasks or ["pick"] * len(episode_lengths)

    episode_indices: list[int] = []
    frame_indices: list[int] = []
    for episode_index, length in enumerate(episode_lengths):
        episode_indices.extend([episode_index] * length)
        frame_indices.extend(range(length))

    frames = datasets.Dataset.from_dict(
        {
            "episode_index": episode_indices,
            "frame_index": frame_indices,
            "complementary_info.is_intervention": interventions,
        }
    )
    episodes = datasets.Dataset.from_dict(
        {
            "episode_index": list(range(len(episode_lengths))),
            "length": episode_lengths,
            "tasks": [[task] for task in tasks],
            "episode_success": labels,
        }
    )
    return SimpleNamespace(hf_dataset=frames, meta=SimpleNamespace(episodes=episodes))


def test_time_takeover_penalizes_only_transition_before_human_control():
    dataset = _dataset([6], [0, 0, 1, 1, 0, 0])
    reward = ValueRewardConfig(mode="time_takeover", takeover_cost=2.0)

    targets, scales = build_successful_value_targets(dataset, reward, "episode_success")

    # Rewards are [-1, -3, -1, -1, -1, 0]. The takeover penalty is charged
    # once to frame 1, whose action transitions from policy to human control.
    expected_returns = np.array([-7, -6, -3, -2, -1, 0], dtype=np.float32)
    assert scales == {"pick": 7.0}
    assert np.allclose(targets, expected_returns / 7.0)


def test_time_takeover_ignores_initial_human_control_and_terminal_entry():
    dataset = _dataset([5], [1, 1, 0, 0, 1])
    reward = ValueRewardConfig(mode="time_takeover", takeover_cost=20.0)

    targets, scales = build_successful_value_targets(dataset, reward, "episode_success")

    # Initial human control has no preceding policy action. A 0->1 transition
    # into the terminal observation is b_T=0 by definition.
    assert scales == {"pick": 4.0}
    assert np.allclose(targets, np.array([-1.0, -0.75, -0.5, -0.25, 0.0]))


def test_time_takeover_counts_each_distinct_takeover_once():
    dataset = _dataset([7], [0, 1, 1, 0, 1, 0, 0])
    reward = ValueRewardConfig(mode="time_takeover", takeover_cost=2.0)

    targets, scales = build_successful_value_targets(dataset, reward, "episode_success")

    # Takeovers occur on transitions 0->1 and 3->4.
    expected_returns = np.array([-10, -7, -6, -5, -2, -1, 0], dtype=np.float32)
    assert scales == {"pick": 10.0}
    assert np.allclose(targets, expected_returns / 10.0)


def test_saved_task_scale_is_reused_and_out_of_range_returns_are_rejected():
    dataset = _dataset([4], [0, 1, 1, 0])
    reward = ValueRewardConfig(mode="time_takeover", takeover_cost=2.0)

    targets, scales = build_successful_value_targets(
        dataset,
        reward,
        "episode_success",
        task_scales={"pick": 10.0},
    )

    assert scales == {"pick": 10.0}
    assert np.allclose(targets, np.array([-0.5, -0.2, -0.1, 0.0]))

    with pytest.raises(ValueError, match="exceed the saved reward scale"):
        build_successful_value_targets(
            dataset,
            reward,
            "episode_success",
            task_scales={"pick": 4.0},
        )


@pytest.mark.parametrize("label", ["failure", "", "unknown"])
def test_paper_reward_rejects_non_success_episodes(label: str):
    dataset = _dataset([3], [0, 0, 0], labels=[label])
    reward = ValueRewardConfig(mode="time_takeover")

    with pytest.raises((ValueError, TypeError), match="success"):
        build_successful_value_targets(dataset, reward, "episode_success")


def test_time_takeover_requires_binary_intervention_flags():
    dataset = _dataset([3], [0, 2, 0])
    reward = ValueRewardConfig(mode="time_takeover")

    with pytest.raises(ValueError, match="binary 0/1"):
        build_successful_value_targets(dataset, reward, "episode_success")
