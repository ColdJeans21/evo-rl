import json
from pathlib import Path

import pytest
import torch

from lerobot.configs.train import EvoRLSelectionConfig
from lerobot.rl.evo_selection import EvoRLSample, EvoRLSelectedDataset, make_evo_selection_dataset_and_sampler


class _DummyDataset:
    def __init__(self, root: Path, length: int = 8):
        self.root = root
        self.num_frames = length
        self.episodes = None

    def __getitem__(self, index: int):
        return {
            "index": torch.tensor(index),
            "action": torch.zeros(5, 2),
            "action_is_pad": torch.zeros(5, dtype=torch.bool),
        }


def test_selected_dataset_masks_actions_at_segment_boundary(tmp_path):
    dataset = _DummyDataset(tmp_path)
    selected = EvoRLSelectedDataset(
        dataset,
        [EvoRLSample(frame_index=1, segment_end=3, source="intervention")],
    )

    item = selected[0]
    assert torch.equal(item["action_is_pad"], torch.tensor([False, False, True, True, True]))


def test_selection_sampler_respects_source_probabilities(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset_root": str(tmp_path),
                "total_frames": 8,
                "pools": {
                    "demo_base": [[0, 2]],
                    "intervention": [[2, 4]],
                    "demo_selected": [[4, 6]],
                    "auto_selected": [[6, 8]],
                },
                "stats": {},
            }
        )
    )
    cfg = EvoRLSelectionConfig(enable=True, plan_path=str(plan_path), samples_per_epoch=10_000)
    dataset, sampler, summary = make_evo_selection_dataset_and_sampler(_DummyDataset(tmp_path), cfg, seed=7)

    counts = dict.fromkeys(summary["source_probabilities"], 0)
    for slot in sampler:
        counts[dataset.samples[slot].source] += 1
    ratios = {source: count / 10_000 for source, count in counts.items()}

    assert ratios["demo_base"] == pytest.approx(0.50, abs=0.02)
    assert ratios["intervention"] == pytest.approx(0.25, abs=0.02)
    assert ratios["demo_selected"] == pytest.approx(0.125, abs=0.02)
    assert ratios["auto_selected"] == pytest.approx(0.125, abs=0.02)


def test_selection_config_requires_normalized_positive_weights():
    with pytest.raises(ValueError, match="sum to 1"):
        EvoRLSelectionConfig(enable=True, plan_path="plan.json", demo_base_weight=0.4).validate()
    with pytest.raises(ValueError, match="> 0"):
        EvoRLSelectionConfig(
            enable=True,
            plan_path="plan.json",
            demo_base_weight=0.625,
            auto_selected_weight=0.0,
        ).validate()
