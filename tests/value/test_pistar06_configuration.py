#!/usr/bin/env python

import pytest

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.factory import make_policy_config
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config


def test_pistar06_config_from_dict():
    payload = {
        "type": "pistar06",
        "num_bins": 101,
        "bin_min": -1.0,
        "bin_max": 0.0,
        "task_index_feature": "task_index",
        "task_field": "task",
        "camera_features": ["observation.images.front"],
        "language_repo_id": "google/gemma-3-270m",
        "vision_repo_id": "google/siglip-so400m-patch14-384",
        "dropout": 0.2,
    }
    cfg = make_policy_config(payload.pop("type"), **payload)
    assert isinstance(cfg, Pistar06Config)
    assert cfg.type == "pistar06"
    assert cfg.num_bins == 101
    assert cfg.camera_features == ["observation.images.front"]
    assert cfg.loss_weight_key == "observation.value_loss_weight"


def test_pistar06_preset_uses_cosine_decay_with_warmup():
    cfg = Pistar06Config()
    scheduler_cfg = cfg.get_scheduler_preset()
    assert isinstance(scheduler_cfg, CosineDecayWithWarmupSchedulerConfig)
    assert scheduler_cfg.peak_lr == cfg.optimizer_lr
    assert scheduler_cfg.decay_lr == cfg.scheduler_decay_lr
    assert scheduler_cfg.num_warmup_steps == cfg.scheduler_warmup_steps
    assert scheduler_cfg.num_decay_steps == cfg.scheduler_decay_steps


def test_pistar06_history_offsets_define_observation_window():
    cfg = Pistar06Config(history_frame_offsets=[-25, -12])
    assert cfg.observation_delta_indices == [-25, -12, 0]
    assert cfg.n_obs_steps == 3


@pytest.mark.parametrize(
    "offsets,match",
    [
        ([0], "negative"),
        ([-12, -25], "strictly increasing"),
        ([-12, -12], "strictly increasing"),
    ],
)
def test_pistar06_rejects_invalid_history_offsets(offsets, match):
    with pytest.raises(ValueError, match=match):
        Pistar06Config(history_frame_offsets=offsets)
