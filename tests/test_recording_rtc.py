#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.scripts.lerobot_record import ACPInferenceConfig, DatasetRecordConfig, RecordConfig
from lerobot.scripts.recording_hil import RTCPolicyActionProvider
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


class _IdentityProcessor:
    def __call__(self, value):
        return value


class _FakeRTCPolicy:
    name = "pi05"

    def __init__(self):
        self.config = SimpleNamespace(device="cpu", use_amp=False, chunk_size=6, rtc_config=None)
        self.requests = []
        self.rtc_processor_initialized = False

    def init_rtc_processor(self):
        self.rtc_processor_initialized = True

    def predict_action_chunk(self, observation, **kwargs):
        request_number = len(self.requests) + 1
        self.requests.append(
            {
                "observation": observation,
                "inference_delay": kwargs["inference_delay"],
                "prev_chunk_left_over": kwargs["prev_chunk_left_over"],
            }
        )
        time.sleep(0.005)
        return torch.full((1, 6, 2), float(request_number), dtype=torch.float32)


def _dataset_config() -> DatasetRecordConfig:
    return DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )


def _pi05_config() -> PI05Config:
    return PI05Config(device="cpu", chunk_size=50, n_action_steps=25)


def test_rtc_provider_replans_with_leftover_chunk():
    policy = _FakeRTCPolicy()
    rtc = RTCConfig(enabled=True, execution_horizon=2)
    provider = RTCPolicyActionProvider(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        rtc_config=rtc,
        fps=100,
        action_queue_size_to_get_new_actions=3,
        action_timeout_s=1.0,
    )
    observation = {"observation.state": np.zeros(2, dtype=np.float32)}

    try:
        actions = [
            provider.get_action(observation_frame=observation, task="test", robot_type="mock")
            for _ in range(4)
        ]
        deadline = time.perf_counter() + 1.0
        while len(policy.requests) < 2 and time.perf_counter() < deadline:
            time.sleep(0.001)
        actions.extend(
            provider.get_action(observation_frame=observation, task="test", robot_type="mock")
            for _ in range(4)
        )
    finally:
        provider.close()

    assert policy.rtc_processor_initialized
    assert policy.config.rtc_config is rtc
    assert len(policy.requests) >= 2
    assert policy.requests[0]["prev_chunk_left_over"] is None
    assert policy.requests[1]["prev_chunk_left_over"] is not None
    assert policy.requests[1]["prev_chunk_left_over"].shape[-1] == 2
    assert torch.equal(actions[0], torch.ones(2))
    assert any(torch.equal(action, torch.full((2,), 2.0)) for action in actions)


def test_record_config_rejects_rtc_without_policy():
    with pytest.raises(ValueError, match="rtc.enabled=true.*policy"):
        RecordConfig(
            robot=MockRobotConfig(),
            teleop=MockTeleopConfig(),
            dataset=_dataset_config(),
            rtc=RTCConfig(enabled=True),
        )


def test_record_config_rejects_rtc_with_acp():
    with pytest.raises(ValueError, match="does not currently support ACP"):
        RecordConfig(
            robot=MockRobotConfig(),
            teleop=MockTeleopConfig(),
            policy=_pi05_config(),
            dataset=_dataset_config(),
            rtc=RTCConfig(enabled=True),
            acp_inference=ACPInferenceConfig(enable=True),
        )


@pytest.mark.parametrize("queue_threshold", [10, 50])
def test_record_config_validates_rtc_queue_threshold(queue_threshold):
    expected_message = "greater than" if queue_threshold == 10 else "smaller than"
    with pytest.raises(ValueError, match=expected_message):
        RecordConfig(
            robot=MockRobotConfig(),
            teleop=MockTeleopConfig(),
            policy=_pi05_config(),
            dataset=_dataset_config(),
            rtc=RTCConfig(enabled=True, execution_horizon=10),
            rtc_action_queue_size_to_get_new_actions=queue_threshold,
        )


def test_record_config_applies_rtc_to_policy():
    policy = _pi05_config()
    rtc = RTCConfig(enabled=True, execution_horizon=10)

    RecordConfig(
        robot=MockRobotConfig(),
        teleop=MockTeleopConfig(),
        policy=policy,
        dataset=_dataset_config(),
        rtc=rtc,
        rtc_action_queue_size_to_get_new_actions=30,
    )

    assert policy.rtc_config is rtc
