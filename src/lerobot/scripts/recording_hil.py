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

"""Human-in-loop recording helpers used by `lerobot_record.py`."""

import logging
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotAction
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


@dataclass
class ACPInferenceConfig:
    enable: bool = False
    use_cfg: bool = False
    cfg_beta: float = 1.0


class RTCPolicyActionProvider:
    """Generate and merge RTC action chunks without blocking the robot control loop."""

    SUPPORTED_POLICY_NAMES = {"pi0", "pi05", "smolvla"}

    def __init__(
        self,
        *,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        rtc_config: RTCConfig,
        fps: int,
        action_queue_size_to_get_new_actions: int,
        action_timeout_s: float,
    ) -> None:
        if not rtc_config.enabled:
            raise ValueError("RTCPolicyActionProvider requires `rtc_config.enabled=true`.")
        if policy.name not in self.SUPPORTED_POLICY_NAMES:
            supported = ", ".join(sorted(self.SUPPORTED_POLICY_NAMES))
            raise ValueError(f"RTC recording supports policies [{supported}], got {policy.name!r}.")
        if action_queue_size_to_get_new_actions <= rtc_config.execution_horizon:
            raise ValueError(
                "`rtc_action_queue_size_to_get_new_actions` must be greater than `rtc.execution_horizon`."
            )
        if action_timeout_s <= 0:
            raise ValueError("`rtc_action_timeout_s` must be positive.")

        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is not None and action_queue_size_to_get_new_actions >= chunk_size:
            raise ValueError(
                "`rtc_action_queue_size_to_get_new_actions` must be smaller than the policy chunk size "
                f"({action_queue_size_to_get_new_actions} >= {chunk_size})."
            )

        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.rtc_config = rtc_config
        self.fps = fps
        self.action_queue_size_to_get_new_actions = action_queue_size_to_get_new_actions
        self.action_timeout_s = action_timeout_s
        self.device = get_safe_torch_device(policy.config.device)
        self.use_amp = policy.config.use_amp

        # Keep the model and queue configuration in sync even when a caller enables RTC
        # after loading a checkpoint whose saved config had RTC disabled.
        self.policy.config.rtc_config = rtc_config
        self.policy.init_rtc_processor()

        self._condition = Condition()
        self._queue = ActionQueue(rtc_config)
        self._latency_tracker = LatencyTracker()
        self._request: tuple[int, dict[str, np.ndarray], str | None, str | None] | None = None
        self._inference_active = False
        self._error: BaseException | None = None
        self._closed = False
        self._epoch = 0
        self._thread = Thread(target=self._worker, name="rtc-policy-inference", daemon=True)
        self._thread.start()

    @staticmethod
    def _copy_observation(observation_frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.array(value, copy=True) for key, value in observation_frame.items()}

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("RTC policy inference worker failed.") from self._error

    def _request_chunk_if_needed(
        self,
        observation_frame: dict[str, np.ndarray],
        task: str | None,
        robot_type: str | None,
    ) -> None:
        with self._condition:
            self._raise_worker_error()
            if (
                self._queue.qsize() <= self.action_queue_size_to_get_new_actions
                and self._request is None
                and not self._inference_active
            ):
                self._request = (
                    self._epoch,
                    self._copy_observation(observation_frame),
                    task,
                    robot_type,
                )
                self._condition.notify_all()

    def get_action(
        self,
        *,
        observation_frame: dict[str, np.ndarray],
        task: str | None,
        robot_type: str | None,
    ) -> PolicyAction:
        """Return the next postprocessed action, waiting only if the queue runs dry."""
        self._request_chunk_if_needed(observation_frame, task, robot_type)

        deadline = time.perf_counter() + self.action_timeout_s
        while True:
            action = self._queue.get()
            if action is not None:
                return action

            with self._condition:
                self._raise_worker_error()
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    raise TimeoutError(
                        f"Timed out after {self.action_timeout_s:.1f}s waiting for an RTC action chunk."
                    )
                self._condition.wait(timeout=remaining_s)

    def reset(self, *, wait_for_inference: bool) -> None:
        """Discard queued/stale actions and optionally wait for an in-flight request."""
        with self._condition:
            self._epoch += 1
            self._queue = ActionQueue(self.rtc_config)
            self._request = None
            self._error = None
            self._latency_tracker.reset()
            self._condition.notify_all()
            if wait_for_inference:
                self._condition.wait_for(lambda: not self._inference_active)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
            self._request = None
            self._condition.notify_all()
        self._thread.join(timeout=max(self.action_timeout_s, 1.0))
        if self._thread.is_alive():
            logging.warning("RTC inference thread did not stop within %.1fs.", self.action_timeout_s)

    def _worker(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._request is not None)
                if self._closed:
                    return
                request = self._request
                self._request = None
                self._inference_active = True

            assert request is not None
            epoch, observation_frame, task, robot_type = request
            queue = self._queue
            action_index_before_inference = queue.get_action_index()
            prev_chunk_left_over = queue.get_left_over()
            inference_delay = math.ceil(self._latency_tracker.max() * self.fps)
            started_t = time.perf_counter()

            try:
                observation = prepare_observation_for_inference(
                    observation_frame,
                    self.device,
                    task,
                    robot_type,
                )
                observation = self.preprocessor(observation)
                autocast_context = (
                    torch.autocast(device_type=self.device.type)
                    if self.device.type == "cuda" and self.use_amp
                    else nullcontext()
                )
                with autocast_context:
                    normalized_actions = self.policy.predict_action_chunk(
                        observation,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_chunk_left_over,
                    )
                    original_actions = normalized_actions.squeeze(0).detach().clone()
                    processed_actions = self.postprocessor(normalized_actions).squeeze(0)

                latency_s = time.perf_counter() - started_t

                with self._condition:
                    if epoch == self._epoch and queue is self._queue and not self._closed:
                        self._latency_tracker.add(latency_s)
                        actions_consumed_during_inference = max(
                            queue.get_action_index() - action_index_before_inference,
                            0,
                        )
                        if (
                            self.action_queue_size_to_get_new_actions
                            < self.rtc_config.execution_horizon + actions_consumed_during_inference
                        ):
                            logging.warning(
                                "RTC queue threshold (%d) is too small for the observed inference delay "
                                "(%d steps) plus execution horizon (%d); increase "
                                "`rtc_action_queue_size_to_get_new_actions`.",
                                self.action_queue_size_to_get_new_actions,
                                actions_consumed_during_inference,
                                self.rtc_config.execution_horizon,
                            )
                        queue.merge(
                            original_actions,
                            processed_actions,
                            actions_consumed_during_inference,
                            action_index_before_inference,
                        )
                        logging.debug(
                            "RTC chunk ready: latency=%.3fs predicted_delay=%d actual_delay=%d queue=%d",
                            latency_s,
                            inference_delay,
                            actions_consumed_during_inference,
                            queue.qsize(),
                        )
            except BaseException as error:
                with self._condition:
                    if epoch == self._epoch and not self._closed:
                        self._error = error
            finally:
                with self._condition:
                    self._inference_active = False
                    self._condition.notify_all()


POLICY_RUNTIME_STATE_KEYS = ("_action_queue", "_queues", "_prev_mean")


INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0


def _get_torch_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _set_torch_rng_state(
    device: torch.device, cpu_state: torch.Tensor, cuda_state: torch.Tensor | None
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _clone_runtime_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, deque):
        return deque((_clone_runtime_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return deepcopy(value)


def _capture_policy_runtime_state(policy: PreTrainedPolicy) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in POLICY_RUNTIME_STATE_KEYS:
        if hasattr(policy, key):
            state[key] = _clone_runtime_value(getattr(policy, key))
    return state


def _restore_policy_runtime_state(policy: PreTrainedPolicy, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(policy, key, _clone_runtime_value(value))


def _predict_policy_action_with_runtime_state(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    runtime_state: dict[str, Any],
) -> PolicyAction:
    _restore_policy_runtime_state(policy, runtime_state)
    action = predict_action(
        observation=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
    )
    runtime_state.clear()
    runtime_state.update(_capture_policy_runtime_state(policy))
    return action


def _predict_policy_action_with_acp_inference(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    acp_inference: ACPInferenceConfig,
    cond_runtime_state: dict[str, Any] | None = None,
    uncond_runtime_state: dict[str, Any] | None = None,
) -> PolicyAction:
    if not acp_inference.enable:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    conditional_task = build_acp_tagged_task(task, is_positive=True)
    if not acp_inference.use_cfg:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=conditional_task,
            robot_type=robot_type,
        )

    if cond_runtime_state is None or uncond_runtime_state is None:
        raise ValueError("CFG inference requires cond/uncond runtime states.")

    cpu_state, cuda_state = _get_torch_rng_state(device)
    action_cond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=conditional_task,
        robot_type=robot_type,
        runtime_state=cond_runtime_state,
    )
    _set_torch_rng_state(device, cpu_state, cuda_state)
    action_uncond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        runtime_state=uncond_runtime_state,
    )
    return action_uncond + acp_inference.cfg_beta * (action_cond - action_uncond)


class PolicySyncDualArmExecutor:
    """Broadcast one policy-derived robot action to follower + teleop arm."""

    def __init__(self, robot: Robot, teleop: Teleoperator, parallel_dispatch: bool = True):
        self.robot = robot
        self.teleop = teleop
        self.parallel_dispatch = parallel_dispatch
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel_dispatch else None

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._pool is None:
            sent_action = self.robot.send_action(action)
            self.teleop.send_feedback(action)
            return sent_action

        robot_future = self._pool.submit(self.robot.send_action, action)
        teleop_future = self._pool.submit(self.teleop.send_feedback, action)
        sent_action = robot_future.result()
        teleop_future.result()
        return sent_action

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
