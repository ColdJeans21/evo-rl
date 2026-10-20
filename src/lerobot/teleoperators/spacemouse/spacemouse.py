from __future__ import annotations

import importlib
import time
from typing import Any

from lerobot.processor import RobotAction

from ..teleoperator import Teleoperator
from .config_spacemouse import SpaceMouseTeleopConfig


AXES = ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz")


def _deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value >= 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def _buttons(state: Any) -> list[int]:
    return list(getattr(state, "buttons", getattr(state, "button", [])) or [])


class SpaceMouseTeleop(Teleoperator):
    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._driver: Any | None = None
        self._device: Any | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {"enabled": bool, **{axis: float for axis in AXES}, "gripper_vel": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._device is not None and bool(getattr(self._device, "connected", True))

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._driver = importlib.import_module("pyspacemouse")
        self._device = self._driver.open(device=self.config.device)
        if self._device is None:
            raise ConnectionError("No supported SpaceMouse device found.")

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> RobotAction:
        if not self.is_connected:
            raise ConnectionError("SpaceMouseTeleop is not connected.")
        state = None
        for _ in range(max(1, self.config.read_drain_count)):
            next_state = self._device.read()
            if next_state is None:
                break
            state = next_state
        values = {axis: 0.0 for axis in AXES}
        now = time.perf_counter()
        stale = (
            state is None
            or getattr(state, "t", -1.0) >= 0.0
            and self.config.input_timeout_s > 0.0
            and now - float(state.t) > self.config.input_timeout_s
        )
        buttons = [] if stale else _buttons(state)
        if state is not None and not stale:
            for output, source, sign in (
                ("target_x", self.config.x_axis, self.config.x_sign),
                ("target_y", self.config.y_axis, self.config.y_sign),
                ("target_z", self.config.z_axis, self.config.z_sign),
                ("target_wx", self.config.wx_axis, self.config.wx_sign),
                ("target_wy", self.config.wy_axis, self.config.wy_sign),
                ("target_wz", self.config.wz_axis, self.config.wz_sign),
            ):
                raw = float(getattr(state, source, 0.0) or 0.0)
                raw = max(-self.config.max_axis_value, min(self.config.max_axis_value, raw))
                values[output] = sign * _deadzone(raw, self.config.deadzone)
        gripper = 0.0
        if self.config.use_gripper:
            if self.config.gripper_open_button is not None and self.config.gripper_open_button < len(buttons):
                gripper += float(bool(buttons[self.config.gripper_open_button]))
            if self.config.gripper_close_button is not None and self.config.gripper_close_button < len(buttons):
                gripper -= float(bool(buttons[self.config.gripper_close_button]))
        if self.config.require_enable_button:
            enabled = self.config.enable_button is not None and self.config.enable_button < len(buttons) and bool(buttons[self.config.enable_button])
        else:
            enabled = self.config.idle_enabled or any(values.values()) or bool(gripper)
        return {"enabled": enabled, **values, "gripper_vel": gripper}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        if self._device is not None and hasattr(self._device, "close"):
            self._device.close()
        self._device = None
        self._driver = None
