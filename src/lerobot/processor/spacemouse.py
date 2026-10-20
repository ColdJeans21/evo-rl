from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.core import RobotAction, RobotObservation, TransitionKey
from lerobot.processor.pipeline import RobotActionProcessorStep, RobotProcessorPipeline
from lerobot.utils.rotation import Rotation

from .converters import robot_action_observation_to_transition, transition_to_robot_action


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class SpaceMouseToSO101Step(RobotActionProcessorStep):
    def __init__(self, config, robot):
        self.config = config
        self.motor_names = MOTOR_NAMES
        urdf_path = config.urdf_path or Path(__file__).parent.parent / "teleoperators" / "spacemouse" / "urdfs" / "so101_new_calib.urdf"
        self.kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=self.motor_names[:-1],
        )
        self._use_degrees = bool(getattr(getattr(robot, "config", None), "use_degrees", False))
        self._bus = robot.bus
        self._target = None
        self._gripper = None
        self._robot = robot

    def _native_to_degrees(self, values: np.ndarray) -> np.ndarray:
        if self._use_degrees:
            return np.array(values, dtype=float, copy=True)
        degrees = []
        for value, name in zip(values, self.motor_names[:-1]):
            calibration = self._bus.calibration[name]
            motor = self._bus.motors[name]
            resolution = self._bus.model_resolution_table[motor.model] - 1
            raw = calibration.range_min + ((float(value) + 100.0) / 200.0) * (
                calibration.range_max - calibration.range_min
            )
            midpoint = (calibration.range_min + calibration.range_max) / 2.0
            degrees.append((raw - midpoint) * 360.0 / resolution)
        return np.asarray(degrees, dtype=float)

    def _degrees_to_native(self, values: np.ndarray) -> np.ndarray:
        if self._use_degrees:
            return np.array(values, dtype=float, copy=True)
        normalized = []
        for value, name in zip(values, self.motor_names[:-1]):
            calibration = self._bus.calibration[name]
            motor = self._bus.motors[name]
            resolution = self._bus.model_resolution_table[motor.model] - 1
            midpoint = (calibration.range_min + calibration.range_max) / 2.0
            raw = float(value) * resolution / 360.0 + midpoint
            normalized.append(
                np.clip(
                    ((raw - calibration.range_min) / (calibration.range_max - calibration.range_min)) * 200.0
                    - 100.0,
                    -100.0,
                    100.0,
                )
            )
        return np.asarray(normalized, dtype=float)

    def action(self, action: RobotAction) -> RobotAction:
        observation: RobotObservation | None = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("SpaceMouse IK requires robot observation.")
        current = np.asarray([float(observation[f"{name}.pos"]) for name in self.motor_names], dtype=float)
        current_degrees = self._native_to_degrees(current[:-1])
        if self._target is None:
            self._target = np.array(self.kinematics.forward_kinematics(current_degrees), copy=True)
            self._gripper = current[-1]
        enabled = bool(action.get("enabled", False))
        if enabled:
            delta_position = np.asarray(
                [float(action.get(f"target_{axis}", 0.0)) for axis in ("x", "y", "z")], dtype=float
            ) * self.config.translation_step_m
            delta_rotation = np.asarray(
                [float(action.get(f"target_w{axis}", 0.0)) for axis in ("x", "y", "z")], dtype=float
            ) * self.config.rotation_step_rad
            self._target[:3, 3] += delta_position
            self._target[:3, :3] = self._target[:3, :3] @ Rotation.from_rotvec(delta_rotation).as_matrix()
            self._gripper = float(
                np.clip(
                    self._gripper + float(action.get("gripper_vel", 0.0)) * self.config.gripper_speed_factor,
                    self.config.gripper_min,
                    self.config.gripper_max,
                )
            )
        else:
            self._target = self.kinematics.forward_kinematics(current_degrees)
            self._gripper = current[-1]
        solved = self.kinematics.inverse_kinematics(
            current_degrees,
            self._target,
            position_weight=self.config.position_weight,
            orientation_weight=self.config.orientation_weight,
        )
        if self.config.max_joint_step_deg > 0:
            solved = current_degrees + np.clip(solved - current_degrees, -self.config.max_joint_step_deg, self.config.max_joint_step_deg)
        solved_native = self._degrees_to_native(solved)
        return {f"{name}.pos": float(value) for name, value in zip(self.motor_names[:-1], solved_native)} | {
            "gripper.pos": float(self._gripper)
        }

    def transform_features(self, features):
        return features

    def reset(self) -> None:
        self._target = None
        self._gripper = None


def make_spacemouse_robot_action_processor(config, robot) -> RobotProcessorPipeline:
    return RobotProcessorPipeline(
        steps=[SpaceMouseToSO101Step(config, robot)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
