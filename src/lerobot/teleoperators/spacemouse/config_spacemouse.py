from dataclasses import dataclass
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass(kw_only=True)
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    device: str | None = None
    deadzone: float = 0.08
    max_axis_value: float = 1.0
    read_drain_count: int = 32
    input_timeout_s: float = 0.08
    x_axis: str = "y"
    y_axis: str = "x"
    z_axis: str = "z"
    wx_axis: str = "roll"
    wy_axis: str = "pitch"
    wz_axis: str = "yaw"
    x_sign: float = 1.0
    y_sign: float = -1.0
    z_sign: float = 1.0
    wx_sign: float = 1.0
    wy_sign: float = 1.0
    wz_sign: float = 1.0
    use_gripper: bool = True
    gripper_open_button: int | None = 1
    gripper_close_button: int | None = 0
    require_enable_button: bool = False
    enable_button: int | None = None
    idle_enabled: bool = False
    urdf_path: Path | None = None
    translation_step_m: float = 0.001
    rotation_step_rad: float = 0.0025
    gripper_speed_factor: float = 0.5
    gripper_min: float = 0.0
    gripper_max: float = 100.0
    max_joint_step_deg: float = 2.0
    position_weight: float = 1.0
    orientation_weight: float = 0.01
