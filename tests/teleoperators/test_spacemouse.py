from types import SimpleNamespace
from unittest.mock import patch

from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig


def test_spacemouse_action_mapping():
    state = SimpleNamespace(y=0.5, x=-0.5, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=[0, 1])
    device = SimpleNamespace(connected=True, read=lambda: state, close=lambda: None)
    with patch("importlib.import_module", return_value=SimpleNamespace(open=lambda **_: device)):
        teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(read_drain_count=1))
        teleop.connect()
        action = teleop.get_action()
        teleop.disconnect()

    assert action["enabled"] is True
    assert action["target_x"] > 0
    assert action["target_y"] > 0
    assert action["gripper_vel"] == 1.0


def test_spacemouse_features_are_stable():
    features = SpaceMouseTeleop(SpaceMouseTeleopConfig()).action_features
    assert set(features) == {
        "enabled",
        "target_x",
        "target_y",
        "target_z",
        "target_wx",
        "target_wy",
        "target_wz",
        "gripper_vel",
    }


def test_stale_spacemouse_state_is_stopped():
    stale_state = SimpleNamespace(t=0.0, y=1.0, x=1.0, z=1.0, roll=1.0, pitch=1.0, yaw=1.0, buttons=[1, 1])
    device = SimpleNamespace(connected=True, read=lambda: stale_state, close=lambda: None)
    with patch("importlib.import_module", return_value=SimpleNamespace(open=lambda **_: device)):
        teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(read_drain_count=1, input_timeout_s=0.01))
        teleop.connect()
        action = teleop.get_action()
        teleop.disconnect()

    assert action["enabled"] is False
    assert all(action[key] == 0.0 for key in action if key != "enabled")
