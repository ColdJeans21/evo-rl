"""Reward settings shared by value training and inference."""

import math
from dataclasses import dataclass


@dataclass
class ValueRewardConfig:
    mode: str = "legacy"
    takeover_cost: float = 150.0
    intervention_field: str = "complementary_info.is_intervention"

    def validate(self) -> None:
        if self.mode not in {"legacy", "time", "time_takeover"}:
            raise ValueError("'reward.mode' must be legacy, time or time_takeover.")
        if not math.isfinite(self.takeover_cost) or self.takeover_cost < 0:
            raise ValueError("'reward.takeover_cost' must be finite and non-negative.")
        if self.mode == "time_takeover" and not self.intervention_field:
            raise ValueError("'reward.intervention_field' must be non-empty for time_takeover.")
