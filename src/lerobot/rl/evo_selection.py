#!/usr/bin/env python

"""Dataset view and weighted sampler for paper-style Evo-RL refinement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from lerobot.configs.train import EvoRLSelectionConfig


@dataclass(frozen=True)
class EvoRLSample:
    frame_index: int
    segment_end: int
    source: str


class EvoRLSelectedDataset(Dataset):
    """Map sampling slots to source frames and mask actions beyond a segment."""

    def __init__(self, dataset: Dataset, samples: list[EvoRLSample]):
        if not samples:
            raise ValueError("Evo-RL selected dataset cannot be empty.")
        self.dataset = dataset
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, slot: int) -> dict[str, Any]:
        sample = self.samples[slot]
        item = self.dataset[sample.frame_index]
        action_is_pad = item.get("action_is_pad")
        if not isinstance(action_is_pad, torch.Tensor) or action_is_pad.dtype != torch.bool:
            raise TypeError("Paper-style Evo-RL selection requires a boolean action_is_pad tensor.")
        if action_is_pad.ndim != 1:
            raise ValueError(
                f"Expected one action horizon, got action_is_pad shape={tuple(action_is_pad.shape)}."
            )

        valid_steps = sample.segment_end - sample.frame_index
        if valid_steps <= 0:
            raise ValueError(f"Invalid Evo-RL sample boundary: {sample}.")
        selection_padding = torch.arange(action_is_pad.shape[0]) >= valid_steps
        item["action_is_pad"] = action_is_pad | selection_padding
        return item

    def __getattr__(self, name: str) -> Any:
        # Preserve the LeRobotDataset interface used by logging and processors.
        return getattr(self.dataset, name)

    @property
    def num_frames(self) -> int:
        return len(self.samples)


def _expand_segments(source: str, segments: list[list[int]], total_frames: int) -> list[EvoRLSample]:
    samples = []
    for bounds in segments:
        if len(bounds) != 2:
            raise ValueError(f"Invalid segment in source '{source}': {bounds}")
        start, end = (int(bounds[0]), int(bounds[1]))
        if start < 0 or end <= start or end > total_frames:
            raise ValueError(f"Out-of-range segment in source '{source}': {bounds}")
        samples.extend(
            EvoRLSample(frame_index=index, segment_end=end, source=source) for index in range(start, end)
        )
    return samples


def make_evo_selection_dataset_and_sampler(
    dataset: Dataset,
    cfg: EvoRLSelectionConfig,
    seed: int | None,
) -> tuple[EvoRLSelectedDataset, WeightedRandomSampler, dict[str, Any]]:
    if not cfg.enable or cfg.plan_path is None:
        raise ValueError("Evo-RL selection must be enabled with a plan path.")
    if getattr(dataset, "episodes", None) is not None:
        raise ValueError("Evo-RL selection plans require loading the complete source dataset.")

    plan_path = Path(cfg.plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get("version") != 1:
        raise ValueError(f"Unsupported Evo-RL selection plan version: {plan.get('version')}")
    total_frames = int(plan["total_frames"])
    dataset_total_frames = int(dataset.num_frames)
    if dataset_total_frames != total_frames:
        raise ValueError(
            f"Selection plan/dataset frame mismatch: plan={total_frames}, dataset={dataset_total_frames}."
        )
    dataset_root = Path(dataset.root).resolve()
    if dataset_root != Path(plan["dataset_root"]).resolve():
        raise ValueError(f"Selection plan targets {plan['dataset_root']}, loaded dataset is {dataset_root}.")

    source_weights = {
        "demo_base": cfg.demo_base_weight,
        "intervention": cfg.intervention_weight,
        "demo_selected": cfg.demo_selected_weight,
        "auto_selected": cfg.auto_selected_weight,
    }
    samples: list[EvoRLSample] = []
    weights: list[float] = []
    source_counts: dict[str, int] = {}
    for source, source_weight in source_weights.items():
        source_samples = _expand_segments(source, plan["pools"].get(source, []), total_frames)
        if not source_samples:
            raise ValueError(f"Selection pool '{source}' is empty.")
        source_counts[source] = len(source_samples)
        samples.extend(source_samples)
        weights.extend([source_weight / len(source_samples)] * len(source_samples))

    selected_dataset = EvoRLSelectedDataset(dataset, samples)
    num_samples = cfg.samples_per_epoch if cfg.samples_per_epoch > 0 else total_frames
    generator = torch.Generator()
    generator.manual_seed(0 if seed is None else seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    summary = {
        "plan_path": str(plan_path),
        "sampling_slots": len(samples),
        "samples_per_epoch": num_samples,
        "source_frame_counts": source_counts,
        "source_probabilities": source_weights,
        "plan_stats": plan.get("stats", {}),
    }
    return selected_dataset, sampler, summary
