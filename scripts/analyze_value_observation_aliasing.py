#!/usr/bin/env python
"""Diagnose single-frame value aliasing on held-out LeRobot episodes.

The tool is intentionally read-only with respect to the source dataset. It finds
pairs whose current-frame representations are very similar even though their
Monte-Carlo value targets differ, then measures whether a value checkpoint
collapses that difference. For manual review it exports both current frames and
their recent histories in side-by-side montages.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.modeling_pistar06 import Pistar06Policy
from lerobot.values.pistar06.processor_pistar06 import PISTAR06_IMAGES_KEY
from lerobot.values.rewards import build_successful_value_targets


DEFAULT_VALIDATION_EPISODES = [1, 4, 10, 21, 25, 46, 50, 55, 56, 67, 71, 72, 89, 91, 92, 93]


@dataclass(frozen=True)
class FrameRecord:
    sample_index: int
    dataset_position: int
    absolute_index: int
    episode_index: int
    frame_index: int
    target: float


@dataclass(frozen=True)
class AliasingPair:
    rank: int
    left_sample_index: int
    right_sample_index: int
    left_episode: int
    left_frame: int
    right_episode: int
    right_frame: int
    representation_similarity: float
    pixel_similarity: float
    target_left: float
    target_right: float
    target_gap: float
    prediction_left: dict[str, float]
    prediction_right: dict[str, float]
    prediction_gap: dict[str, float]
    gap_recovery_ratio: dict[str, float]
    ordering_correct: dict[str, bool]
    score: float
    montage: str


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return values


def checkpoint_name(path: Path) -> str:
    if path.name == "pretrained_model" and path.parent.name.isdigit():
        return path.parent.name
    return path.name


def load_checkpoint(path: Path, dataset: LeRobotDataset, device: torch.device):
    cfg = PreTrainedConfig.from_pretrained(path)
    if not isinstance(cfg, Pistar06Config):
        raise TypeError(f"Expected a Pistar06 checkpoint at {path}, got {type(cfg).__name__}.")
    if cfg.mode != "vision_only":
        raise ValueError("This diagnostic currently requires a vision_only Pistar06 checkpoint.")
    cfg.pretrained_path = path
    cfg.device = device.type
    policy = Pistar06Policy.from_pretrained(path, config=cfg).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device.type}},
        dataset_stats=dataset.meta.stats,
    )
    return policy, cfg, preprocessor


def normalized_pixel_descriptor(images: torch.Tensor, size: int = 24) -> torch.Tensor:
    images = images.detach().to(dtype=torch.float32, device="cpu")
    if images.ndim != 4:
        raise ValueError(f"Expected image batch [B,C,H,W], got {tuple(images.shape)}.")
    small = F.interpolate(images, size=(size, size), mode="area")
    flat = small.flatten(start_dim=1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    return F.normalize(flat, dim=1)


@torch.inference_mode()
def infer_checkpoint(
    checkpoint: Path,
    dataset: LeRobotDataset,
    loader: DataLoader,
    camera_key: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy, _, preprocessor = load_checkpoint(checkpoint, dataset, device)
    predictions: list[torch.Tensor] = []
    representations: list[torch.Tensor] = []
    pixel_descriptors: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def capture_final_representation(_module, _inputs, output):
        captured.append(output.detach().to(dtype=torch.float32, device="cpu"))

    hook = policy.model.final_norm.register_forward_hook(capture_final_representation)
    try:
        for raw_batch in loader:
            pixel_descriptors.append(normalized_pixel_descriptor(raw_batch[camera_key]))
            processed = preprocessor(raw_batch)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = policy.predict_value(processed)
            if len(captured) != 1:
                raise RuntimeError(f"Expected one captured representation, got {len(captured)}.")
            predictions.append(prediction.detach().to(dtype=torch.float32, device="cpu"))
            representations.append(F.normalize(captured.pop(), dim=1))
    finally:
        hook.remove()
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        torch.cat(predictions).numpy(),
        torch.cat(representations).numpy(),
        torch.cat(pixel_descriptors).numpy(),
    )


def candidate_pairs(
    representations: np.ndarray,
    records: list[FrameRecord],
    predictions: dict[str, np.ndarray],
    pixel_descriptors: np.ndarray,
    nearest_neighbors: int,
    min_target_gap: float,
) -> list[dict[str, Any]]:
    features = torch.from_numpy(representations).to(dtype=torch.float32)
    similarity = features @ features.T
    episodes = torch.tensor([record.episode_index for record in records])
    similarity.masked_fill_(episodes[:, None] == episodes[None, :], -torch.inf)
    k = min(nearest_neighbors, max(1, len(records) - 1))
    top_similarity, top_index = torch.topk(similarity, k=k, dim=1)

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left in range(len(records)):
        for sim, right_tensor in zip(top_similarity[left].tolist(), top_index[left].tolist(), strict=True):
            right = int(right_tensor)
            key = (min(left, right), max(left, right))
            if key in seen or not np.isfinite(sim):
                continue
            seen.add(key)
            target_delta = records[left].target - records[right].target
            target_gap = abs(target_delta)
            if target_gap < min_target_gap:
                continue

            pred_gap = {name: abs(float(values[left] - values[right])) for name, values in predictions.items()}
            recovery = {name: gap / target_gap for name, gap in pred_gap.items()}
            ordering = {
                name: bool((float(values[left] - values[right]) * target_delta) > 0)
                for name, values in predictions.items()
            }
            primary = next(iter(predictions))
            # High representation similarity, a large true gap, and a collapsed
            # predicted gap jointly prioritize the clearest aliasing examples.
            collapse = max(0.0, 1.0 - min(recovery[primary], 1.0))
            score = float(max(sim, 0.0) * target_gap * (1.0 + collapse))
            pixel_similarity = float(np.dot(pixel_descriptors[left], pixel_descriptors[right]))
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "similarity": float(sim),
                    "pixel_similarity": pixel_similarity,
                    "target_gap": target_gap,
                    "prediction_gap": pred_gap,
                    "gap_recovery_ratio": recovery,
                    "ordering_correct": ordering,
                    "score": score,
                }
            )
    return sorted(pairs, key=lambda item: item["score"], reverse=True)


def select_diverse_pairs(
    candidates: list[dict[str, Any]],
    records: list[FrameRecord],
    max_pairs: int,
    max_per_episode_pair: int,
    max_per_frame: int,
) -> list[dict[str, Any]]:
    selected = []
    episode_pair_counts: dict[tuple[int, int], int] = {}
    frame_counts: dict[int, int] = {}
    for candidate in candidates:
        left = int(candidate["left"])
        right = int(candidate["right"])
        episode_pair = tuple(sorted((records[left].episode_index, records[right].episode_index)))
        if episode_pair_counts.get(episode_pair, 0) >= max_per_episode_pair:
            continue
        if frame_counts.get(left, 0) >= max_per_frame or frame_counts.get(right, 0) >= max_per_frame:
            continue
        selected.append(candidate)
        episode_pair_counts[episode_pair] = episode_pair_counts.get(episode_pair, 0) + 1
        frame_counts[left] = frame_counts.get(left, 0) + 1
        frame_counts[right] = frame_counts.get(right, 0) + 1
        if len(selected) == max_pairs:
            break
    return selected


def tensor_to_image(tensor: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    tensor = tensor.detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(tensor.shape)}.")
    if tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.LANCZOS)


def render_pair_montage(
    output_path: Path,
    dataset: LeRobotDataset,
    record_by_episode_frame: dict[tuple[int, int], int],
    left: FrameRecord,
    right: FrameRecord,
    camera_key: str,
    history_offsets: list[int],
) -> None:
    offsets = sorted(history_offsets) + [0]
    tile_size = (320, 180)
    header_height = 34
    canvas = Image.new("RGB", (tile_size[0] * len(offsets), 2 * (tile_size[1] + header_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, record in enumerate((left, right)):
        row_y = row * (tile_size[1] + header_height)
        draw.text(
            (5, row_y + 4),
            f"episode={record.episode_index} frame={record.frame_index} target={record.target:.4f}",
            fill="black",
        )
        for column, offset in enumerate(offsets):
            query_frame = max(0, record.frame_index + offset)
            position = record_by_episode_frame[(record.episode_index, query_frame)]
            image = tensor_to_image(dataset[position][camera_key], tile_size)
            x = column * tile_size[0]
            y = row_y + header_height
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + 105, y + 19), fill="black")
            draw.text((x + 3, y + 3), f"offset={offset:+d}", fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    records: list[FrameRecord],
    predictions: dict[str, np.ndarray],
    pairs: list[AliasingPair],
) -> None:
    metrics = {}
    targets = np.asarray([record.target for record in records], dtype=np.float32)
    for name, values in predictions.items():
        metrics[name] = {
            "mae": float(np.mean(np.abs(values - targets))),
            "rmse": float(np.sqrt(np.mean((values - targets) ** 2))),
            "correlation": float(np.corrcoef(values, targets)[0, 1]),
        }

    payload = {
        "configuration": {
            "dataset_root": str(args.dataset_root),
            "episodes": args.episodes,
            "stride": args.stride,
            "min_frame_index": args.min_frame_index,
            "history_offsets": args.history_offsets,
            "nearest_neighbors": args.nearest_neighbors,
            "min_target_gap": args.min_target_gap,
        },
        "sample_count": len(records),
        "metrics": metrics,
        "aliasing_pair_count": len(pairs),
        "pairs": [asdict(pair) for pair in pairs],
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (output_dir / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = [asdict(pair) for pair in pairs]
        if rows:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Single-frame value observation-aliasing report",
        "",
        f"- Held-out sampled frames: {len(records)}",
        f"- Candidate aliasing pairs retained: {len(pairs)}",
        f"- Sampling stride: {args.stride} frames",
        f"- Minimum frame index: {args.min_frame_index}",
        f"- Manual-review history offsets: {args.history_offsets} frames",
        "",
        "## Checkpoint metrics",
        "",
        "| checkpoint | MAE | RMSE | Pearson correlation |",
        "|---|---:|---:|---:|",
    ]
    for name, result in metrics.items():
        lines.append(
            f"| {name} | {result['mae']:.6f} | {result['rmse']:.6f} | {result['correlation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Highest-priority manual-review pairs",
            "",
            "A small prediction-gap/target-gap ratio means the single-frame model collapsed states "
            "that the return target considers different. The montage provides history only for diagnosis; "
            "the evaluated checkpoints still received one current frame.",
            "",
        ]
    )
    for pair in pairs:
        primary = next(iter(pair.prediction_gap))
        lines.extend(
            [
                f"### Pair {pair.rank}",
                "",
                f"- Frames: ep{pair.left_episode}/f{pair.left_frame} vs "
                f"ep{pair.right_episode}/f{pair.right_frame}",
                f"- Representation/pixel similarity: {pair.representation_similarity:.4f} / "
                f"{pair.pixel_similarity:.4f}",
                f"- Target gap: {pair.target_gap:.4f}",
                f"- {primary} predicted gap / recovery: {pair.prediction_gap[primary]:.4f} / "
                f"{pair.gap_recovery_ratio[primary]:.3f}",
                f"- [Montage]({pair.montage})",
                "",
            ]
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="success_valueModelTrain")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--episodes",
        type=parse_int_list,
        default=DEFAULT_VALIDATION_EPISODES,
        help="Comma-separated held-out episode indices.",
    )
    parser.add_argument("--camera-key", default="observation.images.front")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min-frame-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nearest-neighbors", type=int, default=30)
    parser.add_argument("--min-target-gap", type=float, default=0.15)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--max-per-episode-pair", type=int, default=2)
    parser.add_argument("--max-per-frame", type=int, default=3)
    parser.add_argument(
        "--history-offsets",
        type=parse_int_list,
        default=[-25, -12],
        help="Comma-separated frame offsets used only in manual-review montages.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stride <= 0 or args.batch_size <= 0 or args.num_workers < 0 or args.min_frame_index < 0:
        parser.error(
            "stride/batch-size must be positive; num-workers/min-frame-index must be non-negative."
        )
    if args.nearest_neighbors <= 0 or args.max_pairs <= 0 or args.min_target_gap <= 0:
        parser.error("nearest-neighbors, max-pairs and min-target-gap must be positive.")
    if any(offset >= 0 for offset in args.history_offsets):
        parser.error("history-offsets must contain only negative frame offsets.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=args.episodes,
        video_backend=args.video_backend,
    )
    raw_frames = dataset.hf_dataset.with_format(None)
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)
    sample_positions = np.flatnonzero(
        (frame_indices % args.stride == 0) & (frame_indices >= args.min_frame_index)
    ).tolist()
    loader = DataLoader(
        Subset(dataset, sample_positions),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint_configs = [PreTrainedConfig.from_pretrained(path) for path in args.checkpoint]
    first_cfg = checkpoint_configs[0]
    if not isinstance(first_cfg, Pistar06Config) or first_cfg.reward is None:
        raise ValueError("The first checkpoint must contain Pistar06 reward configuration.")
    targets, _ = build_successful_value_targets(
        dataset,
        first_cfg.reward,
        "episode_success",
        task_scales=first_cfg.reward_scales,
    )
    records = [
        FrameRecord(
            sample_index=sample_index,
            dataset_position=position,
            absolute_index=int(absolute_indices[position]),
            episode_index=int(episode_indices[position]),
            frame_index=int(frame_indices[position]),
            target=float(targets[position]),
        )
        for sample_index, position in enumerate(sample_positions)
    ]

    predictions: dict[str, np.ndarray] = {}
    reference_representation = None
    pixel_descriptors = None
    for checkpoint in args.checkpoint:
        name = checkpoint_name(checkpoint)
        logging.info("Evaluating checkpoint %s on %d sampled frames", name, len(records))
        prediction, representation, pixels = infer_checkpoint(
            checkpoint=checkpoint,
            dataset=dataset,
            loader=loader,
            camera_key=args.camera_key,
            device=device,
        )
        predictions[name] = prediction
        if reference_representation is None:
            reference_representation = representation
            pixel_descriptors = pixels

    assert reference_representation is not None and pixel_descriptors is not None
    all_candidate_pairs = candidate_pairs(
        representations=reference_representation,
        records=records,
        predictions=predictions,
        pixel_descriptors=pixel_descriptors,
        nearest_neighbors=args.nearest_neighbors,
        min_target_gap=args.min_target_gap,
    )
    raw_pairs = select_diverse_pairs(
        candidates=all_candidate_pairs,
        records=records,
        max_pairs=args.max_pairs,
        max_per_episode_pair=args.max_per_episode_pair,
        max_per_frame=args.max_per_frame,
    )

    record_by_episode_frame = {
        (int(episode), int(frame)): position
        for position, (episode, frame) in enumerate(zip(episode_indices, frame_indices, strict=True))
    }
    pairs = []
    for rank, raw in enumerate(raw_pairs, start=1):
        left = records[raw["left"]]
        right = records[raw["right"]]
        montage_relative = Path("montages") / f"pair_{rank:03d}.png"
        render_pair_montage(
            output_path=args.output_dir / montage_relative,
            dataset=dataset,
            record_by_episode_frame=record_by_episode_frame,
            left=left,
            right=right,
            camera_key=args.camera_key,
            history_offsets=args.history_offsets,
        )
        pairs.append(
            AliasingPair(
                rank=rank,
                left_sample_index=left.sample_index,
                right_sample_index=right.sample_index,
                left_episode=left.episode_index,
                left_frame=left.frame_index,
                right_episode=right.episode_index,
                right_frame=right.frame_index,
                representation_similarity=raw["similarity"],
                pixel_similarity=raw["pixel_similarity"],
                target_left=left.target,
                target_right=right.target,
                target_gap=raw["target_gap"],
                prediction_left={name: float(values[raw["left"]]) for name, values in predictions.items()},
                prediction_right={name: float(values[raw["right"]]) for name, values in predictions.items()},
                prediction_gap=raw["prediction_gap"],
                gap_recovery_ratio=raw["gap_recovery_ratio"],
                ordering_correct=raw["ordering_correct"],
                score=raw["score"],
                montage=str(montage_relative),
            )
        )

    write_report(args.output_dir, args, records, predictions, pairs)
    logging.info("Wrote diagnostic report to %s", args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
