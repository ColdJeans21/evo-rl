#!/usr/bin/env python
"""Build a paper-style Evo-RL stage-2 segment-selection plan.

The source dataset is read-only. The generated JSON identifies four sampling
sources: complete demonstrations, boosted high-advantage demonstration windows,
high-advantage successful autonomous windows, and contiguous human-correction
segments from successful intervention episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_parquets(root: Path, subdir: str, columns: list[str] | None = None) -> pa.Table:
    paths = sorted((root / subdir).rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root / subdir}")
    return pa.concat_tables([pq.read_table(path, columns=columns) for path in paths]).combine_chunks()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_true_segments(mask: np.ndarray, offset: int) -> list[list[int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [[offset + int(start), offset + int(end)] for start, end in zip(starts, ends, strict=True)]


def merge_forward_windows(starts: list[int], episode_end: int, window_size: int) -> list[list[int]]:
    windows = [[start, min(start + window_size, episode_end)] for start in sorted(set(starts))]
    if not windows:
        return []
    merged = [windows[0]]
    for start, end in windows[1:]:
        # Half-open windows are adjacent when the next start equals the current end.
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def segment_frame_count(segments: list[list[int]]) -> int:
    return sum(end - start for start, end in segments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--indicator-field", required=True)
    parser.add_argument("--advantage-field", required=True)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    info_path = root / "meta/info.json"
    merge_manifest_path = root / "merge_manifest.json"
    if not info_path.is_file() or not merge_manifest_path.is_file():
        raise FileNotFoundError("Dataset requires meta/info.json and merge_manifest.json")

    info = json.loads(info_path.read_text())
    merge_manifest = json.loads(merge_manifest_path.read_text())
    manifest_episodes = {int(item["episode_index"]): item for item in merge_manifest["episodes"]}
    episodes = read_parquets(root, "meta/episodes").sort_by("episode_index")
    required = [
        "index",
        "episode_index",
        "frame_index",
        "complementary_info.is_intervention",
        args.indicator_field,
        args.advantage_field,
    ]
    frames = read_parquets(root, "data", required).sort_by("index")
    if len(frames) != int(info["total_frames"]):
        raise ValueError("Frame count does not match meta/info.json")
    absolute_indices = frames["index"].to_numpy().astype(np.int64, copy=False)
    if not np.array_equal(absolute_indices, np.arange(len(frames))):
        raise ValueError("Stage-2 selection requires contiguous absolute frame indices")

    pools: dict[str, list[list[int]]] = {
        "demo_base": [],
        "demo_selected": [],
        "auto_selected": [],
        "intervention": [],
    }
    selected_start_counts = {"demo_selected": 0, "auto_selected": 0}
    episode_counts = {"demonstration": 0, "autonomous": 0, "intervention": 0}

    indicators = frames[args.indicator_field].to_numpy().astype(np.int64, copy=False)
    advantages = frames[args.advantage_field].to_numpy().astype(np.float32, copy=False)
    interventions = frames["complementary_info.is_intervention"].to_numpy().astype(np.float32, copy=False)

    for episode in episodes.to_pylist():
        episode_index = int(episode["episode_index"])
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        length = end - start
        source = manifest_episodes.get(episode_index)
        if source is None:
            raise ValueError(f"Episode {episode_index} is missing from merge_manifest.json")
        if length != int(source["frames"]):
            raise ValueError(f"Episode {episode_index} length disagrees with merge manifest")

        source_type = source["source_type"]
        human_frames = int(source["human_flag_frames"])
        if source_type == "demonstration":
            episode_counts["demonstration"] += 1
            pools["demo_base"].append([start, end])
            selection_pool = "demo_selected"
        elif source_type == "rollout" and human_frames == 0:
            episode_counts["autonomous"] += 1
            selection_pool = "auto_selected"
        elif source_type == "rollout" and human_frames > 0:
            episode_counts["intervention"] += 1
            episode_mask = interventions[start:end] > 0.5
            measured_human = int(episode_mask.sum())
            if measured_human != human_frames:
                raise ValueError(
                    f"Episode {episode_index} intervention count changed: {measured_human} != {human_frames}"
                )
            pools["intervention"].extend(contiguous_true_segments(episode_mask, start))
            continue
        else:
            raise ValueError(f"Unsupported source classification for episode {episode_index}: {source}")

        # The terminal observation is never a selection start. Requiring a
        # strictly positive advantage is checked again even if inference used
        # require_positive_advantage=true, making the plan self-validating.
        local_selected = np.flatnonzero(
            (indicators[start:end] == 1) & np.isfinite(advantages[start:end]) & (advantages[start:end] > 0.0)
        )
        selected_starts = [start + int(local) for local in local_selected if int(local) < length - 1]
        selected_start_counts[selection_pool] += len(selected_starts)
        pools[selection_pool].extend(merge_forward_windows(selected_starts, end, args.window_size))

    for name, segments in pools.items():
        if any(start < 0 or end <= start or end > len(frames) for start, end in segments):
            raise ValueError(f"Invalid segment bounds in pool {name}")

    stats = {
        "episode_counts": episode_counts,
        "selected_start_counts": selected_start_counts,
        "segment_counts": {name: len(segments) for name, segments in pools.items()},
        "frame_counts": {name: segment_frame_count(segments) for name, segments in pools.items()},
    }
    empty_required = [
        name for name in ("demo_base", "demo_selected", "auto_selected", "intervention") if not pools[name]
    ]
    if empty_required:
        raise ValueError(f"Required selection pools are empty: {empty_required}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selection plan: {output}")
    plan = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "repo_id": info["codebase_version"] if "repo_id" not in merge_manifest else merge_manifest["repo_id"],
        "total_frames": len(frames),
        "fps": int(info["fps"]),
        "indicator_field": args.indicator_field,
        "advantage_field": args.advantage_field,
        "window_size": args.window_size,
        "source_info_sha256": sha256(info_path),
        "source_merge_manifest_sha256": sha256(merge_manifest_path),
        "pools": pools,
        "stats": stats,
    }
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
