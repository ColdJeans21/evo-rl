#!/usr/bin/env python
"""Build a verified local success-only LeRobot v3 dataset without re-encoding videos."""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.configs.value_reward import ValueRewardConfig
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
from lerobot.values.rewards import build_successful_value_targets

INTERVENTION = "complementary_info.is_intervention"


def read_parquets(root: Path, subdir: str) -> pa.Table:
    files = sorted((root / subdir).rglob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files in {root / subdir}")
    return pa.concat_tables([pq.read_table(path) for path in files]).combine_chunks()


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def confirm_success(parent: Path, episode_ids: list[int]) -> list[dict]:
    """Apply only caller-confirmed labels, retaining immutable original files."""
    if not episode_ids:
        return []
    plans = []
    for episode_id in episode_ids:
        root = parent / f"selfRoll_episode{episode_id}"
        paths = sorted((root / "meta/episodes").rglob("*.parquet"))
        if len(paths) != 1:
            raise ValueError(f"Expected one metadata file for {root}")
        table = pq.read_table(paths[0])
        labels = table["episode_success"].to_pylist()
        if labels not in (["failure"], ["success"]):
            raise ValueError(f"Expected one explicitly labelled episode in {root}: {labels}")
        plans.append((episode_id, paths[0], table, labels[0]))

    backup = Path(tempfile.mkdtemp(prefix="_success_label_backup_", dir=parent))
    records = []
    for episode_id, path, table, old_label in plans:
        record = {"episode": episode_id, "old_label": old_label, "new_label": "success"}
        record["source"] = str(path)
        record["sha256_before"] = digest(path)
        saved = backup / path.relative_to(parent)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        record["backup"] = str(saved)
        if old_label != "success":
            index = table.schema.get_field_index("episode_success")
            field = table.schema.field(index)
            updated = table.set_column(index, field, pa.array(["success"], type=field.type))
            fd, temp_name = tempfile.mkstemp(prefix=".success-label-", suffix=".parquet", dir=path.parent)
            os.close(fd)
            pq.write_table(updated, temp_name)
            if not pq.read_table(temp_name).equals(updated):
                raise ValueError(f"Label verification failed: {temp_name}")
            os.replace(temp_name, path)
        record["sha256_after"] = digest(path)
        records.append(record)
        # Save after each edit so even an interrupted run has an audit trail.
        (backup / "relabel_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def validate_source(root: Path) -> tuple[dict, pa.Table, pa.Table]:
    info = json.loads((root / "meta/info.json").read_text())
    meta = read_parquets(root, "meta/episodes").sort_by("episode_index")
    data = read_parquets(root, "data").sort_by("index")
    if len(meta) != info["total_episodes"] or len(data) != info["total_frames"]:
        raise ValueError(f"Metadata/frame counts disagree in {root}")
    if not np.array_equal(data["index"].to_numpy(), np.arange(len(data))):
        raise ValueError(f"Non-contiguous global frame index in {root}")
    if not np.array_equal(meta["episode_index"].to_numpy(), np.arange(len(meta))):
        raise ValueError(f"Non-contiguous episode index in {root}")
    if not np.isin(data[INTERVENTION].to_numpy(), [0, 1]).all():
        raise ValueError(f"Invalid intervention flags in {root}")
    expected_from = 0
    for episode in meta.to_pylist():
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        if start != expected_from or end - start != episode["length"]:
            raise ValueError(f"Invalid episode bounds in {root}: {episode['episode_index']}")
        frames = data.slice(start, end - start)
        if not np.all(frames["episode_index"].to_numpy() == episode["episode_index"]):
            raise ValueError(f"Wrong episode index in {root}")
        if not np.array_equal(frames["frame_index"].to_numpy(), np.arange(len(frames))):
            raise ValueError(f"Non-contiguous episode frames in {root}")
        if not np.allclose(frames["timestamp"].to_numpy(), np.arange(len(frames)) / info["fps"], atol=1e-4):
            raise ValueError(f"Invalid timestamps in {root}")
        expected_from = end
    if expected_from != len(data):
        raise ValueError(f"Trailing frames without metadata in {root}")
    return info, meta, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("/home/xiaozhi/so101_collect_data"))
    parser.add_argument("--demo-name", default="pi05_demo_57_v1")
    parser.add_argument("--output-name", default="success_valueModelTrain")
    parser.add_argument("--max-rollout", type=int, default=52)
    parser.add_argument("--confirm-success", type=int, nargs="*", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    parent = args.parent.resolve()
    # Names, not arbitrary paths, keep source selection and output location explicit.
    for name in (args.demo_name, args.output_name):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid dataset directory name: {name!r}")
    destination = parent / args.output_name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if not 1 <= args.max_rollout:
        raise ValueError("--max-rollout must be positive")

    relabels = confirm_success(parent, sorted(set(args.confirm_success)))
    included, excluded = [], []
    roots = [parent / args.demo_name] + [
        parent / f"selfRoll_episode{index}" for index in range(1, args.max_rollout + 1)
    ]
    for root in roots:
        info, meta, data = validate_source(root)
        labels = meta["episode_success"].to_pylist()
        if not all(label == "success" for label in labels):
            if root.name == args.demo_name or any(label not in {"success", "failure"} for label in labels):
                raise ValueError(f"Unverified demonstration or label in {root}: {labels}")
            if "success" in labels:
                raise ValueError(f"Mixed labels in {root}; split explicitly before merging")
            excluded.append({"root": str(root), "labels": labels, "reason": "failure"})
            continue
        included.append((root, info, meta, data))

    source_hashes = {
        str(path): digest(path)
        for root, *_ in included
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".parquet", ".mp4"}
    }
    work = Path(tempfile.mkdtemp(prefix=f".{args.output_name}.build-", dir=parent))
    staging = work / args.output_name
    logging.info("Building %s from %d sources; failed builds remain at %s", destination, len(included), work)
    aggregate_datasets(
        repo_ids=[root.name for root, *_ in included],
        roots=[root for root, *_ in included],
        aggr_repo_id=args.output_name,
        aggr_root=staging,
        # Zero forces file rotation, so videos are copied byte-for-byte rather than concatenated.
        video_files_size_in_mb=0,
    )
    _, merged_meta, merged_data = validate_source(staging)
    if merged_meta["episode_success"].to_pylist() != ["success"] * len(merged_meta):
        raise ValueError("Merged output contains non-success labels")

    camera_keys = [key for key, feature in included[0][1]["features"].items() if feature["dtype"] == "video"]
    episodes, video_hashes = [], {}
    frame_offset = episode_offset = 0
    for root, _, meta, data in included:
        merged_part = merged_data.slice(frame_offset, len(data))
        unchanged_columns = [
            key for key in data.column_names if key not in {"index", "episode_index", "task_index"}
        ]
        original_values = data.select(unchanged_columns)
        # The upstream pandas-based merger can change fixed-size lists to lists;
        # cast back to the source schema before comparing every stored value.
        merged_values = merged_part.select(unchanged_columns).cast(original_values.schema)
        if not original_values.equals(merged_values):
            raise ValueError(f"Frame content changed while merging {root}")
        for source_episode in meta.to_pylist():
            source_id = source_episode["episode_index"]
            target_id = episode_offset + source_id
            target_episode = merged_meta.slice(target_id, 1).to_pylist()[0]
            if target_episode["tasks"] != source_episode["tasks"]:
                raise ValueError("Episode tasks changed")
            for camera in camera_keys:
                def video_path(dataset_root, episode):
                    return dataset_root / DEFAULT_VIDEO_PATH.format(
                        video_key=camera,
                        chunk_index=episode[f"videos/{camera}/chunk_index"],
                        file_index=episode[f"videos/{camera}/file_index"],
                    )

                source_video = video_path(root, source_episode)
                target_video = video_path(staging, target_episode)
                key = str(target_video.relative_to(staging))
                if key not in video_hashes:
                    video_hashes[key] = digest(target_video)
                if source_hashes[str(source_video)] != video_hashes[key]:
                    raise ValueError(f"Video changed: {source_video}")
                for boundary in ("from_timestamp", "to_timestamp"):
                    field = f"videos/{camera}/{boundary}"
                    if source_episode[field] != target_episode[field]:
                        raise ValueError(f"Video timestamp changed: {source_video}")
            frame_slice = data.slice(source_episode["dataset_from_index"], source_episode["length"])
            human = frame_slice[INTERVENTION].to_numpy().astype(bool)
            episodes.append({
                "episode_index": target_id,
                "source_root": str(root),
                "source_episode_index": source_id,
                "source_type": "demonstration" if root.name == args.demo_name else "rollout",
                "episode_success": "success",
                "frames": source_episode["length"],
                "human_flag_frames": int(human.sum()),
                "takeover_events": int(np.sum(~human[:-2] & human[1:-1])),
            })
        frame_offset += len(data)
        episode_offset += len(meta)
    if (frame_offset, episode_offset) != (len(merged_data), len(merged_meta)):
        raise ValueError("Source/output totals disagree")

    dataset = LeRobotDataset(args.output_name, root=staging, video_backend="pyav")
    targets, scales = build_successful_value_targets(
        dataset, ValueRewardConfig(mode="time_takeover"), "episode_success"
    )
    decoded = 0
    for episode in merged_meta.to_pylist():
        start, length = episode["dataset_from_index"], episode["length"]
        for index in sorted({start, start + length // 2, start + length - 1}):
            sample = dataset[index]
            for camera in camera_keys:
                if not np.isfinite(sample[camera].numpy()).all():
                    raise ValueError(f"Invalid decoded image at frame {index}")
            decoded += 1
    for path, before in source_hashes.items():
        if digest(Path(path)) != before:
            raise ValueError(f"Source changed while building: {path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(destination),
        "repo_id": args.output_name,
        "episodes": episodes,
        "excluded_sources": excluded,
        "confirmed_success_edits": relabels,
        "source_sha256": source_hashes,
        "video_sha256": video_hashes,
        "summary": {
            "episodes": len(merged_meta),
            "frames": len(merged_data),
            "fps": dataset.fps,
            "demonstration_episodes": sum(e["source_type"] == "demonstration" for e in episodes),
            "rollout_episodes": sum(e["source_type"] == "rollout" for e in episodes),
            "human_flag_frames": sum(e["human_flag_frames"] for e in episodes),
            "takeover_events": sum(e["takeover_events"] for e in episodes),
        },
        "verification": {
            "frame_values_unchanged": True,
            "videos_byte_identical": True,
            "video_offsets_unchanged": True,
            "source_files_unchanged_after_confirmed_relabels": True,
            "decoded_samples": decoded,
            "reward_mode": "time_takeover",
            "takeover_cost": 150,
            "reward_scales": scales,
            "target_min": float(targets.min()),
            "target_max": float(targets.max()),
        },
        "notes": [
            "All successful episodes are included; no train/validation split has been applied.",
            "Demonstration is_intervention flags are preserved as recorded (all zero); "
            "source_type identifies demonstrations.",
            "Controller flags must not be used to distinguish original demonstrations "
            "from autonomous rollouts.",
        ],
    }
    (staging / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.rename(staging, destination)
    work.rmdir()
    logging.info("Verified dataset ready: %s", destination)
    print(json.dumps({"summary": manifest["summary"], "verification": manifest["verification"]}, indent=2))


if __name__ == "__main__":
    main()
