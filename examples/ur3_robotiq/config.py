"""Utilities for loading and validating data capture configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class TopicSpec:
    """Configuration for a topic that should be captured."""

    name: str
    type: str
    mode: str  # "image" or "csv"
    encoding: Optional[str] = None  # Only used for image topics


@dataclass
class BagRecorderConfig:
    """rosbag2 recording options."""

    storage: str = "sqlite3"
    compression_mode: Optional[str] = None
    compression_format: Optional[str] = None


@dataclass
class CaptureConfig:
    """Top-level configuration for a capture session."""

    output_root: Path
    task_name: str
    prompt: str
    viewer_topics: list[str]
    image_throttle_hz: float
    other_throttle_hz: float
    enable_image_compression: bool
    stop_key: str
    discard_key: str
    topics: List[TopicSpec]
    bag: BagRecorderConfig
    raw: dict[str, Any]
    source_path: Path


def _validate_topic(entry: dict[str, Any]) -> TopicSpec:
    if "name" not in entry or "type" not in entry or "mode" not in entry:
        raise ValueError(
            "Each topic entry must contain 'name', 'type', and 'mode' fields"
        )

    mode = entry["mode"].lower()
    if mode not in {"image", "csv"}:
        raise ValueError("Topic mode must be either 'image' or 'csv'")

    encoding = entry.get("encoding") if mode == "image" else None

    return TopicSpec(
        name=str(entry["name"]),
        type=str(entry["type"]),
        mode=mode,
        encoding=encoding,
    )


def load_capture_config(config_path: str | Path) -> CaptureConfig:
    """Load and validate a capture configuration YAML file."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Capture config '{path}' does not exist")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    topics_data = data.get("topics")
    if not topics_data:
        raise ValueError("Config must list at least one topic under 'topics'")

    topics = [_validate_topic(entry) for entry in topics_data]

    bag_cfg = data.get("bag_record", {})
    bag = BagRecorderConfig(
        storage=bag_cfg.get("storage", "sqlite3"),
        compression_mode=bag_cfg.get("compression_mode"),
        compression_format=bag_cfg.get("compression_format"),
    )

    output_root = Path(data.get("output_root", "./data")).expanduser().resolve()
    task_name = str(data.get("task_name", "pick_and_place"))
    prompt = str(
        data.get(
            "prompt",
            "pick the blue tape and place it in the orange box.",
        )
    )
    viewer_topics = [str(item) for item in data.get("viewer_topics", [])]
    image_throttle_hz = float(data.get("image_throttle_hz", 10.0))
    other_throttle_hz = float(data.get("other_throttle_hz", 100.0))
    enable_image_compression = bool(data.get("enable_image_compression", True))
    stop_key = str(data.get("stop_key", "q"))
    discard_key = str(data.get("discard_key", "x"))

    return CaptureConfig(
        output_root=output_root,
        task_name=task_name,
        prompt=prompt,
        viewer_topics=viewer_topics,
        image_throttle_hz=image_throttle_hz,
        other_throttle_hz=other_throttle_hz,
        enable_image_compression=enable_image_compression,
        stop_key=stop_key,
        discard_key=discard_key,
        topics=topics,
        bag=bag,
        raw=data,
        source_path=path,
    )
