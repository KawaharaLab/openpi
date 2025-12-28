"""Convert rosbag2 recordings into image files and CSV logs without ROS2."""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.23.0",
#   "opencv-python>=4.5.0",
#   "pyyaml>=6.0",
#   "rosbags>=0.10.0",
# ]
# ///

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import cv2
import numpy as np
import yaml
from rosbags.interfaces import MessageDefinitionFormat
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys import get_types_from_idl, get_types_from_msg

from config import CaptureConfig, TopicSpec, load_capture_config


def _sanitize_topic(topic: str) -> str:
    return topic.strip("/").replace("/", "_") or "root"


def _flatten(value, prefix: str = "", out: Optional[MutableMapping[str, object]] = None):
    if out is None:
        out = {}

    if isinstance(value, Mapping):
        for key, inner in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(inner, next_prefix, out)
    elif isinstance(value, (list, tuple)):
        for idx, inner in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            _flatten(inner, next_prefix, out)
    else:
        key = prefix if prefix else "value"
        out[key] = value
    return out


def _load_csv_field_map(path: Optional[Path], logger=None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    topics = data.get("topics", {}) if isinstance(data, dict) else {}
    field_map: dict[str, dict[str, str]] = {}
    for topic, entry in topics.items():
        if not isinstance(entry, dict):
            continue
        fields = entry.get("fields", {})
        if isinstance(fields, dict):
            field_map[str(topic)] = {str(k): str(v) for k, v in fields.items()}

    if logger:
        logger.info(f"Loaded CSV field map for {len(field_map)} topics from {path}")
    return field_map


def _message_to_mapping(message):
    if is_dataclass(message):
        return asdict(message)
    if hasattr(message, "__dict__"):
        return dict(message.__dict__)
    return message


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _format_csv_value(value):
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _maybe_decompress_file_bag(
    bag_uri: Path, config: CaptureConfig, logger=None
) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """Decompress bags recorded with file compression if needed."""

    metadata_path = bag_uri / "metadata.yaml"
    if not metadata_path.exists():
        return bag_uri, None

    metadata = yaml.safe_load(metadata_path.read_text())
    bag_info = metadata.get("rosbag2_bagfile_information", {})
    relative_paths = bag_info.get("relative_file_paths", []) or []
    mode = str(bag_info.get("compression_mode", "")).lower()
    has_zstd = any(path.endswith(".zstd") for path in relative_paths) or any(
        bag_uri.glob("*.zstd")
    )
    if mode != "file" and not has_zstd:
        return bag_uri, None

    temp_dir = tempfile.TemporaryDirectory(prefix="bag_decompress_")
    target_dir = Path(temp_dir.name) / bag_uri.name
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        _decompress_bag_contents(
            bag_uri,
            target_dir,
            metadata,
            bag_info,
            logger,
        )
    except Exception:
        temp_dir.cleanup()
        raise

    return target_dir, temp_dir


def _decompress_bag_contents(
    source_dir: Path,
    target_dir: Path,
    metadata: dict,
    bag_info: dict,
    logger=None,
) -> None:
    """Copy bag contents and expand any *.zstd segments using the zstd CLI."""

    zstd_bin = shutil.which("zstd")
    if zstd_bin is None:
        raise RuntimeError(
            "Unable to decompress rosbag: 'zstd' command not found. Install the 'zstd' package or record without file compression."
        )

    if logger:
        logger.info(
            "Detected file-compressed bag. Falling back to 'zstd -d' to expand segments."
        )

    # Copy any auxiliary files (e.g., calibration data) verbatim.
    for item in source_dir.iterdir():
        if item.name == "metadata.yaml" or item.name.endswith(".zstd"):
            continue
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    rel_paths = bag_info.get("relative_file_paths", []) or [
        path.name for path in source_dir.glob("*.db3*")
    ]

    updated_relative_paths: list[str] = []
    for rel_path in rel_paths:
        rel_path = str(rel_path)
        src_path = source_dir / rel_path
        if not src_path.exists():
            raise FileNotFoundError(
                f"metadata.yaml references '{rel_path}', but the file does not exist in {source_dir}"
            )

        if rel_path.endswith(".zstd"):
            dest_name = rel_path[: -len(".zstd")]
            dest_path = target_dir / dest_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                zstd_bin,
                "-d",
                "--force",
                "-o",
                str(dest_path),
                str(src_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            if logger and result.stderr:
                logger.debug(result.stderr.strip())
            updated_relative_paths.append(dest_name)
        else:
            dest_path = target_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            updated_relative_paths.append(rel_path)

    bag_info["relative_file_paths"] = updated_relative_paths
    bag_info["compression_mode"] = "NONE"
    bag_info["compression_format"] = "NONE"

    for file_entry in bag_info.get("files", []) or []:
        path_value = file_entry.get("path")
        if isinstance(path_value, str) and path_value.endswith(".zstd"):
            file_entry["path"] = path_value[: -len(".zstd")]

    target_metadata = target_dir / "metadata.yaml"
    target_metadata.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


_ENCODING_MAP = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16uc1": (np.uint16, 1),
    "16sc1": (np.int16, 1),
    "32fc1": (np.float32, 1),
}


def _convert_color(image: np.ndarray, src: str, dst: str) -> np.ndarray:
    if src == dst:
        return image
    conversions = {
        ("rgb8", "bgr8"): cv2.COLOR_RGB2BGR,
        ("bgr8", "rgb8"): cv2.COLOR_BGR2RGB,
        ("rgba8", "bgra8"): cv2.COLOR_RGBA2BGRA,
        ("bgra8", "rgba8"): cv2.COLOR_BGRA2RGBA,
    }
    code = conversions.get((src, dst))
    if code is None:
        return image
    return cv2.cvtColor(image, code)


def _decode_compressed_image(message) -> np.ndarray:
    data = np.frombuffer(message.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Compressed image decode failed")
    return image


def _decode_raw_image(message, desired_encoding: Optional[str]) -> np.ndarray:
    source_encoding = str(message.encoding or "").lower()
    if source_encoding not in _ENCODING_MAP:
        raise ValueError(f"Unsupported image encoding '{source_encoding}'")

    dtype, channels = _ENCODING_MAP[source_encoding]
    data = np.frombuffer(message.data, dtype=dtype)
    if channels == 1:
        image = data.reshape((message.height, message.width))
    else:
        image = data.reshape((message.height, message.width, channels))

    if desired_encoding:
        image = _convert_color(
            image,
            source_encoding,
            str(desired_encoding).lower(),
        )
    return image


_FALLBACK_MSG_DEFS = {
    "control_msgs/msg/JointTrajectoryControllerState": (
        "std_msgs/Header header\n"
        "string[] joint_names\n"
        "trajectory_msgs/JointTrajectoryPoint reference\n"
        "trajectory_msgs/JointTrajectoryPoint feedback\n"
        "trajectory_msgs/JointTrajectoryPoint error\n"
        "trajectory_msgs/JointTrajectoryPoint output\n"
        "trajectory_msgs/JointTrajectoryPoint desired\n"
        "trajectory_msgs/JointTrajectoryPoint actual\n"
        "string[] multi_dof_joint_names\n"
        "trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_reference\n"
        "trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_feedback\n"
        "trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_error\n"
        "trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_output\n"
        "trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_desired\n"
        "trajectory_msgs/MultiDOFJointTrajectoryPoint multi_dof_actual\n"
    ),
}


def _ensure_typestore_type(connection, typestore, logger=None, warned_types=None) -> None:
    msgtype = connection.msgtype
    if msgtype in typestore.fielddefs:
        return

    msgdef = connection.msgdef
    if not msgdef or msgdef.format == MessageDefinitionFormat.NONE:
        fallback = _FALLBACK_MSG_DEFS.get(msgtype)
        if fallback:
            try:
                typs = get_types_from_msg(fallback, msgtype)
                typestore.register(typs)
                if logger:
                    logger.info(f"Registered fallback message type {msgtype}")
            except Exception as exc:  # noqa: BLE001
                if logger:
                    logger.warning(
                        f"Failed to register fallback type {msgtype} ({exc})"
                    )
        else:
            if logger:
                if warned_types is None or msgtype not in warned_types:
                    logger.warning(
                        f"Skipping type registration for {msgtype}: message definition missing"
                    )
        if warned_types is not None:
            warned_types.add(msgtype)
        return

    try:
        if msgdef.format == MessageDefinitionFormat.IDL:
            typs = get_types_from_idl(msgdef.data)
        else:
            typs = get_types_from_msg(msgdef.data, msgtype)
        typestore.register(typs)
        if logger:
            logger.info(f"Registered message type {msgtype} from bag definition")
    except Exception as exc:  # noqa: BLE001
        if logger:
            if warned_types is None or msgtype not in warned_types:
                logger.warning(
                    f"Failed to register message type {msgtype} from bag definition ({exc})"
                )
        if warned_types is not None:
            warned_types.add(msgtype)


def _deserialize_cdr_lenient(rawdata: bytes | memoryview, typename: str, typestore):
    little_endian = bool(rawdata[1])
    msgdef = typestore.get_msgdef(typename)
    func = msgdef.deserialize_cdr_le if little_endian else msgdef.deserialize_cdr_be
    message, _pos = func(rawdata[4:], 0, msgdef.cls, typestore)
    return message


def convert_bag_to_dataset(
    bag_uri: Path,
    output_dir: Path,
    config: CaptureConfig,
    logger=None,
    cutoff_stamp_ns: Optional[int] = None,
) -> None:
    """Read a rosbag2 recording and emit PNG/CSV artifacts."""

    if logger:
        logger.info(f"Converting bag '{bag_uri}' into '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    bag_to_read, temp_handle = _maybe_decompress_file_bag(bag_uri, config, logger)

    try:
        topic_specs = {spec.name: spec for spec in config.topics}
        read_topic_map: dict[str, tuple[TopicSpec, str]] = {}
        for name, spec in topic_specs.items():
            if name.endswith("/image_raw"):
                read_topic_map[f"{name}/compressed"] = (spec, name)
            else:
                read_topic_map[name] = (spec, name)
        typestore = get_typestore(Stores.ROS2_FOXY)

        image_dirs: Dict[str, Path] = {}
        csv_rows_by_topic: dict[str, list[dict[str, object]]] = defaultdict(list)
        skipped_messages: dict[str, int] = defaultdict(int)
        warned_types: set[str] = set()
        warned_lenient: set[str] = set()
        csv_field_map = _load_csv_field_map(
            getattr(config, "csv_config_path", None),
            logger,
        )

        with Reader(str(bag_to_read)) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic in read_topic_map
            ]
            for connection, stamp, rawdata in reader.messages(connections=connections):
                if cutoff_stamp_ns is not None and stamp > cutoff_stamp_ns:
                    continue

                entry = read_topic_map.get(connection.topic)
                if entry is None:
                    continue
                spec, output_topic = entry

                try:
                    _ensure_typestore_type(connection, typestore, logger, warned_types)
                    message = typestore.deserialize_cdr(rawdata, connection.msgtype)
                except AssertionError:
                    try:
                        message = _deserialize_cdr_lenient(
                            rawdata, connection.msgtype, typestore
                        )
                        if logger and connection.msgtype not in warned_lenient:
                            logger.warning(
                                "Lenient CDR decode used for %s; trailing bytes ignored",
                                connection.msgtype,
                            )
                            warned_lenient.add(connection.msgtype)
                    except Exception as exc:  # noqa: BLE001
                        skipped_messages[connection.topic] += 1
                        if logger and skipped_messages[connection.topic] == 1:
                            logger.warning(
                                f"Skipping messages for topic {connection.topic}: deserialization failed ({type(exc).__name__}: {exc!r})"
                            )
                        continue
                except Exception as exc:  # noqa: BLE001
                    skipped_messages[connection.topic] += 1
                    if logger and skipped_messages[connection.topic] == 1:
                        logger.warning(
                            f"Skipping messages for topic {connection.topic}: deserialization failed ({type(exc).__name__}: {exc!r})"
                        )
                    continue

                is_compressed = connection.msgtype.endswith("CompressedImage")
                if spec.mode == "image" or is_compressed:
                    folder = image_dirs.get(output_topic)
                    if folder is None:
                        folder = output_dir / "images" / _sanitize_topic(output_topic)
                        folder.mkdir(parents=True, exist_ok=True)
                        image_dirs[output_topic] = folder

                    try:
                        if is_compressed:
                            image = _decode_compressed_image(message)
                        else:
                            image = _decode_raw_image(message, spec.encoding)
                    except Exception as exc:  # noqa: BLE001
                        skipped_messages[connection.topic] += 1
                        if logger and skipped_messages[connection.topic] == 1:
                            logger.warning(
                                f"Skipping images for topic {connection.topic}: decode failed ({exc})"
                            )
                        continue

                    filename = folder / f"{stamp}.png"
                    if not cv2.imwrite(str(filename), image) and logger:
                        logger.warning(
                            f"Failed to write image for topic {connection.topic} at stamp {stamp}"
                        )
                else:
                    row = {"stamp_ns": stamp}
                    flattened = _flatten(_message_to_mapping(message))
                    prefix = _sanitize_topic(output_topic)
                    for key, value in flattened.items():
                        column = f"{prefix}.{key}" if key else prefix
                        row[column] = _format_csv_value(value)
                    field_map = csv_field_map.get(output_topic)
                    if field_map:
                        remapped = {}
                        for src, dst in field_map.items():
                            if src == "stamp_ns":
                                remapped[dst] = stamp
                            else:
                                remapped[dst] = row.get(src)
                        row = remapped
                    csv_rows_by_topic[prefix].append(row)

        csv_dir = output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        for prefix, rows in csv_rows_by_topic.items():
            if not rows:
                continue
            data_fields = sorted(
                {key for row in rows for key in row.keys() if key != "stamp_ns"}
            )
            fieldnames = ["stamp_ns", *data_fields]
            csv_path = csv_dir / f"{prefix}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            if logger:
                logger.info(f"Wrote {len(rows)} rows to {csv_path}")

        if logger and skipped_messages:
            total_skipped = sum(skipped_messages.values())
            topics = ", ".join(
                f"{name} ({count})" for name, count in skipped_messages.items()
            )
            logger.warning(
                f"Skipped {total_skipped} messages due to deserialization errors: {topics}"
            )
    finally:
        if temp_handle:
            temp_handle.cleanup()


def cli_main() -> None:
    """Entry point for manual conversion without ROS2."""

    import argparse

    parser = argparse.ArgumentParser(description="Convert rosbag2 data into PNG/CSV outputs")
    parser.add_argument("--bag", help="Path to a single rosbag2 directory (metadata.yaml parent)")
    parser.add_argument(
        "--lan-root",
        default="../data/lan",
        help="Root directory that contains per-session folders under */bag/session_* (batch mode)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If set, process every bag under --lan-root matching */bag/session_*",
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Path to the capture YAML config (defaults to output_dir/config.yaml)",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Unused. Output directory is fixed to bag_dir/../..",
    )
    parser.add_argument("--cutoff-ns", type=int, default=None, help="Optional cutoff timestamp in nanoseconds")
    parser.add_argument(
        "--csv-config",
        required=False,
        help="CSV field map YAML (defaults to output_dir/csv_fields.yaml if present)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("bag_converter_uv")

    if args.all:
        lan_root = Path(args.lan_root).resolve()
        bag_dirs = sorted(lan_root.glob("*/bag/session_*"))
        if not bag_dirs:
            logger.error("No bags found under %s", lan_root)
            return
        logger.info("Found %d bag(s) under %s", len(bag_dirs), lan_root)
    else:
        if not args.bag:
            parser.error("--bag is required unless --all is specified")
        bag_dirs = [Path(args.bag).resolve()]

    for bag_path in bag_dirs:
        output_dir = bag_path.parent.parent
        done_flag = output_dir / "done.txt"
        if done_flag.exists():
            logger.info("Skipping %s (done.txt present)", bag_path)
            continue

        config_path = Path(args.config) if args.config else output_dir / "config.yaml"
        cfg = load_capture_config(config_path)
        csv_config_path = Path(args.csv_config) if args.csv_config else output_dir / "csv_fields.yaml"
        cfg.csv_config_path = csv_config_path if csv_config_path.exists() else None

        logger.info("Processing %s", bag_path)
        try:
            convert_bag_to_dataset(
                bag_path,
                output_dir,
                cfg,
                logger=logger,
                cutoff_stamp_ns=args.cutoff_ns,
            )
        except Exception:
            logger.exception("Failed while processing %s", bag_path)
            continue

        try:
            done_flag.touch(exist_ok=True)
            logger.info("Marked done: %s", done_flag)
        except Exception:
            logger.exception("Failed to write done.txt for %s", bag_path)


if __name__ == "__main__":
    cli_main()
