"""
Convert the recorded UR3 + Robotiq pick-and-place runs into the LeRobot format.

This script mirrors the other conversion utilities in this repository and consumes
the already-decoded artifacts produced by the data capture tools:

data/pick_and_place/<session_id>/
  bag/                     # raw rosbag (not used here)
  config.yaml              # capture configuration
  images/
    camera_camera_fixed_color_image_raw_compressed/<stamp>.png
    camera_camera_wrist_color_image_raw_compressed/<stamp>.png
  csv/
    timeseries.csv         # aggregated topic data (joint states, actions, gripper)
  prompt.txt               # task instruction to store as the LeRobot task

Images are used as the temporal anchor (≈15 FPS). Joint states and actions are
aligned to the nearest image timestamp within a configurable tolerance.

Usage:
uv run examples/ur3_robotiq/convert_pick_and_place_to_lerobot.py \\
  --repo-id <your_hf_username/pick_and_place_ur3> \\
  --data-root data/pick_and_place

Notes:
- The script prefers pre-decoded PNG/CSV artifacts. If you want direct rosbag
  decoding instead, provide per-topic CSVs or ensure rosbag2_py + cv_bridge are
  available and we can extend this script.
- Gripper distance/action values are stored in meters (converted from mm).
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import logging
from pathlib import Path
import shutil
from typing import Generic, Sequence, TypeVar

import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro


T = TypeVar("T")


@dataclasses.dataclass(slots=True)
class TimeSeriesIndex(Generic[T]):
    """Lightweight time-indexed lookup with nearest-neighbor queries."""

    stamps: list[int]
    values: list[T]

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[int, T]]) -> "TimeSeriesIndex[T]":
        ordered = sorted(pairs, key=lambda p: p[0])
        stamps, values = zip(*ordered) if ordered else ([], [])
        return cls(list(stamps), list(values))

    def nearest(self, stamp: int, tolerance_ns: int) -> T | None:
        """Return the closest sample within the tolerance, otherwise None."""
        if not self.stamps:
            return None

        idx = bisect.bisect_left(self.stamps, stamp)
        candidates: list[tuple[int, T]] = []
        for j in (idx - 1, idx):
            if 0 <= j < len(self.stamps):
                delta = abs(self.stamps[j] - stamp)
                if delta <= tolerance_ns:
                    candidates.append((delta, self.values[j]))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]


@dataclasses.dataclass(slots=True)
class SessionData:
    session_dir: Path
    prompt: str
    wrist_frames: list[tuple[int, Path]]
    fixed_frames: list[tuple[int, Path]]
    joint_index: TimeSeriesIndex[np.ndarray]
    action_index: TimeSeriesIndex[np.ndarray]
    gripper_action_index: TimeSeriesIndex[float]
    finger_index: TimeSeriesIndex[float]


@dataclasses.dataclass
class ConvertConfig:
    data_root: Path = Path("data/pick_and_place")
    repo_id: str = "uzumi-bi/pick_and_place_ur3"
    fps: float | None = None
    image_tolerance_ns: int = int(1e8)  # 100 ms camera-to-camera pairing window
    state_tolerance_ns: int = int(5e7)  # 50 ms camera-to-state pairing window
    action_tolerance_ns: int = int(5e7)  # 50 ms camera-to-action pairing window
    push_to_hub: bool = False
    local_output_dirname: str = "pick_and_place_lerobot"


def _maybe_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_array(row: dict[str, str], prefix: str, count: int) -> list[float] | None:
    values: list[float] = []
    for i in range(count):
        val = _maybe_float(row[f"{prefix}[{i}]"])
        if val is None:
            return None
        values.append(val)
    return values


def _reorder_joints_to_feature_order(values: np.ndarray) -> np.ndarray:
    """Reorder joint arrays from ROS order to the feature_names order."""
    if values.shape[-1] != 6:
        raise ValueError(f"Expected 6 joint values, got shape {values.shape}")
    # See joint_reorder_idx in _create_dataset for the definition.
    return values[..., (5, 0, 1, 2, 3, 4)]


def _load_timeseries(csv_path: Path) -> tuple[TimeSeriesIndex[np.ndarray], TimeSeriesIndex[np.ndarray], TimeSeriesIndex[float], TimeSeriesIndex[float]]:
    joint_samples: list[tuple[int, np.ndarray]] = []
    action_samples: list[tuple[int, np.ndarray]] = []
    gripper_goal_samples: list[tuple[int, float]] = []
    finger_samples: list[tuple[int, float]] = []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = int(row["stamp_ns"])

            positions = _read_array(row, "joint_states.position", 6)
            if positions is not None:
                reordered = _reorder_joints_to_feature_order(np.asarray(positions, dtype=np.float32))
                joint_samples.append((stamp, reordered))

            actions = _read_array(row, "forward_position_controller_commands.data", 6)
            if actions is not None:
                reordered = _reorder_joints_to_feature_order(np.asarray(actions, dtype=np.float32))
                action_samples.append((stamp, reordered))

            gripper_goal = _maybe_float(row.get("robotiq_2f_gripper_action_goal.data[0]", ""))
            if gripper_goal is not None:
                # Convert mm -> meters to match the finger distance scale used for state.
                gripper_goal_samples.append((stamp, float(gripper_goal) / 1000.0))

            finger_distance_mm = _maybe_float(row.get("robotiq_2f_gripper_finger_distance_mm.data", ""))
            if finger_distance_mm is not None:
                finger_samples.append((stamp, float(finger_distance_mm) / 1000.0))

    return (
        TimeSeriesIndex.from_pairs(joint_samples),
        TimeSeriesIndex.from_pairs(action_samples),
        TimeSeriesIndex.from_pairs(gripper_goal_samples),
        TimeSeriesIndex.from_pairs(finger_samples),
    )


def _load_camera_frames(images_dir: Path) -> list[tuple[int, Path]]:
    frames = []
    for path in images_dir.glob("*.png"):
        try:
            stamp = int(path.stem)
        except ValueError:
            continue
        frames.append((stamp, path))
    return sorted(frames, key=lambda p: p[0])


def _pair_camera_frames(
    wrist_frames: Sequence[tuple[int, Path]],
    fixed_frames: Sequence[tuple[int, Path]],
    tolerance_ns: int,
) -> list[tuple[int, Path, Path]]:
    """Greedily pair wrist images with the nearest fixed images."""
    pairs: list[tuple[int, Path, Path]] = []
    fixed_idx = 0

    for wrist_stamp, wrist_path in wrist_frames:
        while (
            fixed_idx + 1 < len(fixed_frames)
            and abs(fixed_frames[fixed_idx + 1][0] - wrist_stamp) <= abs(fixed_frames[fixed_idx][0] - wrist_stamp)
        ):
            fixed_idx += 1

        if fixed_idx >= len(fixed_frames):
            break

        fixed_stamp, fixed_path = fixed_frames[fixed_idx]
        if abs(fixed_stamp - wrist_stamp) <= tolerance_ns:
            pairs.append((wrist_stamp, wrist_path, fixed_path))

    return pairs


def _load_session(session_dir: Path, cfg: ConvertConfig) -> SessionData:
    prompt_path = session_dir / "prompt.txt"
    prompt = prompt_path.read_text().strip()

    wrist_dir = session_dir / "images" / "camera_camera_wrist_color_image_raw_compressed"
    fixed_dir = session_dir / "images" / "camera_camera_fixed_color_image_raw_compressed"
    wrist_frames = _load_camera_frames(wrist_dir)
    fixed_frames = _load_camera_frames(fixed_dir)

    if not wrist_frames:
        raise FileNotFoundError(f"No wrist frames found under {wrist_dir}")
    if not fixed_frames:
        raise FileNotFoundError(f"No fixed frames found under {fixed_dir}")

    csv_path = session_dir / "csv" / "timeseries.csv"
    joint_idx, action_idx, gripper_goal_idx, finger_idx = _load_timeseries(csv_path)

    return SessionData(
        session_dir=session_dir,
        prompt=prompt,
        wrist_frames=wrist_frames,
        fixed_frames=fixed_frames,
        joint_index=joint_idx,
        action_index=action_idx,
        gripper_action_index=gripper_goal_idx,
        finger_index=finger_idx,
    )


def _compute_fps_from_frames(frames: Sequence[tuple[int, Path]]) -> float:
    stamps = [stamp for stamp, _ in frames]
    if len(stamps) < 2:
        return 15.0
    deltas = np.diff(stamps)
    median_ns = float(np.median(deltas))
    return 1e9 / median_ns if median_ns > 0 else 15.0


def _create_dataset(repo_id: str, fps: float) -> tuple[LeRobotDataset, Path]:
    feature_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "robotiq_finger_distance_m",
    ]

    # Raw ROS 2 joint_states order in the logged CSV is:
    # shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan
    # but we want shoulder_pan first. This index map reorders from the raw order
    # into the target feature_names order above.
    joint_reorder_idx = (5, 0, 1, 2, 3, 4)

    features = {
        "images.cam_high": {
            "dtype": "image",
            "shape": (360, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "images.cam_left_wrist": {
            "dtype": "image",
            "shape": (360, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (len(feature_names),),
            "names": [feature_names],
        },
        "actions": {
            "dtype": "float32",
            "shape": (len(feature_names),),
            "names": [feature_names],
        },
    }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur3_robotiq",
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )
    return dataset, output_path


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _convert_session(session: SessionData, dataset: LeRobotDataset, cfg: ConvertConfig) -> dict[str, int]:
    camera_pairs = _pair_camera_frames(session.wrist_frames, session.fixed_frames, cfg.image_tolerance_ns)
    if not camera_pairs:
        raise RuntimeError(f"Failed to pair cameras for session {session.session_dir}")

    stats = {"frames_written": 0, "frames_skipped_missing_state": 0, "frames_skipped_missing_action": 0}
    last_finger = 0.0

    for stamp, wrist_path, fixed_path in tqdm(camera_pairs, desc=f"Session {session.session_dir.name}"):
        joint_state = session.joint_index.nearest(stamp, cfg.state_tolerance_ns)
        if joint_state is None:
            stats["frames_skipped_missing_state"] += 1
            continue

        finger = session.finger_index.nearest(stamp, cfg.state_tolerance_ns)
        if finger is not None:
            last_finger = finger
        state = np.concatenate([joint_state, np.array([last_finger], dtype=np.float32)], dtype=np.float32)

        action = session.action_index.nearest(stamp, cfg.action_tolerance_ns)
        gripper_cmd = session.gripper_action_index.nearest(stamp, cfg.action_tolerance_ns)
        if gripper_cmd is None:
            gripper_cmd = last_finger

        if action is None:
            stats["frames_skipped_missing_action"] += 1
            continue

        actions = np.concatenate([action, np.array([gripper_cmd], dtype=np.float32)], dtype=np.float32)

        frame = {
            "images.cam_high": _load_rgb(fixed_path),
            "images.cam_left_wrist": _load_rgb(wrist_path),
            "state": state,
            "actions": actions,
        }
        dataset.add_frame(frame, task=session.prompt)
        stats["frames_written"] += 1

    dataset.save_episode()
    return stats


def main(cfg: ConvertConfig):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_root = cfg.data_root
    sessions = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No sessions found under {data_root}")

    first_session = _load_session(sessions[0], cfg)
    fps = cfg.fps or _compute_fps_from_frames(first_session.wrist_frames)
    logging.info(f"Using FPS={fps:.2f}")
    dataset, dataset_path = _create_dataset(cfg.repo_id, fps=fps)

    # Process the first session (already loaded) then the rest.
    all_sessions = [first_session] + [_load_session(session_dir, cfg) for session_dir in sessions[1:]]
    for session in all_sessions:
        stats = _convert_session(session, dataset, cfg)
        logging.info(
            f"{session.session_dir.name}: wrote {stats['frames_written']} frames "
            f"(missing_state={stats['frames_skipped_missing_state']}, "
            f"missing_action={stats['frames_skipped_missing_action']})"
        )

    if cfg.push_to_hub:
        dataset.push_to_hub(
            tags=["ur3", "robotiq", "pick_and_place"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    else:
        target_root = data_root.resolve().parent / cfg.local_output_dirname
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(dataset_path, target_root)
        logging.info(f"Saved local LeRobot dataset to {target_root}")


if __name__ == "__main__":
    tyro.cli(main)
