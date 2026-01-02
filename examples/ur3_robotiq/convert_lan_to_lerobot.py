"""
Convert decoded UR3 + Robotiq LAN captures into the LeRobot format at 5 Hz.

Input (per session):
data/lan/<session_id>/
    images/
        camera_fixed_realsense2_camera_color_image_raw/<stamp>.png
        camera_wrist_realsense2_camera_color_image_raw/<stamp>.png
    csv/
        force_torque_left.csv
        force_torque_right.csv
        joint_states.csv
        scaled_joint_trajectory_controller_controller_state.csv
        robotiq_2f_gripper_action_goal.csv
        robotiq_2f_gripper_finger_distance_mm.csv
    prompt.txt

Synchronization rule (5 Hz):
- Find the earliest timestamp at which every required stream (both cameras and
    all CSV topics) has produced at least one sample.
- From that start time, step forward every 0.2 s (200 ms).
- At each step, use the latest sample at-or-before the step for every topic
    (carry-forward). If either camera is missing a frame for the step, skip it.

Outputs:
- A LeRobot dataset with frames containing:
    * images.cam_fixed (RGB), images.cam_wrist (RGB)
    * state: [joint_pos(6), finger_distance_m]
    * force_left(6)
    * force_right(6)
    * actions: [ref_joint_pos(6), gripper_goal]
- Optionally push to HF Hub (disabled by default); otherwise copied to
    <data_root>/../pick_and_place_lerobot.
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import json
import logging
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro

GRIPPER_ACTION_OPEN = 0.140 # m
GRIPPER_ACTION_CLOSE = 0.002 # m


@dataclasses.dataclass
class ConvertConfig:
    data_root: Path = Path("../data/lan")
    repo_id: str = "uzumi-bi/lan_ur3"
    fps: float = 5.0  # fixed 5 Hz resampling
    action_horizon: int = 50  # number of future actions to include
    action_hz: float = 20.0  # action sampling rate (Hz) for horizon
    push_to_hub: bool = False
    local_output_dirname: str = "lan_ur3_lerobot"


def _reorder_joints_to_feature_order(values: np.ndarray) -> np.ndarray:
    """
    Reorder joint arrays from ROS order to the feature_names order.
    state only; not for actions.
    """
    if values.shape[-1] != 6:
        raise ValueError(f"Expected 6 joint values, got shape {values.shape}")
    # See joint_reorder_idx in _create_dataset for the definition.
    return values[..., (5, 0, 1, 2, 3, 4)]
def _parse_array(text: str, expected_len: int | None = None) -> np.ndarray:
    values = np.asarray(json.loads(text), dtype=np.float32)
    if expected_len is not None and values.shape[0] != expected_len:
        raise ValueError(f"Expected array length {expected_len}, got {values.shape[0]}")
    return values


def _feature_name_lists():
    joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    state_names: list[str] = [
        *[f"joint_pos.{n}" for n in joints],
        "finger_distance_m",
    ]

    action_names: list[str] = [
        *[f"ref_joint_pos.{n}" for n in joints],
        "gripper_goal_m",
    ]
    return state_names, action_names


def _create_dataset(
    repo_id: str,
    fps: float,
    img_shapes: dict[str, tuple[int, int, int]],
    state_names: list[str],
    action_names: list[str],
    action_horizon: int,
) -> tuple[LeRobotDataset, Path]:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    features = {
        "images.cam_fixed": {
            "dtype": "image",
            "shape": img_shapes["fixed"],
            "names": ["height", "width", "channel"],
        },
        "images.cam_wrist": {
            "dtype": "image",
            "shape": img_shapes["wrist"],
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": [state_names],
        },
        "left_ft": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
        "right_ft": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_horizon, len(action_names)),
            "names": ["horizon", "component"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur3_robotiq",
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )
    return dataset, output_path


def _load_rgb(path: Path) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except OSError as exc:  # truncated or unreadable image
        logging.warning("Skipping unreadable image: %s (%s)", path, exc)
        return None


def _load_frames(images_dir: Path) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for path in images_dir.glob("*.png"):
        try:
            stamp = int(path.stem)
        except ValueError:
            continue
        frames.append((stamp, path))
    return sorted(frames, key=lambda p: p[0])


def _load_csv_pairs(path: Path, builder) -> list[tuple[int, object]]:
    pairs: list[tuple[int, object]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = int(row["stamp_ns"])
            pairs.append((stamp, builder(row)))
    return pairs


def _first_last(stamps: list[int]) -> tuple[int, int]:
    return stamps[0], stamps[-1]


def _find_image_dir(session_dir: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        candidate = session_dir / "images" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image directory found under {session_dir}/images (tried {candidates})")


def _prepare_streams(session_dir: Path):
    # Required CSVs
    csv_root = session_dir / "csv"

    left_force = _load_csv_pairs(
        csv_root / "force_torque_left.csv",
        lambda row: np.array(
            [
                float(row["force_left_x"]),
                float(row["force_left_y"]),
                float(row["force_left_z"]),
                float(row["torque_left_x"]),
                float(row["torque_left_y"]),
                float(row["torque_left_z"]),
            ],
            dtype=np.float32,
        ),
    )

    right_force = _load_csv_pairs(
        csv_root / "force_torque_right.csv",
        lambda row: np.array(
            [
                float(row["force_right_x"]),
                float(row["force_right_y"]),
                float(row["force_right_z"]),
                float(row["torque_right_x"]),
                float(row["torque_right_y"]),
                float(row["torque_right_z"]),
            ],
            dtype=np.float32,
        ),
    )

    joint_states = _load_csv_pairs(
        csv_root / "joint_states.csv",
        lambda row: {
            "pos": _reorder_joints_to_feature_order(_parse_array(row["joint_positions"], expected_len=6)), # reorder states only, do not include velocities
        },
    )

    controller_state = _load_csv_pairs(
        csv_root / "scaled_joint_trajectory_controller_controller_state.csv",
        lambda row: {
            "ref_pos": _parse_array(row["reference_joint_positions"], expected_len=6), # DO NOT reorder actions, do not include velocities
        },
    )

    gripper_goal = _load_csv_pairs(
        csv_root / "robotiq_2f_gripper_action_goal.csv",
        # Normalize so fully open -> 0, fully closed -> 1.
        lambda row: (GRIPPER_ACTION_OPEN - float(json.loads(row["gripper_goal"])[0]))
        / (GRIPPER_ACTION_OPEN - GRIPPER_ACTION_CLOSE),
    )

    finger_distance = _load_csv_pairs(
        csv_root / "robotiq_2f_gripper_finger_distance_mm.csv",
        # Normalize so fully open -> 0, fully closed -> 1.
        lambda row: (GRIPPER_ACTION_OPEN - float(row["gripper_finger_distance_mm"]) / 1000.0)
        / (GRIPPER_ACTION_OPEN - GRIPPER_ACTION_CLOSE),
    )

    return {
        "left_force": left_force,
        "right_force": right_force,
        "joint_states": joint_states,
        "controller_state": controller_state,
        "gripper_goal": gripper_goal,
        "finger_distance": finger_distance,
    }


def _determine_time_range(streams, cameras):
    first_stamps = []
    last_stamps = []
    for _, data in streams.items():
        stamps = [s for s, _ in data]
        first, last = _first_last(stamps)
        first_stamps.append(first)
        last_stamps.append(last)

    for name, frames in cameras.items():
        stamps = [s for s, _ in frames]
        first, last = _first_last(stamps)
        first_stamps.append(first)
        last_stamps.append(last)

    start = max(first_stamps)
    end = min(last_stamps)
    return start, end


def _convert_session(session_dir: Path, dataset: LeRobotDataset, cfg: ConvertConfig):
    prompt = (session_dir / "prompt.txt").read_text().strip()

    camera_fixed_dir = _find_image_dir(
        session_dir,
        [
            "camera_fixed_realsense2_camera_color_image_raw",
            "camera_camera_fixed_color_image_raw_compressed",
        ],
    )
    camera_wrist_dir = _find_image_dir(
        session_dir,
        [
            "camera_wrist_realsense2_camera_color_image_raw",
            "camera_camera_wrist_color_image_raw_compressed",
        ],
    )

    camera_fixed = _load_frames(camera_fixed_dir)
    camera_wrist = _load_frames(camera_wrist_dir)
    if not camera_fixed or not camera_wrist:
        raise FileNotFoundError(f"Missing camera frames in {session_dir}")

    streams = _prepare_streams(session_dir)
    start, end = _determine_time_range(streams, {"fixed": camera_fixed, "wrist": camera_wrist})

    step_ns = int(1e9 / cfg.fps)
    action_step_ns = int(1e9 / cfg.action_hz)
    horizon_span = action_step_ns * cfg.action_horizon
    effective_end = end - horizon_span
    if effective_end <= start:
        logging.warning("Skipping %s: insufficient future horizon", session_dir)
        return
    time_grid = range(start, effective_end + 1, step_ns)

    # Pointers and last-values for carry-forward.
    idx = {k: 0 for k in streams.keys()}
    last = {k: None for k in streams.keys()}
    cam_idx = {"fixed": 0, "wrist": 0}
    cam_last = {"fixed": None, "wrist": None}

    frames_written = 0
    skipped = 0

    ctrl_stamps = [s for s, _ in streams["controller_state"]]
    ctrl_vals = [v["ref_pos"] for _, v in streams["controller_state"]]
    goal_stamps = [s for s, _ in streams["gripper_goal"]]
    goal_vals = [v for _, v in streams["gripper_goal"]]

    for t in tqdm(time_grid, desc=f"Session {session_dir.name}"):
        for key, data in streams.items():
            while idx[key] < len(data) and data[idx[key]][0] <= t:
                last[key] = data[idx[key]][1]
                idx[key] += 1

        for key, data in {"fixed": camera_fixed, "wrist": camera_wrist}.items():
            while cam_idx[key] < len(data) and data[cam_idx[key]][0] <= t:
                cam_last[key] = data[cam_idx[key]][1]
                cam_idx[key] += 1

        if any(v is None for v in last.values()) or any(v is None for v in cam_last.values()):
            skipped += 1
            continue

        joint = last["joint_states"]

        state_vec = np.concatenate(
            [
                joint["pos"],
                np.array([last["finger_distance"]], dtype=np.float32),
            ],
            dtype=np.float32,
        )

        # Build future action horizon as deltas from current state.
        action_seq = np.zeros((cfg.action_horizon, 7), dtype=np.float32)
        missing_action = False
        for i in range(cfg.action_horizon):
            target_t = t + (i + 1) * action_step_ns
            idx_ctrl = bisect.bisect_right(ctrl_stamps, target_t) - 1
            idx_goal = bisect.bisect_right(goal_stamps, target_t) - 1
            if idx_ctrl < 0 or idx_goal < 0:
                missing_action = True
                break
            ref_pos = ctrl_vals[idx_ctrl]
            goal = goal_vals[idx_goal]
            # deltas: desired minus current state except for gripper
            action_seq[i, :6] = ref_pos - joint["pos"]
            action_seq[i, 6] = goal

        if missing_action:
            skipped += 1
            continue

        fixed_img = _load_rgb(cam_last["fixed"])
        wrist_img = _load_rgb(cam_last["wrist"])
        if fixed_img is None or wrist_img is None:
            skipped += 1
            continue

        frame = {
            "images.cam_fixed": fixed_img,
            "images.cam_wrist": wrist_img,
            "state": state_vec,
            "left_ft": last["left_force"],
            "right_ft": last["right_force"],
            "actions": action_seq,
        }

        dataset.add_frame(frame, task=prompt)
        frames_written += 1

    dataset.save_episode()
    logging.info(
        f"{session_dir.name}: wrote {frames_written} frames (skipped {skipped} steps with missing data)"
    )


def main(cfg: ConvertConfig):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sessions = sorted([p for p in cfg.data_root.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No sessions found under {cfg.data_root}")

    state_names, action_names = _feature_name_lists()

    # Determine image shapes from the first session (first frame of each camera).
    first_session = sessions[0]
    fixed_dir = _find_image_dir(
        first_session,
        [
            "camera_fixed_realsense2_camera_color_image_raw",
            "camera_camera_fixed_color_image_raw_compressed",
        ],
    )
    wrist_dir = _find_image_dir(
        first_session,
        [
            "camera_wrist_realsense2_camera_color_image_raw",
            "camera_camera_wrist_color_image_raw_compressed",
        ],
    )
    fixed_img = _load_rgb(_load_frames(fixed_dir)[0][1])
    wrist_img = _load_rgb(_load_frames(wrist_dir)[0][1])
    img_shapes = {
        "fixed": fixed_img.shape,
        "wrist": wrist_img.shape,
    }

    dataset, dataset_path = _create_dataset(cfg.repo_id, fps=cfg.fps, img_shapes=img_shapes, state_names=state_names, action_names=action_names, action_horizon=cfg.action_horizon)

    for session_dir in sessions:
        if not (session_dir / "done.txt").exists():
            logging.warning("Skipping %s: missing done.txt (decode likely failed)", session_dir.name)
            continue
        _convert_session(session_dir, dataset, cfg)

    if cfg.push_to_hub:
        dataset.push_to_hub(
            tags=["VLA", "ur3", "robotiq", "force-torque", "insertion"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    else:
        target_root = cfg.data_root.resolve().parent / cfg.local_output_dirname
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(dataset_path, target_root)
        logging.info(f"Saved local LeRobot dataset to {target_root}")


if __name__ == "__main__":
    tyro.cli(main)
