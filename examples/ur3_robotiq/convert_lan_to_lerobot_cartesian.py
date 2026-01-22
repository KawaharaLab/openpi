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
from collections import deque
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro

GRIPPER_ACTION_OPEN = 0.140  # m
GRIPPER_ACTION_CLOSE = 0.002  # m
FT_HORIZON = 1  # number of recent force/torque samples to retain per frame

@dataclasses.dataclass
class ConvertConfig:
    data_root: Path = Path("../data/lan")
    repo_id: str = "uzumi-bi/lan_ur3_cartesian"
    fps: float = 5.0  # fixed 5 Hz resampling
    action_horizon: int = 50  # number of future actions to include
    action_hz: float = 20.0  # action sampling rate (Hz) for horizon
    push_to_hub: bool = False
    local_output_dirname: str = "lan_ur3_lerobot_cartesian"
    ft_horizon: int = 1  # number of recent force/torque samples to retain per frame


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
    # Cartesian end-effector pose (position + orientation quaternion) plus gripper
    state_names: list[str] = [
        "ee_pos.x",
        "ee_pos.y",
        "ee_pos.z",
        "ee_quat.x",
        "ee_quat.y",
        "ee_quat.z",
        "ee_quat.w",
        "finger_distance_m",
    ]

    # Actions are target ee pose (abs) plus gripper goal
    action_names: list[str] = [
        "ee_target.x",
        "ee_target.y",
        "ee_target.z",
        "ee_target.qx",
        "ee_target.qy",
        "ee_target.qz",
        "ee_target.qw",
        "gripper_goal_m",
    ]
    return state_names, action_names


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _fk_ur3_pose(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute UR3 tool pose (position [m], quaternion [xyzw]) from 6 joint angles (rad)."""
    if joints.shape[-1] != 6:
        raise ValueError(f"Expected 6 joint values, got shape {joints.shape}")
    # Standard UR3 DH parameters (meters, radians)
    a = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0], dtype=np.float32)
    d = np.array([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819], dtype=np.float32)
    alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    for i in range(6):
        T = T @ _dh_transform(a[i], alpha[i], d[i], joints[i])

    pos = T[:3, 3]

    # Rotation matrix to quaternion (xyzw)
    R = T[:3, :3]
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    # Normalize to guard against numerical drift
    quat = quat / np.linalg.norm(quat)
    return pos, quat


def _create_dataset(
    repo_id: str,
    fps: float,
    img_shapes: dict[str, tuple[int, int, int]],
    state_names: list[str],
    action_names: list[str],
    action_horizon: int,
    ft_horizon: int,
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
            "shape": (6, ft_horizon),
            "names": ["component", "time"],
        },
        "right_ft": {
            "dtype": "float32",
            "shape": (6, ft_horizon),
            "names": ["component", "time"],
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
            "pos": _reorder_joints_to_feature_order(_parse_array(row["joint_positions"], expected_len=6)),  # reorder states only, do not include velocities
        },
    )

    controller_state = _load_csv_pairs(
        csv_root / "scaled_joint_trajectory_controller_controller_state.csv",
        lambda row: {
            "ref_pos": _parse_array(row["reference_joint_positions"], expected_len=6),  # DO NOT reorder actions, do not include velocities
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
    ft_hist = {"left_force": deque(maxlen=cfg.ft_horizon), "right_force": deque(maxlen=cfg.ft_horizon)}
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
                sample = data[idx[key]][1]
                last[key] = sample
                if key in ft_hist:
                    ft_hist[key].append(sample)
                idx[key] += 1

        for key, data in {"fixed": camera_fixed, "wrist": camera_wrist}.items():
            while cam_idx[key] < len(data) and data[cam_idx[key]][0] <= t:
                cam_last[key] = data[cam_idx[key]][1]
                cam_idx[key] += 1

        if any(v is None for v in last.values()) or any(v is None for v in cam_last.values()):
            skipped += 1
            continue

        joint = last["joint_states"]

        ee_pos, ee_quat = _fk_ur3_pose(joint["pos"])
        finger_dist = np.array([last["finger_distance"]], dtype=np.float32)
        state_vec = np.concatenate(
            [
                ee_pos,
                ee_quat,
                finger_dist,
            ],
            dtype=np.float32,
        )

        # Build future action horizon: pos/quat as deltas, gripper as absolute open distance.
        action_seq = np.zeros((cfg.action_horizon, 8), dtype=np.float32)
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
            ee_target_pos, ee_target_quat = _fk_ur3_pose(ref_pos)
            action_seq[i, 0:3] = ee_target_pos - ee_pos
            action_seq[i, 3:7] = ee_target_quat - ee_quat
            action_seq[i, 7] = goal

        if missing_action:
            skipped += 1
            continue

        fixed_img = _load_rgb(cam_last["fixed"])
        wrist_img = _load_rgb(cam_last["wrist"])
        if fixed_img is None or wrist_img is None:
            skipped += 1
            continue

        def _stack_ft(history: deque[np.ndarray]) -> np.ndarray:
            # Stack last N samples; if insufficient history (e.g., first ~3s),
            # front-fill with the oldest available sample instead of zeros.
            if not history:
                return np.zeros((6, cfg.ft_horizon), dtype=np.float32)

            tail = np.asarray(history, dtype=np.float32)
            tail = tail[-cfg.ft_horizon :]
            if tail.shape[0] < cfg.ft_horizon:
                pad_len = cfg.ft_horizon - tail.shape[0]
                pad = np.repeat(tail[0:1], pad_len, axis=0)  # repeat oldest sample
                tail = np.concatenate([pad, tail], axis=0)
            return tail.T  # (6, horizon)

        frame = {
            "images.cam_fixed": fixed_img,
            "images.cam_wrist": wrist_img,
            "state": state_vec,
            "left_ft": _stack_ft(ft_hist["left_force"]),
            "right_ft": _stack_ft(ft_hist["right_force"]),
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

    dataset, dataset_path = _create_dataset(
        cfg.repo_id,
        fps=cfg.fps,
        img_shapes=img_shapes,
        state_names=state_names,
        action_names=action_names,
        action_horizon=cfg.action_horizon,
        ft_horizon=cfg.ft_horizon,
    )

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
