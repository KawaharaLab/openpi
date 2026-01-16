"""Transforms for running Pi0/Pi05 on a UR3 arm with a Robotiq gripper (position-only Cartesian state/actions)."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model


_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "data" / "example"
_EXAMPLE_CSV = _EXAMPLE_ROOT / "example.csv"
_STATE_KEYS = (
    "ee_pos_x",
    "ee_pos_y",
    "ee_pos_z",
    "robotiq_finger_distance",
)


def make_ur3_example() -> dict:
    """Create a representative observation for sanity checks."""

    if not _EXAMPLE_CSV.exists():
        raise FileNotFoundError(f"UR3 example CSV not found: {_EXAMPLE_CSV}")

    with _EXAMPLE_CSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            row = next(reader)
        except StopIteration as exc:
            raise ValueError("UR3 example CSV is empty") from exc

    state = np.array([float(row[key]) for key in _STATE_KEYS], dtype=np.float32)

    def load_image(column: str) -> np.ndarray:
        rel_path = row[column]
        image_path = (_EXAMPLE_ROOT / rel_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Example image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def load_force_torque(column: str) -> np.ndarray:
        rel_path = row[column]
        ft_path = (_EXAMPLE_ROOT / rel_path).resolve()
        if not ft_path.is_file():
            raise FileNotFoundError(f"Example force/torque data not found: {ft_path}")
        return np.loadtxt(ft_path, delimiter=",", dtype=np.float32)
    images = {
        "cam_high": load_image("fixed_image"),
        "cam_left_wrist": load_image("wrist_image"),
    }

    force_torques = {
        "left": load_force_torque("left_ft"),
        "right": load_force_torque("right_ft"),
    }

    return {
        "state": state,
        "images": images,
        "force_torques": force_torques,
        "prompt": "Pick up the blue cube.",
    }


@dataclasses.dataclass(frozen=True)
class Ur3RobotiqInputs(transforms.DataTransformFn):
    """Map UR3 observations (Cartesian) into the Pi0/Pi05 observation schema."""

    model_type: _model.ModelType = _model.ModelType.PI0

    IMAGE_KEYS: ClassVar[dict[str, str]] = {
        "base": "cam_high",
        "wrist": "cam_left_wrist",
    }

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] >= 4:
            # Keep xyz and gripper; drop orientation components if present.
            state = np.concatenate([state[..., :3], state[..., -1:]], axis=-1)
        images = data.get("images", {})

        base_img = _prepare_image(images.get(self.IMAGE_KEYS["base"]))
        wrist_img = _prepare_image(images.get(self.IMAGE_KEYS["wrist"]), fallback=base_img)

        left_ft = _prepare_ft(data.get("force_torques", {}).get("left"))
        right_ft = _prepare_ft(data.get("force_torques", {}).get("right"))
        # Pi0 reserves two wrist slots. We populate the left slot with the real wrist camera
        # and keep the right slot as an all-zero image with its mask cleared so the model
        # ignores it.
        right_placeholder = np.zeros_like(base_img)

        image_dict = {
            "base_0_rgb": base_img,
            "left_wrist_0_rgb": wrist_img,
            "right_wrist_0_rgb": right_placeholder,
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }
        ft_dict = {
            "left_ft": left_ft,
            "right_ft": right_ft,
        }
        output = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask,
            "force_torques": ft_dict,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] >= 4:
                actions = np.concatenate([actions[..., :3], actions[..., -1:]], axis=-1)
            output["actions"] = actions
        if "prompt" in data:
            output["prompt"] = data["prompt"]

        return output


@dataclasses.dataclass(frozen=True)
class Ur3RobotiqOutputs(transforms.DataTransformFn):
    """Extract the UR3 Cartesian slice from Pi0 action chunks."""

    action_dims: int = 4

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions[..., : self.action_dims]}


def _prepare_image(image: np.ndarray | None, *, fallback: np.ndarray | None = None) -> np.ndarray:
    if image is None:
        if fallback is None:
            raise ValueError("Missing required image and no fallback provided")
        return np.zeros_like(fallback)

    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")

    # Convert to HWC if inputs arrive as CHW.
    if arr.shape[0] in (1, 3) and arr.shape[0] <= 4 and arr.shape[0] != arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        # If values look normalized to [0, 1], scale back to [0, 255] before casting.
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return arr

def _prepare_ft(ft: np.ndarray | None) -> np.ndarray:
    """Normalize force/torque to shape (6, 80) and forward/backward fill missing values.

    Expected input: 6 channels over 80 timesteps. If values are missing (None/NaN),
    fill from the nearest available sample: first forward-fill, then backward-fill
    to cover any leading gaps. If all samples are missing for a channel, zeros are used.
    """

    def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        # Forward fill
        last = np.nan
        for i in range(out.shape[0]):
            if np.isfinite(out[i]):
                last = out[i]
            elif np.isfinite(last):
                out[i] = last
        # Backward fill for leading gaps
        last = np.nan
        for i in range(out.shape[0] - 1, -1, -1):
            if np.isfinite(out[i]):
                last = out[i]
            elif np.isfinite(last):
                out[i] = last
        return out

    horizon = getattr(_model, "FT_HORIZON", None)
    horizon = 200 if horizon is None else int(horizon)

    if ft is None:
        return np.zeros((6, horizon), dtype=np.float32)

    arr = np.asarray(ft, dtype=np.float32)

    # Accept (horizon, 6) and transpose to (6, horizon)
    if arr.shape == (horizon, 6):
        arr = arr.T
    # Accept (6,) by broadcasting across time (unlikely but more forgiving)
    if arr.shape == (6,):
        arr = np.repeat(arr[:, None], horizon, axis=1)

    if arr.shape != (6, horizon):
        raise ValueError(f"Expected force/torque array of shape (6, {horizon}), got {arr.shape}")

    filled = np.empty_like(arr)
    for c in range(6):
        channel = arr[c]
        if not np.isfinite(channel).any():
            filled[c] = 0.0
            continue
        filled[c] = _fill_nan_1d(channel)

    return filled
