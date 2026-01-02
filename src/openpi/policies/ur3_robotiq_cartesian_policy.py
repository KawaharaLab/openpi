"""Transforms for running Pi0/Pi05 on a UR3 arm with a Robotiq gripper (Cartesian state/actions)."""

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
    "ee_quat_x",
    "ee_quat_y",
    "ee_quat_z",
    "ee_quat_w",
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

    images = {
        "cam_high": load_image("fixed_image"),
        "cam_left_wrist": load_image("wrist_image"),
    }

    return {
        "state": state,
        "images": images,
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
        images = data.get("images", {})

        base_img = _prepare_image(images.get(self.IMAGE_KEYS["base"]))
        wrist_img = _prepare_image(images.get(self.IMAGE_KEYS["wrist"]), fallback=base_img)

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

        output = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask,
        }

        if "actions" in data:
            output["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            output["prompt"] = data["prompt"]

        return output


@dataclasses.dataclass(frozen=True)
class Ur3RobotiqOutputs(transforms.DataTransformFn):
    """Extract the UR3 Cartesian slice from Pi0 action chunks."""

    action_dims: int = 8

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
