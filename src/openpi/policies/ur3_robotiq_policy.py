"""Transforms for running Pi0 on a UR3 arm with a Robotiq gripper."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur3_example() -> dict:
    """Create a representative observation for sanity checks."""

    return {
        "state": np.zeros(7, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "prompt": "Pick up the object",
    }


@dataclasses.dataclass(frozen=True)
class Ur3RobotiqInputs(transforms.DataTransformFn):
    """Map UR3 observations into the Pi0 observation schema."""

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
    """Extract the UR3-relevant slice from Pi0 action chunks."""

    action_dims: int = 7

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
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return arr
