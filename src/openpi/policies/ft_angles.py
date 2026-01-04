"""Transforms for UR3 + Robotiq with joint-angle state/actions and force-torque inputs."""

from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model


# Image key mapping (module-level to match other policies)
_IMAGE_KEYS = {
    "base": "cam_high",
    "wrist": "cam_left_wrist",
}


@dataclasses.dataclass(frozen=True)
class Ur3RobotiqInputs(transforms.DataTransformFn):
    """Map UR3 observations (joint space) with FT into the Pi0/Pi05 schema."""

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        images = data.get("images", {})

        base_img = _prepare_image(images.get(_IMAGE_KEYS["base"]))
        wrist_img = _prepare_image(images.get(_IMAGE_KEYS["wrist"]), fallback=base_img)

        left_ft = _prepare_ft(data.get("force_torques", {}).get("left"))
        right_ft = _prepare_ft(data.get("force_torques", {}).get("right"))

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
            output["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            output["prompt"] = data["prompt"]

        return output


@dataclasses.dataclass(frozen=True)
class Ur3RobotiqOutputs(transforms.DataTransformFn):
    """Extract the UR3 joint action slice from Pi0 action chunks."""

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

    if arr.shape[0] in (1, 3) and arr.shape[0] <= 4 and arr.shape[0] != arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return arr


def _prepare_ft(ft: np.ndarray | None) -> np.ndarray:
    """Normalize force/torque to shape (6, horizon) with forward/backward fill."""

    def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        last = np.nan
        for i in range(out.shape[0]):
            if np.isfinite(out[i]):
                last = out[i]
            elif np.isfinite(last):
                out[i] = last
        last = np.nan
        for i in range(out.shape[0] - 1, -1, -1):
            if np.isfinite(out[i]):
                last = out[i]
            elif np.isfinite(last):
                out[i] = last
        return out

    horizon = getattr(_model, "FT_HORIZON", None)
    horizon = 300 if horizon is None else int(horizon)

    if ft is None:
        return np.zeros((6, horizon), dtype=np.float32)

    arr = np.asarray(ft, dtype=np.float32)

    if arr.shape == (horizon, 6):
        arr = arr.T
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
