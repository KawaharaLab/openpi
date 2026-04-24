import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


STATE_SLICE = slice(7, 13)
ACTION_DIM = STATE_SLICE.stop - STATE_SLICE.start


def make_aic_example() -> dict:
    """Creates a random input example for the AIC policy."""
    return {
        "observation.images.left_camera": np.random.randint(256, size=(1024, 1152, 3), dtype=np.uint8),
        "observation.images.center_camera": np.random.randint(256, size=(1024, 1152, 3), dtype=np.uint8),
        "observation.images.right_camera": np.random.randint(256, size=(1024, 1152, 3), dtype=np.uint8),
        "observation.state": np.random.rand(26).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _slice_state(state) -> np.ndarray:
    state = np.asarray(state)
    return state[..., STATE_SLICE]


def _slice_actions(actions) -> np.ndarray:
    actions = np.asarray(actions)
    return actions[..., STATE_SLICE]


@dataclasses.dataclass(frozen=True)
class AICInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.center_camera"])
        left_image = _parse_image(data["observation.images.left_camera"])
        right_image = _parse_image(data["observation.images.right_camera"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, left_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, left_image, right_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": _slice_state(data["observation.state"]),
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = _slice_actions(data["actions"])
        elif "action" in data:
            inputs["actions"] = _slice_actions(data["action"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AICOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
