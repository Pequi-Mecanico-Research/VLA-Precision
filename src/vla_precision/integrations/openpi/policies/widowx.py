import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_widowx_example() -> dict:
    """Creates a random input example for the WidowX AI policy."""
    return {
        # 14D: 7 joint positions then 7 external efforts — see WidowXRobot.observations() and
        # docs/widowx-setup.md. The upstream openpi fork's own version of this helper uses a
        # misleading 7D example; don't copy that mistake here.
        "observation/state": np.random.rand(14),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/low_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WidowXInputs(transforms.DataTransformFn):
    """
    Converts WidowX AI dataset/robot inputs to the format expected by the model. Used for both
    training and inference.

    Unlike UR5e/Franka, all three of the model's image slots are real cameras here (cam_high,
    cam_wrist, cam_low) — there is no padded/masked slot.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])  # cam_high
        wrist_image = _parse_image(data["observation/wrist_image"])  # cam_wrist
        low_image = _parse_image(data["observation/low_image"])  # cam_low

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": low_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class WidowXOutputs(transforms.DataTransformFn):
    """
    Converts model outputs back to the WidowX AI action space. Used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # WidowX AI action space is 7D: joint_0..joint_5 plus the gripper
        # (left_carriage_joint), unpadded from the model's action dimension.
        return {"actions": np.asarray(data["actions"][:, :7])}
