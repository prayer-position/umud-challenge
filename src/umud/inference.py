from pathlib import Path

import cv2
import numpy as np
import torch

from umud.data.transforms import build_val_transform
from umud.models.unet import build_model_from_config


def load_model(checkpoint_path: Path, config: dict, device: torch.device) -> torch.nn.Module:
    # Pretrained encoder weights are overwritten by the checkpoint anyway.
    model = build_model_from_config(config, encoder_weights=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


def prepare_input(image: np.ndarray, image_size: tuple[int, int]) -> torch.Tensor:
    """Resize + normalize exactly as MaskDataset does for training."""
    h, w = image_size
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return build_val_transform(image_size)(image=resized)["image"]


@torch.no_grad()
def predict_prob(model: torch.nn.Module, batch: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Foreground probabilities (N, H, W) for a normalized (N, 3, H, W) batch.
    With tta, averages in the horizontally flipped prediction."""
    with torch.autocast(device_type=batch.device.type, enabled=batch.device.type == "cuda"):
        prob = torch.sigmoid(model(batch).float())
        if tta:
            flipped = torch.sigmoid(model(batch.flip(-1)).float()).flip(-1)
            prob = (prob + flipped) / 2
    return prob[:, 0]


def upsample_and_threshold(prob: np.ndarray, out_shape: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    """Resize a probability map to `out_shape` before thresholding, so thin
    structures come back as smooth lines rather than blocky pixels."""
    prob = cv2.resize(prob, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8)


def predict_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    config: dict,
    device: torch.device,
    out_shape: tuple[int, int] | None = None,
    tta: bool = False,
    threshold: float = 0.5,
) -> np.ndarray:
    """Binary mask for one RGB image, at `out_shape` (default: the image's
    own size, so pixel measurements are comparable across resolutions)."""
    batch = prepare_input(image, tuple(config["image_size"])).unsqueeze(0).to(device)
    prob = predict_prob(model, batch, tta=tta)[0].cpu().numpy()
    return upsample_and_threshold(prob, out_shape or image.shape[:2], threshold)
