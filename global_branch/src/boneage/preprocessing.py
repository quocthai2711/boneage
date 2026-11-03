"""Deterministic evaluation and conservative training preprocessing."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def percentile_normalization(image: np.ndarray, lower: float = 1, upper: float = 99) -> np.ndarray:
    pixels = image.astype(np.float32)
    p_low, p_high = np.percentile(pixels, [lower, upper])
    if p_high <= p_low:
        raise ValueError("Image has no usable intensity range")
    normalized = np.clip((pixels - p_low) / (p_high - p_low), 0, 1)
    return np.rint(normalized * 255).astype(np.uint8)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def conservative_foreground_crop(image: np.ndarray, margin_fraction: float = 0.08) -> np.ndarray:
    """Crop obvious empty borders while retaining a generous wrist/fingertip margin."""
    threshold = max(5, int(np.percentile(image, 30)))
    mask = (image > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return image
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component]
    if area < image.size * 0.05:
        return image
    margin = int(max(width, height) * margin_fraction)
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image.shape[1], x + width + margin)
    bottom = min(image.shape[0], y + height + 2 * margin)
    crop = image[top:bottom, left:right]
    return crop if crop.size else image


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    return TF.pad(image, (left, top, side - width - left, side - height - top), fill=0)


class BoneAgePreprocess:
    VALID_MODES = {"P0", "P1", "P2"}

    def __init__(self, mode: str = "P1", target_size: int = 224, is_train: bool = False) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        if target_size < 64:
            raise ValueError("target_size must be at least 64")
        self.mode = mode
        self.target_size = target_size
        self.is_train = is_train

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.finalize(self.base_image(image))

    def base_image(self, image: Image.Image) -> Image.Image:
        """Deterministic part of the pipeline (safe to cache); no augmentation."""
        pixels = np.asarray(image.convert("L"))
        pixels = percentile_normalization(pixels)
        if self.mode in {"P1", "P2"}:
            pixels = apply_clahe(pixels)
        if self.mode == "P2":
            pixels = conservative_foreground_crop(pixels)

        output = pad_to_square(Image.fromarray(pixels))
        return TF.resize(
            output, (self.target_size, self.target_size), interpolation=Image.Resampling.BILINEAR
        )

    def finalize(self, output: Image.Image) -> torch.Tensor:
        """Train-time augmentation (if enabled) followed by tensor conversion."""
        if self.is_train:
            output = TF.affine(
                output,
                angle=random.uniform(-10, 10),
                translate=(
                    int(random.uniform(-0.04, 0.04) * self.target_size),
                    int(random.uniform(-0.04, 0.04) * self.target_size),
                ),
                scale=random.uniform(0.94, 1.06),
                shear=(0.0, 0.0),
                fill=0,
            )
            output = TF.adjust_gamma(output, gamma=random.uniform(0.85, 1.15))

        tensor = TF.to_tensor(output.convert("RGB"))
        return TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
